import numpy as np
import pytest
import torch

from multipose2 import vit_sam
from multipose2.roles import ROLES, RoleMixer


MODALITIES = [("he", (0, 1, 2)), ("uchl1", (3,)), ("fabp7", (4,))]


def _mixer(**kw):
    torch.manual_seed(0)
    return RoleMixer(MODALITIES, **kw)


def _inputs(n=2, c=5, s=32):
    torch.manual_seed(1)
    return torch.randn(n, c, s, s)


def _base(n=2, s=32):
    torch.manual_seed(2)
    return torch.randn(n, 3, s, s)


def test_untrained_mixer_leaves_the_base_prediction_effectively_unchanged():
    """The parity property: an untrained mixer must not move the baseline."""
    mixer, base, x = _mixer(), _base(), _inputs()
    with torch.no_grad():
        out = mixer(base, x)
    # signed coefficients start at exactly zero; non-negative ones at
    # softplus(-10) = 4.5e-5, so the base shifts only by that much
    assert torch.allclose(out, base, atol=1e-3)


def test_disabled_mixer_is_exactly_a_no_op():
    mixer, base, x = _mixer(), _base(), _inputs()
    mixer.enabled = False
    with torch.no_grad():
        assert torch.equal(mixer(base, x), base)


def test_veto_can_only_subtract_from_cell_probability():
    """Necessity, which an additive mixture cannot express."""
    mixer = _mixer(roles=("veto",))
    with torch.no_grad():
        mixer.coefficients["uchl1__veto"].fill_(2.0)  # softplus(2) ~ 2.13
        base, x = _base(), _inputs()
        out = mixer(base, x)
    delta = (out - base)[:, -1:]
    assert (delta <= 1e-6).all(), "a veto term must never add existence evidence"
    assert delta.min() < -1e-3, "and it must actually subtract"
    # flows untouched by an existence role
    assert torch.allclose(out[:, -3:-1], base[:, -3:-1])


def test_support_is_signed_and_touches_only_cell_probability():
    mixer = _mixer(roles=("support",))
    with torch.no_grad():
        mixer.coefficients["uchl1__support"].fill_(1.0)
        base, x = _base(), _inputs()
        out = mixer(base, x)
    delta = (out - base)[:, -1:]
    assert delta.max() > 0 and delta.min() < 0, "support must be able to add or subtract"
    assert torch.allclose(out[:, -3:-1], base[:, -3:-1])


def test_potential_and_edge_touch_only_the_flow_field():
    for role in ("potential", "edge"):
        mixer = _mixer(roles=(role,))
        with torch.no_grad():
            mixer.coefficients[f"fabp7__{role}"].fill_(1.5)
            base, x = _base(), _inputs()
            out = mixer(base, x)
        assert not torch.allclose(out[:, -3:-1], base[:, -3:-1]), role
        assert torch.allclose(out[:, -1:], base[:, -1:]), role


def test_potential_sign_flips_the_flow_direction():
    """A positive coefficient attracts, a negative one repels."""
    mixer = _mixer(roles=("potential",))
    base, x = _base(), _inputs()
    with torch.no_grad():
        mixer.coefficients["uchl1__potential"].fill_(1.0)
        attract = mixer(base, x) - base
        mixer.coefficients["uchl1__potential"].fill_(-1.0)
        repel = mixer(base, x) - base
    assert torch.allclose(attract[:, -3:-1], -repel[:, -3:-1], atol=1e-5)


def test_coefficients_receive_gradient_from_a_zero_start():
    """A zero-initialised coefficient must still be able to grow."""
    mixer = _mixer()
    base, x = _base(), _inputs()
    mixer(base, x).square().mean().backward()
    for name in ("he", "uchl1", "fabp7"):
        for role in ROLES:
            g = mixer.coefficients[f"{name}__{role}"].grad
            assert g is not None and torch.isfinite(g).all()
            assert g.abs() > 0, f"{name}/{role} coefficient has no gradient"


def test_coefficient_table_reports_constrained_and_signed_roles():
    mixer = _mixer()
    with torch.no_grad():
        mixer.coefficients["uchl1__veto"].fill_(3.0)
        mixer.coefficients["fabp7__potential"].fill_(-2.0)
    table = mixer.coefficient_table()
    assert set(table) == {"he", "uchl1", "fabp7"}
    assert table["uchl1"]["veto"] == pytest.approx(torch.nn.functional.softplus(
        torch.tensor(3.0)).item())
    assert table["fabp7"]["potential"] == pytest.approx(-2.0)
    # non-negative roles can never report a negative value
    assert all(t["veto"] >= 0 and t["edge"] >= 0 for t in table.values())


def test_l1_is_zero_ish_at_init_and_grows_with_use():
    mixer = _mixer()
    start = float(mixer.l1().detach())
    with torch.no_grad():
        mixer.coefficients["he__support"].fill_(4.0)
    assert float(mixer.l1().detach()) > start + 3.9


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError, match="unknown roles"):
        RoleMixer(MODALITIES, roles=("veto", "vibes"))
    with pytest.raises(ValueError, match="at least one role"):
        RoleMixer(MODALITIES, roles=())


def test_resolution_mismatch_is_rejected():
    mixer = _mixer()
    with pytest.raises(ValueError, match="full resolution"):
        mixer(_base(s=32), _inputs(s=64))


def test_transformer_without_a_mixer_is_unchanged():
    net = vit_sam.Transformer(in_channels=5, bsize=64)
    net.eval()
    x = torch.randn(1, 5, 64, 64)
    with torch.no_grad():
        before = net(x)[0]
        net.set_role_mixer(RoleMixer(MODALITIES, nout=3))
        net.role_mixer.enabled = False
        after = net(x)[0]
    assert torch.equal(before, after), "a disabled mixer must not alter forward()"


def test_transformer_with_an_untrained_mixer_stays_near_baseline():
    net = vit_sam.Transformer(in_channels=5, bsize=64)
    net.eval()
    x = torch.randn(1, 5, 64, 64)
    with torch.no_grad():
        before = net(x)[0]
        net.set_role_mixer(RoleMixer(MODALITIES, nout=3))
        after = net(x)[0]
    assert torch.allclose(before, after, atol=1e-3)


def test_coefficient_is_the_only_carrier_of_scale():
    """Closes a scale degeneracy that made the coefficient meaningless.

    With an unbounded branch, coefficient * branch(x) can hold any magnitude at
    any coefficient, so an L1 penalty drives the coefficient toward zero while
    the branch grows to compensate. Squashing the branch to [-1, 1] bounds each
    term by its coefficient, so the table reports real magnitudes.
    """
    for role, bound in (("support", 1.0), ("edge", 1.0)):
        mixer = _mixer(roles=(role,))
        base, x = _base(), _inputs()
        with torch.no_grad():
            if role in ("veto", "edge"):
                mixer.coefficients[f"uchl1__{role}"].fill_(-10.0)  # ~0
            coef = 0.5
            raw = torch.log(torch.expm1(torch.tensor(coef))) if role == "edge" \
                else torch.tensor(coef)
            mixer.coefficients[f"uchl1__{role}"].copy_(raw)
            small = (mixer(base, x) - base).abs().max()
            # blow the branch weights up 100x; the term must not follow
            for prm in mixer.branches[f"uchl1__{role}"].parameters():
                prm.mul_(100.0)
            large = (mixer(base, x) - base).abs().max()
        effective = float(mixer.coefficient("uchl1", role).detach())
        assert large <= effective * bound + 1e-4, (
            f"{role}: term grew to {large:.4f} with coefficient {effective:.4f}")
        assert small <= effective * bound + 1e-4


def test_veto_magnitude_is_bounded_by_gain_times_coefficient():
    from multipose2.roles import VETO_GAIN
    mixer = _mixer(roles=("veto",))
    base, x = _base(), _inputs()
    with torch.no_grad():
        mixer.coefficients["uchl1__veto"].fill_(2.0)
        for prm in mixer.branches["uchl1__veto"].parameters():
            prm.mul_(100.0)
        delta = (mixer(base, x) - base)[:, -1:]
    coef = float(mixer.coefficient("uchl1", "veto").detach())
    assert (delta <= 1e-6).all()
    assert delta.abs().max() <= coef * (VETO_GAIN + 0.1)


# --- trainable modes and sparsity penalties ---------------------------------

def _net_with_mixer(nchan=5):
    net = vit_sam.Transformer(in_channels=nchan, bsize=64)
    net.set_role_mixer(RoleMixer(MODALITIES, nout=3))
    return net


def test_roles_only_mode_trains_just_the_mixer():
    from multipose2 import train
    net = _net_with_mixer()
    train.set_trainable_parameters(net, trainable_mode="roles_only")
    names = [n for n, p in net.named_parameters() if p.requires_grad]
    assert names, "roles_only selected nothing"
    assert all(n.startswith("role_mixer.") for n in names)
    assert any("coefficients" in n for n in names)
    assert any("branches" in n for n in names)


def test_roles_head_and_adapter_roles_head_select_the_right_groups():
    from multipose2 import train
    net = _net_with_mixer()

    train.set_trainable_parameters(net, trainable_mode="roles_head")
    names = [n for n, p in net.named_parameters() if p.requires_grad]
    assert any(n.startswith("out.") for n in names)
    assert any(n.startswith("role_mixer.") for n in names)
    assert not any(n.startswith("input_adapter.") for n in names)

    train.set_trainable_parameters(net, trainable_mode="adapter_roles_head")
    names = [n for n, p in net.named_parameters() if p.requires_grad]
    for prefix in ("input_adapter.", "out.", "role_mixer."):
        assert any(n.startswith(prefix) for n in names), prefix
    assert not any(n.startswith("encoder.") for n in names)


def test_existing_modes_never_select_the_mixer():
    from multipose2 import train
    net = _net_with_mixer()
    for mode in ("adapter_head", "adapter_only", "head_only",
                 "adapter_head_last_blocks"):
        train.set_trainable_parameters(net, trainable_mode=mode)
        names = [n for n, p in net.named_parameters() if p.requires_grad]
        assert not any(n.startswith("role_mixer.") for n in names), mode


def test_adapter_channel_norms_start_at_identity():
    net = vit_sam.Transformer(in_channels=6, bsize=64)
    norms = net.input_adapter.channel_norms().detach()
    assert norms.shape == (6,)
    # identity init gives the first three channels unit norm and the rest none
    assert torch.allclose(norms[:3], torch.ones(3))
    assert torch.allclose(norms[3:], torch.zeros(3))
    assert float(net.input_adapter.channel_l1().detach()) == pytest.approx(3.0)


def test_channel_l1_is_a_group_lasso_not_elementwise():
    """Whole channels, not individual weights, are what should go to zero."""
    net = vit_sam.Transformer(in_channels=4, bsize=64)
    with torch.no_grad():
        net.input_adapter.proj.weight.zero_()
        # one channel with three small weights vs one with a single large weight
        net.input_adapter.proj.weight[:, 0, 0, 0] = 0.6     # norm sqrt(3)*0.6
        net.input_adapter.proj.weight[0, 1, 0, 0] = 1.0     # norm 1.0
    norms = net.input_adapter.channel_norms().detach()
    assert float(norms[0]) == pytest.approx(0.6 * 3 ** 0.5, abs=1e-5)
    assert float(norms[1]) == pytest.approx(1.0, abs=1e-5)
    assert float(norms[2]) == 0.0 and float(norms[3]) == 0.0


def test_sparsity_penalty_is_none_by_default_and_tracks_frozen_groups():
    from multipose2 import train
    net = _net_with_mixer()
    assert train._sparsity_penalty(net, 0., 0.) is None

    train.set_trainable_parameters(net, trainable_mode="roles_only")
    # the mixer is trainable, the adapter is not, so only the role term counts
    roles_only = train._sparsity_penalty(net, 1., 1.)
    assert roles_only is not None
    assert float(roles_only.detach()) == pytest.approx(float(net.role_mixer.l1().detach()))

    train.set_trainable_parameters(net, trainable_mode="adapter_only")
    adapter_only = train._sparsity_penalty(net, 1., 1.)
    assert float(adapter_only.detach()) == pytest.approx(
        float(net.input_adapter.channel_l1().detach()))


def test_sparsity_penalty_sums_both_terms_when_both_are_trainable():
    from multipose2 import train
    net = _net_with_mixer()
    train.set_trainable_parameters(net, trainable_mode="adapter_roles_head")
    both = float(train._sparsity_penalty(net, 1., 1.).detach())
    expected = (float(net.role_mixer.l1().detach())
                + float(net.input_adapter.channel_l1().detach()))
    assert both == pytest.approx(expected)
    # and the weights scale independently
    scaled = float(train._sparsity_penalty(net, 2., 0.).detach())
    assert scaled == pytest.approx(2 * float(net.role_mixer.l1().detach()))
