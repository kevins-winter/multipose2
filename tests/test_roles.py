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


def test_foreground_suppress_can_only_subtract_from_cell_probability():
    """Necessity, which an additive mixture cannot express."""
    mixer = _mixer(roles=("foreground_suppress",))
    with torch.no_grad():
        mixer.coefficients["uchl1__foreground_suppress"].fill_(2.0)  # softplus(2) ~ 2.13
        base, x = _base(), _inputs()
        out = mixer(base, x)
    delta = (out - base)[:, -1:]
    assert (delta <= 1e-6).all(), "a veto term must never add existence evidence"
    assert delta.min() < -1e-3, "and it must actually subtract"
    # flows untouched by an existence role
    assert torch.allclose(out[:, -3:-1], base[:, -3:-1])


def test_foreground_support_direction_lives_in_the_coefficient():
    """The sigmoid branch puts the sign in the coefficient, not the pattern.

    A tanh branch would let one modality raise the logit in one place and lower
    it in another: more flexible, but it makes the coefficient's sign
    meaningless, because negating both coefficient and branch gives identical
    output.
    """
    mixer = _mixer(roles=("foreground_support",))
    base, x = _base(), _inputs()
    with torch.no_grad():
        mixer.coefficients["uchl1__foreground_support"].fill_(1.0)
        positive = (mixer(base, x) - base)[:, -1:].clone()
        mixer.coefficients["uchl1__foreground_support"].fill_(-1.0)
        negative = (mixer(base, x) - base)[:, -1:].clone()
        out = mixer(base, x)
    assert (positive >= -1e-6).all(), "a positive coefficient must only add"
    assert (negative <= 1e-6).all(), "a negative coefficient must only subtract"
    assert float(positive.max()) > 1e-3
    assert torch.allclose(positive, -negative, atol=1e-6)
    assert torch.allclose(out[:, -3:-1], base[:, -3:-1])


def test_sign_degeneracy_is_closed_for_signed_roles():
    """Negating coefficient and branch together must now change the output.

    With a tanh branch it did not -- the two were bit-identical, so the sign of
    a flow_center coefficient carried no information about attraction versus
    repulsion. A sigmoid branch removes that symmetry.
    """
    for role in ("foreground_support", "flow_center"):
        mixer = _mixer(roles=(role,))
        base, x = _base(), _inputs()
        with torch.no_grad():
            mixer.coefficients[f"uchl1__{role}"].fill_(1.0)
            before = mixer(base, x).clone()
            mixer.coefficients[f"uchl1__{role}"].fill_(-1.0)
            for prm in mixer.branches[f"uchl1__{role}"].parameters():
                prm.mul_(-1.0)
            after = mixer(base, x).clone()
        assert not torch.allclose(before, after, atol=1e-5), (
            f"{role}: sign degeneracy is still present")


def test_flow_center_and_flow_refine_touch_only_the_flow_field():
    for role in ("flow_center", "flow_refine"):
        mixer = _mixer(roles=(role,))
        with torch.no_grad():
            mixer.coefficients[f"fabp7__{role}"].fill_(1.5)
            base, x = _base(), _inputs()
            out = mixer(base, x)
        assert not torch.allclose(out[:, -3:-1], base[:, -3:-1]), role
        assert torch.allclose(out[:, -1:], base[:, -1:]), role


def test_flow_center_sign_flips_the_flow_direction():
    """A positive coefficient attracts, a negative one repels."""
    mixer = _mixer(roles=("flow_center",))
    base, x = _base(), _inputs()
    with torch.no_grad():
        mixer.coefficients["uchl1__flow_center"].fill_(1.0)
        attract = mixer(base, x) - base
        mixer.coefficients["uchl1__flow_center"].fill_(-1.0)
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
        mixer.coefficients["uchl1__foreground_suppress"].fill_(3.0)
        mixer.coefficients["fabp7__flow_center"].fill_(-2.0)
    table = mixer.coefficient_table()
    assert set(table) == {"he", "uchl1", "fabp7"}
    assert table["uchl1"]["foreground_suppress"] == pytest.approx(torch.nn.functional.softplus(
        torch.tensor(3.0)).item())
    assert table["fabp7"]["flow_center"] == pytest.approx(-2.0)
    # non-negative roles can never report a negative value
    assert all(t["foreground_suppress"] >= 0 and t["flow_refine"] >= 0 for t in table.values())


def test_l1_is_zero_ish_at_init_and_grows_with_use():
    mixer = _mixer()
    start = float(mixer.l1().detach())
    with torch.no_grad():
        mixer.coefficients["he__foreground_support"].fill_(4.0)
    assert float(mixer.l1().detach()) > start + 3.9


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError, match="unknown roles"):
        RoleMixer(MODALITIES, roles=("foreground_suppress", "vibes"))
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
    for role, bound in (("foreground_support", 1.0), ("flow_refine", 1.0)):
        mixer = _mixer(roles=(role,))
        base, x = _base(), _inputs()
        with torch.no_grad():
            if role in ("foreground_suppress", "flow_refine"):
                mixer.coefficients[f"uchl1__{role}"].fill_(-10.0)  # ~0
            coef = 0.5
            raw = torch.log(torch.expm1(torch.tensor(coef))) if role == "flow_refine" \
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


def test_foreground_suppress_magnitude_is_bounded_by_gain_times_coefficient():
    from multipose2.roles import SUPPRESS_GAIN
    mixer = _mixer(roles=("foreground_suppress",))
    base, x = _base(), _inputs()
    with torch.no_grad():
        mixer.coefficients["uchl1__foreground_suppress"].fill_(2.0)
        for prm in mixer.branches["uchl1__foreground_suppress"].parameters():
            prm.mul_(100.0)
        delta = (mixer(base, x) - base)[:, -1:]
    coef = float(mixer.coefficient("uchl1", "foreground_suppress").detach())
    assert (delta <= 1e-6).all()
    assert delta.abs().max() <= coef * (SUPPRESS_GAIN + 0.1)


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


def test_contribution_report_measures_what_forward_applies():
    """Effect size, not coefficients: the quantity worth reporting."""
    mixer = _mixer()
    x = _inputs()
    with torch.no_grad():
        mixer.coefficients["uchl1__foreground_support"].fill_(1.5)
        report = mixer.contribution_report(x)

    assert set(report) == {"he", "uchl1", "fabp7"}
    assert set(report["uchl1"]) == set(mixer.roles)
    entry = report["uchl1"]["foreground_support"]
    assert entry["target"] == "cellprob"
    assert 0. < entry["mean_abs"] <= entry["max_abs"]
    assert 0. <= entry["frac_active"] <= 1.
    # a flow_center term is differentiated before measuring, so it is reported
    # against the output it actually corrects
    assert report["uchl1"]["flow_center"]["target"] == "flows"
    assert report["uchl1"]["flow_refine"]["target"] == "flows"
    assert report["uchl1"]["foreground_suppress"]["target"] == "cellprob"


def test_contribution_report_separates_terms_a_coefficient_cannot():
    """Two branches sharing a coefficient can differ hugely in total influence.

    This is why the coefficient is a diagnostic and the measured contribution is
    the reportable number.
    """
    mixer = _mixer(roles=("foreground_support",))
    x = _inputs()
    with torch.no_grad():
        mixer.coefficients["uchl1__foreground_support"].fill_(1.0)
        last = mixer.branches["uchl1__foreground_support"].net[-1]
        last.weight.mul_(0.)
        last.bias.fill_(-8.0)          # sigmoid ~ 0, barely acts
        quiet = mixer.contribution_report(x)["uchl1"]["foreground_support"]
        last.bias.fill_(8.0)           # sigmoid ~ 1, acts everywhere
        loud = mixer.contribution_report(x)["uchl1"]["foreground_support"]

    coef = float(mixer.coefficient("uchl1", "foreground_support").detach())
    assert coef == pytest.approx(1.0)
    assert loud["mean_abs"] > 100 * quiet["mean_abs"], (
        "the report must distinguish what one coefficient cannot")


def test_suppression_is_bounded_and_never_reaches_zero_probability():
    """Suppression multiplies the odds; it is not a logical gate."""
    from multipose2.roles import SUPPRESS_GAIN
    mixer = _mixer(roles=("foreground_suppress",))
    base, x = _base(), _inputs()
    with torch.no_grad():
        mixer.coefficients["uchl1__foreground_suppress"].fill_(2.0)
        for prm in mixer.branches["uchl1__foreground_suppress"].parameters():
            prm.mul_(1000.0)
        delta = (mixer(base, x) - base)[:, -1:]
    coef = float(mixer.coefficient("uchl1", "foreground_suppress").detach())
    assert (delta <= 1e-6).all(), "suppression must never add foreground evidence"
    worst = float(delta.min())
    assert worst >= -coef * (SUPPRESS_GAIN + 0.1), "bounded by gain x coefficient"
    # a finite logit penalty is a finite odds ratio, never a probability of zero
    assert float(torch.exp(torch.tensor(worst))) > 0.


# --- capacity-matched control ------------------------------------------------

def test_control_is_sized_to_match_the_mixer():
    from multipose2.roles import UnrestrictedCorrection
    mixer = _mixer()
    ctrl = UnrestrictedCorrection.matched_to(mixer, in_channels=5)
    target = sum(p.numel() for p in mixer.parameters())
    realized = sum(p.numel() for p in ctrl.parameters())
    assert ctrl.matched_target == target
    assert ctrl.matched_error == realized - target
    # within a couple of percent, which is what makes the arms comparable
    assert abs(realized - target) / target < 0.05


def test_control_starts_as_an_exact_no_op_but_still_learns():
    from multipose2.roles import UnrestrictedCorrection
    ctrl = UnrestrictedCorrection(in_channels=5, width=12)
    base, x = _base(), _inputs()
    ctrl.eval()
    with torch.no_grad():
        assert torch.equal(ctrl(base, x), base), "both arms must start identical"
    # a zero-initialised head keeps its own gradient, so the layer learns first
    ctrl(base, x).square().mean().backward()
    assert float(ctrl.net.head.weight.grad.abs().sum()) > 0


def test_control_writes_every_output_channel_without_structure():
    from multipose2.roles import UnrestrictedCorrection
    ctrl = UnrestrictedCorrection(in_channels=5, width=12)
    base, x = _base(), _inputs()
    with torch.no_grad():
        ctrl.net.head.weight.normal_(0., 0.2)
        ctrl.net.head.bias.normal_(0., 0.2)
        delta = ctrl(base, x) - base
    # unlike the mixer, nothing confines it to one output or one sign
    assert delta[:, -1:].abs().max() > 0
    assert delta[:, -3:-1].abs().max() > 0
    assert delta.min() < 0 < delta.max()


def test_control_respects_enabled_and_resolution():
    from multipose2.roles import UnrestrictedCorrection
    ctrl = UnrestrictedCorrection(in_channels=5, width=12)
    base, x = _base(), _inputs()
    ctrl.enabled = False
    with torch.no_grad():
        assert torch.equal(ctrl(base, x), base)
    ctrl.enabled = True
    with pytest.raises(ValueError, match="full resolution"):
        ctrl(_base(s=32), _inputs(s=64))


def test_groups_divides_every_width():
    from multipose2.roles import _groups, UnrestrictedCorrection
    for w in range(1, 128):
        assert w % _groups(w) == 0
        UnrestrictedCorrection(in_channels=4, width=w, dilations=(1,))


def test_sparsity_penalty_skips_a_control_with_no_coefficients():
    from multipose2 import train
    from multipose2.roles import UnrestrictedCorrection
    net = vit_sam.Transformer(in_channels=5, bsize=64)
    net.set_role_mixer(UnrestrictedCorrection(in_channels=5, width=8))
    train.set_trainable_parameters(net, trainable_mode="roles_only")
    # no coefficient penalty applies; weight decay already covers its parameters
    assert train._sparsity_penalty(net, 1., 0.) is None


# --- availability versus observed-but-empty ---------------------------------

def test_unavailable_and_observed_empty_are_different_statements():
    """Zeroing values says "measured, saw nothing"; absence says "no channel"."""
    mixer = _mixer(roles=("foreground_support",))
    with torch.no_grad():
        mixer.coefficients["uchl1__foreground_support"].fill_(1.0)
    base, x = _base(), _inputs()
    mixer.eval()
    with torch.no_grad():
        zeroed = x.clone()
        zeroed[:, 3] = 0.
        observed_empty = mixer(base, zeroed)
        mixer.set_presence(torch.tensor([1., 0., 1.]))
        unavailable = mixer(base, x)
        mixer.set_presence(None)
        restored = mixer(base, x)

    assert not torch.allclose(observed_empty, base, atol=1e-6), \
        "an observed channel reading zero is still evidence"
    assert torch.equal(unavailable[:, -1:], base[:, -1:]), \
        "an unavailable modality must contribute exactly nothing"
    assert not torch.allclose(observed_empty, unavailable, atol=1e-6)
    assert not torch.allclose(restored, base, atol=1e-6), "clearing must restore"


def test_presence_shape_is_validated():
    mixer = _mixer()
    with pytest.raises(ValueError, match="must cover 3 modalities"):
        mixer.set_presence(torch.tensor([1., 0.]))


def test_modality_dropout_only_applies_in_training_mode():
    mixer = _mixer(modality_dropout=0.9)
    mixer.eval()
    assert float(mixer._presence(32, torch.device("cpu"), torch.float32).mean()) == 1.0
    mixer.train()
    mask = mixer._presence(256, torch.device("cpu"), torch.float32)
    assert 0. < float(mask.mean()) < 1.
    # never drop every modality at once, which would supervise nothing
    assert bool(mask.any(dim=1).all())


def test_invalid_modality_dropout_is_rejected():
    with pytest.raises(ValueError, match="modality_dropout must be"):
        _mixer(modality_dropout=1.0)
    with pytest.raises(ValueError, match="modality_dropout must be"):
        _mixer(modality_dropout=-0.1)


# --- receptive field, context conditioning, and interactions ----------------

CORR_MODS = [("he", (0, 1, 2)), ("uchl1", (3,)), ("fabp7", (4,))]


def _corr(**kw):
    from multipose2.roles import ModalityCorrections
    torch.manual_seed(0)
    return ModalityCorrections(CORR_MODS, nout=3, **kw).eval()


def _ctx(n=2, d=256, s=8):
    torch.manual_seed(3)
    return torch.randn(n, d, s, s)


def _saturate(m, value=1.5):
    with torch.no_grad():
        for key in m.coefficients:
            m.coefficients[key].fill_(value)
    return m


def test_branch_receptive_field_covers_a_soma():
    """Two 3x3 convolutions see 5 pixels; a soma spans tens."""
    from multipose2.roles import MultiscaleBranch
    narrow = MultiscaleBranch(3, 8, width=8, dilations=(1, 1))
    assert narrow.receptive_field == 5
    wide = MultiscaleBranch(3, 8, width=8)
    assert wide.receptive_field == 63
    # and the width comes nearly free, because the stack is depthwise-separable
    assert sum(p.numel() for p in wide.parameters()) < 4000


def test_independent_arm_is_exactly_additive_across_modalities():
    """Arm 1: a modality's correction does not depend on the others."""
    m = _saturate(_corr())
    base, x = _base(s=64), _inputs(c=5, s=64)

    def delta(on):
        m.set_presence(torch.tensor([1. if n in on else 0.
                                     for n in m.modality_names]))
        with torch.no_grad():
            out = (m(base, x) - base).clone()
        m.set_presence(None)
        return out

    he, tx, both = delta({"he"}), delta({"uchl1"}), delta({"he", "uchl1"})
    assert torch.allclose(both, he + tx, atol=1e-5)
    # masking features alone would leak the route head's bias, so the term is
    # masked too and an absent modality contributes exactly nothing
    assert torch.allclose(delta({"he"}), he, atol=1e-6)


def test_cross_modality_arm_is_not_additive():
    """Arm 3 exists to express conjunctions, so it must break additivity."""
    m = _saturate(_corr(context_dim=256, cross_modality=True))
    base, x, ctx = _base(s=64), _inputs(c=5, s=64), _ctx()

    def delta(on):
        m.set_presence(torch.tensor([1. if n in on else 0.
                                     for n in m.modality_names]))
        with torch.no_grad():
            out = (m(base, x, context=ctx) - base).clone()
        m.set_presence(None)
        return out

    he, tx, both = delta({"he"}), delta({"uchl1"}), delta({"he", "uchl1"})
    assert not torch.allclose(both, he + tx, atol=1e-4)
    # attribution is given up in exchange, so there is one entry, not three
    assert list(m.coefficient_table()) == ["mixed"]


def test_context_changes_the_correction_only_when_conditioned():
    m = _saturate(_corr(context_dim=256))
    base, x = _base(s=64), _inputs(c=5, s=64)
    with torch.no_grad():
        # FiLM is zero-initialised, so context has no effect until it trains
        same = (m(base, x, context=_ctx()) - base)
        other = (m(base, x, context=torch.randn_like(_ctx())) - base)
        assert torch.allclose(same, other, atol=1e-6)
        for film in m.context_film.values():
            film.weight.normal_(0., 0.3)
            film.bias.normal_(0., 0.1)
        a = (m(base, x, context=_ctx()) - base).clone()
        b = (m(base, x, context=torch.randn_like(_ctx())) - base).clone()
    assert not torch.allclose(a, b, atol=1e-5)

    plain = _corr()
    assert plain.context_dim is None


def test_conditioned_module_requires_context():
    m = _corr(context_dim=256)
    with pytest.raises(ValueError, match="needs the frozen"):
        m(_base(s=64), _inputs(c=5, s=64))


def test_all_three_arms_start_as_no_ops():
    base, x, ctx = _base(s=64), _inputs(c=5, s=64), _ctx()
    for kw in ({}, {"context_dim": 256}, {"context_dim": 256, "cross_modality": True}):
        m = _corr(**kw)
        with torch.no_grad():
            out = m(base, x, context=ctx if kw.get("context_dim") else None)
        assert torch.allclose(out, base, atol=1e-6), kw
        m.enabled = False
        with torch.no_grad():
            assert torch.equal(m(base, x, context=ctx if kw.get("context_dim") else None), base)


def test_foreground_constraints_behave_as_documented():
    base, x = _base(s=64), _inputs(c=5, s=64)

    only_down = _saturate(_corr(foreground_constraint="suppress_only"))
    with torch.no_grad():
        d = (only_down(base, x) - base)[:, -1:]
    assert (d <= 1e-6).all() and d.min() < -1e-3

    both_ways = _saturate(_corr(foreground_constraint="none"))
    with torch.no_grad():
        d = (both_ways(base, x) - base)[:, -1:]
    assert d.max() > 0 > d.min(), "the default must be able to raise and lower"

    signed = _corr(foreground_constraint="sign_identifiable")
    with torch.no_grad():
        signed.coefficients["he__foreground"].fill_(1.0)
        up = (signed(base, x) - base)[:, -1:].clone()
        signed.coefficients["he__foreground"].fill_(-1.0)
        down = (signed(base, x) - base)[:, -1:].clone()
    assert (up >= -1e-6).all() and (down <= 1e-6).all()


def test_flow_parameterizations_both_move_only_the_flows():
    base, x = _base(s=64), _inputs(c=5, s=64)
    for param in ("direct", "potential"):
        m = _saturate(_corr(flow_parameterization=param))
        with torch.no_grad():
            out = m(base, x)
        assert not torch.allclose(out[:, -3:-1], base[:, -3:-1], atol=1e-4), param


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="foreground_constraint must be"):
        _corr(foreground_constraint="hinge")
    with pytest.raises(ValueError, match="flow_parameterization must be"):
        _corr(flow_parameterization="curl")


def test_corrections_attach_to_the_transformer_and_receive_context():
    from multipose2 import train
    net = vit_sam.Transformer(in_channels=5, bsize=64)
    net.set_role_mixer(_corr(context_dim=256))
    train.set_trainable_parameters(net, trainable_mode="roles_only")
    names = [n for n, p in net.named_parameters() if p.requires_grad]
    assert names and all(n.startswith("role_mixer.") for n in names)
    net.eval()
    with torch.no_grad():
        y = net(torch.randn(1, 5, 64, 64))[0]
    assert y.shape == (1, 3, 64, 64)


def test_control_matches_the_new_corrections_too():
    from multipose2.roles import UnrestrictedCorrection
    m = _corr(context_dim=256)
    ctrl = UnrestrictedCorrection.matched_to(m, in_channels=5)
    target = sum(p.numel() for p in m.parameters())
    realized = sum(p.numel() for p in ctrl.parameters())
    assert abs(realized - target) / target < 0.05


def test_stochastic_depth_follows_encoder_trainability():
    from multipose2 import train
    net = vit_sam.Transformer(in_channels=5, bsize=64)
    net.set_role_mixer(_corr())
    x = torch.randn(1, 5, 64, 64)

    train.set_trainable_parameters(net, trainable_mode="roles_only")
    assert net.deterministic_trunk is True
    net.train()
    with torch.no_grad():
        a = net.forward_trunk(net.input_adapter(x))
        b = net.forward_trunk(net.input_adapter(x))
    assert torch.allclose(a, b, atol=1e-6), "a frozen base must be reproducible"

    train.set_trainable_parameters(net, trainable_mode="all")
    assert net.deterministic_trunk is False
    net.train()
    with torch.no_grad():
        a = net.forward_trunk(net.input_adapter(x))
        b = net.forward_trunk(net.input_adapter(x))
    assert not torch.allclose(a, b, atol=1e-6), "layer dropping should resume"


def test_control_matches_receptive_field_not_only_parameters():
    """Otherwise a win for the structured arm could be a receptive-field win."""
    from multipose2.roles import UnrestrictedCorrection
    m = _corr(context_dim=256)
    ctrl = UnrestrictedCorrection.matched_to(m, in_channels=5)
    assert ctrl.receptive_field == m.receptive_field == 63
    target = sum(p.numel() for p in m.parameters())
    assert abs(sum(p.numel() for p in ctrl.parameters()) - target) / target < 0.05


def test_control_param_formula_matches_a_built_model():
    from multipose2.roles import UnrestrictedCorrection
    for width in (1, 7, 16, 21, 64):
        built = UnrestrictedCorrection(in_channels=6, nout=3, width=width)
        assert sum(p.numel() for p in built.parameters()) == \
            UnrestrictedCorrection._param_count(6, 3, width, 5)
