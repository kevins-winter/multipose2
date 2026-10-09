"""Per-modality functional roles over the outputs Cellpose already supervises.

Cellpose supervises two separate outputs: a cell probability logit and a flow
field. The flow field is not arbitrary -- masks_to_flows builds it as the
gradient of a diffused potential peaking at the cell centre -- so between them
those two outputs can express *existence* and *geometry* independently.

Nothing in the stock architecture routes modalities differently to them. A 1x1
input adapter mixes every channel additively into one 3-channel trunk input, and
additive mixing cannot express necessity: strong morphology can always
compensate for an absent marker. This module adds a small basis of operators
whose per-modality coefficients are learned, so role assignment is discovered
rather than declared:

    veto       ``logsigmoid(g(x))``, coefficient >= 0, term <= 0.
               Acts on the cell probability logit, and can only subtract, so a
               modality can veto a cell but never conjure one. This is what
               "necessary" means, and it is what an additive mixture cannot say.

    support    ``h(x)``, signed, on the cell probability logit.
               Ordinary evidence for or against a cell being present.

    potential  ``phi(x)``, signed, added to the flow field as ``grad(phi)``.
               Predicting a scalar potential and differentiating it yields a
               curl-free field consistent with how Cellpose defines flows. A
               positive coefficient makes the modality a centre attractor, a
               negative one makes it a surround to be excluded.

    edge       ``e(x)``, coefficient >= 0, added to the flow field directly at
               full resolution. Sharpens where a boundary falls without any say
               over whether a cell exists there.

Because the coefficients start at (effectively) zero, an untrained mixer leaves
the base prediction alone: the parity property that makes a stock Cellpose model
the baseline survives. Because every role term reads one modality's channels
only, a coefficient measures that modality's direct effect rather than something
the trunk already entangled.

Roles can disagree with an ablation, and when they do the ablation is right. A
learned role predicts a specific failure -- suppressing a high-veto modality
should cost recall, suppressing a high-edge modality should cost F1 at strict
IoU thresholds but not at 0.5 -- so the two together are evidence in a way
neither is alone.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

roles_logger = logging.getLogger(__name__)

ROLES = ("veto", "support", "potential", "edge")

# roles whose coefficient must stay non-negative for the role to mean anything:
# a negative veto would add existence evidence, collapsing it into support
NONNEGATIVE_ROLES = ("veto", "edge")

# softplus(-10) = 4.5e-5, so a non-negative coefficient starts effectively off
# while still receiving gradient. Signed coefficients can start at exactly zero.
_OFF = -10.0

# Every branch output is squashed to [-1, 1] before its coefficient scales it.
# Without that, coefficient * branch(x) has a scale degeneracy: an L1 penalty
# drives the coefficient toward zero while the branch grows to compensate, and
# the coefficient stops measuring anything. Bounding the branch leaves the
# coefficient as the only carrier of magnitude, so it is identifiable and the
# table is readable. The branch still determines the spatial pattern.
#
# A veto needs more headroom than +/-1 to suppress a confident base logit, so
# its bounded opinion is amplified by a fixed gain before logsigmoid: the term
# then spans roughly [-VETO_GAIN * coefficient, 0].
VETO_GAIN = 4.0


def _sobel_kernels(dtype=torch.float32):
    """Fixed (d/dy, d/dx) kernels, shaped for a 1-channel input."""
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      dtype=dtype) / 8.
    return torch.stack((kx.t(), kx)).unsqueeze(1)  # (2, 1, 3, 3), dy then dx


class _RoleBranch(nn.Module):
    """A small full-resolution CNN mapping one modality to ``nout`` maps.

    Deliberately tiny. With a few hundred annotated cells the interpretable
    quantity is the coefficient, not the branch, and a large branch would both
    overfit and make the coefficient harder to read.
    """

    def __init__(self, in_channels, nout=1, width=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.GroupNorm(min(4, width), width),
            nn.GELU(),
            nn.Conv2d(width, nout, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class RoleMixer(nn.Module):
    """Adds learned per-modality role terms to a base Cellpose prediction.

    Args:
        modalities (sequence of tuple): ``(name, channels)`` per modality, where
            channels index the fused input stack.
        roles (sequence of str, optional): Which of ROLES to instantiate.
        width (int, optional): Hidden width of each role branch.
        nout (int, optional): Output channel count of the base prediction, so
            that the cell-probability and flow slices are located the same way
            the loss locates them.
    """

    def __init__(self, modalities, roles=ROLES, width=16, nout=3):
        super().__init__()
        self.modality_names = [str(name) for name, _ in modalities]
        self.modality_channels = [tuple(int(c) for c in ch) for _, ch in modalities]
        bad = set(roles) - set(ROLES)
        if bad:
            raise ValueError(f"unknown roles {sorted(bad)}; expected {ROLES}")
        self.roles = tuple(r for r in ROLES if r in set(roles))
        if not self.roles:
            raise ValueError("RoleMixer needs at least one role")
        self.nout = int(nout)

        # one branch and one coefficient per (modality, role)
        self.branches = nn.ModuleDict()
        self.coefficients = nn.ParameterDict()
        for name, channels in zip(self.modality_names, self.modality_channels):
            for role in self.roles:
                n_in, n_out = len(channels), 2 if role == "edge" else 1
                self.branches[f"{name}__{role}"] = _RoleBranch(n_in, n_out, width)
                init = _OFF if role in NONNEGATIVE_ROLES else 0.
                self.coefficients[f"{name}__{role}"] = nn.Parameter(
                    torch.tensor(init, dtype=torch.float32))

        self.register_buffer("sobel", _sobel_kernels(), persistent=False)
        # when False the terms are skipped entirely, so the module is exactly a
        # no-op and the parity assertion against the stock model is exact
        self.enabled = True

    def coefficient(self, name, role):
        """The effective coefficient, after any non-negativity constraint."""
        raw = self.coefficients[f"{name}__{role}"]
        return F.softplus(raw) if role in NONNEGATIVE_ROLES else raw

    def coefficient_table(self):
        """Learned role assignment as ``{modality: {role: float}}``.

        This is the interpretation. Read a column to see which modalities took a
        role, and a row to see what one modality learned to do.
        """
        return {name: {role: float(self.coefficient(name, role).detach())
                       for role in self.roles}
                for name in self.modality_names}

    def l1(self):
        """Sum of coefficient magnitudes, for sparsity regularization.

        Without it a modality spreads itself thinly over every role and the
        table stops being readable; the prior that a marker does one or two
        things is both correct and what makes the result interpretable.
        """
        total = None
        for name in self.modality_names:
            for role in self.roles:
                term = self.coefficient(name, role).abs()
                total = term if total is None else total + term
        return total

    def forward(self, base, inputs):
        """Add role terms to a base prediction.

        Args:
            base (torch.Tensor): Base prediction, ``[N x nout x Ly x Lx]``.
            inputs (torch.Tensor): The fused input stack, ``[N x C x Ly x Lx]``,
                at the same spatial size as ``base``.

        Returns:
            torch.Tensor: ``base`` with the role terms added.
        """
        if not self.enabled:
            return base
        if base.shape[-2:] != inputs.shape[-2:]:
            raise ValueError(
                f"role terms are added at full resolution, but base is "
                f"{tuple(base.shape[-2:])} and inputs are "
                f"{tuple(inputs.shape[-2:])}")

        cellprob = torch.zeros_like(base[:, -1:])
        flows = torch.zeros_like(base[:, -3:-1])
        potential = None

        for name, channels in zip(self.modality_names, self.modality_channels):
            x_m = inputs[:, list(channels)]
            for role in self.roles:
                coef = self.coefficient(name, role).to(base.dtype)
                raw = self.branches[f"{name}__{role}"](x_m).to(base.dtype)
                # bounded to [-1, 1] so the coefficient carries all the scale
                opinion = torch.tanh(raw)
                if role == "veto":
                    # logsigmoid <= 0 and coefficient >= 0, so this can only
                    # subtract: a veto, never evidence
                    cellprob = cellprob + coef * F.logsigmoid(VETO_GAIN * opinion)
                elif role == "support":
                    cellprob = cellprob + coef * opinion
                elif role == "edge":
                    flows = flows + coef * opinion
                else:
                    term = coef * opinion
                    potential = term if potential is None else potential + term

        if potential is not None:
            # one gradient of the summed potential, so attractors and surrounds
            # compose into a single consistent field
            sobel = self.sobel.to(potential.dtype)
            flows = flows + F.conv2d(potential, sobel, padding=1)

        out = base.clone()
        out[:, -1:] = out[:, -1:] + cellprob
        out[:, -3:-1] = out[:, -3:-1] + flows
        return out

    def extra_repr(self):
        return (f"modalities={self.modality_names}, roles={list(self.roles)}, "
                f"enabled={self.enabled}")
