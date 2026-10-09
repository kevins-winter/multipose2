"""Modality-specific corrections to foreground prediction and spatial refinement.

Cellpose supervises two outputs separately: a per-pixel foreground logit
(cellprob) and a flow field. Nothing in the stock architecture lets a modality
influence one without the other -- a 1x1 adapter mixes every channel into a
single 3-channel trunk input, so all modalities reach both outputs through the
same compressed path.

This module adds explicit per-modality routes to each output, so the hypothesis
that different signals contribute differently can be *tested*: a transcript
channel might help identify neuronal regions while morphology helps place their
boundaries. Four operators, each with a learned per-modality coefficient:

    foreground_suppress   reduces the foreground logit, and only reduces it.
    foreground_support    raises or lowers the foreground logit.
    flow_center           added to the flow field as the gradient of a scalar
                          potential, which draws flows toward or away from where
                          that modality's signal is high.
    flow_refine           added to the flow field directly, at full resolution.

What these coefficients are not
-------------------------------
They are not measurements of biological necessity, and they are not uniquely
identifiable. Four distinct reasons, all of which have been checked against this
implementation rather than assumed:

1. A suppression term is bounded. ``coefficient * logsigmoid(gain * opinion)``
   is at worst ``-gain * coefficient`` logits, which multiplies the *odds* by
   ``exp(-gain * coefficient)`` -- not the probability, and never to zero. A
   large positive base logit can survive it, and a support term can offset it.
   This is strong suppression, not a logical gate.

2. The coefficient bounds a term's magnitude per pixel, not its total influence.
   A branch is free to saturate on 1% of pixels or on all of them; measured on
   random inputs, two branches sharing one coefficient differed 50-fold in mean
   absolute effect.

3. Correlated modalities substitute for one another, so a coefficient is only
   as meaningful as the channels are independent. Duplicate channels -- a
   single-channel modality stored as JPEG becomes three identical ones -- make
   the split between them arbitrary.

4. Absent transcript signal is not absent biology. Detection depends on
   expression level, panel sensitivity and which plane the section cut, so a
   modality reading zero is evidence of weaker confidence, not proof of absence.
   An unavailable channel and an observed-but-empty channel are different
   statements and should not produce the same correction.

So ``coefficient_table()`` is a diagnostic, never evidence. Use
``contribution_report()``, which measures each term's effect on held-out data,
and settle questions about what a modality does by perturbing it and measuring
the outcome.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

roles_logger = logging.getLogger(__name__)

ROLES = ("foreground_suppress", "foreground_support", "flow_center", "flow_refine")

# Roles whose coefficient is constrained non-negative, because the operator
# itself carries the direction: suppression must not be able to add foreground
# evidence, and a refinement magnitude has no meaningful sign.
NONNEGATIVE_ROLES = ("foreground_suppress", "flow_refine")

# Which output each role corrects.
ROLE_TARGET = {
    "foreground_suppress": "cellprob",
    "foreground_support": "cellprob",
    "flow_center": "potential",
    "flow_refine": "flows",
}

# How each branch output is squashed before its coefficient scales it.
#
# Both bounds exist to stop the coefficient and branch trading scale: without
# one, an L1 penalty drives the coefficient toward zero while the branch grows
# to compensate, and the coefficient measures nothing.
#
# The choice between them closes a sign degeneracy. For a signed coefficient,
# tanh output in [-1, 1] makes (coefficient, branch) and (-coefficient, -branch)
# produce bit-identical output, so the sign carries no information -- verified,
# not assumed. A sigmoid output in [0, 1] leaves the sign only in the
# coefficient, at the cost of a modality no longer raising the logit in one
# place and lowering it in another. For a coefficient already constrained
# non-negative there is no such degeneracy, so tanh is kept where a signed
# spatial pattern is wanted: a flow correction is a vector and needs both signs.
ROLE_SQUASH = {
    "foreground_suppress": torch.tanh,     # coefficient >= 0, no degeneracy
    "foreground_support": torch.sigmoid,   # signed coefficient, needs [0, 1]
    "flow_center": torch.sigmoid,          # signed coefficient, needs [0, 1]
    "flow_refine": torch.tanh,             # coefficient >= 0, signed field
}

# softplus(-10) = 4.5e-5, so a non-negative coefficient starts effectively off
# while still receiving gradient. Signed coefficients can start at exactly zero.
_OFF = -10.0

# A suppression term needs headroom beyond +/-1 to move a confident base logit,
# so its bounded opinion is amplified before logsigmoid. The term then spans
# roughly [-SUPPRESS_GAIN * coefficient, 0].
SUPPRESS_GAIN = 4.0


def _sobel_kernels(dtype=torch.float32):
    """Fixed (d/dy, d/dx) kernels, shaped for a 1-channel input."""
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      dtype=dtype) / 8.
    return torch.stack((kx.t(), kx)).unsqueeze(1)  # (2, 1, 3, 3), dy then dx


class _RoleBranch(nn.Module):
    """A small full-resolution CNN mapping one modality to ``nout`` maps.

    Deliberately tiny. With a few hundred annotated cells a large branch would
    overfit, and the quantity worth reporting is the measured contribution
    rather than anything inside the branch.
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
    """Adds learned per-modality corrections to a base Cellpose prediction.

    Args:
        modalities (sequence of tuple): ``(name, channels)`` per modality, where
            channels index the fused input stack.
        roles (sequence of str, optional): Which of ROLES to instantiate.
        width (int, optional): Hidden width of each role branch.
        nout (int, optional): Output channel count of the base prediction, so
            the foreground and flow slices are located as the loss locates them.
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

        self.branches = nn.ModuleDict()
        self.coefficients = nn.ParameterDict()
        for name, channels in zip(self.modality_names, self.modality_channels):
            for role in self.roles:
                n_out = 2 if role == "flow_refine" else 1
                self.branches[f"{name}__{role}"] = _RoleBranch(
                    len(channels), n_out, width)
                init = _OFF if role in NONNEGATIVE_ROLES else 0.
                self.coefficients[f"{name}__{role}"] = nn.Parameter(
                    torch.tensor(init, dtype=torch.float32))

        self.register_buffer("sobel", _sobel_kernels(), persistent=False)
        # when False the terms are skipped entirely, so the module is exactly a
        # no-op and the comparison against the stock model is exact
        self.enabled = True

    def coefficient(self, name, role):
        """The effective coefficient, after any non-negativity constraint."""
        raw = self.coefficients[f"{name}__{role}"]
        return F.softplus(raw) if role in NONNEGATIVE_ROLES else raw

    def coefficient_table(self):
        """Learned coefficients as ``{modality: {role: float}}``.

        A diagnostic, not evidence. These are not unique, not comparable across
        modalities, and not measurements of biological necessity -- see this
        module's docstring for the four specific reasons. Report
        contribution_report() alongside it, and settle claims about what a
        modality does with perturbation experiments.
        """
        return {name: {role: float(self.coefficient(name, role).detach())
                       for role in self.roles}
                for name in self.modality_names}

    def l1(self):
        """Sum of coefficient magnitudes, for sparsity regularization.

        Without it a modality spreads thinly over every role. Sparsity does not
        make the coefficients identifiable; it only keeps the table legible.
        """
        total = None
        for name in self.modality_names:
            for role in self.roles:
                term = self.coefficient(name, role).abs()
                total = term if total is None else total + term
        return total

    def _terms(self, inputs, dtype):
        """Yield ``(modality, role, target, term)`` for every active role term.

        Shared by forward() and contribution_report(), so what is measured is
        exactly what is applied.
        """
        for name, channels in zip(self.modality_names, self.modality_channels):
            x_m = inputs[:, list(channels)]
            for role in self.roles:
                coef = self.coefficient(name, role).to(dtype)
                raw = self.branches[f"{name}__{role}"](x_m).to(dtype)
                opinion = ROLE_SQUASH[role](raw)
                if role == "foreground_suppress":
                    term = coef * F.logsigmoid(SUPPRESS_GAIN * opinion)
                else:
                    term = coef * opinion
                yield name, role, ROLE_TARGET[role], term

    def _flows_from_potential(self, potential):
        return F.conv2d(potential, self.sobel.to(potential.dtype), padding=1)

    def forward(self, base, inputs):
        """Add role corrections to a base prediction.

        Args:
            base (torch.Tensor): Base prediction, ``[N x nout x Ly x Lx]``.
            inputs (torch.Tensor): Fused input stack, ``[N x C x Ly x Lx]``, at
                the same spatial size as ``base``.

        Returns:
            torch.Tensor: ``base`` with the role corrections added.
        """
        if not self.enabled:
            return base
        if base.shape[-2:] != inputs.shape[-2:]:
            raise ValueError(
                f"role corrections are added at full resolution, but base is "
                f"{tuple(base.shape[-2:])} and inputs are "
                f"{tuple(inputs.shape[-2:])}")

        cellprob = torch.zeros_like(base[:, -1:])
        flows = torch.zeros_like(base[:, -3:-1])
        potential = None
        for _, _, target, term in self._terms(inputs, base.dtype):
            if target == "cellprob":
                cellprob = cellprob + term
            elif target == "flows":
                flows = flows + term
            else:
                potential = term if potential is None else potential + term

        if potential is not None:
            # one gradient of the summed potential, so the centre terms compose
            # into a single consistent field
            flows = flows + self._flows_from_potential(potential)

        out = base.clone()
        out[:, -1:] = out[:, -1:] + cellprob
        out[:, -3:-1] = out[:, -3:-1] + flows
        return out

    @torch.no_grad()
    def contribution_report(self, inputs):
        """Measured effect size of each role term on this batch.

        This is the quantity worth reporting. A coefficient bounds a term per
        pixel but says nothing about how much of the image it acts on, so two
        branches sharing a coefficient can differ by orders of magnitude in
        total influence. Here each term is applied and its effect on the output
        it actually corrects is measured, with a flow_center term differentiated
        first so it is comparable with flow_refine.

        Average over held-out batches before reporting, and read it next to a
        perturbation experiment rather than instead of one.

        Args:
            inputs (torch.Tensor): Fused input stack, ``[N x C x Ly x Lx]``.

        Returns:
            dict: ``{modality: {role: {"target", "mean_abs", "max_abs",
            "frac_active"}}}``, where ``frac_active`` is the fraction of pixels
            whose contribution exceeds a hundredth of that term's own maximum.
        """
        report = {name: {} for name in self.modality_names}
        for name, role, target, term in self._terms(inputs, inputs.dtype):
            effect = self._flows_from_potential(term) if target == "potential" else term
            mag = effect.abs()
            peak = float(mag.max())
            thresh = peak / 100. if peak > 0 else float("inf")
            report[name][role] = {
                "target": "flows" if target == "potential" else target,
                "mean_abs": float(mag.mean()),
                "max_abs": peak,
                "frac_active": float((mag > thresh).to(torch.float32).mean()),
            }
        return report

    def extra_repr(self):
        return (f"modalities={self.modality_names}, roles={list(self.roles)}, "
                f"enabled={self.enabled}")
