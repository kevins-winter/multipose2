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


def _groups(width):
    """Largest group count up to four that divides ``width``.

    GroupNorm requires the channel count to be divisible by the group count, so
    a fixed min(4, width) fails for widths like 6, 10 or 21 -- which the
    capacity-matched control picks freely.
    """
    for g in (4, 2, 1):
        if width % g == 0:
            return g
    return 1


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
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
            nn.Conv2d(width, nout, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


FOREGROUND_CONSTRAINTS = ("none", "suppress_only", "sign_identifiable")
FLOW_PARAMETERIZATIONS = ("direct", "potential")


class MultiscaleBranch(nn.Module):
    """A depthwise-separable dilated stack at full resolution.

    Two stacked 3x3 convolutions see 5 pixels. A sensory neuron soma spans tens
    of pixels, so a branch asked to judge transcript density over a whole cell
    through a 5-pixel window cannot do it however well it is trained --
    receptive field, not depth or width, is the binding constraint here.

    Dilating a depthwise stack widens the receptive field geometrically for
    roughly the parameter count of a plain two-layer branch, and every layer
    stays at full resolution so the correction remains per-pixel.
    """

    def __init__(self, in_channels, nout, width=16, dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        if not dilations:
            raise ValueError("MultiscaleBranch needs at least one dilation")
        self.dilations = tuple(int(d) for d in dilations)
        self.stem = nn.Conv2d(in_channels, width, 1)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(width, width, 3, padding=d, dilation=d, groups=width),
                nn.GroupNorm(_groups(width), width),
                nn.GELU(),
                nn.Conv2d(width, width, 1),
            ) for d in self.dilations)
        self.head = nn.Conv2d(width, nout, 1)

    @property
    def receptive_field(self):
        """Side length in pixels that one output value can see."""
        return 1 + 2 * sum(self.dilations)

    def forward(self, x, modulation=None):
        h = self.stem(x)
        if modulation is not None:
            # FiLM: the frozen backbone's features scale and shift this
            # modality's features, so its correction depends on what the
            # backbone saw rather than on its own channels alone
            scale, shift = modulation
            h = h * (1. + scale) + shift
        for block in self.blocks:
            h = h + block(h)
        return self.head(h)


class ModalityCorrections(nn.Module):
    """Per-modality corrections to foreground and flows, optionally conditioned.

    Three configurations, which are the three arms worth comparing:

    ``context_dim=None, cross_modality=False``
        Independent corrections. A modality's correction is the same for the
        same input whatever the other modalities say.
    ``context_dim=D, cross_modality=False``
        Each modality's features are modulated by the frozen backbone's, so a
        transcript correction can depend on the morphology around it. Still
        attributable per modality.
    ``context_dim=D, cross_modality=True``
        Modality features mix before the route heads, so conjunctions between
        modalities are expressible. Attribution is given up in exchange.

    The constraints that used to define four named operators are arguments here
    instead, because separate foreground and flow routes are the defensible part
    of the hypothesis and the rest are extra assumptions worth testing:

    ``foreground_constraint``
        ``"none"`` signed coefficient over a tanh branch, so one modality may
        raise the logit in one place and lower it in another. ``"suppress_only"``
        a non-negative coefficient over a log-sigmoid, which can only subtract.
        ``"sign_identifiable"`` a signed coefficient over a sigmoid, so the sign
        means something but the pattern cannot change sign.
    ``flow_parameterization``
        ``"direct"`` adds a two-channel correction to the flow field.
        ``"potential"`` predicts a scalar and adds its gradient, which is how
        Cellpose builds flows in the first place.

    Defaults are the least assumptive of each. Flow corrections are not purely
    cosmetic: mask reconstruction follows the flows, so they can merge, split or
    remove instances as well as move a boundary.
    """

    def __init__(self, modalities, width=16, nout=3, context_dim=None,
                 cross_modality=False, dilations=(1, 2, 4, 8, 16),
                 foreground_constraint="none", flow_parameterization="direct",
                 context_width=32, modality_dropout=0.):
        super().__init__()
        if foreground_constraint not in FOREGROUND_CONSTRAINTS:
            raise ValueError(
                f"foreground_constraint must be one of {FOREGROUND_CONSTRAINTS}, "
                f"got {foreground_constraint!r}")
        if flow_parameterization not in FLOW_PARAMETERIZATIONS:
            raise ValueError(
                f"flow_parameterization must be one of {FLOW_PARAMETERIZATIONS}, "
                f"got {flow_parameterization!r}")
        if not 0. <= modality_dropout < 1.:
            raise ValueError("modality_dropout must be in [0, 1)")
        if cross_modality and context_dim is None:
            roles_logger.info(
                ">>> cross_modality without context_dim: modalities will mix, "
                "but neither they nor the mix can see the backbone's features")

        self.modality_names = [str(name) for name, _ in modalities]
        self.modality_channels = [tuple(int(c) for c in ch) for _, ch in modalities]
        self.width = int(width)
        self.nout = int(nout)
        self.context_dim = context_dim
        self.cross_modality = bool(cross_modality)
        self.foreground_constraint = foreground_constraint
        self.flow_parameterization = flow_parameterization
        self.modality_dropout = float(modality_dropout)
        self._presence_override = None
        self.enabled = True

        # one feature extractor per modality; the route heads come after, so a
        # modality costs one branch rather than one per route
        self.branches = nn.ModuleDict({
            name: MultiscaleBranch(len(channels), width, width=width,
                                   dilations=dilations)
            for name, channels in zip(self.modality_names, self.modality_channels)})

        if context_dim is not None:
            self.context_proj = nn.Conv2d(int(context_dim), context_width, 1)
            self.context_film = nn.ModuleDict({
                name: nn.Conv2d(context_width, 2 * width, 1)
                for name in self.modality_names})
            for film in self.context_film.values():
                nn.init.zeros_(film.weight)
                nn.init.zeros_(film.bias)

        flow_out = 1 if flow_parameterization == "potential" else 2
        if self.cross_modality:
            # a single 1x1 convolution would be linear in the concatenated
            # features, so the only cross-modality nonlinearity would be the
            # head's. Mix nonlinearly, or the arm cannot represent the
            # conjunctions it exists to test.
            self.mix = nn.Sequential(
                nn.Conv2d(width * len(self.modality_names), width, 1),
                nn.GroupNorm(_groups(width), width),
                nn.GELU(),
                nn.Conv2d(width, width, 3, padding=1),
            )
            coefs = ["mixed"]
        else:
            self.mix = None
            coefs = list(self.modality_names)
        self.foreground_head = nn.ModuleDict({
            key: nn.Conv2d(width, 1, 1) for key in coefs})
        self.flow_head = nn.ModuleDict({
            key: nn.Conv2d(width, flow_out, 1) for key in coefs})
        self.coefficients = nn.ParameterDict()
        for key in coefs:
            for route in ("foreground", "flow"):
                init = (_OFF if (route == "foreground"
                                 and foreground_constraint == "suppress_only")
                        else 0.)
                self.coefficients[f"{key}__{route}"] = nn.Parameter(
                    torch.tensor(init, dtype=torch.float32))

        self.register_buffer("sobel", _sobel_kernels(), persistent=False)

    @property
    def receptive_field(self):
        return next(iter(self.branches.values())).receptive_field

    @property
    def _keys(self):
        return ["mixed"] if self.cross_modality else self.modality_names

    def coefficient(self, key, route):
        raw = self.coefficients[f"{key}__{route}"]
        if route == "foreground" and self.foreground_constraint == "suppress_only":
            return F.softplus(raw)
        return raw

    def coefficient_table(self):
        """Learned coefficients. A diagnostic, never evidence.

        Not unique and not comparable across modalities: the coefficient bounds
        a term per pixel but not how much of the image it acts on, correlated
        modalities substitute for one another, and under cross_modality there is
        one entry for the mix rather than one per modality.
        """
        return {key: {route: float(self.coefficient(key, route).detach())
                      for route in ("foreground", "flow")}
                for key in self._keys}

    def l1(self):
        total = None
        for key in self._keys:
            for route in ("foreground", "flow"):
                term = self.coefficient(key, route).abs()
                total = term if total is None else total + term
        return total

    def _modality_features(self, inputs, context, dtype):
        """Per-modality feature maps, FiLM-modulated by context when given."""
        modulation = {}
        if self.context_dim is not None and context is not None:
            ctx = F.gelu(self.context_proj(context.to(dtype)))
            ctx = F.interpolate(ctx, size=inputs.shape[-2:], mode="bilinear",
                                align_corners=False)
            for name in self.modality_names:
                scale, shift = self.context_film[name](ctx).chunk(2, dim=1)
                modulation[name] = (scale, shift)

        presence = self._presence(inputs.shape[0], inputs.device, dtype)
        feats, avail = {}, {}
        for i, (name, channels) in enumerate(zip(self.modality_names,
                                                 self.modality_channels)):
            x_m = inputs[:, list(channels)].to(dtype)
            h = self.branches[name](x_m, modulation.get(name))
            mask = presence[:, i].reshape(-1, 1, 1, 1)
            # masking the features makes an absent modality absent from the mix
            feats[name] = h * mask
            avail[name] = mask
        return feats, avail

    def _route_terms(self, inputs, context, dtype):
        """Yield ``(key, route, term)`` for each correction term."""
        feats, avail = self._modality_features(inputs, context, dtype)
        if self.cross_modality:
            mixed = self.mix(torch.cat([feats[n] for n in self.modality_names], 1))
            sources, masks = {"mixed": mixed}, {"mixed": None}
        else:
            sources, masks = feats, avail
        for key, h in sources.items():
            # masking the features is not enough on its own: a route head has a
            # bias, so head(0) is non-zero and an absent modality would still
            # contribute. Mask the term too, which also keeps the independent
            # arm exactly additive across modalities.
            mask = masks[key]
            fg = self.foreground_head[key](h)
            if self.foreground_constraint == "suppress_only":
                fg = F.logsigmoid(SUPPRESS_GAIN * torch.tanh(fg))
            elif self.foreground_constraint == "sign_identifiable":
                fg = torch.sigmoid(fg)
            else:
                fg = torch.tanh(fg)
            fg_term = self.coefficient(key, "foreground").to(dtype) * fg
            yield key, "foreground", fg_term if mask is None else fg_term * mask

            fl = self.flow_head[key](h)
            if self.flow_parameterization == "potential":
                fl = torch.sigmoid(fl)
            else:
                fl = torch.tanh(fl)
            fl_term = self.coefficient(key, "flow").to(dtype) * fl
            yield key, "flow", fl_term if mask is None else fl_term * mask

    def forward(self, base, inputs, context=None):
        if not self.enabled:
            return base
        if base.shape[-2:] != inputs.shape[-2:]:
            raise ValueError(
                f"corrections are added at full resolution, but base is "
                f"{tuple(base.shape[-2:])} and inputs are "
                f"{tuple(inputs.shape[-2:])}")
        if self.context_dim is not None and context is None:
            raise ValueError(
                "this module was built with context_dim, so it needs the frozen "
                "trunk's features; Transformer.forward passes them automatically")

        foreground = torch.zeros_like(base[:, -1:])
        flows = torch.zeros_like(base[:, -3:-1])
        potential = None
        for _, route, term in self._route_terms(inputs, context, base.dtype):
            if route == "foreground":
                foreground = foreground + term
            elif self.flow_parameterization == "potential":
                potential = term if potential is None else potential + term
            else:
                flows = flows + term
        if potential is not None:
            flows = flows + F.conv2d(potential, self.sobel.to(potential.dtype),
                                     padding=1)

        out = base.clone()
        out[:, -1:] = out[:, -1:] + foreground
        out[:, -3:-1] = out[:, -3:-1] + flows
        return out

    @torch.no_grad()
    def contribution_report(self, inputs, context=None):
        """Measured effect of each term on the output it corrects."""
        report = {key: {} for key in self._keys}
        for key, route, term in self._route_terms(inputs, context, inputs.dtype):
            effect = term
            if route == "flow" and self.flow_parameterization == "potential":
                effect = F.conv2d(term, self.sobel.to(term.dtype), padding=1)
            mag = effect.abs()
            peak = float(mag.max())
            report[key][route] = {
                "mean_abs": float(mag.mean()),
                "max_abs": peak,
                "frac_active": float(
                    (mag > (peak / 100. if peak > 0 else float("inf"))
                     ).to(torch.float32).mean()),
            }
        return report

    def extra_repr(self):
        return (f"modalities={self.modality_names}, width={self.width}, "
                f"receptive_field={self.receptive_field}, "
                f"context_dim={self.context_dim}, "
                f"cross_modality={self.cross_modality}, "
                f"foreground_constraint={self.foreground_constraint!r}, "
                f"flow_parameterization={self.flow_parameterization!r}, "
                f"enabled={self.enabled}")


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

    def __init__(self, modalities, roles=ROLES, width=16, nout=3,
                 modality_dropout=0.):
        super().__init__()
        if not 0. <= modality_dropout < 1.:
            raise ValueError("modality_dropout must be in [0, 1)")
        self.modality_dropout = float(modality_dropout)
        self._presence_override = None
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

    def set_presence(self, presence):
        """Declare which modalities are available, or None to clear.

        An unavailable modality and an observed modality reading zero are
        different statements, and they must not produce the same correction.
        Zeroing a channel's values says "we measured here and saw nothing",
        which is evidence a branch can legitimately learn from. Marking it
        absent says "this channel does not exist for this sample", which should
        contribute nothing at all.

        Args:
            presence: None, or a tensor broadcastable to ``[N, n_modalities]``
                where non-zero means available. A 1-D tensor of length
                n_modalities applies to every sample.
        """
        if presence is None:
            self._presence_override = None
            return self
        presence = torch.as_tensor(presence)
        if presence.ndim == 1:
            presence = presence.unsqueeze(0)
        if presence.shape[-1] != len(self.modality_names):
            raise ValueError(
                f"presence must cover {len(self.modality_names)} modalities, "
                f"got {presence.shape[-1]}")
        self._presence_override = presence.to(torch.float32)
        return self

    def _presence(self, n, device, dtype):
        """Per-sample availability, ``[N, n_modalities]``.

        An explicit override wins. Otherwise modalities are all available,
        except that during training modality_dropout marks some absent so the
        model sees both regimes and does not come to depend on a channel that
        may be missing at inference.
        """
        m = len(self.modality_names)
        if self._presence_override is not None:
            out = self._presence_override.to(device=device, dtype=dtype)
            return out.expand(n, m) if out.shape[0] == 1 else out
        if self.training and self.modality_dropout > 0.:
            keep = (torch.rand((n, m), device=device) >= self.modality_dropout)
            # never drop every modality at once; that supervises nothing
            empty = ~keep.any(dim=1)
            if empty.any():
                keep[empty, torch.randint(0, m, (int(empty.sum()),),
                                          device=device)] = True
            return keep.to(dtype)
        return torch.ones((n, m), device=device, dtype=dtype)

    def _terms(self, inputs, dtype):
        """Yield ``(modality, role, target, term)`` for every active role term.

        Shared by forward() and contribution_report(), so what is measured is
        exactly what is applied.
        """
        presence = self._presence(inputs.shape[0], inputs.device, dtype)
        for i, (name, channels) in enumerate(zip(self.modality_names,
                                                 self.modality_channels)):
            x_m = inputs[:, list(channels)]
            # [N, 1, 1, 1], so an absent modality contributes exactly nothing
            avail = presence[:, i].reshape(-1, 1, 1, 1)
            for role in self.roles:
                coef = self.coefficient(name, role).to(dtype)
                raw = self.branches[f"{name}__{role}"](x_m).to(dtype)
                opinion = ROLE_SQUASH[role](raw)
                if role == "foreground_suppress":
                    term = coef * F.logsigmoid(SUPPRESS_GAIN * opinion)
                else:
                    term = coef * opinion
                yield name, role, ROLE_TARGET[role], term * avail

    def _flows_from_potential(self, potential):
        return F.conv2d(potential, self.sobel.to(potential.dtype), padding=1)

    def forward(self, base, inputs, context=None):
        """Add role corrections to a base prediction.

        Args:
            base (torch.Tensor): Base prediction, ``[N x nout x Ly x Lx]``.
            inputs (torch.Tensor): Fused input stack, ``[N x C x Ly x Lx]``, at
                the same spatial size as ``base``.
            context: Accepted and ignored, so every correction module takes the
                same call. This one reads raw channels only; ModalityCorrections
                is the variant that uses the trunk's features.

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
                f"modality_dropout={self.modality_dropout}, "
                f"enabled={self.enabled}")


class UnrestrictedCorrection(nn.Module):
    """A correction network with the same parameter budget and no structure.

    This is the control arm. RoleMixer imposes a specific structure -- one
    branch per modality, separate routes to the foreground logit and the flow
    field, a sign constraint on suppression, flows entered as the gradient of a
    potential. Any improvement it shows over a frozen baseline could come from
    that structure, or simply from the parameters the structure brought with it.

    Without this control the comparison cannot distinguish the two, so a result
    showing RoleMixer beats the baseline would say nothing about whether the
    roles matter. This module reads every modality jointly, writes all output
    channels directly, and is sized to match a given RoleMixer, so the only
    difference between the arms is the structure.

    Like RoleMixer it starts as a no-op, so both arms depart from an identical
    baseline: the last convolution is zero-initialized, which leaves its own
    gradient intact so the layer learns first and the rest follows.

    Args:
        in_channels (int): Channels in the fused input stack.
        nout (int, optional): Output channel count of the base prediction.
        width (int, optional): Hidden width.
        depth (int, optional): Number of hidden convolutions.
    """

    def __init__(self, in_channels, nout=3, width=16,
                 dilations=(1, 2, 4, 8, 16)):
        super().__init__()
        # the same stack the per-modality branches use, so the arms differ in
        # structure alone. With a plain two-convolution body the control would
        # see 5 pixels against their 63, and a win for the structured arm could
        # be explained by receptive field rather than by structure.
        self.net = MultiscaleBranch(int(in_channels), int(nout), width=width,
                                    dilations=dilations)
        with torch.no_grad():
            self.net.head.weight.zero_()
            self.net.head.bias.zero_()
        self.nout = int(nout)
        self.enabled = True

    @property
    def receptive_field(self):
        return self.net.receptive_field

    @staticmethod
    def _param_count(in_channels, nout, width, n_dilations):
        """Parameters of the MultiscaleBranch this wraps, without building it."""
        return (in_channels * width + width
                + n_dilations * (13 * width + width * width)
                + width * nout + nout)

    @classmethod
    def matched_to(cls, corrections, in_channels, nout=3,
                   dilations=(1, 2, 4, 8, 16), max_width=512):
        """Build one whose parameter count is as close as possible.

        Args:
            corrections (nn.Module): The correction module this is the control
                for; any of RoleMixer or ModalityCorrections.
            in_channels (int): Channels in the fused input stack.
            nout (int, optional): Output channel count.
            dilations (tuple, optional): Matched to the structured arm's, so the
                two have the same receptive field.
            max_width (int, optional): Largest width to consider.

        Returns:
            UnrestrictedCorrection: Sized to match, with its realized parameter
            count recorded on ``matched_target`` and ``matched_error``.
        """
        target = sum(p.numel() for p in corrections.parameters())
        n_d = len(dilations)
        best = min(range(1, max_width + 1),
                   key=lambda w: abs(cls._param_count(in_channels, nout, w, n_d)
                                     - target))
        model = cls(in_channels, nout=nout, width=best, dilations=dilations)
        realized = sum(p.numel() for p in model.parameters())
        model.matched_target = int(target)
        model.matched_error = int(realized - target)
        roles_logger.info(
            ">>> capacity-matched control: width=%d, receptive field %d px, "
            "%d parameters vs %d (%+d)", best, model.receptive_field, realized,
            target, realized - target)
        return model

    def forward(self, base, inputs, context=None):
        """``context`` is accepted and ignored, so every arm takes the same call."""
        if not self.enabled:
            return base
        if base.shape[-2:] != inputs.shape[-2:]:
            raise ValueError(
                f"corrections are added at full resolution, but base is "
                f"{tuple(base.shape[-2:])} and inputs are "
                f"{tuple(inputs.shape[-2:])}")
        return base + self.net(inputs.to(base.dtype)).to(base.dtype)

    def extra_repr(self):
        return (f"nout={self.nout}, receptive_field={self.receptive_field}, "
                f"enabled={self.enabled}")


# Availability handling is identical for both correction modules: an unavailable
# modality contributes nothing, while an observed modality reading zero is still
# evidence a branch may learn from.
ModalityCorrections.set_presence = RoleMixer.set_presence
ModalityCorrections._presence = RoleMixer._presence
