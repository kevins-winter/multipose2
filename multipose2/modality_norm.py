"""Per-modality normalization with parameters fitted once on a training set.

Cellpose normalizes every image by its own 1st and 99th percentiles. That is the
right choice for a generalist facing unknown acquisition scales, and the wrong
choice for a fixed platform carrying heterogeneous modalities:

- For morphology (H&E) per-crop percentiles are correct. Staining intensity
  varies between sections, that variation is nuisance, and relative contrast
  carries the signal.

- For transcript counts they are destructive, because absolute density *is* the
  signal. A blank crop holding two stray transcripts has those two dots
  stretched across the full dynamic range, while a crop holding two hundred maps
  to the same [0, 1]. Per-crop normalization inverts the information it is
  meant to preserve, and makes background indistinguishable from a
  transcript-rich cell.

So each modality declares its own mode, and the modes whose parameters are
global are fitted once over the training set and kept, so training and inference
apply an identical transform.

Modes:
    ``percentile``   per-crop lower/upper percentile rescaling, as Cellpose does.
                     Nothing to fit. For morphology.
    ``log1p_fixed``  ``log1p(x) / s``, with ``s`` fitted as a percentile of
                     ``log1p(x)`` pooled over the training set. Variance
                     stabilizing, and absolute scale is preserved across crops.
                     For counts.
    ``fixed``        ``(x - lo) / (hi - lo)``, with both fitted globally. For
                     channels wanting linear rescaling on a common scale.
    ``identity``     left alone. For channels already in sensible units.
"""
import dataclasses
import json
import logging
from pathlib import Path

import numpy as np

from .transforms import normalize99

modality_norm_logger = logging.getLogger(__name__)

MODES = ("percentile", "log1p_fixed", "fixed", "identity")
NEEDS_FIT = ("log1p_fixed", "fixed")

DEFAULT_PARAMS = {
    "percentile": {"lower": 1.0, "upper": 99.0},
    "log1p_fixed": {"percentile": 99.9},
    "fixed": {"lower": 0.1, "upper": 99.9},
    "identity": {},
}


@dataclasses.dataclass
class ModalitySpec:
    """How one modality's channels are normalized.

    Attributes:
        name (str): Modality name, for reporting and for the stored manifest.
        channels (tuple of int): Indices of this modality in the fused stack.
        mode (str): One of MODES.
        params (dict): Mode configuration, defaulted from DEFAULT_PARAMS.
        fitted (dict): Parameters measured by fit(); empty until then.
    """

    name: str
    channels: tuple
    mode: str = "percentile"
    params: dict = dataclasses.field(default_factory=dict)
    fitted: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        self.channels = tuple(int(c) for c in self.channels)
        if not self.channels:
            raise ValueError(f"modality {self.name!r} has no channels")
        self.params = {**DEFAULT_PARAMS[self.mode], **(self.params or {})}

    @property
    def needs_fit(self):
        return self.mode in NEEDS_FIT

    @property
    def is_fitted(self):
        return not self.needs_fit or bool(self.fitted)

    def fit(self, pooled):
        """Measure global parameters from pooled training pixels of this modality."""
        if not self.needs_fit:
            return
        if self.mode == "log1p_fixed":
            scale = float(np.percentile(np.log1p(np.maximum(pooled, 0.)),
                                        self.params["percentile"]))
            if not np.isfinite(scale) or scale <= 0:
                modality_norm_logger.warning(
                    "modality %r: log1p scale came out as %r, so this modality is "
                    "flat across the whole training set; using 1.0 and leaving the "
                    "values unscaled", self.name, scale)
                scale = 1.
            self.fitted = {"scale": scale}
        else:
            lo, hi = np.percentile(pooled, [self.params["lower"],
                                            self.params["upper"]])
            if not np.isfinite(hi - lo) or hi - lo <= 0:
                modality_norm_logger.warning(
                    "modality %r: fitted range is empty (lo=%r hi=%r), so this "
                    "modality is flat across the whole training set; using a unit "
                    "range", self.name, lo, hi)
                lo, hi = float(lo), float(lo) + 1.
            self.fitted = {"lo": float(lo), "hi": float(hi)}

    def apply(self, block):
        """Normalize a channels-first block of this modality's channels."""
        if self.mode == "identity":
            return block
        if self.mode == "percentile":
            for c in range(block.shape[0]):
                block[c] = normalize99(block[c], lower=self.params["lower"],
                                       upper=self.params["upper"], copy=False,
                                       downsample=True)
            return block
        if not self.is_fitted:
            raise RuntimeError(
                f"modality {self.name!r} uses mode {self.mode!r}, whose parameters "
                "are fitted on the training set; call fit() or load a fitted "
                "normalizer before applying it"
            )
        if self.mode == "log1p_fixed":
            return np.log1p(np.maximum(block, 0.)) / self.fitted["scale"]
        lo, hi = self.fitted["lo"], self.fitted["hi"]
        return (block - lo) / (hi - lo)


class ModalityNormalizer:
    """Applies a per-modality normalization, with globals fitted once.

    Args:
        specs (sequence of ModalitySpec): One per modality. Channel indices must
            not overlap.
    """

    def __init__(self, specs):
        self.specs = list(specs)
        if not self.specs:
            raise ValueError("ModalityNormalizer needs at least one ModalitySpec")
        seen = {}
        for spec in self.specs:
            for c in spec.channels:
                if c in seen:
                    raise ValueError(
                        f"channel {c} is claimed by both {seen[c]!r} and "
                        f"{spec.name!r}")
                seen[c] = spec.name
        self.n_channels = max(seen) + 1
        if sorted(seen) != list(range(self.n_channels)):
            missing = sorted(set(range(self.n_channels)) - set(seen))
            raise ValueError(f"channels {missing} are not covered by any modality")

    @classmethod
    def from_layout(cls, layout):
        """Build from a compact description, assigning channels in order.

        Args:
            layout (sequence of dict): Each with ``name``, either ``channels``
                (explicit indices) or ``n_channels`` (assigned in order), and
                optionally ``mode`` and ``params``. The order must match the
                order the modalities were fused in.

        Returns:
            ModalityNormalizer
        """
        specs, start = [], 0
        for entry in layout:
            channels = entry.get("channels")
            if channels is None:
                n = int(entry["n_channels"])
                channels = tuple(range(start, start + n))
            start = max(start, max(channels) + 1)
            specs.append(ModalitySpec(name=entry["name"], channels=channels,
                                      mode=entry.get("mode", "percentile"),
                                      params=entry.get("params")))
        return cls(specs)

    @property
    def is_fitted(self):
        return all(spec.is_fitted for spec in self.specs)

    def fit(self, images, channel_axis=0, max_pixels_per_image=200_000):
        """Fit global parameters over a training set.

        Args:
            images (sequence of ndarray): Training images, all with the same
                channel count.
            channel_axis (int, optional): Which axis holds channels.
            max_pixels_per_image (int, optional): Pixels sampled per image per
                modality, on a regular stride, to bound memory.

        Returns:
            ModalityNormalizer: self, so this can be chained.
        """
        pending = [spec for spec in self.specs if spec.needs_fit]
        if not pending:
            return self
        pools = {spec.name: [] for spec in pending}
        for img in images:
            moved = np.moveaxis(np.asarray(img), channel_axis, 0)
            if moved.shape[0] != self.n_channels:
                raise ValueError(
                    f"expected {self.n_channels} channels, got {moved.shape[0]}")
            for spec in pending:
                flat = np.asarray(moved[list(spec.channels)],
                                  dtype=np.float64).reshape(-1)
                if flat.size > max_pixels_per_image:
                    flat = flat[::max(1, flat.size // max_pixels_per_image)]
                pools[spec.name].append(flat)
        for spec in pending:
            spec.fit(np.concatenate(pools[spec.name]))
            modality_norm_logger.info(">>> fitted %s (%s): %s", spec.name,
                                      spec.mode, spec.fitted)
        return self

    def __call__(self, img, channel_axis=0):
        """Normalize one image, returning float32 with the same axis order."""
        moved = np.moveaxis(np.asarray(img), channel_axis, 0)
        if moved.shape[0] != self.n_channels:
            raise ValueError(
                f"expected {self.n_channels} channels, got {moved.shape[0]}")
        out = moved.astype(np.float32, copy=True)
        for spec in self.specs:
            idx = list(spec.channels)
            out[idx] = spec.apply(out[idx])
        return np.moveaxis(out, 0, channel_axis)

    def channel_report(self, img, channel_axis=0):
        """Per-channel standard deviation before and after, for diagnostics.

        A channel whose normalized standard deviation is zero contributes
        nothing, which is the failure this module exists to make visible.
        """
        raw = np.moveaxis(np.asarray(img), channel_axis, 0).astype(np.float32)
        norm = np.moveaxis(self(img, channel_axis=channel_axis), channel_axis, 0)
        owner = {c: spec for spec in self.specs for c in spec.channels}
        rows = []
        for c in range(self.n_channels):
            spec = owner[c]
            rows.append({"channel": c, "modality": spec.name, "mode": spec.mode,
                         "raw_std": float(raw[c].std()),
                         "norm_std": float(norm[c].std()),
                         "dead": float(norm[c].std()) == 0.})
        return rows

    def to_dict(self):
        return {"n_channels": self.n_channels,
                "specs": [dataclasses.asdict(spec) for spec in self.specs]}

    @classmethod
    def from_dict(cls, data):
        return cls([ModalitySpec(**spec) for spec in data["specs"]])

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        return Path(path)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def __repr__(self):
        parts = ", ".join(f"{s.name}:{s.mode}{list(s.channels)}"
                          for s in self.specs)
        return f"ModalityNormalizer({parts}, fitted={self.is_fitted})"
