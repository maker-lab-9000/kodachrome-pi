"""Per-image white balance and exposure normalisation.

This is the "dynamic" half of the Kodachrome pipeline. The LUT fitted by the
trainer expects input at a neutral white point and a fixed exposure, so the
same normalisation runs on every capture: a tungsten-lit room and a cloudy
street both reach the LUT looking like the images it was fitted on. The
trainer applies this exact code to the corpora (``normalize_float``); the Pi
applies the same maths through three 256-entry lookup tables
(``normalize_u8``).

Why three 1D tables suffice
---------------------------
White balance is a per-channel gain in linear light and exposure is a scalar
gain in linear light, so their composite sRGB-to-sRGB map is three
independent monotone functions of one byte. ``cv2.LUT`` applies that in
milliseconds on a Pi 400.

Targets versus sources
----------------------
Kodachrome scans are normalised with ``white_balance=False``: the film's
daylight balance and warm cast are part of the look being learned. Only the
per-slide exposure lottery is removed.

Idempotence, approximately
--------------------------
Normalising an already-normalised image is close to, but not exactly, a
no-op. The statistics mask is recomputed on the transformed pixels, so the
second pass averages a slightly different subset and applies a small
correction. Repeated passes converge rather than drift. Nothing here
normalises twice, so this is documented rather than engineered away:
iterating to a fixed point would cost time on every frame to remove an
error of about two 8-bit levels that no caller ever sees.

Reporting clamps
----------------
Both gains are clamped to sane ranges so a night shot is not amplified into
daylight. When a clamp bites, the resulting image is *not* fully normalised,
and the LUT then sees input it was not fitted on. ``Gains.clamped`` records
that so the trainer can publish a clamp rate per corpus and the capture log
can explain a shot that came out wrong, instead of the limit acting
silently.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields

import numpy as np

from ._cv2 import require_cv2
from .color import LUMA_709, linear_to_srgb, srgb_to_linear

cv2 = require_cv2()

_EPS = 1e-6


def _check_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


@dataclass
class NormalizeParams:
    white_balance: bool = True
    wb_gain_min: float = 0.6
    wb_gain_max: float = 1.6
    exposure_target_median: float = 0.25
    exposure_gain_min: float = 0.5
    exposure_gain_max: float = 3.0
    stats_lum_min: float = 0.02
    stats_lum_max: float = 0.90
    # Levels: a training-side normaliser for scanned targets. Archival scans
    # are deliberately flat -- across 150 LoC Kodachromes the white point sat
    # at a median 0.72 linear luminance -- and learning from them unstretched
    # taught the LUT to push white to 0.85. Levels puts p0.5 at black and
    # p99.5 at white, then a gamma puts the median on the exposure target
    # without moving either end (a gain would drag the white point again).
    # Never used on the capture path; ``normalize_u8`` refuses it.
    levels: bool = False
    levels_low_pct: float = 0.5
    levels_high_pct: float = 99.5
    levels_max_stretch: float = 4.0
    levels_gamma_min: float = 0.5
    levels_gamma_max: float = 2.0

    def __post_init__(self) -> None:
        for f in fields(self):
            if f.name not in ("white_balance", "levels"):
                _check_finite(f.name, getattr(self, f.name))
        if self.levels and self.white_balance:
            raise ValueError(
                "levels normalisation does not combine with white_balance; targets keep "
                "their cast (see docs/decisions.md), so pass white_balance=False"
            )
        if not 0.0 <= self.levels_low_pct < self.levels_high_pct <= 100.0:
            raise ValueError(
                f"levels_low_pct ({self.levels_low_pct}) must be below "
                f"levels_high_pct ({self.levels_high_pct}), both within [0, 100]"
            )
        if self.levels_max_stretch < 1.0:
            raise ValueError(
                f"levels_max_stretch must be at least 1, got {self.levels_max_stretch}"
            )
        if not 0.0 < self.levels_gamma_min < self.levels_gamma_max:
            raise ValueError(
                f"levels_gamma_min ({self.levels_gamma_min}) must be positive and below "
                f"levels_gamma_max ({self.levels_gamma_max})"
            )
        if self.wb_gain_min <= 0:
            raise ValueError(f"wb_gain_min must be positive, got {self.wb_gain_min}")
        if self.exposure_gain_min <= 0:
            raise ValueError(f"exposure_gain_min must be positive, got {self.exposure_gain_min}")
        if self.wb_gain_min >= self.wb_gain_max:
            raise ValueError(
                f"wb_gain_min ({self.wb_gain_min}) must be below wb_gain_max ({self.wb_gain_max})"
            )
        if self.exposure_gain_min >= self.exposure_gain_max:
            raise ValueError(
                f"exposure_gain_min ({self.exposure_gain_min}) must be below "
                f"exposure_gain_max ({self.exposure_gain_max})"
            )
        if not 0.0 < self.exposure_target_median < 1.0:
            raise ValueError(
                f"exposure_target_median must be in (0, 1), got {self.exposure_target_median}"
            )
        if not 0.0 <= self.stats_lum_min < self.stats_lum_max <= 1.0:
            raise ValueError(
                f"stats_lum_min ({self.stats_lum_min}) must be below "
                f"stats_lum_max ({self.stats_lum_max}), both within [0, 1]"
            )

    @classmethod
    def from_dict(cls, d: dict) -> NormalizeParams:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Gains:
    """Per-channel white balance gains and a scalar exposure gain, in linear light."""

    wb: np.ndarray
    exposure: float
    clamped: dict = field(default_factory=lambda: {"wb": False, "exposure": False})
    # Set only by the levels path: the black/white points found, the stretch
    # and gamma applied. ``exposure`` stays 1.0 there, because no gain was.
    levels: dict | None = None

    @property
    def combined(self) -> np.ndarray:
        return (np.asarray(self.wb, dtype=np.float32) * np.float32(self.exposure)).astype(
            np.float32
        )

    def to_dict(self) -> dict:
        d = {
            "wb": [round(float(g), 4) for g in self.wb],
            "exposure": round(float(self.exposure), 4),
            "clamped": dict(self.clamped),
        }
        if self.levels is not None:
            d["levels"] = {k: round(float(v), 4) for k, v in self.levels.items()}
        return d


def _require_float_image(rgb: np.ndarray, caller: str) -> np.ndarray:
    """Guard the float path against uint8 input, which it would destroy silently.

    ``srgb_to_linear`` clips to [0, 1], so every uint8 value from 1 to 255
    collapses to 1.0 and the image's entire content is lost with no error
    raised — a trainer handing over the result of ``load_rgb`` would get a
    flat frame and never know. Every other public entry point in this package
    validates its input; this one must too.
    """
    rgb = np.asarray(rgb)
    if not np.issubdtype(rgb.dtype, np.floating):
        raise ValueError(
            f"{caller} expects a float image in [0, 1], got dtype {rgb.dtype}. "
            "Divide a uint8 array by 255 first, or use normalize_u8."
        )
    return rgb


def compute_gains(rgb: np.ndarray, params: NormalizeParams) -> Gains:
    """Grey-world white balance and median-to-target exposure from an sRGB float image."""
    rgb = _require_float_image(rgb, "compute_gains")
    lin = srgb_to_linear(rgb).reshape(-1, 3)
    lum = lin @ LUMA_709
    mask = (lum >= params.stats_lum_min) & (lum <= params.stats_lum_max)
    if mask.mean() < 0.01:
        mask = np.ones_like(mask)
    sel = lin[mask]

    wb_clamped = False
    if params.white_balance:
        means = np.maximum(sel.mean(axis=0), _EPS)
        raw = float(means @ LUMA_709) / means
        wb = np.clip(raw, params.wb_gain_min, params.wb_gain_max).astype(np.float32)
        # Compare against the bounds, not against the clipped array: `raw` and
        # `wb` can differ by a float32 cast alone, which is not a clamp.
        wb_clamped = bool(
            np.any(raw < params.wb_gain_min) or np.any(raw > params.wb_gain_max)
        )
    else:
        wb = np.ones(3, dtype=np.float32)

    median_lum = float(np.median((sel * wb) @ LUMA_709))
    raw_exposure = params.exposure_target_median / max(median_lum, _EPS)
    exposure = float(np.clip(raw_exposure, params.exposure_gain_min, params.exposure_gain_max))
    return Gains(
        wb=wb,
        exposure=exposure,
        clamped={
            "wb": wb_clamped,
            "exposure": not params.exposure_gain_min <= raw_exposure <= params.exposure_gain_max,
        },
    )


def apply_gains_float(rgb: np.ndarray, gains: Gains) -> np.ndarray:
    rgb = _require_float_image(rgb, "apply_gains_float")
    return linear_to_srgb(np.clip(srgb_to_linear(rgb) * gains.combined, 0.0, 1.0))


def levels_float(rgb: np.ndarray, params: NormalizeParams) -> tuple[np.ndarray, Gains]:
    """Stretch p_low..p_high of luminance to 0..1, then gamma the median to target.

    Everything happens in linear light with ONE black point and ONE stretch
    shared by all three channels, so a neutral input stays neutral. The gamma
    is likewise per channel with a shared exponent. Both the stretch and the
    gamma are clamped and the clamp is recorded, never silent.
    """
    rgb = _require_float_image(rgb, "levels_float")
    lin = srgb_to_linear(rgb)
    lum = lin @ LUMA_709
    # ONE black point and ONE stretch for all channels, both from luminance.
    # A per-channel black point was tried (2026-09-04) because dark greys came
    # out olive: it made the fit worse than identity. Subtracting a different
    # offset per channel and clipping leaves 0.5% of pixels at zero in one
    # channel but not the others -- saturated dark colours the film never
    # produced, seeded straight into the target distribution. The spread of
    # the per-channel black points is still measured and recorded, so a
    # scan with a strongly coloured base is visible in the corpus report.
    pct = np.percentile(lum, [params.levels_low_pct, params.levels_high_pct])
    lo, hi = float(pct[0]), float(pct[1])
    lo_c = np.percentile(lin.reshape(-1, 3), params.levels_low_pct, axis=0)
    raw_stretch = 1.0 / max(hi - lo, _EPS)
    stretch = min(raw_stretch, params.levels_max_stretch)
    lin = np.clip((lin - lo) * stretch, 0.0, 1.0)

    lum = lin @ LUMA_709
    mask = (lum >= params.stats_lum_min) & (lum <= params.stats_lum_max)
    if mask.mean() < 0.01:
        mask = np.ones_like(mask)
    median = float(np.clip(np.median(lum[mask]), _EPS, 1.0 - _EPS))
    raw_gamma = math.log(params.exposure_target_median) / math.log(median)
    gamma = float(np.clip(raw_gamma, params.levels_gamma_min, params.levels_gamma_max))
    out = linear_to_srgb(np.power(lin, np.float32(gamma)))

    clamped = raw_stretch > params.levels_max_stretch or not (
        params.levels_gamma_min <= raw_gamma <= params.levels_gamma_max
    )
    gains = Gains(
        wb=np.ones(3, dtype=np.float32),
        exposure=1.0,
        clamped={"wb": False, "exposure": False, "levels": bool(clamped)},
        levels={"low": lo, "high": hi, "stretch": stretch, "gamma": gamma,
                "black_rgb_spread": float(lo_c.max() - lo_c.min())},
    )
    return out.astype(np.float32), gains


def normalize_float(rgb: np.ndarray, params: NormalizeParams) -> tuple[np.ndarray, Gains]:
    """Reference path used by the trainer. ``rgb`` is float32 sRGB in [0, 1]."""
    if params.levels:
        return levels_float(rgb, params)
    gains = compute_gains(rgb, params)
    return apply_gains_float(rgb, gains), gains


def gains_to_luts(gains: Gains) -> np.ndarray:
    """Bake the gains into three 256-entry uint8 tables, one per channel."""
    lin = srgb_to_linear(np.arange(256, dtype=np.float32) / 255.0)
    luts = np.empty((3, 256), dtype=np.uint8)
    for c in range(3):
        out = linear_to_srgb(np.clip(lin * gains.combined[c], 0.0, 1.0))
        luts[c] = np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)
    return luts


def normalize_u8(
    rgb_u8: np.ndarray, params: NormalizeParams, max_stats_pixels: int = 300_000
) -> tuple[np.ndarray, Gains]:
    """Fast path for the Pi: statistics from a strided subsample, applied with ``cv2.LUT``."""
    if params.levels:
        raise ValueError(
            "levels normalisation is a training-side step for scanned targets; "
            "the capture path does not support it"
        )
    h, w = rgb_u8.shape[:2]
    step = max(1, int(np.ceil(np.sqrt(h * w / max_stats_pixels))))
    gains = compute_gains(rgb_u8[::step, ::step].astype(np.float32) / 255.0, params)
    table = np.ascontiguousarray(gains_to_luts(gains).T).reshape(256, 1, 3)
    return cv2.LUT(np.ascontiguousarray(rgb_u8), table), gains
