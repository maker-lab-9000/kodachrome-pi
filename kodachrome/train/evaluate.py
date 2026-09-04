"""Measuring whether the fitted LUT actually did what we claim, and is safe.

Two jobs live here, and both exist because the obvious way to do them is
wrong.

**Paired measurement.** The headline number is "distance from the graded
images to the Kodachrome colour cloud, before and after". Computing that
twice with fresh random samples and fresh random projections means the two
numbers differ partly because the LUT changed the pixels and partly because
the sampling changed - and on a good day the sampling noise is the same size
as the effect. ``Evaluator`` therefore fixes the sample indices and the
projection directions once, and both measurements reuse them. A test asserts
that an identity LUT produces *exactly* equal before and after values, which
is only true if the evaluator is genuinely paired.

Alongside it, ``swd_seed_spread`` re-runs the measurement across five
evaluator seeds and reports the spread, so a claimed improvement can be
compared against the noise floor instead of being asserted.

Metrics are computed on **held-out images**. Values on the training pool are
reported too, prefixed ``train_``, and are only useful for spotting
overfitting.

**Safety gates.** A LUT can reduce the distribution distance and still be
unusable. Checking only that neutral grey keeps rising in luminance, as the
first version did, misses a LUT that tints greys blue, one whose red channel
folds back on itself while overall luminance still climbs, and one that
crushes most of the colour cube onto the gamut boundary. Each of those is
checked directly, with a numeric threshold agreed before tuning so the gates
cannot be quietly relaxed to fit whatever the fit produced.

``transport_gamut_clip_deltaE`` is separated from ``lut_fit_rms_deltaE``
deliberately: the first says "the transport asked for colours outside sRGB",
the second says "a smooth LUT could not express what was asked". Folding
them together, as the first version did, made a gamut problem look like a
fitting problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..color import luminance, oklab_to_lch, oklab_to_srgb, srgb_to_linear, srgb_to_oklab
from ..lut import LUT3D
from .dataset import PixelPool
from .transport import hue_bin_index, hue_histogram

# Numeric acceptance thresholds, fixed before tuning (spec 6.5).
NOISE_MARGIN = 3.0
MAX_NEUTRAL_CHROMA = 0.02
MAX_CLIPPED_VOLUME = 0.05


@dataclass
class Evaluator:
    """A fixed sample and projection set, so before/after differ only by the LUT."""

    src_idx: np.ndarray
    tgt_points: np.ndarray
    directions: np.ndarray

    @classmethod
    def build(
        cls,
        src_pool: PixelPool,
        tgt_pool: PixelPool,
        tgt_weights: np.ndarray | None = None,
        n_proj: int = 64,
        max_points: int = 100_000,
        seed: int = 0,
    ) -> Evaluator:
        rng = np.random.default_rng(seed)
        n = min(len(src_pool.srgb), len(tgt_pool.srgb), max_points)
        src_idx = rng.choice(len(src_pool.srgb), n, replace=False)
        if tgt_weights is not None:
            p = np.asarray(tgt_weights, dtype=np.float64)
            tgt_idx = rng.choice(len(tgt_pool.srgb), n, replace=True, p=p / p.sum())
        else:
            tgt_idx = rng.choice(len(tgt_pool.srgb), n, replace=False)
        directions = rng.standard_normal((n_proj, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        return cls(
            src_idx=src_idx,
            tgt_points=np.sort(
                np.asarray(tgt_pool.lab, dtype=np.float64)[tgt_idx] @ directions.T, axis=0
            ),
            directions=directions,
        )

    def distance(self, src_lab: np.ndarray) -> float:
        """Sliced Wasserstein distance from these source pixels to the fixed target sample."""
        projected = np.sort(
            np.asarray(src_lab, dtype=np.float64)[self.src_idx] @ self.directions.T, axis=0
        )
        return float(np.sqrt(np.mean((projected - self.tgt_points) ** 2)))


def swd_seed_spread(
    src_pool: PixelPool,
    tgt_pool: PixelPool,
    tgt_weights: np.ndarray | None,
    lut: LUT3D,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> tuple[float, float]:
    """Mean and standard deviation of the graded distance across evaluator seeds."""
    graded_lab = srgb_to_oklab(lut.apply_numpy(src_pool.srgb))
    values = [
        Evaluator.build(src_pool, tgt_pool, tgt_weights, seed=s).distance(graded_lab) for s in seeds
    ]
    return float(np.mean(values)), float(np.std(values))


def _grey_ramp(n: int = 256) -> np.ndarray:
    return np.repeat(np.linspace(0, 1, n, dtype=np.float32)[:, None], 3, axis=1)


def grey_axis_is_monotone(lut: LUT3D, tolerance: float = 1e-3) -> bool:
    """Luminance must never fall as neutral input rises."""
    lum = luminance(srgb_to_linear(lut.apply_numpy(_grey_ramp())))
    return bool(np.all(np.diff(lum) >= -tolerance))


def channels_are_monotone(lut: LUT3D, tolerance: float = 1e-3) -> bool:
    """Each output channel must not fall as its own input axis rises.

    The grey-axis check alone passes a LUT whose red channel folds back while
    total luminance keeps climbing, which shows up as posterised or inverted
    colour in a gradient.
    """
    table = lut.table
    for axis in range(3):
        moved = np.moveaxis(table[..., axis], axis, 0)
        if np.min(np.diff(moved, axis=0)) < -tolerance:
            return False
    return True


def neutral_axis_max_chroma(lut: LUT3D) -> float:
    """Largest Oklab chroma the LUT gives a neutral input: a tint detector."""
    return float(oklab_to_lch(srgb_to_oklab(lut.apply_numpy(_grey_ramp())))[:, 1].max())


def clipped_volume_fraction(lut: LUT3D, eps: float = 1e-4) -> float:
    """Fraction of **interior** input nodes whose output is pinned to the gamut boundary.

    Interior only, deliberately. The six faces of the cube are inputs that
    already sit at 0 or 1 in some channel, so an identity LUT pins them too:
    counting them would report 53% clipping at size 9 and 17% at size 33 for
    a LUT that changes nothing, and would make any fixed threshold depend on
    the grid size. Restricting to interior nodes gives exactly 0.0 for the
    identity and for an ordinary tone curve, and rises only when the LUT is
    genuinely crushing colours onto the boundary.
    """
    n = lut.size
    if n < 3:
        return 0.0
    inner = lut.table[1:-1, 1:-1, 1:-1]
    pinned = (inner <= eps) | (inner >= 1.0 - eps)
    return float(np.any(pinned, axis=-1).mean())


def hue_bin_shifts(
    src_srgb: np.ndarray, lut: LUT3D, n_bins: int = 24, chroma_floor: float = 0.03
) -> list[dict]:
    """Per-hue-bin mean change in lightness, chroma and hue, in plain units."""
    before = srgb_to_oklab(src_srgb)
    after = srgb_to_oklab(lut.apply_numpy(src_srgb))
    idx = hue_bin_index(before, n_bins, chroma_floor)
    lch_b, lch_a = oklab_to_lch(before), oklab_to_lch(after)
    d_hue = np.degrees(np.angle(np.exp(1j * (lch_a[:, 2] - lch_b[:, 2]))))
    out = []
    for b in range(n_bins + 1):
        sel = idx == b
        count = int(sel.sum())
        centre = (b + 0.5) * 360.0 / n_bins if b < n_bins else None
        if count == 0:
            out.append(
                {"bin": b, "hue_deg": centre, "count": 0, "delta_L": 0.0,
                 "chroma_ratio": 1.0, "delta_hue_deg": 0.0}
            )
            continue
        chroma_before = max(float(lch_b[sel, 1].mean()), 1e-6)
        out.append(
            {
                "bin": b,
                "hue_deg": centre,
                "count": count,
                "delta_L": round(float((lch_a[sel, 0] - lch_b[sel, 0]).mean()), 4),
                "chroma_ratio": round(float(lch_a[sel, 1].mean()) / chroma_before, 3),
                "delta_hue_deg": round(float(d_hue[sel].mean()), 2) if b < n_bins else 0.0,
            }
        )
    return out


def hue_hist_residual(
    src_lab: np.ndarray,
    tgt_lab: np.ndarray,
    weights: np.ndarray,
    n_bins: int,
    chroma_floor: float,
) -> float:
    """How well reweighting actually equalised the hue histograms (0 = perfectly)."""
    h_src = hue_histogram(src_lab, n_bins, chroma_floor)
    h_tgt = hue_histogram(tgt_lab, n_bins, chroma_floor, weights=weights)
    return float(np.abs(h_src - h_tgt).max())


@dataclass
class Gate:
    name: str
    value: float | bool
    threshold: float | bool
    passed: bool
    detail: str


def check_gates(metrics: dict) -> list[Gate]:
    """The numeric bar an artifact must clear, fixed before tuning."""
    margin = NOISE_MARGIN * float(metrics.get("swd_seed_spread", 0.0))
    improvement = float(metrics["swd_before"]) - float(metrics["swd_after"])
    gates = [
        Gate(
            "improvement_exceeds_noise",
            round(improvement, 6),
            round(margin, 6),
            improvement > margin,
            f"held-out distance fell by {improvement:.5f}; needs to beat "
            f"{NOISE_MARGIN}x the seed spread ({margin:.5f})",
        ),
        Gate(
            "grey_axis_monotone",
            bool(metrics["grey_axis_monotone"]),
            True,
            bool(metrics["grey_axis_monotone"]),
            "neutral greys must not darken as input brightens",
        ),
        Gate(
            "channel_monotone",
            bool(metrics["channel_monotone"]),
            True,
            bool(metrics["channel_monotone"]),
            "each output channel must rise with its own input",
        ),
        Gate(
            "neutral_axis_chroma",
            round(float(metrics["neutral_axis_max_chroma"]), 5),
            MAX_NEUTRAL_CHROMA,
            float(metrics["neutral_axis_max_chroma"]) < MAX_NEUTRAL_CHROMA,
            "neutral input must stay close to neutral output",
        ),
        Gate(
            "clipped_volume",
            round(float(metrics["clipped_volume_fraction"]), 5),
            MAX_CLIPPED_VOLUME,
            float(metrics["clipped_volume_fraction"]) < MAX_CLIPPED_VOLUME,
            "too much of the colour cube is pinned to the gamut boundary",
        ),
    ]
    return gates


def evaluate(
    lut: LUT3D,
    val_src: PixelPool,
    val_tgt: PixelPool,
    val_weights: np.ndarray | None,
    train_src: PixelPool,
    train_tgt: PixelPool,
    train_weights: np.ndarray | None,
    transported_lab: np.ndarray,
    n_bins: int = 24,
    chroma_floor: float = 0.03,
    seed: int = 0,
) -> dict:
    """The metrics block of params.json. Primary numbers are held-out."""
    identity = LUT3D.identity(lut.size)

    val_ev = Evaluator.build(val_src, val_tgt, val_weights, seed=seed)
    swd_before = val_ev.distance(val_src.lab)
    swd_after = val_ev.distance(srgb_to_oklab(lut.apply_numpy(val_src.srgb)))
    swd_identity = val_ev.distance(srgb_to_oklab(identity.apply_numpy(val_src.srgb)))
    _mean, spread = swd_seed_spread(val_src, val_tgt, val_weights, lut)

    train_ev = Evaluator.build(train_src, train_tgt, train_weights, seed=seed)
    train_before = train_ev.distance(train_src.lab)
    train_after = train_ev.distance(srgb_to_oklab(lut.apply_numpy(train_src.srgb)))

    # Separate "the transport wanted out-of-gamut colours" from "the LUT could not fit".
    clipped_partners_lab = srgb_to_oklab(np.clip(oklab_to_srgb(transported_lab), 0.0, 1.0))
    clip_error = float(
        np.sqrt(np.mean(np.sum((clipped_partners_lab - transported_lab) ** 2, axis=1)))
    )
    graded_train_lab = srgb_to_oklab(lut.apply_numpy(train_src.srgb))
    fit_error = float(
        np.sqrt(np.mean(np.sum((graded_train_lab - clipped_partners_lab) ** 2, axis=1)))
    )

    return {
        "swd_before": round(swd_before, 6),
        "swd_after": round(swd_after, 6),
        "swd_identity": round(swd_identity, 6),
        "swd_seed_spread": round(spread, 6),
        "train_swd_before": round(train_before, 6),
        "train_swd_after": round(train_after, 6),
        "transport_gamut_clip_deltaE": round(clip_error, 6),
        "lut_fit_rms_deltaE": round(fit_error, 6),
        "grey_axis_monotone": grey_axis_is_monotone(lut),
        "channel_monotone": channels_are_monotone(lut),
        "neutral_axis_max_chroma": round(neutral_axis_max_chroma(lut), 6),
        "clipped_volume_fraction": round(clipped_volume_fraction(lut), 6),
        "hue_bins": hue_bin_shifts(val_src.srgb, lut, n_bins, chroma_floor),
        "n_val_source_pixels": int(len(val_src.srgb)),
        "n_val_target_pixels": int(len(val_tgt.srgb)),
    }
