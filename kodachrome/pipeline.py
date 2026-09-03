"""The Kodachrome pipeline: normalise, apply the LUT, add grain.

The order is fixed and matters. The LUT was fitted on normalised input, so
normalisation comes first; grain is a property of the developed film, so it
goes on last. The same ``Pipeline`` serves full-resolution captures, the
low-resolution live preview (``grain=False``) and batch reprocessing, which
is what guarantees the preview shows the grade the capture will get.

``info`` returns the gains that were applied, whether either gain hit its
clamp, and the LUT's content hash. The capture app writes all of it to its
log so a surprising frame can be explained after the fact.
"""

from __future__ import annotations

import numpy as np

from .artifacts import Artifacts
from .grain import add_grain
from .normalize import normalize_u8


class Pipeline:
    def __init__(self, artifacts: Artifacts) -> None:
        self.artifacts = artifacts
        self._filter = artifacts.lut.to_pillow()

    def process(
        self, rgb_u8: np.ndarray, *, grain: bool = True, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, dict]:
        if rgb_u8.dtype != np.uint8 or rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
            raise ValueError(
                f"process() expects an RGB uint8 array of shape (H, W, 3); "
                f"got dtype {rgb_u8.dtype} shape {rgb_u8.shape}"
            )
        normalised, gains = normalize_u8(rgb_u8, self.artifacts.normalize)
        graded = self.artifacts.lut.apply_pillow(normalised, self._filter)
        if grain and self.artifacts.grain.enabled:
            graded = add_grain(graded, self.artifacts.grain, rng)
        return graded, {
            "wb_gains": [round(float(g), 4) for g in gains.wb],
            "exposure_gain": round(float(gains.exposure), 4),
            "clamped": dict(gains.clamped),
            "lut_sha1": self.artifacts.lut_sha1,
        }
