"""Fine film grain, added after the LUT.

Kodachrome 25 and 64 were among the finest-grained colour films made, so the
default here is subtle. The model is deliberately simple:

* Noise goes on **luminance only**. Film grain is a density variation of the
  dye layers seen together; chroma noise reads as a digital sensor artefact.
* The noise field is Gaussian-blurred by ``blur_sigma`` and renormalised to
  unit variance. Pixel-independent noise looks like high-ISO noise;
  slightly correlated noise looks like grain clumps.
* An envelope ``4Y(1 - Y)`` scales it: zero at black and white, one at
  mid-grey, because real grain is least visible in deep shadow and in fully
  exposed highlights.

``strength`` is the noise standard deviation in luminance units at the
mid-grey peak; 0.025 is about six 8-bit levels.

Reproducibility: ``add_grain`` takes an explicit generator. The capture app
draws a seed per shot and records it, so a graded file can be regenerated
from its original (spec 7.2).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import numpy as np

from ._cv2 import require_cv2

cv2 = require_cv2()


@dataclass
class GrainParams:
    strength: float = 0.025
    blur_sigma: float = 0.7
    enabled: bool = True

    def __post_init__(self) -> None:
        for name in ("strength", "blur_sigma"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
            if value < 0:
                raise ValueError(f"{name} must not be negative, got {value}")

    @classmethod
    def from_dict(cls, d: dict) -> GrainParams:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)


def add_grain(
    rgb_u8: np.ndarray, params: GrainParams, rng: np.random.Generator | None = None
) -> np.ndarray:
    if not params.enabled or params.strength <= 0:
        return rgb_u8.copy()
    rng = rng if rng is not None else np.random.default_rng()

    ycc = cv2.cvtColor(np.ascontiguousarray(rgb_u8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    luma = ycc[..., 0] / 255.0

    noise = rng.standard_normal(luma.shape, dtype=np.float32)
    if params.blur_sigma > 0:
        noise = cv2.GaussianBlur(noise, (0, 0), params.blur_sigma)
        noise /= max(float(noise.std()), 1e-6)

    envelope = 4.0 * luma * (1.0 - luma)
    ycc[..., 0] = np.clip(luma + params.strength * envelope * noise, 0.0, 1.0) * 255.0
    return cv2.cvtColor(np.clip(np.round(ycc), 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)
