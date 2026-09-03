"""3D colour lookup tables: the exported form of the Kodachrome look.

A 3D LUT is a grid of N x N x N output colours indexed by the input colour,
with trilinear interpolation between nodes. The table is held in memory as
``table[r, g, b, channel]``. Two external conventions both order the flat
file **red fastest**:

* ``.cube`` (Adobe, Resolve, everyone): ``LUT_3D_SIZE N`` then N^3 lines
  ``r g b``, the first being the output for input (0, 0, 0).
* Pillow's ``ImageFilter.Color3DLUT``: "channels are changed first, then
  first dimension, then second, then third" - the same order.

``apply_numpy`` is the readable reference used in tests and the trainer;
``apply_pillow`` is the C path used on the Pi. Pillow stores the table in
16-bit fixed point, so the two can differ by one 8-bit level.

Validation
----------
Every invariant the rest of the code assumes is checked here rather than
trusted: cubic shape, size 2..65 (Pillow's limits), all values finite and
inside [0, 1]. ``read_cube`` also refuses a domain other than the unit cube,
because the in-memory contract has no domain field - silently ignoring
``DOMAIN_MIN``/``DOMAIN_MAX`` would misapply such a file.

``sha1_hex`` identifies a LUT by content. ``params.json`` records it and
``Artifacts.load`` verifies it, so a half-written artifact pair cannot load
(see ``artifacts.py``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


class CubeError(ValueError):
    """Malformed or unsupported .cube file."""


@dataclass
class LUT3D:
    table: np.ndarray  # (N, N, N, 3) float32, indexed [r, g, b, channel], values in [0, 1]

    def __post_init__(self) -> None:
        t = np.asarray(self.table, dtype=np.float32)
        if t.ndim != 4 or t.shape[3] != 3 or not (t.shape[0] == t.shape[1] == t.shape[2]):
            raise ValueError(f"LUT table must have shape (N, N, N, 3), got {t.shape}")
        if not 2 <= t.shape[0] <= 65:
            raise ValueError(f"LUT size must be in 2..65, got {t.shape[0]}")
        if not np.isfinite(t).all():
            raise ValueError("LUT table must be finite; found NaN or infinity")
        if t.min() < 0.0 or t.max() > 1.0:
            raise ValueError(f"LUT values must lie in [0, 1], got [{t.min()}, {t.max()}]")
        self.table = t

    @property
    def size(self) -> int:
        return int(self.table.shape[0])

    @classmethod
    def identity(cls, size: int = 33) -> LUT3D:
        grid = np.linspace(0.0, 1.0, size, dtype=np.float32)
        r, g, b = np.meshgrid(grid, grid, grid, indexing="ij")
        return cls(np.stack([r, g, b], axis=-1))

    def to_flat(self) -> np.ndarray:
        """(N^3, 3) rows ordered red-fastest, then green, then blue."""
        return np.ascontiguousarray(self.table.transpose(2, 1, 0, 3)).reshape(-1, 3)

    @classmethod
    def from_flat(cls, flat: np.ndarray, size: int) -> LUT3D:
        bgr_major = np.asarray(flat, dtype=np.float32).reshape(size, size, size, 3)
        return cls(np.ascontiguousarray(bgr_major.transpose(2, 1, 0, 3)))

    def apply_numpy(self, rgb: np.ndarray) -> np.ndarray:
        """Trilinear interpolation in NumPy. ``rgb`` is float sRGB in [0, 1]."""
        rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
        n = self.size
        x = rgb * (n - 1)
        i0 = np.minimum(np.floor(x).astype(np.int64), n - 2)
        f = x - i0
        i1 = i0 + 1
        t = self.table
        r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
        r1, g1, b1 = i1[..., 0], i1[..., 1], i1[..., 2]
        fr, fg, fb = f[..., 0, None], f[..., 1, None], f[..., 2, None]
        c00 = t[r0, g0, b0] * (1 - fr) + t[r1, g0, b0] * fr
        c10 = t[r0, g1, b0] * (1 - fr) + t[r1, g1, b0] * fr
        c01 = t[r0, g0, b1] * (1 - fr) + t[r1, g0, b1] * fr
        c11 = t[r0, g1, b1] * (1 - fr) + t[r1, g1, b1] * fr
        c0 = c00 * (1 - fg) + c10 * fg
        c1 = c01 * (1 - fg) + c11 * fg
        return (c0 * (1 - fb) + c1 * fb).astype(np.float32)

    def to_pillow(self) -> ImageFilter.Color3DLUT:
        flat = np.ascontiguousarray(self.to_flat().ravel(), dtype=np.float32)
        return ImageFilter.Color3DLUT(self.size, flat, channels=3)

    def apply_pillow(
        self, rgb_u8: np.ndarray, filt: ImageFilter.Color3DLUT | None = None
    ) -> np.ndarray:
        """Fast path. Build ``filt`` once with ``to_pillow()`` when processing many frames."""
        filt = filt if filt is not None else self.to_pillow()
        im = Image.fromarray(np.ascontiguousarray(rgb_u8), "RGB")
        return np.asarray(im.filter(filt))


def sha1_hex(lut: LUT3D) -> str:
    """Content hash of the canonical flat table, used to identify an artifact."""
    return hashlib.sha1(np.ascontiguousarray(lut.to_flat(), dtype=np.float32).tobytes()).hexdigest()


def write_cube(lut: LUT3D, path: str | Path, title: str = "kodachrome") -> None:
    lines = [
        f'TITLE "{title}"',
        f"LUT_3D_SIZE {lut.size}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
    ]
    lines.extend(f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in lut.to_flat())
    Path(path).write_text("\n".join(lines) + "\n")


def _parse_triplet(parts: list[str], path: Path, lineno: int, key: str) -> tuple[float, ...]:
    try:
        return tuple(float(p) for p in parts)
    except ValueError as exc:
        raise CubeError(f"{path}: non-numeric {key} on line {lineno}") from exc


def read_cube(path: str | Path) -> LUT3D:
    path = Path(path)
    size: int | None = None
    domain_min = (0.0, 0.0, 0.0)
    domain_max = (1.0, 1.0, 1.0)
    rows: list[list[float]] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key = line.split()[0].upper()
        if key == "TITLE":
            continue
        if key == "LUT_1D_SIZE":
            raise CubeError(f"{path}: 1D LUTs are not supported (line {lineno})")
        if key == "LUT_3D_SIZE":
            try:
                size = int(line.split()[1])
            except (IndexError, ValueError) as exc:
                raise CubeError(f"{path}: bad LUT_3D_SIZE on line {lineno}") from exc
            if not 2 <= size <= 65:
                raise CubeError(f"{path}: LUT_3D_SIZE must be in 2..65, got {size} (line {lineno})")
            continue
        if key == "DOMAIN_MIN":
            domain_min = _parse_triplet(line.split()[1:], path, lineno, "DOMAIN_MIN")
            continue
        if key == "DOMAIN_MAX":
            domain_max = _parse_triplet(line.split()[1:], path, lineno, "DOMAIN_MAX")
            continue
        parts = line.split()
        if len(parts) != 3:
            raise CubeError(f"{path}: expected 3 values on line {lineno}, got {len(parts)}")
        try:
            rows.append([float(p) for p in parts])
        except ValueError as exc:
            raise CubeError(f"{path}: non-numeric value on line {lineno}: {line!r}") from exc
    if size is None:
        raise CubeError(f"{path}: missing LUT_3D_SIZE")
    if len(rows) != size**3:
        raise CubeError(f"{path}: expected {size**3} rows for LUT_3D_SIZE {size}, got {len(rows)}")
    if domain_min != (0.0, 0.0, 0.0) or domain_max != (1.0, 1.0, 1.0):
        raise CubeError(
            f"{path}: only the unit DOMAIN is supported, got MIN {domain_min} MAX {domain_max}"
        )
    table = np.array(rows, dtype=np.float32)
    if not np.isfinite(table).all():
        raise CubeError(f"{path}: table must be finite; found NaN or infinity")
    if table.min() < 0.0 or table.max() > 1.0:
        raise CubeError(f"{path}: values must lie in [0, 1], got [{table.min()}, {table.max()}]")
    return LUT3D.from_flat(table, size)
