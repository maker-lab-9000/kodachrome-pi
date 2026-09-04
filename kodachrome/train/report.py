"""Human-readable evidence that a fitted LUT does what we claim and is safe.

Numbers alone do not tell you whether a grade looks right, and pictures alone
do not tell you whether it generalises. The report gives both:

* ``contact_sheet.png`` - **held-out** source images, normalised beside
  graded, with a strip of real Kodachrome scans underneath. The question to
  ask is whether the graded row belongs in the same family as the strip.
  Held-out matters: showing training images would flatter the fit. When the
  corpus was too small to hold any back, the row labels say TRAINING in full
  and ``summary.txt`` carries a warning, so the sheet never claims otherwise.
* ``ramps.png`` - grey ramp and three hue sweeps, before over after. The
  grey ramp shows the learned tone curve; the sweeps show saturation and hue
  movement. Banding or a wobble here means the smoothness weight is too low.
* ``diagnostics.png`` - white balance and exposure gain histograms per
  corpus with the clamp rate. If normalisation is clamping often, the LUT was
  fitted on input the Pi will rarely reproduce.
* ``metrics.json`` and ``summary.txt`` - the full metric block plus the
  pass/fail gates in plain language, so nobody has to interpret a number to
  learn whether the artifact is acceptable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..color import lch_to_oklab, oklab_to_srgb
from ..imageio import load_rgb
from ..lut import LUT3D
from ..normalize import NormalizeParams
from .dataset import CorpusSplit, PixelPool, SampleConfig, prepare_image
from .evaluate import Gate

_BG = (16, 16, 16)
_FG = (220, 220, 220)


def _to_u8(rgb_float: np.ndarray) -> np.ndarray:
    return np.clip(np.round(rgb_float * 255.0), 0, 255).astype(np.uint8)


def _thumb(rgb_u8: np.ndarray, size: int) -> Image.Image:
    im = Image.fromarray(rgb_u8, "RGB")
    im.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), (24, 24, 24))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return canvas


def render_contact_sheet(
    source_paths: Sequence[Path],
    target_paths: Sequence[Path],
    lut: LUT3D,
    source_normalize: NormalizeParams,
    target_normalize: NormalizeParams,
    cfg: SampleConfig,
    out_path: str | Path,
    n: int = 8,
    thumb: int = 240,
    rng: np.random.Generator | None = None,
    held_out: bool = True,
) -> Path:
    """Draw the sheet, labelling honestly which images it actually used.

    ``held_out=False`` means the caller had no validation split and is showing
    training images. The labels say so in full, because a sheet captioned
    "held-out" while showing images the fit was trained on is exactly the
    flattering picture this artifact exists to avoid.
    """
    rng = rng if rng is not None else np.random.default_rng(0)

    def pick(paths: Sequence[Path]) -> list[Path]:
        if not paths:
            return []
        idx = rng.choice(len(paths), min(n, len(paths)), replace=False)
        return [paths[i] for i in idx]

    filt = lut.to_pillow()
    normalised, graded = [], []
    for p in pick(source_paths):
        prepared, _gains = prepare_image(load_rgb(p)[0], source_normalize, cfg)
        norm_u8 = _to_u8(prepared)
        normalised.append(_thumb(norm_u8, thumb))
        graded.append(_thumb(lut.apply_pillow(norm_u8, filt), thumb))
    kodachrome = [
        _thumb(_to_u8(prepare_image(load_rgb(p)[0], target_normalize, cfg)[0]), thumb)
        for p in pick(target_paths)
    ]

    pad, label_h = 8, 18
    cols = max(len(normalised), len(kodachrome), 1)
    sheet = Image.new(
        "RGB",
        (pad + cols * (thumb + pad), 3 * (label_h + thumb + pad) + pad),
        _BG,
    )
    draw = ImageDraw.Draw(sheet)
    origin = "Held-out" if held_out else "TRAINING (corpus too small to hold any back)"
    rows = [
        (f"{origin} source, normalised", normalised),
        (f"{origin} source, graded with the fitted LUT", graded),
        ("Real Kodachrome scans (exposure-normalised)", kodachrome),
    ]
    for r, (label, images) in enumerate(rows):
        y = pad + r * (label_h + thumb + pad)
        draw.text((pad, y), label, fill=_FG)
        for j, im in enumerate(images):
            sheet.paste(im, (pad + j * (thumb + pad), y + label_h))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def _grey_ramp_strip(width: int) -> np.ndarray:
    return np.repeat(np.linspace(0, 1, width, dtype=np.float32)[None, :, None], 3, axis=2)


def _hue_sweep(width: int, lightness: float, chroma: float) -> np.ndarray:
    hue = np.linspace(-np.pi, np.pi, width, dtype=np.float32)
    lch = np.stack(
        [np.full(width, lightness, np.float32), np.full(width, chroma, np.float32), hue], axis=1
    )
    return np.clip(oklab_to_srgb(lch_to_oklab(lch)), 0, 1)[None, :, :]


def render_ramps(lut: LUT3D, out_path: str | Path, width: int = 768, band: int = 36) -> Path:
    strips = [("grey ramp", _grey_ramp_strip(width))] + [
        (f"hue sweep L={lum:.1f} C=0.12", _hue_sweep(width, lum, 0.12)) for lum in (0.4, 0.6, 0.8)
    ]
    label_h, pad = 16, 6
    img = Image.new("RGB", (width, len(strips) * (label_h + 2 * band + pad) + pad), _BG)
    draw = ImageDraw.Draw(img)
    y = pad
    for label, line in strips:
        draw.text((4, y), f"{label}: before (top) / after (bottom)", fill=_FG)
        y += label_h
        img.paste(Image.fromarray(np.repeat(_to_u8(line), band, axis=0), "RGB"), (0, y))
        img.paste(
            Image.fromarray(np.repeat(_to_u8(lut.apply_numpy(line)), band, axis=0), "RGB"),
            (0, y + band),
        )
        y += 2 * band + pad
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def _histogram_bars(
    draw: ImageDraw.ImageDraw, values: Sequence[float], x: int, y: int, w: int, h: int, label: str
) -> None:
    draw.text((x, y), label, fill=_FG)
    y += 14
    draw.rectangle([x, y, x + w, y + h], outline=(90, 90, 90))
    if not len(values):
        return
    counts, _edges = np.histogram(np.asarray(values, dtype=float), bins=20)
    peak = max(int(counts.max()), 1)
    bar_w = max(1, w // len(counts))
    for i, c in enumerate(counts):
        bh = int(h * c / peak)
        draw.rectangle(
            [x + i * bar_w, y + h - bh, x + (i + 1) * bar_w - 1, y + h], fill=(120, 170, 220)
        )


def render_diagnostics(
    source_pool: PixelPool, target_pool: PixelPool, out_path: str | Path
) -> Path:
    width, height = 760, 420
    img = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(img)
    draw.text((8, 8), "Normalisation diagnostics", fill=_FG)

    for col, (name, pool) in enumerate((("source", source_pool), ("target", target_pool))):
        x = 8 + col * 380
        draw.text(
            (x, 30),
            f"{name}: {pool.n_images} images, clamp rate {pool.clamp_rate:.0%}",
            fill=_FG,
        )
        wb = [g for gains in pool.wb_gains for g in gains]
        _histogram_bars(draw, wb, x, 50, 340, 120, "white balance gains")
        _histogram_bars(draw, pool.exposure_gains, x, 200, 340, 120, "exposure gains")
        profiles = ", ".join(f"{k}: {v}" for k, v in sorted(pool.profiles.items())) or "none"
        draw.text((x, 340), f"ICC profiles: {profiles}"[:70], fill=_FG)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def write_report(
    out_dir: str | Path,
    lut: LUT3D,
    metrics: dict,
    gates: Sequence[Gate],
    source_split: CorpusSplit,
    target_split: CorpusSplit,
    source_normalize: NormalizeParams,
    target_normalize: NormalizeParams,
    cfg: SampleConfig,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # fit() records what it actually measured; fall back to the paths for
    # direct callers that assemble metrics themselves.
    held_out = bool(metrics.get("held_out_eval", bool(source_split.val_paths)))
    render_contact_sheet(
        source_split.val_paths or source_split.train_paths,
        target_split.val_paths or target_split.train_paths,
        lut,
        source_normalize,
        target_normalize,
        cfg,
        out_dir / "contact_sheet.png",
        held_out=held_out,
    )
    render_ramps(lut, out_dir / "ramps.png")
    render_diagnostics(
        source_split.train_pool, target_split.train_pool, out_dir / "diagnostics.png"
    )

    payload = {**metrics, "gates": [vars(g) for g in gates]}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "Kodachrome fit summary",
        "",
        f"{'held-out' if held_out else 'TRAINING'} distance to Kodachrome: "
        f"{metrics['swd_before']:.5f} before "
        f"-> {metrics['swd_after']:.5f} after "
        f"(seed spread {metrics['swd_seed_spread']:.5f})",
        f"training-pool distance:          {metrics['train_swd_before']:.5f} -> "
        f"{metrics['train_swd_after']:.5f}",
        f"transport clipped out of gamut:  {metrics['transport_gamut_clip_deltaE']:.5f} dE",
        f"LUT fit residual:                {metrics['lut_fit_rms_deltaE']:.5f} dE",
        "",
        "Gates:",
    ]
    if not held_out:
        lines.insert(
            2,
            "WARNING: a corpus was too small to hold any images back, so the contact "
            "sheet shows images the fit was trained on and the distances below "
            "measure memorisation rather than generalisation.",
        )
    for g in gates:
        lines.append(f"  [{'PASS' if g.passed else 'FAIL'}] {g.name}: {g.value} - {g.detail}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    return out_dir
