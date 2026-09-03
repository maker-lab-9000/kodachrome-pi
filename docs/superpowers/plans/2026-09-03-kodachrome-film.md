# Kodachrome Film Look Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Raspberry Pi 400 with an Innomaker U20CAM-1080P-WDR saves every capture twice, original and Kodachrome-graded, where the grade is a 3D LUT learned on a Mac from real Kodachrome scans.

**Architecture:** One Python package `kodachrome` with shared colour, normalisation, LUT and grain modules. A Mac-side trainer downloads public-domain Kodachrome scans from Wikimedia Commons, transports the colour distribution of camera samples onto the Kodachrome distribution in Oklab (hue-reweighted iterative distribution transfer), then fits a smooth 33³ LUT to the resulting pairs by regularised sparse least squares. The Pi runtime normalises each frame (white balance and exposure as three `cv2.LUT` tables), applies the LUT via Pillow's C `Color3DLUT`, adds luminance grain, and saves.

**Tech Stack:** Python 3.11+, NumPy, Pillow, OpenCV (apt `python3-opencv` on the Pi, `opencv-python` wheel on the Mac), SciPy and requests for the trainer only, pytest.

**Spec:** `docs/superpowers/specs/2026-09-03-kodachrome-film-design.md` — read it first; every task below cites the section it implements.

## Global Constraints

- `requires-python = ">=3.11"`. Develop on the Mac with `/usr/local/bin/python3.12` in `.venv`.
- Base pip dependencies are only `numpy` and `Pillow`. OpenCV is imported at runtime but is **not** a pip dependency of the package (spec 7.4). `[train]` adds `scipy>=1.12`, `requests`, `tqdm`. `[dev]` adds `opencv-python`, `pytest`, `ruff`.
- Image arrays are **RGB**, `uint8` at boundaries, `float32` in `[0, 1]` inside algorithms. BGR appears only inside `camera.py` at the OpenCV boundary (spec 4, "Channel and value conventions").
- LUT tables are indexed `table[r, g, b, channel]` in memory; `.cube` and Pillow flat order is red-fastest (spec 4).
- All perceptual statistics are in Oklab; the exported LUT maps sRGB to sRGB (spec 4).
- Tests never touch the network or camera hardware. Use `FakeCamera` and fake HTTP sessions.
- Documentation is updated in the same task as the code: README, `docs/decisions.md`, module docstrings that explain the *why*. The user asked for this explicitly.
- Commit after every task. Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Run tests with `.venv/bin/pytest -q` from the repo root. Run `.venv/bin/ruff check kodachrome tests` before each commit.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `pyproject.toml` | package metadata, extras, CLI entry points | 1 |
| `kodachrome/color.py` | sRGB/linear/Oklab/LCh conversions | 2 |
| `kodachrome/normalize.py` | white balance + exposure, float and `cv2.LUT` paths | 3 |
| `kodachrome/lut.py` | `LUT3D`, `.cube` I/O, NumPy trilinear, Pillow path | 4 |
| `kodachrome/grain.py` | luminance film grain | 5 |
| `kodachrome/pipeline.py` | `Artifacts` load/write, `Pipeline.process` | 6 |
| `artifacts/kodachrome.cube`, `artifacts/params.json` | committed artifact (identity first, trained in Task 15) | 6, 15 |
| `kodachrome/imageio.py` | load/save images, list image files | 7 |
| `kodachrome/capture/camera.py` | `Camera` protocol, `V4L2Camera`, `FakeCamera` | 8 |
| `kodachrome/capture/batch.py` | `kodachrome-process` | 9 |
| `kodachrome/capture/app.py` | `kodachrome-capture` session, preview, headless | 10 |
| `kodachrome/train/fetch.py` | `kodachrome-fetch` Commons downloader | 11 |
| `kodachrome/train/dataset.py` | crop/resize/normalise/sample into pixel pools | 12 |
| `kodachrome/train/transport.py` | hue reweighting, IDT, sliced Wasserstein | 13 |
| `kodachrome/train/lutfit.py` | trilinear design matrix, smoothness operator, CG solve | 14 |
| `kodachrome/train/report.py` | contact sheet, ramps, metrics | 15 |
| `kodachrome/train/fit.py` | `kodachrome-train` orchestration and params writing | 16 |
| `README.md`, `docs/decisions.md` | living documentation | every task |

The spec's file tree named a single `train/fit.py`; this plan splits the maths into `transport.py` and `lutfit.py` so each file holds one algorithm and its tests, with `fit.py` as the thin orchestrator. Task 16 updates the spec tree accordingly.

---

### Task 1: Project scaffold and virtual environment

**Files:**
- Create: `pyproject.toml`, `kodachrome/__init__.py`, `kodachrome/train/__init__.py`, `kodachrome/capture/__init__.py`, `tests/conftest.py`, `tests/test_package.py`, `README.md`

**Interfaces:**
- Produces: importable package `kodachrome` with `__version__ = "0.1.0"`; pytest fixture `repo_root` (Path to the repository root) used by later tests.

- [ ] **Step 1: Write pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "kodachrome-film"
version = "0.1.0"
description = "Kodachrome film look for a Raspberry Pi 400 + U20CAM camera, learned from real Kodachrome scans"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.24",
  "Pillow>=10.0",
]

[project.optional-dependencies]
train = ["scipy>=1.12", "requests>=2.31", "tqdm>=4.66"]
dev = ["opencv-python>=4.8", "pytest>=7.4", "ruff>=0.4"]

[project.scripts]
kodachrome-fetch = "kodachrome.train.fetch:main"
kodachrome-train = "kodachrome.train.fit:main"
kodachrome-capture = "kodachrome.capture.app:main"
kodachrome-process = "kodachrome.capture.batch:main"

[tool.setuptools.packages.find]
include = ["kodachrome*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
```

- [ ] **Step 2: Create package files**

`kodachrome/__init__.py`:
```python
"""Kodachrome film look for Raspberry Pi 400 + Innomaker U20CAM-1080P-WDR.

The package has two halves that share code:

* ``kodachrome.color``, ``normalize``, ``lut``, ``grain``, ``pipeline`` are the
  processing core used on the Pi. They depend only on NumPy, Pillow and OpenCV.
* ``kodachrome.train`` fits the LUT on a Mac from real Kodachrome scans and
  needs SciPy and requests (``pip install -e ".[train]"``).

See ``docs/superpowers/specs/2026-09-03-kodachrome-film-design.md`` for the design.
"""

__version__ = "0.1.0"
```

`kodachrome/train/__init__.py`:
```python
"""Mac-side trainer: fetch Kodachrome scans, transport colour distributions, fit the LUT."""
```

`kodachrome/capture/__init__.py`:
```python
"""Pi-side capture: camera access, keypress capture app, batch reprocessing."""
```

`tests/conftest.py`:
```python
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
```

`tests/test_package.py`:
```python
import kodachrome


def test_version():
    assert kodachrome.__version__ == "0.1.0"
```

- [ ] **Step 3: Write the README skeleton**

`README.md`:
````markdown
# kodachrome-film

Kodachrome film look for a Raspberry Pi 400 with an Innomaker U20CAM-1080P-WDR
USB camera. Every capture is saved twice: the camera's original and a version
graded to match real Kodachrome, using a 3D LUT learned from public-domain
Kodachrome scans.

Status: under construction. See `docs/superpowers/specs/` for the design and
`docs/decisions.md` for why things are the way they are.

## Mac setup (development and training)

```bash
/usr/local/bin/python3.12 -m venv .venv
.venv/bin/pip install -e ".[train,dev]"
.venv/bin/pytest -q
```

## Pi setup

Written in a later step.

## Commands

| Command | Where | What |
|---|---|---|
| `kodachrome-fetch` | Mac | download Kodachrome scans from Wikimedia Commons |
| `kodachrome-train` | Mac | fit the LUT and write `artifacts/` |
| `kodachrome-capture` | Pi | live preview, SPACE to capture |
| `kodachrome-process` | either | regrade a folder of originals |
````

- [ ] **Step 4: Create the venv and install**

```bash
/usr/local/bin/python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[train,dev]"
```
Expected: installs numpy, Pillow, scipy, requests, tqdm, opencv-python, pytest, ruff without error. If `opencv-python` has no wheel for this platform, install `opencv-python-headless` instead and note in README that preview is unavailable on the Mac.

- [ ] **Step 5: Run tests and lint**

Run: `.venv/bin/pytest -q && .venv/bin/ruff check kodachrome tests`
Expected: `1 passed`, ruff reports no errors.

- [ ] **Step 6: Set git identity if missing, commit**

```bash
git config user.email >/dev/null || git config user.email "george.babanau@localhost"
git config user.name  >/dev/null || git config user.name  "george.babanau"
git add pyproject.toml kodachrome tests README.md
git commit -m "chore: scaffold kodachrome package, venv, pytest

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Colour conversions (`color.py`)

Implements spec 5.1.

**Files:**
- Create: `kodachrome/color.py`, `tests/test_color.py`

**Interfaces:**
- Produces:
  - `srgb_to_linear(x) -> np.float32 array`, `linear_to_srgb(x)`, both clip to `[0, 1]`
  - `linear_to_oklab(rgb)`, `oklab_to_linear(lab)`, `srgb_to_oklab(rgb)`, `oklab_to_srgb(lab)`
  - `oklab_to_lch(lab)` → `(L, C, h_radians)`, `lch_to_oklab(lch)`
  - `LUMA_709` constant `np.array([0.2126, 0.7152, 0.0722], float32)` and `luminance(rgb_linear) -> array`
  - All accept shape `(..., 3)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_color.py`:
```python
import numpy as np
import pytest

from kodachrome import color


def test_srgb_linear_roundtrip():
    x = np.random.default_rng(0).random((100, 3), dtype=np.float32)
    back = color.linear_to_srgb(color.srgb_to_linear(x))
    assert back.dtype == np.float32
    assert np.allclose(back, x, atol=1e-4)


def test_srgb_to_linear_known_points():
    assert color.srgb_to_linear(np.array([0.0, 1.0]))[1] == pytest.approx(1.0)
    # 18% grey card is about sRGB 0.461
    assert color.srgb_to_linear(np.array([0.4613]))[0] == pytest.approx(0.18, abs=1e-3)


@pytest.mark.parametrize(
    "rgb, expected",
    [
        ((1.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
        ((1.0, 0.0, 0.0), (0.627955, 0.224863, 0.125846)),
        ((0.0, 1.0, 0.0), (0.866440, -0.233888, 0.179498)),
        ((0.0, 0.0, 1.0), (0.452014, -0.032457, -0.311528)),
    ],
)
def test_oklab_reference_values(rgb, expected):
    # Reference values from Björn Ottosson's Oklab article (linear sRGB inputs).
    lab = color.linear_to_oklab(np.array(rgb, dtype=np.float32))
    assert np.allclose(lab, expected, atol=1e-3)


def test_oklab_roundtrip():
    rgb = np.random.default_rng(1).random((500, 3), dtype=np.float32)
    back = color.oklab_to_srgb(color.srgb_to_oklab(rgb))
    assert np.allclose(back, rgb, atol=1e-4)


def test_lch_roundtrip_and_hue_range():
    lab = color.srgb_to_oklab(np.random.default_rng(2).random((200, 3), dtype=np.float32))
    lch = color.oklab_to_lch(lab)
    assert lch[..., 1].min() >= 0
    assert np.all(np.abs(lch[..., 2]) <= np.pi + 1e-6)
    assert np.allclose(color.lch_to_oklab(lch), lab, atol=1e-5)


def test_luminance_weights_sum_to_one():
    assert color.LUMA_709.sum() == pytest.approx(1.0, abs=1e-4)
    assert color.luminance(np.ones(3, dtype=np.float32)) == pytest.approx(1.0, abs=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_color.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'kodachrome.color'`

- [ ] **Step 3: Implement color.py**

```python
"""Colour-space conversions shared by the trainer and the Pi runtime.

Why these spaces
----------------
* **sRGB** is what the camera delivers and what JPEGs store. Its transfer
  curve is roughly gamma 2.2, so arithmetic on sRGB values does not model
  light. Every function here expects sRGB in ``[0, 1]``.
* **Linear RGB** is sRGB with the transfer curve removed. White balance and
  exposure are multiplications of light, so ``normalize.py`` works here.
* **Oklab** (Björn Ottosson, 2020) is a perceptual space: Euclidean distance
  approximates perceived difference and hue angles are far more uniform
  than in CIELAB, which bends visibly in the blues. The trainer computes
  hue histograms, distribution transport and metrics in Oklab so that
  "match the distribution" means "match what the eye sees".
* **Oklch** is Oklab in polar form: lightness L, chroma C, hue h (radians,
  from ``arctan2``, so in ``[-pi, pi]``).

All functions accept arrays of shape ``(..., 3)`` and return ``float32``.
"""

from __future__ import annotations

import numpy as np

# Oklab matrices from https://bottosson.github.io/posts/oklab/ (linear sRGB -> LMS -> Lab).
_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float64,
)
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float64,
)
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)

# Rec. 709 / sRGB luminance weights for linear RGB.
LUMA_709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Remove the sRGB transfer curve. Input is clipped to [0, 1]."""
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Apply the sRGB transfer curve. Input is clipped to [0, 1]."""
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(
        x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055
    ).astype(np.float32)


def linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    lms = np.cbrt(rgb @ _M1.T)
    return (lms @ _M2.T).astype(np.float32)


def oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    lms = (lab @ _M2_INV.T) ** 3
    return (lms @ _M1_INV.T).astype(np.float32)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    return linear_to_oklab(srgb_to_linear(rgb))


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    return linear_to_srgb(oklab_to_linear(lab))


def oklab_to_lch(lab: np.ndarray) -> np.ndarray:
    """Polar Oklab: (L, chroma, hue in radians)."""
    lab = np.asarray(lab, dtype=np.float32)
    a, b = lab[..., 1], lab[..., 2]
    return np.stack([lab[..., 0], np.hypot(a, b), np.arctan2(b, a)], axis=-1).astype(np.float32)


def lch_to_oklab(lch: np.ndarray) -> np.ndarray:
    lch = np.asarray(lch, dtype=np.float32)
    lum, chroma, hue = lch[..., 0], lch[..., 1], lch[..., 2]
    return np.stack([lum, chroma * np.cos(hue), chroma * np.sin(hue)], axis=-1).astype(np.float32)


def luminance(rgb_linear: np.ndarray) -> np.ndarray:
    """Rec. 709 luminance of linear RGB."""
    return np.asarray(rgb_linear, dtype=np.float32) @ LUMA_709
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_color.py -q`
Expected: all pass. If `test_oklab_reference_values` fails by more than 1e-3, check the matrix transcription digit by digit against the article; do not loosen the tolerance.

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/color.py tests/test_color.py
git commit -m "feat: sRGB, linear and Oklab colour conversions

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Normalisation (`normalize.py`)

Implements spec 5.2. This is the "dynamic" per-shot step.

**Files:**
- Create: `kodachrome/normalize.py`, `tests/test_normalize.py`

**Interfaces:**
- Consumes: `color.srgb_to_linear`, `color.linear_to_srgb`, `color.LUMA_709`
- Produces:
  - `@dataclass NormalizeParams` with fields exactly as spec 5.2, plus `from_dict(d)` (ignores unknown keys) and `to_dict()`
  - `@dataclass Gains(wb: np.ndarray(3,), exposure: float)` with `.combined` property and `.to_dict()`
  - `compute_gains(rgb_float, params) -> Gains`
  - `apply_gains_float(rgb_float, gains) -> rgb_float`
  - `normalize_float(rgb_float, params) -> (rgb_float, Gains)`
  - `gains_to_luts(gains) -> np.ndarray (3, 256) uint8`
  - `normalize_u8(rgb_u8, params, max_stats_pixels=300_000) -> (rgb_u8, Gains)`

- [ ] **Step 1: Write the failing tests**

`tests/test_normalize.py`:
```python
import numpy as np
import pytest

from kodachrome import color
from kodachrome.normalize import (
    Gains,
    NormalizeParams,
    compute_gains,
    gains_to_luts,
    normalize_float,
    normalize_u8,
)


def _gradient_image(h=48, w=64, cast=(1.0, 0.9, 0.75)):
    """A smooth scene-like image with a warm cast, sRGB float32."""
    ramp = np.linspace(0.1, 0.9, w, dtype=np.float32)[None, :, None]
    rows = np.linspace(0.8, 1.2, h, dtype=np.float32)[:, None, None]
    img = np.clip(ramp * rows * np.array(cast, dtype=np.float32), 0, 1)
    return img.astype(np.float32)


def test_params_from_dict_ignores_unknown_and_roundtrips():
    p = NormalizeParams.from_dict({"exposure_target_median": 0.2, "bogus": 1})
    assert p.exposure_target_median == 0.2
    assert p.white_balance is True
    assert NormalizeParams.from_dict(p.to_dict()) == p


def test_gains_combined_and_dict():
    g = Gains(wb=np.array([1.0, 1.1, 1.2], dtype=np.float32), exposure=2.0)
    assert np.allclose(g.combined, [2.0, 2.2, 2.4])
    assert g.to_dict() == {"wb": [1.0, 1.1, 1.2], "exposure": 2.0}


def test_grey_world_neutralises_a_cast():
    # Mild cast: the implied gains (about 0.83, 1.04, 1.33) stay inside the clamps,
    # so grey-world can fully neutralise it. A stronger cast would clamp and stay tinted.
    img = np.full((32, 32, 3), (0.5, 0.45, 0.4), dtype=np.float32)
    out, gains = normalize_float(img, NormalizeParams())
    assert np.allclose(out[..., 0], out[..., 1], atol=1 / 255)
    assert np.allclose(out[..., 1], out[..., 2], atol=1 / 255)
    # median linear luminance lands on the 18% target
    lin = color.srgb_to_linear(out)
    assert np.median(color.luminance(lin)) == pytest.approx(0.18, abs=0.005)
    assert gains.wb[0] < 1.0 < gains.wb[2]


def test_white_balance_can_be_disabled():
    img = np.full((8, 8, 3), (0.5, 0.4, 0.3), dtype=np.float32)
    gains = compute_gains(img, NormalizeParams(white_balance=False))
    assert np.array_equal(gains.wb, np.ones(3, dtype=np.float32))
    assert gains.exposure > 1.0


def test_gains_are_clamped():
    p = NormalizeParams()
    dark = np.full((8, 8, 3), 0.02, dtype=np.float32)
    assert compute_gains(dark, p).exposure == pytest.approx(p.exposure_gain_max)
    bright = np.full((8, 8, 3), 0.95, dtype=np.float32)
    # sRGB 0.95 is linear 0.89; the exposure gain would be 0.2, clamped to the minimum
    assert compute_gains(bright, p).exposure == pytest.approx(p.exposure_gain_min)
    red = np.full((8, 8, 3), (0.9, 0.1, 0.1), dtype=np.float32)
    g = compute_gains(red, p)
    assert g.wb[1] == pytest.approx(p.wb_gain_max)
    assert g.wb[0] == pytest.approx(p.wb_gain_min)


def test_normalising_twice_is_stable():
    img = _gradient_image()
    once, _ = normalize_float(img, NormalizeParams())
    twice, gains2 = normalize_float(once, NormalizeParams())
    assert np.allclose(gains2.wb, 1.0, atol=0.02)
    assert gains2.exposure == pytest.approx(1.0, abs=0.02)
    assert np.abs(twice - once).max() < 2 / 255


def test_float_and_u8_paths_agree():
    img_u8 = (_gradient_image() * 255).round().astype(np.uint8)
    p = NormalizeParams()
    out_f, gains_f = normalize_float(img_u8.astype(np.float32) / 255.0, p)
    out_u8, gains_u8 = normalize_u8(img_u8, p)
    assert out_u8.dtype == np.uint8 and out_u8.shape == img_u8.shape
    assert np.allclose(gains_f.combined, gains_u8.combined, atol=1e-6)
    diff = np.abs(out_u8.astype(int) - np.round(out_f * 255).astype(int))
    assert diff.max() <= 1


def test_luts_are_monotone():
    luts = gains_to_luts(Gains(wb=np.array([0.8, 1.0, 1.5], dtype=np.float32), exposure=1.3))
    assert luts.shape == (3, 256) and luts.dtype == np.uint8
    assert np.all(np.diff(luts.astype(int), axis=1) >= 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_normalize.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'kodachrome.normalize'`

- [ ] **Step 3: Implement normalize.py**

```python
"""Per-image white balance and exposure normalisation.

This is the "dynamic" half of the Kodachrome pipeline. The LUT fitted by the
trainer expects input that has been brought to a neutral white point and a
fixed exposure. Doing the same to every capture means the grade does not
fight the scene lighting: a tungsten-lit room and a cloudy street both hit
the LUT looking like the images it was fitted on. The trainer applies this
exact code to the source corpus (``normalize_float``); the Pi applies the
same maths through three 256-entry lookup tables (``normalize_u8``).

Why it can be three 1D tables
-----------------------------
White balance is a per-channel gain in linear light. Exposure is a scalar
gain in linear light. Their product is one gain per channel, so the whole
sRGB -> linear -> gain -> sRGB map is three independent monotone functions of
one byte each. ``cv2.LUT`` applies that in milliseconds on a Pi 400.

Targets vs sources
------------------
Kodachrome scans are normalised with ``white_balance=False``: the film's
daylight balance and warm cast are part of the look being learned. Only the
per-slide exposure lottery is removed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import cv2
import numpy as np

from .color import LUMA_709, linear_to_srgb, srgb_to_linear

_EPS = 1e-6


@dataclass
class NormalizeParams:
    white_balance: bool = True
    wb_gain_min: float = 0.6
    wb_gain_max: float = 1.6
    exposure_target_median: float = 0.18
    exposure_gain_min: float = 0.5
    exposure_gain_max: float = 3.0
    stats_lum_min: float = 0.02
    stats_lum_max: float = 0.90

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

    @property
    def combined(self) -> np.ndarray:
        return (np.asarray(self.wb, dtype=np.float32) * np.float32(self.exposure)).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "wb": [round(float(g), 4) for g in self.wb],
            "exposure": round(float(self.exposure), 4),
        }


def compute_gains(rgb: np.ndarray, params: NormalizeParams) -> Gains:
    """Grey-world white balance and median-to-target exposure from an sRGB float image."""
    lin = srgb_to_linear(rgb).reshape(-1, 3)
    lum = lin @ LUMA_709
    mask = (lum >= params.stats_lum_min) & (lum <= params.stats_lum_max)
    if mask.mean() < 0.01:
        mask = np.ones_like(mask)
    sel = lin[mask]

    if params.white_balance:
        means = np.maximum(sel.mean(axis=0), _EPS)
        mean_lum = float(means @ LUMA_709)
        wb = np.clip(mean_lum / means, params.wb_gain_min, params.wb_gain_max).astype(np.float32)
    else:
        wb = np.ones(3, dtype=np.float32)

    median_lum = float(np.median((sel * wb) @ LUMA_709))
    exposure = float(
        np.clip(
            params.exposure_target_median / max(median_lum, _EPS),
            params.exposure_gain_min,
            params.exposure_gain_max,
        )
    )
    return Gains(wb=wb, exposure=exposure)


def apply_gains_float(rgb: np.ndarray, gains: Gains) -> np.ndarray:
    lin = srgb_to_linear(rgb) * gains.combined
    return linear_to_srgb(np.clip(lin, 0.0, 1.0))


def normalize_float(rgb: np.ndarray, params: NormalizeParams) -> tuple[np.ndarray, Gains]:
    """Reference path used by the trainer. ``rgb`` is float32 sRGB in [0, 1]."""
    gains = compute_gains(rgb, params)
    return apply_gains_float(rgb, gains), gains


def gains_to_luts(gains: Gains) -> np.ndarray:
    """Bake the gains into three 256-entry uint8 tables, one per channel."""
    ramp = np.arange(256, dtype=np.float32) / 255.0
    lin = srgb_to_linear(ramp)
    luts = np.empty((3, 256), dtype=np.uint8)
    for c in range(3):
        out = linear_to_srgb(np.clip(lin * gains.combined[c], 0.0, 1.0))
        luts[c] = np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8)
    return luts


def normalize_u8(
    rgb_u8: np.ndarray, params: NormalizeParams, max_stats_pixels: int = 300_000
) -> tuple[np.ndarray, Gains]:
    """Fast path for the Pi. Statistics come from a strided subsample; the
    tables are applied to every pixel with ``cv2.LUT``."""
    h, w = rgb_u8.shape[:2]
    step = max(1, int(np.ceil(np.sqrt(h * w / max_stats_pixels))))
    small = rgb_u8[::step, ::step].astype(np.float32) / 255.0
    gains = compute_gains(small, params)
    luts = gains_to_luts(gains)
    # A (256, 1, 3) array is a 256-entry, 3-channel table for cv2.LUT.
    table = np.ascontiguousarray(luts.T).reshape(256, 1, 3)
    out = cv2.LUT(np.ascontiguousarray(rgb_u8), table)
    return out, gains
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_normalize.py -q`
Expected: all pass. If `test_float_and_u8_paths_agree` shows a max diff of 2 or more, the cause is almost always the `cv2.LUT` table shape; check that `table.shape == (256, 1, 3)`.

- [ ] **Step 5: Document the decision and commit**

Append to `docs/decisions.md` nothing new (the 1D-lookup decision is already recorded). Commit:

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/normalize.py tests/test_normalize.py
git commit -m "feat: grey-world white balance and exposure normalisation with cv2.LUT fast path

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: 3D LUT (`lut.py`)

Implements spec 5.3.

**Files:**
- Create: `kodachrome/lut.py`, `tests/test_lut.py`

**Interfaces:**
- Produces:
  - `class CubeError(ValueError)`
  - `@dataclass LUT3D(table: np.ndarray)` with `.size`, `LUT3D.identity(size=33)`, `.to_flat() -> (N³, 3)` red-fastest, `LUT3D.from_flat(flat, size)`, `.apply_numpy(rgb_float) -> rgb_float`, `.to_pillow() -> ImageFilter.Color3DLUT`, `.apply_pillow(rgb_u8, filt=None) -> rgb_u8`
  - `read_cube(path) -> LUT3D`, `write_cube(lut, path, title="kodachrome")`

- [ ] **Step 1: Write the failing tests**

`tests/test_lut.py`:
```python
import numpy as np
import pytest

from kodachrome.lut import LUT3D, CubeError, read_cube, write_cube


def _smooth_test_lut(n=17):
    """Identity warped by a smooth, invertible tweak so interpolation is exercised."""
    ident = LUT3D.identity(n).table
    warped = ident.copy()
    warped[..., 0] = ident[..., 0] ** 1.2
    warped[..., 1] = 0.9 * ident[..., 1] + 0.1 * ident[..., 0]
    warped[..., 2] = np.clip(ident[..., 2] * 1.05 - 0.02, 0, 1)
    return LUT3D(warped)


def test_identity_leaves_image_unchanged():
    lut = LUT3D.identity(33)
    rgb = np.random.default_rng(0).random((20, 30, 3), dtype=np.float32)
    assert np.allclose(lut.apply_numpy(rgb), rgb, atol=1e-6)
    u8 = (rgb * 255).round().astype(np.uint8)
    # Pillow works in 16-bit fixed point, so allow one 8-bit level of rounding.
    assert np.abs(lut.apply_pillow(u8).astype(int) - u8.astype(int)).max() <= 1


def test_flat_order_is_red_fastest():
    flat = LUT3D.identity(2).to_flat()
    expected = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        dtype=np.float32,
    )
    assert np.array_equal(flat, expected)
    assert np.array_equal(LUT3D.from_flat(flat, 2).table, LUT3D.identity(2).table)


def test_cube_roundtrip(tmp_path):
    lut = _smooth_test_lut(9)
    path = tmp_path / "t.cube"
    write_cube(lut, path, title="test")
    back = read_cube(path)
    assert back.size == 9
    assert np.allclose(back.table, lut.table, atol=1e-6)
    text = path.read_text()
    assert text.startswith('TITLE "test"\nLUT_3D_SIZE 9\n')


def test_numpy_and_pillow_agree():
    lut = _smooth_test_lut(17)
    u8 = np.random.default_rng(3).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    ref = np.round(lut.apply_numpy(u8.astype(np.float32) / 255.0) * 255).astype(int)
    got = lut.apply_pillow(u8).astype(int)
    diff = np.abs(ref - got)
    assert diff.max() <= 1
    assert diff.mean() < 0.3


@pytest.mark.parametrize(
    "text, message",
    [
        ("LUT_3D_SIZE 1\n0 0 0\n", "2..65"),
        ("LUT_3D_SIZE 2\n0 0 0\n", "expected 8"),
        ("LUT_3D_SIZE 2\n" + "0 0 x\n" * 8, "line 2"),
        ("LUT_1D_SIZE 4\n", "1D"),
        ("0 0 0\n", "LUT_3D_SIZE"),
    ],
)
def test_cube_errors(tmp_path, text, message):
    path = tmp_path / "bad.cube"
    path.write_text(text)
    with pytest.raises(CubeError, match=message):
        read_cube(path)


def test_table_validation():
    with pytest.raises(ValueError):
        LUT3D(np.zeros((3, 3, 2, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        LUT3D(np.zeros((66, 66, 66, 3), dtype=np.float32))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_lut.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'kodachrome.lut'`

- [ ] **Step 3: Implement lut.py**

```python
"""3D colour lookup tables: the exported form of the Kodachrome look.

A 3D LUT is a grid of N x N x N output colours indexed by the input colour.
Colours between grid nodes are trilinearly interpolated. This module keeps
the table in memory as ``table[r, g, b, channel]`` and knows two external
conventions that both order the flat file **red fastest**:

* The ``.cube`` format (Adobe / Resolve / everyone): ``LUT_3D_SIZE N`` then
  N^3 lines ``r g b``; the first line is input (0,0,0), the second is input
  (1/(N-1), 0, 0), and so on.
* Pillow's ``ImageFilter.Color3DLUT``: "channels are changed first, then
  first dimension, then second, then third", which is the same order.

``apply_numpy`` is the readable reference used in tests and the trainer.
``apply_pillow`` is the C-implemented path used on the Pi (about 150 ms for a
1080p frame on a Pi 400). Pillow stores the table in 16-bit fixed point, so
the two paths can differ by one 8-bit level; tests allow exactly that.
"""

from __future__ import annotations

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

    def apply_pillow(self, rgb_u8: np.ndarray, filt: ImageFilter.Color3DLUT | None = None) -> np.ndarray:
        """Fast path. Build ``filt`` once with ``to_pillow()`` when processing many frames."""
        filt = filt if filt is not None else self.to_pillow()
        im = Image.fromarray(np.ascontiguousarray(rgb_u8), "RGB")
        return np.asarray(im.filter(filt))


def write_cube(lut: LUT3D, path: str | Path, title: str = "kodachrome") -> None:
    lines = [
        f'TITLE "{title}"',
        f"LUT_3D_SIZE {lut.size}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
    ]
    lines.extend(f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in lut.to_flat())
    Path(path).write_text("\n".join(lines) + "\n")


def read_cube(path: str | Path) -> LUT3D:
    path = Path(path)
    size: int | None = None
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
        if key in ("DOMAIN_MIN", "DOMAIN_MAX"):
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
    return LUT3D.from_flat(np.array(rows, dtype=np.float32), size)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_lut.py -q`
Expected: all pass. If Pillow raises `TypeError` about the table in `to_pillow`, replace `flat` with `flat.tolist()` and re-run; note the Pillow version in `docs/decisions.md`. If `test_numpy_and_pillow_agree` fails with a max diff of 2, inspect where: a single pixel at a grid boundary may be Pillow rounding; then loosen only `diff.max() <= 2` and record why in `docs/decisions.md`.

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/lut.py tests/test_lut.py
git commit -m "feat: LUT3D with .cube I/O, NumPy trilinear reference and Pillow fast path

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Film grain (`grain.py`)

Implements spec 5.4.

**Files:**
- Create: `kodachrome/grain.py`, `tests/test_grain.py`

**Interfaces:**
- Produces:
  - `@dataclass GrainParams(strength=0.025, blur_sigma=0.7, enabled=True)` with `from_dict`, `to_dict`
  - `add_grain(rgb_u8, params, rng: np.random.Generator | None = None) -> rgb_u8`

- [ ] **Step 1: Write the failing tests**

`tests/test_grain.py`:
```python
import numpy as np
import pytest

from kodachrome.grain import GrainParams, add_grain


def test_disabled_is_identity():
    img = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    out = add_grain(img, GrainParams(enabled=False))
    assert np.array_equal(out, img)
    assert out is not img


def test_preserves_mean_luminance_and_adds_no_colour_bias():
    img = np.full((256, 256, 3), 128, dtype=np.uint8)
    out = add_grain(img, GrainParams(strength=0.05), rng=np.random.default_rng(1))
    assert out.dtype == np.uint8 and out.shape == img.shape
    means = out.reshape(-1, 3).mean(axis=0)
    assert np.allclose(means, 128, atol=0.5)
    assert out.std() > 5  # grain is actually there


def test_strength_scales_noise():
    img = np.full((256, 256, 3), 128, dtype=np.uint8)
    lo = add_grain(img, GrainParams(strength=0.02), rng=np.random.default_rng(2))
    hi = add_grain(img, GrainParams(strength=0.06), rng=np.random.default_rng(2))
    # mid-grey has envelope 1, so std in 8-bit units is about strength * 255
    assert lo[..., 1].std() == pytest.approx(0.02 * 255, rel=0.25)
    assert hi[..., 1].std() == pytest.approx(0.06 * 255, rel=0.25)


def test_black_and_white_are_untouched():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[16:] = 255
    out = add_grain(img, GrainParams(strength=0.1), rng=np.random.default_rng(3))
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 1


def test_seeded_is_reproducible():
    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    a = add_grain(img, GrainParams(), rng=np.random.default_rng(7))
    b = add_grain(img, GrainParams(), rng=np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_params_dict_roundtrip():
    p = GrainParams(strength=0.03, blur_sigma=0.5, enabled=False)
    assert GrainParams.from_dict({**p.to_dict(), "extra": 1}) == p
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_grain.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'kodachrome.grain'`

- [ ] **Step 3: Implement grain.py**

```python
"""Fine film grain added after the LUT.

Kodachrome 25 and 64 were among the finest-grained colour films made, so the
default here is subtle. The model is deliberately simple:

* Noise is added to **luminance only** (the Y of YCrCb). Film grain is a
  density variation of the dye layers seen together; chroma noise reads as a
  digital sensor artefact, not grain.
* The noise field is Gaussian-blurred by ``blur_sigma`` pixels and then
  rescaled back to unit variance. Pixel-independent noise looks like ISO
  noise; slightly correlated noise looks like grain clumps.
* An envelope ``4 * Y * (1 - Y)`` scales the noise: zero at black and at
  white, one at mid-grey. Real grain is least visible in the deepest shadows
  and in fully exposed highlights.

``strength`` is the noise standard deviation in luminance units (0..1) at
the mid-grey peak; 0.025 is about six 8-bit levels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import cv2
import numpy as np


@dataclass
class GrainParams:
    strength: float = 0.025
    blur_sigma: float = 0.7
    enabled: bool = True

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
    luma = np.clip(luma + params.strength * envelope * noise, 0.0, 1.0)
    ycc[..., 0] = luma * 255.0
    ycc_u8 = np.clip(np.round(ycc), 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycc_u8, cv2.COLOR_YCrCb2RGB)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_grain.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/grain.py tests/test_grain.py
git commit -m "feat: luminance film grain with midtone envelope

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Pipeline, artifacts and the committed identity artifact

Implements spec 5.5 and 5.6. After this task the Pi runtime has everything it needs except a camera; the LUT is identity until Task 16 trains it.

**Files:**
- Create: `kodachrome/pipeline.py`, `tests/test_pipeline.py`, `artifacts/kodachrome.cube`, `artifacts/params.json`
- Modify: `README.md` (add "How it works")

**Interfaces:**
- Consumes: `LUT3D`, `read_cube`, `write_cube`, `NormalizeParams`, `normalize_u8`, `GrainParams`, `add_grain`
- Produces:
  - `PARAMS_VERSION = 1`, `class ArtifactsError(Exception)`
  - `@dataclass Artifacts(lut, normalize, grain, training: dict, path: Path)` with `Artifacts.load(dir_path)`
  - `write_params(dir_path, normalize, grain, lut_file="kodachrome.cube", training=None) -> Path`
  - `class Pipeline(artifacts)` with `process(rgb_u8, *, grain=True, rng=None) -> (rgb_u8, info)` where `info == {"wb_gains": [r, g, b], "exposure_gain": float}`

- [ ] **Step 1: Write the failing tests**

`tests/test_pipeline.py`:
```python
import json

import numpy as np
import pytest

from kodachrome.grain import GrainParams
from kodachrome.lut import LUT3D, write_cube
from kodachrome.normalize import NormalizeParams
from kodachrome.pipeline import Artifacts, ArtifactsError, Pipeline, write_params


@pytest.fixture
def identity_artifacts(tmp_path):
    write_cube(LUT3D.identity(9), tmp_path / "kodachrome.cube")
    write_params(tmp_path, NormalizeParams(), GrainParams(), training={"note": "test"})
    return tmp_path


def test_write_and_load_params(identity_artifacts):
    data = json.loads((identity_artifacts / "params.json").read_text())
    assert data["version"] == 1
    assert data["lut_file"] == "kodachrome.cube"
    art = Artifacts.load(identity_artifacts)
    assert art.lut.size == 9
    assert art.normalize == NormalizeParams()
    assert art.grain == GrainParams()
    assert art.training == {"note": "test"}


def test_missing_params_is_clear_error(tmp_path):
    with pytest.raises(ArtifactsError, match="params.json"):
        Artifacts.load(tmp_path)


def test_bad_version_and_bad_json(tmp_path):
    (tmp_path / "params.json").write_text('{"version": 99}')
    with pytest.raises(ArtifactsError, match="version"):
        Artifacts.load(tmp_path)
    (tmp_path / "params.json").write_text("{not json")
    with pytest.raises(ArtifactsError, match="JSON"):
        Artifacts.load(tmp_path)


def test_missing_cube_is_clear_error(tmp_path):
    write_params(tmp_path, NormalizeParams(), GrainParams())
    with pytest.raises(ArtifactsError, match="kodachrome.cube"):
        Artifacts.load(tmp_path)


def test_process_shapes_and_info(identity_artifacts):
    pipe = Pipeline(Artifacts.load(identity_artifacts))
    frame = np.random.default_rng(0).integers(0, 256, (36, 64, 3), dtype=np.uint8)
    out, info = pipe.process(frame, rng=np.random.default_rng(0))
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert set(info) == {"wb_gains", "exposure_gain"}
    assert len(info["wb_gains"]) == 3


def test_grain_can_be_skipped(identity_artifacts):
    pipe = Pipeline(Artifacts.load(identity_artifacts))
    frame = np.full((36, 64, 3), 128, dtype=np.uint8)
    no_grain, _ = pipe.process(frame, grain=False)
    with_grain, _ = pipe.process(frame, grain=True, rng=np.random.default_rng(0))
    assert no_grain.std() < with_grain.std()


def test_process_rejects_wrong_input(identity_artifacts):
    pipe = Pipeline(Artifacts.load(identity_artifacts))
    with pytest.raises(ValueError):
        pipe.process(np.zeros((4, 4, 3), dtype=np.float32))


def test_committed_artifacts_load(repo_root):
    art = Artifacts.load(repo_root / "artifacts")
    assert art.lut.size == 33
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pipeline.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'kodachrome.pipeline'`

- [ ] **Step 3: Implement pipeline.py**

```python
"""The Kodachrome pipeline: normalise, apply the LUT, add grain.

``Artifacts`` is everything the trainer produced: the ``.cube`` LUT and
``params.json`` holding the normalisation the LUT was fitted against, the
grain settings and training provenance. ``Pipeline`` applies them in a
fixed order to an RGB uint8 frame. The order matters: the LUT was fitted on
normalised input, and grain is a property of the developed film so it goes
on last.

The same ``Pipeline`` serves the capture app (full frames), the live
preview (small frames, ``grain=False``) and batch reprocessing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .grain import GrainParams, add_grain
from .lut import LUT3D, CubeError, read_cube
from .normalize import NormalizeParams, normalize_u8

PARAMS_VERSION = 1


class ArtifactsError(Exception):
    """Artifacts directory missing or invalid."""


@dataclass
class Artifacts:
    lut: LUT3D
    normalize: NormalizeParams
    grain: GrainParams
    training: dict
    path: Path

    @classmethod
    def load(cls, dir_path: str | Path) -> Artifacts:
        path = Path(dir_path)
        params_path = path / "params.json"
        if not params_path.is_file():
            raise ArtifactsError(
                f"{params_path} not found. Run kodachrome-train, or restore the committed "
                "artifacts/ directory from git."
            )
        try:
            data = json.loads(params_path.read_text())
        except json.JSONDecodeError as exc:
            raise ArtifactsError(f"{params_path} is not valid JSON: {exc}") from exc
        version = data.get("version")
        if version is None or version > PARAMS_VERSION:
            raise ArtifactsError(
                f"{params_path}: unsupported params version {version!r} "
                f"(this build reads up to {PARAMS_VERSION})"
            )
        lut_path = path / data.get("lut_file", "kodachrome.cube")
        if not lut_path.is_file():
            raise ArtifactsError(f"LUT file {lut_path} not found (named in {params_path})")
        try:
            lut = read_cube(lut_path)
        except CubeError as exc:
            raise ArtifactsError(str(exc)) from exc
        return cls(
            lut=lut,
            normalize=NormalizeParams.from_dict(data.get("normalize", {})),
            grain=GrainParams.from_dict(data.get("grain", {})),
            training=data.get("training", {}),
            path=path,
        )


def write_params(
    dir_path: str | Path,
    normalize: NormalizeParams,
    grain: GrainParams,
    lut_file: str = "kodachrome.cube",
    training: dict | None = None,
) -> Path:
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    data = {
        "version": PARAMS_VERSION,
        "lut_file": lut_file,
        "normalize": normalize.to_dict(),
        "grain": grain.to_dict(),
        "training": training or {},
    }
    out = path / "params.json"
    out.write_text(json.dumps(data, indent=2) + "\n")
    return out


class Pipeline:
    def __init__(self, artifacts: Artifacts) -> None:
        self.artifacts = artifacts
        self._filter = artifacts.lut.to_pillow()

    def process(
        self, rgb_u8: np.ndarray, *, grain: bool = True, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, dict]:
        if rgb_u8.dtype != np.uint8 or rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
            raise ValueError("process() expects an RGB uint8 array of shape (H, W, 3)")
        normalised, gains = normalize_u8(rgb_u8, self.artifacts.normalize)
        graded = self.artifacts.lut.apply_pillow(normalised, self._filter)
        if grain and self.artifacts.grain.enabled:
            graded = add_grain(graded, self.artifacts.grain, rng)
        info = {
            "wb_gains": [round(float(g), 4) for g in gains.wb],
            "exposure_gain": round(float(gains.exposure), 4),
        }
        return graded, info
```

- [ ] **Step 4: Generate the identity artifact**

```bash
.venv/bin/python - <<'EOF'
from kodachrome.grain import GrainParams
from kodachrome.lut import LUT3D, write_cube
from kodachrome.normalize import NormalizeParams
from kodachrome.pipeline import write_params
write_cube(LUT3D.identity(33), "artifacts/kodachrome.cube", title="identity placeholder")
write_params("artifacts", NormalizeParams(), GrainParams(),
             training={"note": "identity placeholder; not trained yet", "proxy_source": True})
EOF
ls -la artifacts
```
Expected: `kodachrome.cube` about 1 MB and `params.json`.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest -q`
Expected: all pass, including `test_committed_artifacts_load`.

- [ ] **Step 6: Document "How it works" in the README**

Replace the `Status:` paragraph in `README.md` with:

````markdown
## How it works

1. **Normalise.** Grey-world white balance and exposure-to-middle-grey,
   computed in linear light and applied as three 256-entry lookups. Every
   capture reaches the LUT with the same white point and exposure the LUT was
   fitted on, so scene lighting does not fight the grade.
2. **LUT.** A 33x33x33 colour lookup table (`artifacts/kodachrome.cube`)
   learned from real Kodachrome scans, applied with Pillow's C implementation.
3. **Grain.** Subtle luminance-only grain with a midtone envelope.

`artifacts/params.json` records the normalisation targets, grain strength and
how the LUT was trained. Until training has run, the committed LUT is an
identity placeholder and its `training.note` says so.
````

- [ ] **Step 7: Commit**

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/pipeline.py tests/test_pipeline.py artifacts README.md
git commit -m "feat: Artifacts loader, Pipeline, identity placeholder artifact

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Image file I/O (`imageio.py`)

Small shared helpers so RGB/BGR confusion cannot creep in: all file I/O goes through Pillow, which is RGB.

**Files:**
- Create: `kodachrome/imageio.py`, `tests/test_imageio.py`

**Interfaces:**
- Produces:
  - `IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}`
  - `load_rgb(path) -> np.ndarray uint8 (H, W, 3)`
  - `save_jpeg(rgb_u8, path, quality=95) -> Path` (creates parent dirs)
  - `list_images(dir_path) -> list[Path]` sorted, matching extensions case-insensitively

- [ ] **Step 1: Write the failing tests**

`tests/test_imageio.py`:
```python
import numpy as np

from kodachrome.imageio import list_images, load_rgb, save_jpeg


def test_save_and_load_roundtrip(tmp_path):
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    img[..., 0] = 200  # red-dominant: proves channel order survives
    out = save_jpeg(img, tmp_path / "nested" / "a.jpg", quality=100)
    assert out.exists()
    back = load_rgb(out)
    assert back.shape == (10, 20, 3) and back.dtype == np.uint8
    assert back[..., 0].mean() > 150 and back[..., 2].mean() < 30


def test_load_converts_modes(tmp_path):
    from PIL import Image

    Image.new("L", (8, 8), 77).save(tmp_path / "grey.png")
    Image.new("RGBA", (8, 8), (1, 2, 3, 255)).save(tmp_path / "rgba.png")
    assert load_rgb(tmp_path / "grey.png").shape == (8, 8, 3)
    assert load_rgb(tmp_path / "rgba.png").shape == (8, 8, 3)


def test_list_images_filters_and_sorts(tmp_path):
    for name in ["b.JPG", "a.jpeg", "c.png", "notes.txt", "d.tif"]:
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in list_images(tmp_path)] == ["a.jpeg", "b.JPG", "c.png", "d.tif"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_imageio.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement imageio.py**

```python
"""Image file I/O through Pillow so arrays are always RGB uint8."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def save_jpeg(rgb_u8: np.ndarray, path: str | Path, quality: int = 95) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(rgb_u8), "RGB").save(path, "JPEG", quality=quality)
    return path


def list_images(dir_path: str | Path) -> list[Path]:
    return sorted(
        p for p in Path(dir_path).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
```

- [ ] **Step 4: Run tests and commit**

Run: `.venv/bin/pytest tests/test_imageio.py -q` → all pass.

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/imageio.py tests/test_imageio.py
git commit -m "feat: RGB image file helpers

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Camera access (`capture/camera.py`)

Implements spec 7.1. The real camera cannot be tested on the Mac; the fake must be a faithful stand-in and the real one must fail loudly and helpfully.

**Files:**
- Create: `kodachrome/capture/camera.py`, `tests/test_camera.py`

**Interfaces:**
- Produces:
  - `class CameraError(Exception)`
  - `class Camera(Protocol)`: `read() -> np.ndarray` RGB uint8, `close() -> None`
  - `synthetic_frame(height=1080, width=1920) -> np.ndarray` gradient plus colour patches
  - `class FakeCamera(frames: list[np.ndarray] | None = None)` cycling through frames
  - `parse_device(device: int | str | None) -> int | None` (`"/dev/video2"` → 2)
  - `class V4L2Camera(device=None, width=1920, height=1080, fps=30, warmup_frames=15)` with `.width`, `.height`
  - `list_video_devices() -> list[str]`

- [ ] **Step 1: Write the failing tests**

`tests/test_camera.py`:
```python
import numpy as np
import pytest

from kodachrome.capture.camera import (
    CameraError,
    FakeCamera,
    V4L2Camera,
    parse_device,
    synthetic_frame,
)


def test_synthetic_frame_shape_and_content():
    f = synthetic_frame(90, 160)
    assert f.shape == (90, 160, 3) and f.dtype == np.uint8
    assert f[..., 0].mean() != f[..., 2].mean()  # has colour, not just grey


def test_fake_camera_defaults_and_cycles():
    cam = FakeCamera()
    a = cam.read()
    assert a.shape == (1080, 1920, 3) and a.dtype == np.uint8
    frames = [np.zeros((4, 4, 3), np.uint8), np.ones((4, 4, 3), np.uint8)]
    cam = FakeCamera(frames)
    assert cam.read().max() == 0
    assert cam.read().max() == 1
    assert cam.read().max() == 0
    cam.read()[0, 0, 0] = 99  # copies, so the stored frame is untouched
    assert frames[0].max() == 0
    cam.close()


@pytest.mark.parametrize(
    "value, expected", [(None, None), (3, 3), ("3", 3), ("/dev/video7", 7)]
)
def test_parse_device(value, expected):
    assert parse_device(value) == expected


def test_parse_device_rejects_garbage():
    with pytest.raises(CameraError):
        parse_device("camera")


def test_v4l2_camera_reports_missing_device():
    with pytest.raises(CameraError, match="video"):
        V4L2Camera(device=99, warmup_frames=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_camera.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement camera.py**

```python
"""Camera access for the Innomaker U20CAM-1080P-WDR (and a fake for tests).

Facts from the vendor manual that shaped this module:

* The camera is a standard UVC device; OpenCV's V4L2 backend drives it.
* 1920x1080 at 30 fps is only available in MJPEG. YUY2 drops to 5 fps at
  1080p, so the FOURCC is forced to ``MJPG``.
* The camera runs its own auto exposure and white balance. They need a few
  frames to settle after opening, hence ``warmup_frames``.
* V4L2 queues frames. A single ``read()`` may return a stale frame, so
  ``read()`` grabs twice and decodes the third.

Frames are returned as **RGB** uint8; the BGR->RGB swap happens here and
nowhere else.
"""

from __future__ import annotations

import glob
import re
from typing import Protocol

import cv2
import numpy as np


class CameraError(Exception):
    """Camera could not be opened or read."""


class Camera(Protocol):
    def read(self) -> np.ndarray: ...

    def close(self) -> None: ...


def synthetic_frame(height: int = 1080, width: int = 1920) -> np.ndarray:
    """A grey gradient with a row of colour patches, for tests and ``--fake`` runs."""
    ramp = np.linspace(0, 255, width, dtype=np.float32)
    frame = np.repeat(np.repeat(ramp[None, :, None], height, axis=0), 3, axis=2).astype(np.uint8)
    patches = [
        (220, 40, 40),
        (40, 180, 60),
        (50, 80, 220),
        (240, 200, 60),
        (230, 180, 150),
        (120, 120, 120),
    ]
    pw = max(1, width // len(patches))
    top, bottom = height // 4, height // 2
    for i, colour in enumerate(patches):
        frame[top:bottom, i * pw : (i + 1) * pw] = colour
    return frame


class FakeCamera:
    def __init__(self, frames: list[np.ndarray] | None = None) -> None:
        self._frames = frames if frames else [synthetic_frame()]
        self._i = 0

    def read(self) -> np.ndarray:
        frame = self._frames[self._i % len(self._frames)]
        self._i += 1
        return frame.copy()

    def close(self) -> None:
        return None


def parse_device(device: int | str | None) -> int | None:
    if device is None:
        return None
    if isinstance(device, int):
        return device
    m = re.fullmatch(r"(?:/dev/video)?(\d+)", device.strip())
    if not m:
        raise CameraError(f"Cannot parse camera device {device!r}; use an index or /dev/videoN")
    return int(m.group(1))


def list_video_devices() -> list[str]:
    return sorted(glob.glob("/dev/video*"))


class V4L2Camera:
    def __init__(
        self,
        device: int | str | None = None,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        warmup_frames: int = 15,
    ) -> None:
        index = parse_device(device)
        candidates = [index] if index is not None else list(range(10))
        self.cap: cv2.VideoCapture | None = None
        for idx in candidates:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            ok, _ = cap.read()
            if ok:
                self.cap = cap
                break
            cap.release()
        if self.cap is None:
            found = list_video_devices()
            hint = f"found {', '.join(found)}" if found else "no /dev/video* devices exist"
            raise CameraError(
                f"No camera delivered a frame (tried index {candidates[0]}"
                + (f"..{candidates[-1]}" if len(candidates) > 1 else "")
                + f"); {hint}. Pass --device N or /dev/videoN."
            )
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (self.width, self.height) != (width, height):
            print(f"warning: camera negotiated {self.width}x{self.height}, not {width}x{height}")
        for _ in range(warmup_frames):
            self.cap.read()

    def read(self) -> np.ndarray:
        assert self.cap is not None
        for _ in range(3):
            self.cap.grab()
            self.cap.grab()
            ok, frame = self.cap.read()
            if ok and frame is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raise CameraError("Failed to read a frame after 3 attempts")

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
```

- [ ] **Step 4: Run tests and commit**

Run: `.venv/bin/pytest tests/test_camera.py -q` → all pass. `test_v4l2_camera_reports_missing_device` must finish in under two seconds; if OpenCV prints a warning about the V4L2 backend on macOS that is fine.

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/capture/camera.py tests/test_camera.py
git commit -m "feat: V4L2 camera wrapper forcing MJPEG 1080p, FakeCamera for tests

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Batch reprocessing (`capture/batch.py`, `kodachrome-process`)

Implements spec 7.3. Also the easiest way to try the pipeline on the Mac with any folder of JPEGs.

**Files:**
- Create: `kodachrome/capture/batch.py`, `tests/test_batch.py`
- Modify: `README.md` (Commands section: document `kodachrome-process`)

**Interfaces:**
- Consumes: `Artifacts`, `Pipeline`, `load_rgb`, `save_jpeg`, `list_images`
- Produces: `process_dir(in_dir, out_dir, artifacts_dir, grain=True) -> list[Path]`, `main(argv=None) -> int`

- [ ] **Step 1: Write the failing tests**

`tests/test_batch.py`:
```python
import numpy as np

from kodachrome.capture.batch import main, process_dir
from kodachrome.imageio import save_jpeg


def _make_inputs(dir_path, n=3):
    dir_path.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n):
        save_jpeg(rng.integers(0, 256, (24, 32, 3), dtype=np.uint8), dir_path / f"img{i}.jpg")
    (dir_path / "ignore.txt").write_text("x")


def test_process_dir_writes_one_output_per_image(tmp_path, repo_root):
    _make_inputs(tmp_path / "in")
    outputs = process_dir(tmp_path / "in", tmp_path / "out", repo_root / "artifacts")
    assert [p.name for p in outputs] == [
        "img0_kodachrome.jpg",
        "img1_kodachrome.jpg",
        "img2_kodachrome.jpg",
    ]
    assert all(p.exists() for p in outputs)


def test_main_returns_zero_and_prints_summary(tmp_path, repo_root, capsys):
    _make_inputs(tmp_path / "in", n=2)
    code = main(
        [str(tmp_path / "in"), str(tmp_path / "out"), "--artifacts", str(repo_root / "artifacts")]
    )
    assert code == 0
    assert "2 image" in capsys.readouterr().out


def test_main_reports_missing_artifacts(tmp_path, capsys):
    _make_inputs(tmp_path / "in", n=1)
    code = main([str(tmp_path / "in"), str(tmp_path / "out"), "--artifacts", str(tmp_path / "none")])
    assert code == 2
    assert "params.json" in capsys.readouterr().err


def test_main_reports_empty_input(tmp_path, repo_root, capsys):
    (tmp_path / "in").mkdir()
    code = main([str(tmp_path / "in"), str(tmp_path / "out"), "--artifacts", str(repo_root / "artifacts")])
    assert code == 1
    assert "no images" in capsys.readouterr().err.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_batch.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement batch.py**

```python
"""``kodachrome-process``: regrade a folder of images with the current artifacts.

Use it to reprocess old originals after retraining, or to try the pipeline on
the Mac without a camera.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ..imageio import list_images, load_rgb, save_jpeg
from ..pipeline import Artifacts, ArtifactsError, Pipeline


def process_dir(
    in_dir: str | Path, out_dir: str | Path, artifacts_dir: str | Path, grain: bool = True
) -> list[Path]:
    pipeline = Pipeline(Artifacts.load(artifacts_dir))
    out_dir = Path(out_dir)
    outputs: list[Path] = []
    for src in list_images(in_dir):
        graded, _ = pipeline.process(load_rgb(src), grain=grain)
        outputs.append(save_jpeg(graded, out_dir / f"{src.stem}_kodachrome.jpg"))
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kodachrome-process", description="Regrade a folder of images with the Kodachrome LUT."
    )
    parser.add_argument("in_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--no-grain", action="store_true", help="skip film grain")
    args = parser.parse_args(argv)

    if not args.in_dir.is_dir() or not list_images(args.in_dir):
        print(f"error: no images found in {args.in_dir}", file=sys.stderr)
        return 1
    t0 = time.perf_counter()
    try:
        outputs = process_dir(args.in_dir, args.out_dir, args.artifacts, grain=not args.no_grain)
    except ArtifactsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    elapsed = time.perf_counter() - t0
    print(f"Processed {len(outputs)} image(s) into {args.out_dir} in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_batch.py -q` → all pass.

- [ ] **Step 5: Document and commit**

In `README.md`, under the Commands table, add:

````markdown
### Regrade a folder

```bash
kodachrome-process ~/Pictures/kodachrome/2026-09-03 /tmp/regraded
```

Every `*_original.jpg` (or any JPEG/PNG) in the input folder is written to the
output folder as `<name>_kodachrome.jpg` using the current `artifacts/`. Run
this after retraining to bring old shots up to the new look.
````

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/capture/batch.py tests/test_batch.py README.md
git commit -m "feat: kodachrome-process batch regrading command

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Capture app (`capture/app.py`, `kodachrome-capture`)

Implements spec 7.2 and the runtime rows of spec 8. The loops are separated from key reading and windowing so the whole flow is testable with `FakeCamera`.

**Files:**
- Create: `kodachrome/capture/app.py`, `tests/test_app.py`
- Modify: `README.md` (document `kodachrome-capture` and the output layout)

**Interfaces:**
- Consumes: `Camera`, `FakeCamera`, `V4L2Camera`, `CameraError`, `Artifacts`, `ArtifactsError`, `Pipeline`, `save_jpeg`
- Produces:
  - `@dataclass CaptureResult(original: Path, kodachrome: Path, info: dict, processing_ms: float)`
  - `class CaptureSession(camera, pipeline, out_root, now=None)` with `capture() -> CaptureResult` and `preview_frame(graded=True, size=(640, 360)) -> np.ndarray`
  - `run_headless_loop(session, read_key: Callable[[], str | None], out=print) -> int` (number of captures)
  - `run_preview_loop(session, window_name=...) -> bool` (False if the OpenCV build has no GUI)
  - `class TerminalKeys` context manager with `.read(timeout=0.1) -> str | None`
  - `has_display() -> bool`, `main(argv=None) -> int`

- [ ] **Step 1: Write the failing tests**

`tests/test_app.py`:
```python
import json
from datetime import datetime

import numpy as np

from kodachrome.capture.app import CaptureSession, main, run_headless_loop
from kodachrome.capture.camera import FakeCamera, synthetic_frame
from kodachrome.pipeline import Artifacts, Pipeline


def _session(tmp_path, repo_root, now=None):
    pipeline = Pipeline(Artifacts.load(repo_root / "artifacts"))
    camera = FakeCamera([synthetic_frame(90, 160)])
    return CaptureSession(camera, pipeline, tmp_path / "shots", now=now)


def test_capture_writes_both_files_and_a_log_line(tmp_path, repo_root):
    fixed = datetime(2026, 9, 3, 21, 5, 7)
    session = _session(tmp_path, repo_root, now=lambda: fixed)
    result = session.capture()
    day = tmp_path / "shots" / "2026-09-03"
    assert result.original == day / "210507_original.jpg"
    assert result.kodachrome == day / "210507_kodachrome.jpg"
    assert result.original.exists() and result.kodachrome.exists()
    assert result.processing_ms > 0
    lines = (day / "captures.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["original"] == "210507_original.jpg"
    assert set(record) >= {"timestamp", "original", "kodachrome", "wb_gains", "exposure_gain", "processing_ms"}


def test_same_second_captures_do_not_collide(tmp_path, repo_root):
    fixed = datetime(2026, 9, 3, 21, 5, 7)
    session = _session(tmp_path, repo_root, now=lambda: fixed)
    a = session.capture()
    b = session.capture()
    assert a.original.name == "210507_original.jpg"
    assert b.original.name == "210507-2_original.jpg"


def test_preview_frame_is_small_rgb(tmp_path, repo_root):
    session = _session(tmp_path, repo_root)
    frame = session.preview_frame(graded=True)
    assert frame.shape == (360, 640, 3) and frame.dtype == np.uint8
    raw = session.preview_frame(graded=False)
    assert raw.shape == (360, 640, 3)


def test_headless_loop_captures_on_space_and_quits_on_q(tmp_path, repo_root):
    session = _session(tmp_path, repo_root)
    keys = iter([None, " ", "x", " ", "q"])
    messages = []
    count = run_headless_loop(session, read_key=lambda: next(keys), out=messages.append)
    assert count == 2
    assert len(list((tmp_path / "shots").rglob("*_kodachrome.jpg"))) == 2
    assert any("Saved" in m for m in messages)


def test_main_headless_without_tty_exits_2(tmp_path, repo_root, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    code = main(["--fake", "--no-preview", "--out", str(tmp_path), "--artifacts", str(repo_root / "artifacts")])
    assert code == 2
    assert "terminal" in capsys.readouterr().err.lower()


def test_main_reports_missing_artifacts(tmp_path, capsys):
    code = main(["--fake", "--no-preview", "--out", str(tmp_path), "--artifacts", str(tmp_path / "nope")])
    assert code == 2
    assert "params.json" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_app.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement app.py**

```python
"""``kodachrome-capture``: live preview, press SPACE, get two JPEGs.

Structure
---------
``CaptureSession`` owns the camera, the pipeline and the output folder, and
knows how to take one capture or produce one preview frame. Two thin loops
drive it:

* ``run_preview_loop`` shows the graded live feed in an OpenCV window and
  reads keys from it. Used when a display is present and the OpenCV build has
  GUI support (apt ``python3-opencv`` on the Pi does; pip's headless wheel
  does not).
* ``run_headless_loop`` reads single keys from the terminal. Used when there
  is no display, or as the fallback when the window cannot be created.

Both take injectable key sources so tests can drive them with ``FakeCamera``.

Output layout
-------------
``OUT/YYYY-MM-DD/HHMMSS_original.jpg`` and ``HHMMSS_kodachrome.jpg`` plus one
JSON line per capture in ``OUT/YYYY-MM-DD/captures.jsonl`` recording the
white balance and exposure gains that were applied and the processing time.
When a shot looks wrong, that line says what the normaliser did to it.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import time
import tty
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..imageio import save_jpeg
from ..pipeline import Artifacts, ArtifactsError, Pipeline
from .camera import Camera, CameraError, FakeCamera, V4L2Camera

DEFAULT_OUT = Path("~/Pictures/kodachrome")
WINDOW_NAME = "Kodachrome  [SPACE capture | P toggle grade | Q quit]"


@dataclass
class CaptureResult:
    original: Path
    kodachrome: Path
    info: dict
    processing_ms: float


class CaptureSession:
    def __init__(
        self,
        camera: Camera,
        pipeline: Pipeline,
        out_root: str | Path,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.camera = camera
        self.pipeline = pipeline
        self.out_root = Path(out_root).expanduser()
        self._now = now or datetime.now

    def _allocate_paths(self) -> tuple[Path, str, datetime]:
        t = self._now()
        day_dir = self.out_root / t.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        base = t.strftime("%H%M%S")
        stem, k = base, 1
        while (day_dir / f"{stem}_original.jpg").exists():
            k += 1
            stem = f"{base}-{k}"
        return day_dir, stem, t

    def capture(self) -> CaptureResult:
        frame = self.camera.read()
        t0 = time.perf_counter()
        graded, info = self.pipeline.process(frame)
        processing_ms = (time.perf_counter() - t0) * 1000.0
        day_dir, stem, t = self._allocate_paths()
        original = save_jpeg(frame, day_dir / f"{stem}_original.jpg")
        kodachrome = save_jpeg(graded, day_dir / f"{stem}_kodachrome.jpg")
        record = {
            "timestamp": t.isoformat(timespec="seconds"),
            "original": original.name,
            "kodachrome": kodachrome.name,
            **info,
            "processing_ms": round(processing_ms, 1),
        }
        with (day_dir / "captures.jsonl").open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        return CaptureResult(original, kodachrome, info, processing_ms)

    def preview_frame(self, graded: bool = True, size: tuple[int, int] = (640, 360)) -> np.ndarray:
        frame = self.camera.read()
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        if graded:
            small, _ = self.pipeline.process(small, grain=False)
        return small


def _announce(result: CaptureResult, out: Callable[[str], None]) -> None:
    out(
        f"Saved {result.kodachrome.name} (+ original) in {result.processing_ms:.0f} ms; "
        f"wb={result.info['wb_gains']} exposure={result.info['exposure_gain']}"
    )


def run_headless_loop(
    session: CaptureSession,
    read_key: Callable[[], str | None],
    out: Callable[[str], None] = print,
) -> int:
    out("Headless mode: SPACE to capture, Q to quit.")
    count = 0
    while True:
        key = read_key()
        if key is None:
            continue
        if key == " ":
            try:
                _announce(session.capture(), out)
                count += 1
            except (CameraError, OSError) as exc:
                out(f"error: {exc}")
        elif key.lower() == "q":
            return count


def run_preview_loop(session: CaptureSession, window_name: str = WINDOW_NAME) -> bool:
    """Returns False if OpenCV cannot open a window (headless build)."""
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    except cv2.error:
        return False
    graded = True
    try:
        while True:
            frame = session.preview_frame(graded)
            cv2.imshow(window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                try:
                    _announce(session.capture(), print)
                except (CameraError, OSError) as exc:
                    print(f"error: {exc}")
            elif key in (ord("p"), ord("P")):
                graded = not graded
            elif key in (ord("q"), ord("Q"), 27):
                return True
    finally:
        cv2.destroyAllWindows()


class TerminalKeys:
    """Put the terminal in cbreak mode and read single keys without Enter."""

    def __enter__(self) -> TerminalKeys:
        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc: object) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def read(self, timeout: float = 0.1) -> str | None:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if ready else None


def has_display() -> bool:
    return sys.platform == "darwin" or bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kodachrome-capture", description="Capture Kodachrome-graded photos from the U20CAM."
    )
    parser.add_argument("--device", default=None, help="camera index or /dev/videoN (default: probe)")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-preview", action="store_true", help="never open a window")
    parser.add_argument("--fake", action="store_true", help="use a synthetic camera (no hardware)")
    args = parser.parse_args(argv)

    try:
        pipeline = Pipeline(Artifacts.load(args.artifacts))
    except ArtifactsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        camera: Camera = FakeCamera() if args.fake else V4L2Camera(args.device)
    except CameraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    session = CaptureSession(camera, pipeline, args.out)
    try:
        if not args.no_preview and has_display():
            if run_preview_loop(session):
                return 0
            print("Preview unavailable (OpenCV built without GUI); falling back to headless mode.")
        if not sys.stdin.isatty():
            print(
                "error: stdin is not a terminal, so keys cannot be read. Run from a terminal, "
                "or use kodachrome-process for files.",
                file=sys.stderr,
            )
            return 2
        with TerminalKeys() as keys:
            run_headless_loop(session, keys.read)
        return 0
    finally:
        camera.close()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_app.py -q` → all pass.

- [ ] **Step 5: Try it for real on the Mac**

```bash
.venv/bin/kodachrome-capture --fake --out /tmp/kodachrome-shots
```
Expected: a window with the synthetic frame; SPACE writes two files and prints a "Saved" line; Q quits. If `opencv-python-headless` was installed in Task 1, it falls back to headless mode and SPACE still works in the terminal.

- [ ] **Step 6: Document and commit**

In `README.md`, under Commands, add:

````markdown
### Capture on the Pi

```bash
kodachrome-capture                 # probes /dev/video*, opens a preview if a display is attached
kodachrome-capture --device 0      # pick a camera explicitly
kodachrome-capture --no-preview    # headless: SPACE and Q read from the terminal
kodachrome-capture --fake          # no hardware: synthetic frames, for trying the app
```

Keys: `SPACE` capture, `P` toggle graded/original preview, `Q` quit.

Output goes to `~/Pictures/kodachrome/YYYY-MM-DD/` as `HHMMSS_original.jpg`
and `HHMMSS_kodachrome.jpg`, plus `captures.jsonl` with one line per capture
recording the white balance gains, exposure gain and processing time. The
originals double as the training corpus for `kodachrome-train --source`.
````

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/capture/app.py tests/test_app.py README.md
git commit -m "feat: kodachrome-capture app with preview and headless loops

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: Commons downloader (`train/fetch.py`, `kodachrome-fetch`)

Implements spec 3 and 6.1. Tests use a fake HTTP session; nothing hits the network.

**Files:**
- Create: `kodachrome/train/fetch.py`, `tests/test_fetch.py`
- Modify: `README.md` (Training section, part 1: fetching)

**Interfaces:**
- Produces:
  - `API_URL`, `USER_AGENT`, `DEFAULT_CATEGORY`, `SKIP_WORDS`, `MIN_LONG_SIDE = 800`
  - `class FetchError(Exception)`
  - `@dataclass FileInfo(title, url, width, height, license, lccn)` with `.filename`
  - `api_get(session, params, retries=3) -> dict`
  - `iter_category_members(session, category, recurse=True) -> Iterator[str]`
  - `select_titles(titles) -> list[str]`
  - `fetch_imageinfo(session, titles, width) -> list[FileInfo]`
  - `download(session, info, out_dir, retries=3) -> Path | None`
  - `fetch_category(session, category, out_dir, width=1024, limit=None, sample=None, seed=0, progress=None) -> list[dict]` (manifest entries, also written to `out_dir/manifest.json`)
  - `main(argv=None) -> int`
  - A session is anything with `.get(url, params=None, headers=None, timeout=None)` returning an object with `.status_code`, `.json()`, `.content`. `requests.Session` qualifies.

- [ ] **Step 1: Write the failing tests**

`tests/test_fetch.py`:
```python
import json

import pytest

from kodachrome.train.fetch import (
    API_URL,
    FileInfo,
    download,
    fetch_category,
    fetch_imageinfo,
    iter_category_members,
    main,
    select_titles,
)

CAT = "Category:Test"
SUB = "Category:Sub"


class FakeResponse:
    def __init__(self, status=200, payload=None, content=b""):
        self.status_code = status
        self._payload = payload
        self.content = content

    def json(self):
        return self._payload


class FakeSession:
    """Routes API calls to a handler and file URLs to a dict of bytes."""

    def __init__(self, handler, files=None, fail_urls=()):
        self.handler = handler
        self.files = files or {}
        self.fail_urls = set(fail_urls)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params))
        assert headers and "kodachrome-film" in headers["User-Agent"]
        if url == API_URL:
            return FakeResponse(payload=self.handler(params))
        if url in self.fail_urls:
            return FakeResponse(status=500)
        return FakeResponse(content=self.files.get(url, b""))


def _members(items, cont=None):
    out = {"query": {"categorymembers": items}}
    if cont:
        out["continue"] = {"cmcontinue": cont, "continue": "-||"}
    return out


def _handler(params):
    if params.get("list") == "categorymembers":
        if params["cmtitle"] == CAT and "cmcontinue" not in params:
            return _members(
                [{"ns": 6, "title": "File:A LCCN2017000001.jpg"}, {"ns": 14, "title": SUB}], cont="c1"
            )
        if params["cmtitle"] == CAT:
            return _members([{"ns": 6, "title": "File:B LCCN2017000002.jpg"}])
        if params["cmtitle"] == SUB:
            return _members(
                [{"ns": 6, "title": "File:C (cropped) LCCN2017000003.jpg"}, {"ns": 14, "title": CAT}]
            )
    if params.get("prop") == "imageinfo":
        pages = {}
        for i, title in enumerate(params["titles"].split("|")):
            big = "small" not in title
            pages[str(i)] = {
                "title": title,
                "imageinfo": [
                    {
                        "url": f"https://upload/{i}.jpg",
                        "thumburl": f"https://upload/thumb/{i}.jpg",
                        "width": 4000 if big else 300,
                        "height": 3000 if big else 200,
                        "mime": "image/jpeg",
                        "extmetadata": {"LicenseShortName": {"value": "Public domain"}},
                    }
                ],
            }
        return {"query": {"pages": pages}}
    raise AssertionError(f"unexpected params {params}")


def test_iter_category_members_follows_continue_and_recurses_once():
    titles = list(iter_category_members(FakeSession(_handler), CAT))
    assert titles == [
        "File:A LCCN2017000001.jpg",
        "File:C (cropped) LCCN2017000003.jpg",
        "File:B LCCN2017000002.jpg",
    ]


def test_select_titles_skips_dedupes_and_orders():
    titles = [
        "File:Zed no lccn.jpg",
        "File:A LCCN2017000001.jpg",
        "File:A again LCCN2017000001.jpg",
        "File:C (cropped) LCCN2017000003.jpg",
        "File:D restored LCCN2017000004.jpg",
        "File:B LCCN2017000002.jpg",
    ]
    assert select_titles(titles) == [
        "File:A LCCN2017000001.jpg",
        "File:B LCCN2017000002.jpg",
        "File:Zed no lccn.jpg",
    ]


def test_fetch_imageinfo_uses_thumb_and_skips_small():
    infos = fetch_imageinfo(FakeSession(_handler), ["File:X LCCN2017000009.jpg", "File:small.jpg"], 1024)
    assert len(infos) == 1
    assert infos[0].url == "https://upload/thumb/0.jpg"
    assert infos[0].lccn == "2017000009"
    assert infos[0].license == "Public domain"
    assert infos[0].filename == "2017000009.jpg"


def test_filename_without_lccn_is_sanitised():
    info = FileInfo("File:Odd name / with: chars.jpg", "u", 1, 1, "PD", None)
    assert info.filename == "Odd_name_with_chars.jpg"


def test_download_skips_existing_and_reports_failure(tmp_path):
    info = FileInfo("File:T LCCN2017000001.jpg", "https://upload/1.jpg", 1, 1, "PD", "2017000001")
    session = FakeSession(_handler, files={"https://upload/1.jpg": b"JPEGDATA"})
    path = download(session, info, tmp_path)
    assert path.read_bytes() == b"JPEGDATA"
    n_calls = len(session.calls)
    assert download(session, info, tmp_path) == path
    assert len(session.calls) == n_calls  # not re-downloaded
    bad = FileInfo("File:U LCCN2017000002.jpg", "https://upload/bad.jpg", 1, 1, "PD", "2017000002")
    assert download(FakeSession(_handler, fail_urls={"https://upload/bad.jpg"}), bad, tmp_path, retries=1) is None


def test_fetch_category_writes_manifest(tmp_path):
    files = {f"https://upload/thumb/{i}.jpg": b"x" * (i + 1) for i in range(3)}
    entries = fetch_category(FakeSession(_handler, files=files), CAT, tmp_path, width=1024)
    assert [e["lccn"] for e in entries] == ["2017000001", "2017000002"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["category"] == CAT
    assert {e["filename"] for e in manifest["files"]} == {"2017000001.jpg", "2017000002.jpg"}
    assert all({"title", "lccn", "url", "width", "height", "license", "filename", "sha1"} <= set(e) for e in entries)


def test_fetch_category_limit_and_sample(tmp_path):
    files = {f"https://upload/thumb/{i}.jpg": b"x" for i in range(3)}
    entries = fetch_category(FakeSession(_handler, files=files), CAT, tmp_path, limit=1)
    assert len(entries) == 1
    entries = fetch_category(FakeSession(_handler, files=files), CAT, tmp_path / "s", sample=1, seed=3)
    assert len(entries) == 1


def test_main_min_files_threshold(tmp_path, monkeypatch, capsys):
    files = {f"https://upload/thumb/{i}.jpg": b"x" for i in range(3)}
    monkeypatch.setattr("kodachrome.train.fetch.make_session", lambda: FakeSession(_handler, files=files))
    assert main(["--out", str(tmp_path), "--category", CAT, "--min-files", "5"]) == 1
    assert "fewer than 5" in capsys.readouterr().err
    assert main(["--out", str(tmp_path), "--category", CAT, "--min-files", "2"]) == 0


@pytest.mark.parametrize("bad", ["File:E edit.jpg", "File:F colorized.jpg", "File:G retouched.jpg"])
def test_skip_words(bad):
    assert select_titles([bad]) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_fetch.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement fetch.py**

```python
"""``kodachrome-fetch``: download public-domain Kodachrome scans from Wikimedia Commons.

Why Commons and not loc.gov
---------------------------
The Library of Congress FSA/OWI colour transparencies are the target corpus,
but loc.gov sits behind a Cloudflare challenge that returns HTTP 403 to
scripted clients (checked 2026-09-03 with several User-Agents). Commons hosts
the same LoC scans in "Category:Color photographs from the Farm Security
Administration", keeps the LoC catalogue number (LCCN) in each filename, tags
each file public domain, and its API welcomes scripted access as long as the
User-Agent identifies the tool.

Selection rules (spec section 3)
--------------------------------
* Skip titles containing cropped / restored / retouched / colorized / edit:
  they are derivatives whose colours were changed by a Commons editor.
* Skip files under 800 px on the long side.
* Files with an LCCN come first and are de-duplicated by LCCN; files without
  one follow. ``--sample`` draws a seeded random subset; ``--limit`` truncates.
* Download at ``--width`` (default 1024) using Commons' thumbnail service.

``manifest.json`` in the output directory records exactly what was fetched:
title, LCCN, URL, size, licence string and file SHA-1. That is the provenance
for whatever LUT gets trained on the folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "kodachrome-film/0.1 (Kodachrome LUT trainer; https://github.com/kodachrome-film) python-requests"
DEFAULT_CATEGORY = "Category:Color photographs from the Farm Security Administration"
SKIP_WORDS = ("cropped", "restored", "retouched", "colorized", "colourized", "edit")
MIN_LONG_SIDE = 800
_LCCN_RE = re.compile(r"LCCN(\d{6,})", re.IGNORECASE)


class FetchError(Exception):
    """The Commons API could not be reached or answered unexpectedly."""


@dataclass
class FileInfo:
    title: str
    url: str
    width: int
    height: int
    license: str
    lccn: str | None

    @property
    def filename(self) -> str:
        if self.lccn:
            return f"{self.lccn}.jpg"
        stem = self.title.removeprefix("File:").rsplit(".", 1)[0]
        stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")[:120]
        return f"{stem}.jpg"


def make_session() -> Any:
    import requests

    return requests.Session()


def api_get(session: Any, params: dict, retries: int = 3) -> dict:
    params = {**params, "format": "json"}
    last = "no response"
    for attempt in range(retries):
        try:
            r = session.get(API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001 - network errors are all retried the same way
            last = repr(exc)
        time.sleep(2**attempt)
    raise FetchError(f"Commons API request failed after {retries} attempts: {last}")


def iter_category_members(
    session: Any, category: str, recurse: bool = True, _seen: set[str] | None = None
) -> Iterator[str]:
    seen = _seen if _seen is not None else set()
    if category in seen:
        return
    seen.add(category)
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": "file|subcat",
        "cmlimit": "500",
    }
    while True:
        data = api_get(session, params)
        for member in data.get("query", {}).get("categorymembers", []):
            if member["ns"] == 6:
                yield member["title"]
            elif member["ns"] == 14 and recurse:
                yield from iter_category_members(session, member["title"], recurse, seen)
        cont = data.get("continue")
        if not cont:
            return
        params = {**params, **cont}


def select_titles(titles: list[str]) -> list[str]:
    with_lccn: list[str] = []
    without: list[str] = []
    seen: set[str] = set()
    for title in titles:
        low = title.lower()
        if any(word in low for word in SKIP_WORDS):
            continue
        m = _LCCN_RE.search(title)
        if m:
            if m.group(1) in seen:
                continue
            seen.add(m.group(1))
            with_lccn.append(title)
        else:
            without.append(title)
    return with_lccn + without


def fetch_imageinfo(session: Any, titles: list[str], width: int) -> list[FileInfo]:
    infos: list[FileInfo] = []
    for start in range(0, len(titles), 50):
        batch = titles[start : start + 50]
        data = api_get(
            session,
            {
                "action": "query",
                "prop": "imageinfo",
                "titles": "|".join(batch),
                "iiprop": "url|size|mime|extmetadata",
                "iiurlwidth": str(width),
            },
        )
        for page in data.get("query", {}).get("pages", {}).values():
            ii = (page.get("imageinfo") or [None])[0]
            if not ii or not str(ii.get("mime", "")).startswith("image/"):
                continue
            if max(int(ii["width"]), int(ii["height"])) < MIN_LONG_SIDE:
                continue
            m = _LCCN_RE.search(page["title"])
            infos.append(
                FileInfo(
                    title=page["title"],
                    url=ii.get("thumburl") or ii["url"],
                    width=int(ii["width"]),
                    height=int(ii["height"]),
                    license=ii.get("extmetadata", {}).get("LicenseShortName", {}).get("value", ""),
                    lccn=m.group(1) if m else None,
                )
            )
    return infos


def download(session: Any, info: FileInfo, out_dir: str | Path, retries: int = 3) -> Path | None:
    path = Path(out_dir) / info.filename
    if path.is_file() and path.stat().st_size > 0:
        return path
    for attempt in range(retries):
        try:
            r = session.get(info.url, headers={"User-Agent": USER_AGENT}, timeout=120)
            if r.status_code == 200 and r.content:
                path.write_bytes(r.content)
                return path
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2**attempt)
    return None


def fetch_category(
    session: Any,
    category: str,
    out_dir: str | Path,
    width: int = 1024,
    limit: int | None = None,
    sample: int | None = None,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
) -> list[dict]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda _msg: None)

    titles = select_titles(list(iter_category_members(session, category)))
    say(f"{len(titles)} candidate files in {category}")
    if sample is not None and sample < len(titles):
        titles = sorted(random.Random(seed).sample(titles, sample), key=titles.index)
    if limit is not None:
        titles = titles[:limit]

    infos = fetch_imageinfo(session, titles, width)
    say(f"{len(infos)} files pass the size filter; downloading at {width}px")

    entries: list[dict] = []
    failed = 0
    for i, info in enumerate(infos, start=1):
        path = download(session, info, out_dir)
        if path is None:
            failed += 1
            continue
        entry = asdict(info)
        entry["filename"] = info.filename
        entry["sha1"] = hashlib.sha1(path.read_bytes()).hexdigest()
        entries.append(entry)
        if i % 25 == 0:
            say(f"  {i}/{len(infos)}")
        time.sleep(0.05)

    manifest = {
        "category": category,
        "width": width,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_files": len(entries),
        "n_failed": failed,
        "files": entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    say(f"done: {len(entries)} files, {failed} failed, manifest at {out_dir / 'manifest.json'}")
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kodachrome-fetch", description="Download public-domain Kodachrome scans from Wikimedia Commons."
    )
    parser.add_argument("--out", type=Path, default=Path("data/kodachrome"))
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="stop after N files")
    parser.add_argument("--sample", type=int, default=None, help="seeded random subset of N files")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-files", type=int, default=200, help="exit 1 if fewer files were obtained")
    args = parser.parse_args(argv)

    try:
        entries = fetch_category(
            make_session(),
            args.category,
            args.out,
            width=args.width,
            limit=args.limit,
            sample=args.sample,
            seed=args.seed,
            progress=print,
        )
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if len(entries) < args.min_files:
        print(
            f"error: obtained {len(entries)} files, fewer than {args.min_files}; "
            "not enough to train on. Check the category name or network.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_fetch.py -q` → all pass. The download tests call `time.sleep(2**attempt)` on failure; with `retries=1` that is one second, acceptable.

- [ ] **Step 5: Document and commit**

In `README.md` add a `## Training (Mac)` section after Commands:

````markdown
## Training (Mac)

### 1. Fetch the Kodachrome scans

```bash
.venv/bin/kodachrome-fetch            # about 1,000 files, ~200 MB, into data/kodachrome/
```

The scans are the Library of Congress FSA/OWI colour transparencies
(1939-1944), public domain, mirrored on Wikimedia Commons. loc.gov itself
blocks scripted downloads, so the tool uses the Commons API with a descriptive
User-Agent. Derivatives (cropped, restored, colorized) and files under 800 px
are skipped. `data/kodachrome/manifest.json` lists every file with its LoC
catalogue number, licence and SHA-1. The command is resumable.

To use your own Kodachrome scans instead, put them in any folder and pass it
to `kodachrome-train --target`.
````

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/train/fetch.py tests/test_fetch.py README.md
git commit -m "feat: kodachrome-fetch downloads FSA Kodachrome scans from Wikimedia Commons

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: Dataset preparation (`train/dataset.py`)

Implements spec 6.2 and 6.3.

**Files:**
- Create: `kodachrome/train/dataset.py`, `tests/test_dataset.py`

**Interfaces:**
- Consumes: `load_rgb`, `list_images`, `normalize_float`, `NormalizeParams`, `srgb_to_oklab`
- Produces:
  - `@dataclass SampleConfig(crop_frac=0.06, max_side=512, pixels_per_image=3000, l_min=0.02, l_max=0.98, max_pixels=400_000, seed=0)`
  - `@dataclass PixelPool(srgb: np.ndarray (M,3) float32, n_images: int)` with `.lab` property (Oklab, computed once and cached)
  - `crop_and_resize(rgb_u8, crop_frac, max_side) -> rgb_u8`
  - `prepare_image(rgb_u8, normalize_params, cfg) -> rgb_float32` (crop, resize, normalise)
  - `sample_pixels(rgb_float, n, l_min, l_max, rng) -> (k, 3) float32 sRGB`
  - `build_pool(paths, normalize_params, cfg, progress=None) -> PixelPool`
  - `dir_fingerprint(paths) -> str` (sha1 over sorted names and sizes)

- [ ] **Step 1: Write the failing tests**

`tests/test_dataset.py`:
```python
import numpy as np

from kodachrome.color import srgb_to_oklab
from kodachrome.imageio import save_jpeg
from kodachrome.normalize import NormalizeParams
from kodachrome.train.dataset import (
    PixelPool,
    SampleConfig,
    build_pool,
    crop_and_resize,
    dir_fingerprint,
    prepare_image,
    sample_pixels,
)


def test_crop_and_resize_geometry():
    img = np.zeros((500, 1000, 3), dtype=np.uint8)
    out = crop_and_resize(img, crop_frac=0.06, max_side=512)
    assert out.shape == (256, 512, 3)
    tall = crop_and_resize(np.zeros((1000, 500, 3), dtype=np.uint8), 0.0, 100)
    assert tall.shape == (100, 50, 3)


def test_crop_removes_border():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[10:90, 10:90] = 200  # 10% black border
    out = crop_and_resize(img, crop_frac=0.1, max_side=80)
    assert out.min() >= 190


def test_prepare_image_normalises():
    # mild cast so the white-balance gains stay inside the clamps (see test_normalize)
    img = np.full((100, 100, 3), (190, 170, 150), dtype=np.uint8)
    out = prepare_image(img, NormalizeParams(), SampleConfig(crop_frac=0.0, max_side=50))
    assert out.dtype == np.float32 and out.shape == (50, 50, 3)
    assert np.allclose(out[..., 0], out[..., 1], atol=1 / 255)  # grey-world made it neutral


def test_sample_pixels_respects_lightness_bounds_and_count():
    img = np.zeros((10, 20, 3), dtype=np.float32)
    img[:, 10:] = 0.5  # left half black (L≈0), right half mid grey
    rng = np.random.default_rng(0)
    px = sample_pixels(img, 1000, 0.02, 0.98, rng)
    assert px.shape == (100, 3)  # only the 100 mid-grey pixels qualify
    assert np.allclose(px, 0.5)
    px = sample_pixels(img, 30, 0.02, 0.98, rng)
    assert px.shape == (30, 3)


def test_build_pool_and_cap(tmp_path):
    rng = np.random.default_rng(0)
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.jpg"
        save_jpeg(rng.integers(30, 220, (40, 60, 3), dtype=np.uint8), p)
        paths.append(p)
    cfg = SampleConfig(crop_frac=0.0, max_side=60, pixels_per_image=500, max_pixels=800)
    pool = build_pool(paths, NormalizeParams(), cfg)
    assert isinstance(pool, PixelPool)
    assert pool.n_images == 3
    assert pool.srgb.shape == (800, 3) and pool.srgb.dtype == np.float32
    assert pool.lab.shape == (800, 3)
    assert np.allclose(pool.lab, srgb_to_oklab(pool.srgb), atol=1e-6)


def test_dir_fingerprint_changes_with_content(tmp_path):
    a = tmp_path / "a.jpg"
    a.write_bytes(b"12")
    f1 = dir_fingerprint([a])
    a.write_bytes(b"123")
    assert dir_fingerprint([a]) != f1
    assert len(f1) == 40
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_dataset.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement dataset.py**

```python
"""Turn folders of images into pixel pools the fitter can work on.

Both corpora (camera samples and Kodachrome scans) go through the same steps:

1. Crop ``crop_frac`` from every edge. Slide scans include the film rebate,
   mount shadow or scanner bed; camera frames may have vignetted corners.
2. Downscale so the long side is ``max_side``. Colour statistics do not need
   full resolution and this keeps the trainer fast.
3. Normalise with :func:`kodachrome.normalize.normalize_float`, using the
   **same** parameters the Pi will use. Sources get white balance; targets
   are passed ``NormalizeParams(white_balance=False)`` by the caller because
   the film's cast is part of the look.
4. Sample up to ``pixels_per_image`` pixels whose Oklab lightness is inside
   ``(l_min, l_max)``. Near-black pixels are borders and crushed shadows;
   near-white pixels are blown highlights or scanner glare. Neither says
   anything about how Kodachrome renders colour.

The pool is capped at ``max_pixels`` by a seeded subsample so runs are
reproducible and the transport step stays in seconds.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..color import srgb_to_oklab
from ..imageio import load_rgb
from ..normalize import NormalizeParams, normalize_float


@dataclass
class SampleConfig:
    crop_frac: float = 0.06
    max_side: int = 512
    pixels_per_image: int = 3000
    l_min: float = 0.02
    l_max: float = 0.98
    max_pixels: int = 400_000
    seed: int = 0


@dataclass
class PixelPool:
    srgb: np.ndarray
    n_images: int
    _lab: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def lab(self) -> np.ndarray:
        if self._lab is None:
            self._lab = srgb_to_oklab(self.srgb)
        return self._lab


def crop_and_resize(rgb_u8: np.ndarray, crop_frac: float, max_side: int) -> np.ndarray:
    h, w = rgb_u8.shape[:2]
    dy, dx = int(round(h * crop_frac)), int(round(w * crop_frac))
    cropped = rgb_u8[dy : h - dy or None, dx : w - dx or None]
    ch, cw = cropped.shape[:2]
    scale = max_side / max(ch, cw)
    if scale >= 1.0:
        return np.ascontiguousarray(cropped)
    size = (max(1, int(round(cw * scale))), max(1, int(round(ch * scale))))
    return cv2.resize(np.ascontiguousarray(cropped), size, interpolation=cv2.INTER_AREA)


def prepare_image(rgb_u8: np.ndarray, normalize_params: NormalizeParams, cfg: SampleConfig) -> np.ndarray:
    small = crop_and_resize(rgb_u8, cfg.crop_frac, cfg.max_side).astype(np.float32) / 255.0
    normalised, _ = normalize_float(small, normalize_params)
    return normalised


def sample_pixels(
    rgb: np.ndarray, n: int, l_min: float, l_max: float, rng: np.random.Generator
) -> np.ndarray:
    flat = rgb.reshape(-1, 3).astype(np.float32)
    lightness = srgb_to_oklab(flat)[:, 0]
    keep = np.flatnonzero((lightness > l_min) & (lightness < l_max))
    if len(keep) > n:
        keep = rng.choice(keep, n, replace=False)
    return flat[keep]


def build_pool(
    paths: Sequence[Path],
    normalize_params: NormalizeParams,
    cfg: SampleConfig,
    progress: Callable[[str], None] | None = None,
) -> PixelPool:
    rng = np.random.default_rng(cfg.seed)
    chunks: list[np.ndarray] = []
    for i, path in enumerate(paths, start=1):
        img = prepare_image(load_rgb(path), normalize_params, cfg)
        chunks.append(sample_pixels(img, cfg.pixels_per_image, cfg.l_min, cfg.l_max, rng))
        if progress and i % 100 == 0:
            progress(f"  {i}/{len(paths)} images sampled")
    pixels = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3), np.float32)
    if len(pixels) > cfg.max_pixels:
        pixels = pixels[rng.choice(len(pixels), cfg.max_pixels, replace=False)]
    return PixelPool(srgb=np.ascontiguousarray(pixels, dtype=np.float32), n_images=len(paths))


def dir_fingerprint(paths: Sequence[Path]) -> str:
    h = hashlib.sha1()
    for p in sorted(Path(x) for x in paths):
        h.update(f"{p.name}:{p.stat().st_size}\n".encode())
    return h.hexdigest()
```

- [ ] **Step 4: Run tests and commit**

Run: `.venv/bin/pytest tests/test_dataset.py -q` → all pass.

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/train/dataset.py tests/test_dataset.py
git commit -m "feat: dataset preparation into normalised pixel pools

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 13: Distribution transport (`train/transport.py`)

Implements spec 6.4 steps 1 and 2 and the sliced Wasserstein metric of 6.5.

**Files:**
- Create: `kodachrome/train/transport.py`, `tests/test_transport.py`

**Interfaces:**
- Consumes: `oklab_to_lch`
- Produces:
  - `hue_bin_index(lab, n_bins, chroma_floor) -> int64 array` (bin `n_bins` is achromatic)
  - `hue_histogram(lab, n_bins, chroma_floor, weights=None) -> float64 array (n_bins + 1,)` summing to 1
  - `hue_weights(src_lab, tgt_lab, n_bins=24, chroma_floor=0.03, w_min=0.2, w_max=5.0) -> float64 (len(tgt),)` with mean 1
  - `weighted_quantile_map(x, y, w) -> float64 array like x`
  - `random_rotation(rng) -> (3, 3) proper rotation`
  - `iterative_distribution_transfer(src_lab, tgt_lab, tgt_w=None, iterations=40, rng=None) -> float32 (len(src), 3)`
  - `sliced_wasserstein(a, b, n_proj=64, rng=None, b_weights=None, max_points=100_000) -> float`

- [ ] **Step 1: Write the failing tests**

`tests/test_transport.py`:
```python
import numpy as np
import pytest

from kodachrome.color import lch_to_oklab
from kodachrome.train.transport import (
    hue_bin_index,
    hue_histogram,
    hue_weights,
    iterative_distribution_transfer,
    random_rotation,
    sliced_wasserstein,
    weighted_quantile_map,
)


def _cloud(rng, n, hue_lo=0.0, hue_hi=2 * np.pi, chroma=0.12):
    hue = rng.uniform(hue_lo, hue_hi, n)
    lum = rng.uniform(0.3, 0.8, n)
    return lch_to_oklab(np.stack([lum, np.full(n, chroma), hue], axis=1))


def test_hue_bins_cover_circle_and_achromatic():
    rng = np.random.default_rng(0)
    lab = _cloud(rng, 5000)
    idx = hue_bin_index(lab, 24, 0.03)
    assert idx.min() == 0 and idx.max() == 23
    grey = np.array([[0.5, 0.0, 0.0], [0.5, 0.01, -0.01]])
    assert np.all(hue_bin_index(grey, 24, 0.03) == 24)


def test_hue_histogram_sums_to_one():
    lab = _cloud(np.random.default_rng(1), 1000)
    h = hue_histogram(lab, 12, 0.03)
    assert h.shape == (13,) and h.sum() == pytest.approx(1.0)


def test_hue_weights_make_target_histogram_match_source():
    rng = np.random.default_rng(2)
    src = _cloud(rng, 20000)  # uniform hue
    tgt = np.concatenate([_cloud(rng, 14000, 0, np.pi), _cloud(rng, 6000, np.pi, 2 * np.pi)])
    w = hue_weights(src, tgt, n_bins=24)
    assert w.mean() == pytest.approx(1.0)
    h_src = hue_histogram(src, 24, 0.03)
    h_tgt_w = hue_histogram(tgt, 24, 0.03, weights=w)
    assert np.abs(h_src - h_tgt_w).max() < 0.01


def test_weighted_quantile_map_matches_moments_and_keeps_order():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 20000)
    y = rng.normal(5, 2, 20000)
    mapped = weighted_quantile_map(x, y, np.ones_like(y))
    assert mapped.mean() == pytest.approx(5, abs=0.1)
    assert mapped.std() == pytest.approx(2, rel=0.05)
    order = np.argsort(x)
    assert np.all(np.diff(mapped[order]) >= 0)


def test_weighted_quantile_map_respects_weights():
    x = np.linspace(0, 1, 1001)
    y = np.concatenate([np.zeros(1000), np.full(1000, 10.0)])
    w = np.concatenate([np.full(1000, 3.0), np.ones(1000)])
    mapped = weighted_quantile_map(x, y, w)
    # three quarters of the weighted mass sits at 0, so the 60th percentile maps to 0
    assert mapped[600] < 0.5
    assert mapped[-1] > 9.5
    unweighted = weighted_quantile_map(x, y, np.ones_like(y))
    assert unweighted[600] > 9.5  # without weights the 60th percentile is already at 10


def test_random_rotation_is_proper():
    r = random_rotation(np.random.default_rng(4))
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-10)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_idt_moves_source_onto_target_distribution():
    rng = np.random.default_rng(5)
    src = rng.normal([0.5, 0.0, 0.0], [0.1, 0.05, 0.05], (20000, 3))
    tgt = rng.normal([0.6, 0.05, -0.05], [0.15, 0.08, 0.04], (20000, 3))
    before = sliced_wasserstein(src, tgt, rng=np.random.default_rng(0))
    moved = iterative_distribution_transfer(src, tgt, iterations=20, rng=np.random.default_rng(1))
    after = sliced_wasserstein(moved, tgt, rng=np.random.default_rng(0))
    assert moved.shape == src.shape and moved.dtype == np.float32
    assert after < before * 0.1
    assert np.allclose(moved.mean(axis=0), tgt.mean(axis=0), atol=0.01)
    assert np.allclose(moved.std(axis=0), tgt.std(axis=0), rtol=0.1)


def test_sliced_wasserstein_zero_for_identical_and_positive_for_shift():
    rng = np.random.default_rng(6)
    a = rng.normal(size=(5000, 3))
    assert sliced_wasserstein(a, a.copy(), rng=np.random.default_rng(0)) == pytest.approx(0.0, abs=1e-9)
    assert sliced_wasserstein(a, a + 1.0, rng=np.random.default_rng(0)) > 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_transport.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement transport.py**

```python
"""Unpaired colour-distribution transport in Oklab.

The problem
-----------
We have two unpaired sets of pixels: colours the camera produces, and
colours Kodachrome produced. Nothing links a particular camera pixel to a
particular film pixel. What we can ask is: "if the camera's cloud of colours
had to become the film's cloud of colours, moving each point as little as
possible, where would each camera colour go?" That is a transport problem;
its answer gives every source pixel a partner, and the LUT is then fitted to
those pairs (see ``lutfit.py``).

Content bias and hue reweighting
--------------------------------
The two corpora do not show the same things. The 1940s FSA photographs are
full of fields, khaki and weathered wood; a modern indoor sample set is not.
Raw distribution matching would happily turn a blue wall green because the
film corpus has more green. ``hue_weights`` prevents that: target pixels are
reweighted so the target's hue histogram matches the source's. The transport
can then only learn how Kodachrome renders each hue (its saturation,
lightness and tone curve, plus local hue shifts smaller than a bin), not how
much of each hue the 1940s contained. This deliberately caps learnable hue
rotation at about one bin (15 degrees for 24 bins).

Iterative distribution transfer (Pitié, Kokaram, Dahyot 2005)
-------------------------------------------------------------
Matching a 3D distribution directly is hard; matching a 1D one is a sort.
IDT repeats: pick a random 3D rotation, project both clouds onto its three
axes, match the source marginal to the (weighted) target marginal along
each axis by quantile mapping, rotate back. Each round moves the source
cloud closer to the target in every direction; a few dozen rounds converge.
Because each round is a monotone map along each axis, pixel identities are
preserved: row ``i`` of the output is where source pixel ``i`` went.

Sliced Wasserstein distance is the same idea used as a metric: average
1D Wasserstein-2 distance over random projections.
"""

from __future__ import annotations

import numpy as np

from ..color import oklab_to_lch


def hue_bin_index(lab: np.ndarray, n_bins: int, chroma_floor: float) -> np.ndarray:
    lch = oklab_to_lch(lab)
    hue = np.mod(lch[..., 2], 2 * np.pi)
    idx = np.minimum(np.floor(hue / (2 * np.pi) * n_bins).astype(np.int64), n_bins - 1)
    idx[lch[..., 1] < chroma_floor] = n_bins
    return idx


def hue_histogram(
    lab: np.ndarray, n_bins: int, chroma_floor: float, weights: np.ndarray | None = None
) -> np.ndarray:
    idx = hue_bin_index(lab, n_bins, chroma_floor)
    hist = np.bincount(idx, weights=weights, minlength=n_bins + 1).astype(np.float64)
    return hist / max(hist.sum(), 1e-12)


def hue_weights(
    src_lab: np.ndarray,
    tgt_lab: np.ndarray,
    n_bins: int = 24,
    chroma_floor: float = 0.03,
    w_min: float = 0.2,
    w_max: float = 5.0,
) -> np.ndarray:
    h_src = hue_histogram(src_lab, n_bins, chroma_floor)
    h_tgt = hue_histogram(tgt_lab, n_bins, chroma_floor)
    ratio = np.where(h_tgt > 0, h_src / np.maximum(h_tgt, 1e-12), 1.0)
    ratio = np.clip(ratio, w_min, w_max)
    w = ratio[hue_bin_index(tgt_lab, n_bins, chroma_floor)]
    return (w / w.mean()).astype(np.float64)


def weighted_quantile_map(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Monotone map sending the empirical distribution of ``x`` onto the
    ``w``-weighted empirical distribution of ``y``."""
    order = np.argsort(y, kind="stable")
    y_sorted = y[order]
    w_sorted = w[order]
    cum = np.cumsum(w_sorted)
    q_y = (cum - 0.5 * w_sorted) / cum[-1]
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[np.argsort(x, kind="stable")] = np.arange(len(x))
    q_x = (ranks + 0.5) / len(x)
    return np.interp(q_x, q_y, y_sorted)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def iterative_distribution_transfer(
    src_lab: np.ndarray,
    tgt_lab: np.ndarray,
    tgt_w: np.ndarray | None = None,
    iterations: int = 40,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng if rng is not None else np.random.default_rng(0)
    x = np.asarray(src_lab, dtype=np.float64).copy()
    y = np.asarray(tgt_lab, dtype=np.float64)
    w = np.ones(len(y)) if tgt_w is None else np.asarray(tgt_w, dtype=np.float64)
    for _ in range(iterations):
        rot = random_rotation(rng)
        xr = x @ rot
        yr = y @ rot
        for axis in range(3):
            xr[:, axis] = weighted_quantile_map(xr[:, axis], yr[:, axis], w)
        x = xr @ rot.T
    return x.astype(np.float32)


def sliced_wasserstein(
    a: np.ndarray,
    b: np.ndarray,
    n_proj: int = 64,
    rng: np.random.Generator | None = None,
    b_weights: np.ndarray | None = None,
    max_points: int = 100_000,
) -> float:
    rng = rng if rng is not None else np.random.default_rng(0)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b), max_points)
    a = a[rng.choice(len(a), n, replace=False)]
    if b_weights is not None:
        p = np.asarray(b_weights, dtype=np.float64)
        b = b[rng.choice(len(b), n, replace=True, p=p / p.sum())]
    else:
        b = b[rng.choice(len(b), n, replace=False)]
    dirs = rng.standard_normal((n_proj, a.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pa = np.sort(a @ dirs.T, axis=0)
    pb = np.sort(b @ dirs.T, axis=0)
    return float(np.sqrt(np.mean((pa - pb) ** 2)))
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_transport.py -q` → all pass. `test_idt_moves_source_onto_target_distribution` takes a few seconds; if it fails on the `after < before * 0.1` bound, print both numbers before changing anything: 20 iterations on Gaussians should get well under 5%.

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/train/transport.py tests/test_transport.py
git commit -m "feat: hue-reweighted iterative distribution transfer and sliced Wasserstein metric

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 14: Smooth LUT regression (`train/lutfit.py`)

Implements spec 6.4 step 3.

**Files:**
- Create: `kodachrome/train/lutfit.py`, `tests/test_lutfit.py`

**Interfaces:**
- Consumes: `LUT3D`
- Produces:
  - `node_index(r, g, b, n) -> int array` matching `table.reshape(-1, 3)` order
  - `trilinear_design_matrix(x_srgb, n) -> scipy.sparse.csr_matrix (M, n³)`
  - `second_difference_operator(n) -> csr_matrix (3·n²·(n-2), n³)`
  - `fit_lut(x_srgb, y_srgb, n=33, lambda_smooth=1e-3, lambda_identity=1e-4, rtol=1e-8, maxiter=5000) -> LUT3D`
  - `class FitConvergenceError(RuntimeError)`

- [ ] **Step 1: Write the failing tests**

`tests/test_lutfit.py`:
```python
import numpy as np
import pytest

from kodachrome.lut import LUT3D
from kodachrome.train.lutfit import (
    fit_lut,
    node_index,
    second_difference_operator,
    trilinear_design_matrix,
)


def test_node_index_matches_table_flattening():
    n = 5
    table = LUT3D.identity(n).table.reshape(-1, 3)
    for r, g, b in [(0, 0, 0), (4, 0, 0), (0, 4, 0), (0, 0, 4), (2, 3, 1)]:
        idx = node_index(np.array([r]), np.array([g]), np.array([b]), n)[0]
        assert np.allclose(table[idx], [r / 4, g / 4, b / 4])


def test_design_matrix_rows_sum_to_one_and_reproduce_identity():
    n = 9
    x = np.random.default_rng(0).random((500, 3), dtype=np.float32)
    a = trilinear_design_matrix(x, n)
    assert a.shape == (500, n**3)
    assert np.allclose(np.asarray(a.sum(axis=1)).ravel(), 1.0)
    assert a.nnz <= 500 * 8
    ident = LUT3D.identity(n).table.reshape(-1, 3)
    assert np.allclose(a @ ident, x, atol=1e-6)


def test_second_difference_operator_kills_linear_luts():
    n = 7
    d = second_difference_operator(n)
    assert d.shape == (3 * n * n * (n - 2), n**3)
    ident = LUT3D.identity(n).table.reshape(-1, 3)
    assert np.allclose(d @ ident, 0.0, atol=1e-6)
    bumpy = ident.copy()
    bumpy[n**3 // 2] += 0.1
    assert np.abs(d @ bumpy).max() > 0.1


def test_fit_recovers_per_channel_curve():
    rng = np.random.default_rng(1)
    x = rng.random((30000, 3), dtype=np.float32)
    y = np.clip(x**1.3, 0, 1).astype(np.float32)
    lut = fit_lut(x, y, n=17, lambda_smooth=1e-3, lambda_identity=1e-4)
    assert lut.size == 17
    held = rng.random((3000, 3), dtype=np.float32)
    err = np.abs(lut.apply_numpy(held) - np.clip(held**1.3, 0, 1)).mean()
    assert err < 0.01


def test_fit_stays_near_identity_where_there_is_no_data():
    rng = np.random.default_rng(2)
    # data only in the dark half of the cube
    x = (rng.random((20000, 3), dtype=np.float32) * 0.5).astype(np.float32)
    y = np.clip(x * 1.1, 0, 1).astype(np.float32)
    lut = fit_lut(x, y, n=9)
    bright = np.array([[0.95, 0.95, 0.95], [0.9, 0.2, 0.9]], dtype=np.float32)
    out = lut.apply_numpy(bright)
    assert np.all(np.isfinite(out))
    assert np.abs(out - bright).max() < 0.25  # extrapolated, but not wild


def test_fit_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        fit_lut(np.zeros((10, 3)), np.zeros((9, 3)), n=5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_lutfit.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement lutfit.py**

```python
"""Fit a smooth 3D LUT to (input colour, target colour) pairs.

After transport (``transport.py``) every source pixel ``x_i`` has a partner
``y_i``. A trilinear LUT is *linear in its node values*: the output for
``x_i`` is a fixed weighted sum of the eight surrounding nodes. So fitting
the LUT is ordinary least squares with a sparse design matrix ``A`` (eight
non-zeros per row), solved once per output channel:

    minimise  (1/M) ||A L - y||^2
            + lambda_smooth   / |D|  * ||D L||^2
            + lambda_identity / N^3  * ||L - I||^2

* The **data term** pulls nodes toward the transported partners.
* ``D`` stacks second-difference operators along the three grid axes. The
  **smoothness term** stops individual nodes chasing noisy partners, which
  would show up as banding or speckle in gradients.
* ``I`` is the identity LUT. The **identity term** is tiny but decides what
  happens to nodes no source pixel ever touches (a saturated magenta the
  camera never saw): they stay where they were instead of drifting.

Each term is divided by its own row count so the lambdas are relative
weights that do not change meaning when the sample count or grid size does.

The normal equations ``(A'A/M + ... ) L = A'y/M + ...`` are symmetric positive
definite (the identity term guarantees it), so they are solved with
conjugate gradients and a Jacobi preconditioner. That was chosen over a
direct sparse solve because the 3D grid's fill-in makes factorisation
memory-hungry at N=33 (35,937 unknowns), while CG converges in a few
thousand cheap sparse products.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg

from ..lut import LUT3D


class FitConvergenceError(RuntimeError):
    """Conjugate gradients did not converge."""


def node_index(r: np.ndarray, g: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    return (r * n + g) * n + b


def trilinear_design_matrix(x_srgb: np.ndarray, n: int) -> sp.csr_matrix:
    x = np.clip(np.asarray(x_srgb, dtype=np.float64), 0.0, 1.0) * (n - 1)
    i0 = np.minimum(np.floor(x).astype(np.int64), n - 2)
    f = x - i0
    m = len(x)
    rows = np.arange(m)
    row_parts, col_parts, val_parts = [], [], []
    for dr in (0, 1):
        wr = f[:, 0] if dr else 1.0 - f[:, 0]
        for dg in (0, 1):
            wg = f[:, 1] if dg else 1.0 - f[:, 1]
            for db in (0, 1):
                wb = f[:, 2] if db else 1.0 - f[:, 2]
                row_parts.append(rows)
                col_parts.append(node_index(i0[:, 0] + dr, i0[:, 1] + dg, i0[:, 2] + db, n))
                val_parts.append(wr * wg * wb)
    a = sp.csr_matrix(
        (np.concatenate(val_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
        shape=(m, n**3),
    )
    a.sum_duplicates()
    return a


def second_difference_operator(n: int) -> sp.csr_matrix:
    idx = np.arange(n**3).reshape(n, n, n)
    blocks = []
    for axis in range(3):
        moved = np.moveaxis(idx, axis, 0)
        prev = moved[:-2].ravel()
        centre = moved[1:-1].ravel()
        nxt = moved[2:].ravel()
        k = len(centre)
        rows = np.arange(k)
        blocks.append(
            sp.csr_matrix(
                (
                    np.concatenate([np.ones(k), -2.0 * np.ones(k), np.ones(k)]),
                    (np.concatenate([rows, rows, rows]), np.concatenate([prev, centre, nxt])),
                ),
                shape=(k, n**3),
            )
        )
    return sp.vstack(blocks).tocsr()


def fit_lut(
    x_srgb: np.ndarray,
    y_srgb: np.ndarray,
    n: int = 33,
    lambda_smooth: float = 1e-3,
    lambda_identity: float = 1e-4,
    rtol: float = 1e-8,
    maxiter: int = 5000,
) -> LUT3D:
    x = np.asarray(x_srgb, dtype=np.float64)
    y = np.clip(np.asarray(y_srgb, dtype=np.float64), 0.0, 1.0)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"x and y must both be (M, 3); got {x.shape} and {y.shape}")
    m = len(x)
    a = trilinear_design_matrix(x, n)
    d = second_difference_operator(n)
    ident = LUT3D.identity(n).table.reshape(-1, 3).astype(np.float64)

    lhs = (a.T @ a) / m
    lhs = lhs + (d.T @ d) * (lambda_smooth / d.shape[0])
    lhs = lhs + sp.identity(n**3, format="csr") * (lambda_identity / n**3)
    lhs = lhs.tocsr()
    precond = sp.diags(1.0 / np.maximum(lhs.diagonal(), 1e-12))

    table = np.empty((n**3, 3), dtype=np.float64)
    for c in range(3):
        rhs = (a.T @ y[:, c]) / m + (lambda_identity / n**3) * ident[:, c]
        sol, info = cg(lhs, rhs, x0=ident[:, c], rtol=rtol, maxiter=maxiter, M=precond)
        if info != 0:
            raise FitConvergenceError(
                f"CG did not converge for channel {c} (info={info}); "
                "raise --lambda-identity or --lambda-smooth slightly"
            )
        table[:, c] = sol
    return LUT3D(np.clip(table, 0.0, 1.0).reshape(n, n, n, 3).astype(np.float32))
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_lutfit.py -q` → all pass. `test_fit_recovers_per_channel_curve` should run in a few seconds at n=17. If `scipy.sparse.linalg.cg` rejects `rtol`, the installed SciPy is older than 1.12; upgrade it (`.venv/bin/pip install "scipy>=1.12"`), do not switch to the deprecated `tol` argument.

- [ ] **Step 5: Record the solver decision and commit**

Append to `docs/decisions.md`:

```markdown
## 2026-09-03: Conjugate gradients on the normal equations for the LUT fit

**Decided:** `fit_lut` forms the normal equations of the regularised least
squares problem and solves them per channel with `scipy.sparse.linalg.cg`
and a Jacobi preconditioner.
**Rejected:** `lsqr` on the stacked system (slower for this shape); direct
`spsolve` (3D grid fill-in is memory-hungry at 33^3).
**Why:** the identity term makes the system positive definite, so CG is
safe, and each iteration is a sparse product over about a million
non-zeros. Fits finish in seconds on the Mac.
```

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/train/lutfit.py tests/test_lutfit.py docs/decisions.md
git commit -m "feat: regularised sparse least-squares LUT fit

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 15: Training report (`train/report.py`)

Implements spec 6.5. The report is how a human judges the fit, so it must be readable, not just present.

**Files:**
- Create: `kodachrome/train/report.py`, `tests/test_report.py`

**Interfaces:**
- Consumes: `LUT3D`, `NormalizeParams`, `SampleConfig`, `prepare_image`, `load_rgb`, `PixelPool`, `hue_bin_index`, `sliced_wasserstein`, `srgb_to_oklab`, `oklab_to_srgb`, `oklab_to_lch`, `lch_to_oklab`, `srgb_to_linear`, `luminance`
- Produces:
  - `render_contact_sheet(source_paths, target_paths, lut, source_normalize, target_normalize, cfg, out_path, n=8, thumb=240, rng=None) -> Path`
  - `render_ramps(lut, out_path) -> Path`
  - `grey_axis_is_monotone(lut) -> bool`
  - `hue_bin_shifts(src_srgb, lut, n_bins=24, chroma_floor=0.03) -> list[dict]` with keys `bin`, `hue_deg`, `count`, `delta_L`, `chroma_ratio`, `delta_hue_deg`
  - `compute_metrics(source_pool, target_pool, target_weights, transported_lab, lut, n_bins, chroma_floor, rng=None) -> dict`
  - `write_report(out_dir, lut, source_pool, target_pool, target_weights, transported_lab, source_paths, target_paths, source_normalize, target_normalize, cfg, n_bins, chroma_floor) -> dict` (the metrics; writes `contact_sheet.png`, `ramps.png`, `metrics.json`)

- [ ] **Step 1: Write the failing tests**

`tests/test_report.py`:
```python
import json

import numpy as np
from PIL import Image

from kodachrome.imageio import save_jpeg
from kodachrome.lut import LUT3D
from kodachrome.normalize import NormalizeParams
from kodachrome.train.dataset import PixelPool, SampleConfig
from kodachrome.train.report import (
    compute_metrics,
    grey_axis_is_monotone,
    hue_bin_shifts,
    render_ramps,
    write_report,
)


def _images(dir_path, n, seed):
    dir_path.mkdir()
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        p = dir_path / f"{i}.jpg"
        save_jpeg(rng.integers(30, 220, (60, 80, 3), dtype=np.uint8), p)
        paths.append(p)
    return paths


def _darkening_lut(n=9):
    t = LUT3D.identity(n).table.copy()
    t[..., :] = t[..., :] ** 1.5
    return LUT3D(t)


def test_grey_axis_monotone_detects_inversion():
    assert grey_axis_is_monotone(LUT3D.identity(9))
    t = LUT3D.identity(9).table.copy()
    t[4, 4, 4] = 0.05  # dip in the middle of the grey axis
    assert not grey_axis_is_monotone(LUT3D(t))


def test_hue_bin_shifts_report_darkening():
    rng = np.random.default_rng(0)
    src = rng.random((5000, 3), dtype=np.float32) * 0.6 + 0.2
    shifts = hue_bin_shifts(src, _darkening_lut(), n_bins=12)
    assert len(shifts) == 13
    populated = [s for s in shifts if s["count"] > 0]
    assert all(s["delta_L"] < 0 for s in populated)
    assert {"bin", "hue_deg", "count", "delta_L", "chroma_ratio", "delta_hue_deg"} <= set(shifts[0])


def test_render_ramps_writes_png(tmp_path):
    out = render_ramps(LUT3D.identity(9), tmp_path / "ramps.png")
    with Image.open(out) as im:
        assert im.size[0] >= 512 and im.size[1] > 100


def test_compute_metrics_keys(tmp_path):
    rng = np.random.default_rng(1)
    src = PixelPool(rng.random((2000, 3), dtype=np.float32), n_images=2)
    tgt = PixelPool(rng.random((2000, 3), dtype=np.float32) * 0.8, n_images=2)
    w = np.ones(2000)
    metrics = compute_metrics(src, tgt, w, tgt.lab, LUT3D.identity(9), 12, 0.03, rng=np.random.default_rng(0))
    assert {"swd_before", "swd_after", "lut_fit_rms_deltaE", "grey_axis_monotone", "hue_bins"} <= set(metrics)
    assert metrics["grey_axis_monotone"] is True


def test_write_report_produces_files(tmp_path):
    src_paths = _images(tmp_path / "src", 3, 0)
    tgt_paths = _images(tmp_path / "tgt", 3, 1)
    rng = np.random.default_rng(2)
    src = PixelPool(rng.random((1000, 3), dtype=np.float32), n_images=3)
    tgt = PixelPool(rng.random((1000, 3), dtype=np.float32), n_images=3)
    out_dir = tmp_path / "report"
    metrics = write_report(
        out_dir,
        _darkening_lut(),
        src,
        tgt,
        np.ones(1000),
        tgt.lab,
        src_paths,
        tgt_paths,
        NormalizeParams(),
        NormalizeParams(white_balance=False),
        SampleConfig(crop_frac=0.0, max_side=80),
        n_bins=12,
        chroma_floor=0.03,
    )
    assert (out_dir / "contact_sheet.png").exists()
    assert (out_dir / "ramps.png").exists()
    saved = json.loads((out_dir / "metrics.json").read_text())
    assert saved["swd_before"] == metrics["swd_before"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_report.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement report.py**

```python
"""Human-readable evidence that a fitted LUT is sane and does what we wanted.

Three artefacts land in ``artifacts/report/``:

* ``contact_sheet.png``: for a handful of source images, the normalised
  input beside the graded output, and below them a strip of real Kodachrome
  scans. The question to ask: do the graded shots sit naturally in the strip?
* ``ramps.png``: a grey ramp and three hue sweeps, before and after the LUT.
  The grey ramp shows the learned tone curve (Kodachrome should deepen
  shadows and hold highlights). The hue sweeps show saturation and hue
  shifts per hue. Banding or a wobble here means the smoothness weight is
  too low.
* ``metrics.json``: sliced Wasserstein distance to the Kodachrome cloud
  before and after grading (lower after is the whole point), the RMS Oklab
  error between the LUT and the transported partners (how much of the
  transport the smooth LUT could express), per-hue-bin shifts in plain
  numbers, and a flag that the grey axis stays monotone.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..color import lch_to_oklab, luminance, oklab_to_lch, oklab_to_srgb, srgb_to_linear, srgb_to_oklab
from ..imageio import load_rgb
from ..lut import LUT3D
from ..normalize import NormalizeParams
from .dataset import PixelPool, SampleConfig, prepare_image
from .transport import hue_bin_index, sliced_wasserstein


def _to_u8(rgb_float: np.ndarray) -> np.ndarray:
    return np.clip(np.round(rgb_float * 255.0), 0, 255).astype(np.uint8)


def _fit_thumb(rgb_u8: np.ndarray, thumb: int) -> Image.Image:
    im = Image.fromarray(rgb_u8, "RGB")
    im.thumbnail((thumb, thumb))
    canvas = Image.new("RGB", (thumb, thumb), (24, 24, 24))
    canvas.paste(im, ((thumb - im.width) // 2, (thumb - im.height) // 2))
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
) -> Path:
    rng = rng if rng is not None else np.random.default_rng(0)
    pick_src = [source_paths[i] for i in rng.choice(len(source_paths), min(n, len(source_paths)), replace=False)]
    pick_tgt = [target_paths[i] for i in rng.choice(len(target_paths), min(n, len(target_paths)), replace=False)]
    filt = lut.to_pillow()
    pad, label_h = 8, 18
    cols = max(len(pick_src), len(pick_tgt), 1)
    width = pad + cols * (thumb + pad)
    height = 3 * (label_h + thumb + pad) + pad
    sheet = Image.new("RGB", (width, height), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)

    def row(y: int, label: str, images: list[Image.Image]) -> None:
        draw.text((pad, y), label, fill=(220, 220, 220))
        for j, im in enumerate(images):
            sheet.paste(im, (pad + j * (thumb + pad), y + label_h))

    normalised, graded = [], []
    for p in pick_src:
        norm_u8 = _to_u8(prepare_image(load_rgb(p), source_normalize, cfg))
        normalised.append(_fit_thumb(norm_u8, thumb))
        graded.append(_fit_thumb(lut.apply_pillow(norm_u8, filt), thumb))
    kodachrome = [_fit_thumb(_to_u8(prepare_image(load_rgb(p), target_normalize, cfg)), thumb) for p in pick_tgt]

    row(pad, "Source, normalised", normalised)
    row(pad + (label_h + thumb + pad), "Source, graded with the fitted LUT", graded)
    row(pad + 2 * (label_h + thumb + pad), "Real Kodachrome scans (exposure-normalised)", kodachrome)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def _grey_ramp(width: int) -> np.ndarray:
    return np.repeat(np.linspace(0, 1, width, dtype=np.float32)[None, :, None], 3, axis=2)


def _hue_sweep(width: int, lightness: float, chroma: float) -> np.ndarray:
    hue = np.linspace(-np.pi, np.pi, width, dtype=np.float32)
    lch = np.stack([np.full(width, lightness, np.float32), np.full(width, chroma, np.float32), hue], axis=1)
    return np.clip(oklab_to_srgb(lch_to_oklab(lch)), 0, 1)[None, :, :]


def render_ramps(lut: LUT3D, out_path: str | Path, width: int = 768, band: int = 36) -> Path:
    strips = [("grey ramp", _grey_ramp(width))] + [
        (f"hue sweep L={lum:.1f} C=0.12", _hue_sweep(width, lum, 0.12)) for lum in (0.4, 0.6, 0.8)
    ]
    label_h, pad = 16, 6
    height = len(strips) * (label_h + 2 * band + pad) + pad
    img = Image.new("RGB", (width, height), (16, 16, 16))
    draw = ImageDraw.Draw(img)
    y = pad
    for label, line in strips:
        draw.text((4, y), f"{label}: before (top) / after (bottom)", fill=(220, 220, 220))
        y += label_h
        before = np.repeat(_to_u8(line), band, axis=0)
        after = np.repeat(_to_u8(lut.apply_numpy(line)), band, axis=0)
        img.paste(Image.fromarray(before, "RGB"), (0, y))
        img.paste(Image.fromarray(after, "RGB"), (0, y + band))
        y += 2 * band + pad
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def grey_axis_is_monotone(lut: LUT3D, tolerance: float = 1e-3) -> bool:
    greys = np.repeat(np.linspace(0, 1, 256, dtype=np.float32)[:, None], 3, axis=1)
    lum = luminance(srgb_to_linear(lut.apply_numpy(greys)))
    return bool(np.all(np.diff(lum) >= -tolerance))


def hue_bin_shifts(
    src_srgb: np.ndarray, lut: LUT3D, n_bins: int = 24, chroma_floor: float = 0.03
) -> list[dict]:
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
            out.append({"bin": b, "hue_deg": centre, "count": 0, "delta_L": 0.0, "chroma_ratio": 1.0, "delta_hue_deg": 0.0})
            continue
        chroma_b = max(float(lch_b[sel, 1].mean()), 1e-6)
        out.append(
            {
                "bin": b,
                "hue_deg": centre,
                "count": count,
                "delta_L": round(float((lch_a[sel, 0] - lch_b[sel, 0]).mean()), 4),
                "chroma_ratio": round(float(lch_a[sel, 1].mean()) / chroma_b, 3),
                "delta_hue_deg": round(float(d_hue[sel].mean()), 2) if b < n_bins else 0.0,
            }
        )
    return out


def compute_metrics(
    source_pool: PixelPool,
    target_pool: PixelPool,
    target_weights: np.ndarray,
    transported_lab: np.ndarray,
    lut: LUT3D,
    n_bins: int,
    chroma_floor: float,
    rng: np.random.Generator | None = None,
) -> dict:
    rng = rng if rng is not None else np.random.default_rng(0)
    graded_lab = srgb_to_oklab(lut.apply_numpy(source_pool.srgb))
    fit_err = np.sqrt(np.mean(np.sum((graded_lab - transported_lab) ** 2, axis=1)))
    return {
        "swd_before": round(sliced_wasserstein(source_pool.lab, target_pool.lab, rng=np.random.default_rng(rng.integers(1 << 30)), b_weights=target_weights), 5),
        "swd_after": round(sliced_wasserstein(graded_lab, target_pool.lab, rng=np.random.default_rng(rng.integers(1 << 30)), b_weights=target_weights), 5),
        "lut_fit_rms_deltaE": round(float(fit_err), 5),
        "grey_axis_monotone": grey_axis_is_monotone(lut),
        "hue_bins": hue_bin_shifts(source_pool.srgb, lut, n_bins, chroma_floor),
    }


def write_report(
    out_dir: str | Path,
    lut: LUT3D,
    source_pool: PixelPool,
    target_pool: PixelPool,
    target_weights: np.ndarray,
    transported_lab: np.ndarray,
    source_paths: Sequence[Path],
    target_paths: Sequence[Path],
    source_normalize: NormalizeParams,
    target_normalize: NormalizeParams,
    cfg: SampleConfig,
    n_bins: int,
    chroma_floor: float,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_contact_sheet(source_paths, target_paths, lut, source_normalize, target_normalize, cfg, out_dir / "contact_sheet.png")
    render_ramps(lut, out_dir / "ramps.png")
    metrics = compute_metrics(source_pool, target_pool, target_weights, transported_lab, lut, n_bins, chroma_floor)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_report.py -q` → all pass. Lines longer than 100 characters will fail ruff; wrap the two `sliced_wasserstein(...)` calls in `compute_metrics` and the `render_contact_sheet(...)` call in `write_report` across lines.

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/train/report.py tests/test_report.py
git commit -m "feat: training report with contact sheet, ramps and metrics

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 16: Trainer orchestration (`train/fit.py`, `kodachrome-train`)

Implements spec 6.4 end to end and the `training` block of spec 5.6. Also updates the spec's file tree for the `transport.py`/`lutfit.py` split.

**Files:**
- Create: `kodachrome/train/fit.py`, `tests/test_fit.py`
- Modify: `README.md` (Training section, part 2), `docs/superpowers/specs/2026-09-03-kodachrome-film-design.md` (section 4 tree and 6.4 file names)

**Interfaces:**
- Consumes: everything from Tasks 12 to 15, `write_cube`, `write_params`, `GrainParams`
- Produces:
  - `@dataclass FitConfig(lut_size=33, iterations=40, hue_bins=24, chroma_floor=0.03, lambda_smooth=1e-3, lambda_identity=1e-4, strength=1.0, seed=0)`
  - `@dataclass FitResult(lut: LUT3D, transported_lab: np.ndarray, target_weights: np.ndarray)`
  - `fit(source_pool, target_pool, cfg, progress=None) -> FitResult`
  - `train(source_dir, target_dir, out_dir, cfg, sample_cfg, grain, proxy_source=False, progress=None) -> dict` (returns metrics; writes `.cube`, `params.json`, report)
  - `main(argv=None) -> int`

- [ ] **Step 1: Write the failing tests**

`tests/test_fit.py`:
```python
import json

import numpy as np
import pytest

from kodachrome.color import lch_to_oklab, oklab_to_lch, oklab_to_srgb, srgb_to_oklab
from kodachrome.imageio import save_jpeg
from kodachrome.pipeline import Artifacts
from kodachrome.train.dataset import PixelPool, SampleConfig
from kodachrome.train.fit import FitConfig, fit, main, train


def _known_transform(lab):
    """A convex-potential map: L -> L^1.15, chroma x1.25. Transport must recover it exactly."""
    lch = oklab_to_lch(lab)
    lch[..., 0] = lch[..., 0] ** 1.15
    lch[..., 1] = lch[..., 1] * 1.25
    return lch_to_oklab(lch)


def test_fit_recovers_a_convex_transform():
    rng = np.random.default_rng(0)
    src_srgb = (rng.random((30000, 3), dtype=np.float32) * 0.7 + 0.15).astype(np.float32)
    tgt_srgb = np.clip(oklab_to_srgb(_known_transform(srgb_to_oklab(rng.random((30000, 3), dtype=np.float32) * 0.7 + 0.15))), 0, 1)
    src, tgt = PixelPool(src_srgb, 1), PixelPool(tgt_srgb.astype(np.float32), 1)
    result = fit(src, tgt, FitConfig(lut_size=17, iterations=30, seed=0))
    held = (rng.random((3000, 3), dtype=np.float32) * 0.7 + 0.15).astype(np.float32)
    expected = _known_transform(srgb_to_oklab(held))
    got = srgb_to_oklab(result.lut.apply_numpy(held))
    delta_e = np.sqrt(np.sum((got - expected) ** 2, axis=1)).mean()
    assert delta_e < 0.02
    assert result.transported_lab.shape == (30000, 3)
    assert result.target_weights.shape == (30000,)


def test_strength_zero_gives_identity_lut():
    rng = np.random.default_rng(1)
    src = PixelPool(rng.random((5000, 3), dtype=np.float32), 1)
    tgt = PixelPool((rng.random((5000, 3), dtype=np.float32) * 0.5).astype(np.float32), 1)
    result = fit(src, tgt, FitConfig(lut_size=9, iterations=5, strength=0.0))
    x = rng.random((500, 3), dtype=np.float32)
    assert np.abs(result.lut.apply_numpy(x) - x).max() < 0.02


def _image_dir(dir_path, n, seed, transform=None):
    dir_path.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        base = np.linspace(0.2, 0.8, 64, dtype=np.float32)[None, :, None] * rng.uniform(0.7, 1.0, 3).astype(np.float32)
        img = np.repeat(base, 48, axis=0)
        if transform:
            img = transform(img)
        save_jpeg((np.clip(img, 0, 1) * 255).astype(np.uint8), dir_path / f"{i}.jpg")


def test_train_end_to_end_writes_loadable_artifacts(tmp_path):
    _image_dir(tmp_path / "src", 4, 0)
    _image_dir(tmp_path / "tgt", 4, 1, transform=lambda im: im**1.3)
    metrics = train(
        tmp_path / "src",
        tmp_path / "tgt",
        tmp_path / "artifacts",
        FitConfig(lut_size=9, iterations=5),
        SampleConfig(crop_frac=0.0, max_side=64, pixels_per_image=500),
        grain=None,
        proxy_source=True,
    )
    art = Artifacts.load(tmp_path / "artifacts")
    assert art.lut.size == 9
    assert art.training["proxy_source"] is True
    assert art.training["n_source_images"] == 4
    assert art.training["lut_size"] == 9
    assert art.training["metrics"]["swd_after"] == metrics["swd_after"]
    assert (tmp_path / "artifacts" / "report" / "contact_sheet.png").exists()
    assert (tmp_path / "artifacts" / "report" / "metrics.json").exists()


def test_main_rejects_identical_dirs_and_missing_dirs(tmp_path, capsys):
    _image_dir(tmp_path / "src", 2, 0)
    assert main(["--source", str(tmp_path / "src"), "--target", str(tmp_path / "src"), "--out", str(tmp_path / "o")]) == 1
    assert "same" in capsys.readouterr().err
    assert main(["--source", str(tmp_path / "missing"), "--target", str(tmp_path / "src"), "--out", str(tmp_path / "o")]) == 1


def test_main_runs_small(tmp_path):
    _image_dir(tmp_path / "src", 3, 0)
    _image_dir(tmp_path / "tgt", 3, 1, transform=lambda im: im**1.2)
    code = main(
        [
            "--source", str(tmp_path / "src"), "--target", str(tmp_path / "tgt"), "--out", str(tmp_path / "o"),
            "--lut-size", "9", "--iterations", "5", "--max-side", "64", "--pixels-per-image", "300", "--proxy-source",
        ]
    )
    assert code == 0
    data = json.loads((tmp_path / "o" / "params.json").read_text())
    assert data["training"]["strength"] == 1.0
    assert data["grain"]["strength"] == pytest.approx(0.025)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_fit.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement fit.py**

```python
"""``kodachrome-train``: fit the Kodachrome LUT from two folders of images.

Pipeline (spec section 6):

1. ``dataset.build_pool`` turns the source folder (camera shots, white
   balanced) and the target folder (Kodachrome scans, exposure-normalised
   only) into pixel pools.
2. ``transport.hue_weights`` reweights the target so its hue histogram
   matches the source's (content-bias control).
3. ``transport.iterative_distribution_transfer`` gives every source pixel a
   Kodachrome partner; ``strength`` blends between staying put (0) and the
   full transport (1).
4. ``lutfit.fit_lut`` fits a smooth LUT to the (source, partner) pairs.
5. ``report.write_report`` renders the evidence; ``params.json`` records
   every setting and metric so the artifact is reproducible.

Why ``--proxy-source`` is a flag
--------------------------------
The trainer cannot tell whether a folder of photos came from the U20CAM.
Pass the flag when training on stand-in photos so ``params.json`` says so
and users know to retrain with real camera shots.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..color import oklab_to_srgb
from ..grain import GrainParams
from ..imageio import list_images
from ..lut import LUT3D, write_cube
from ..normalize import NormalizeParams
from ..pipeline import write_params
from .dataset import PixelPool, SampleConfig, build_pool, dir_fingerprint
from .lutfit import fit_lut
from .report import write_report
from .transport import hue_weights, iterative_distribution_transfer

MIN_SOURCE_IMAGES = 30
MIN_TARGET_IMAGES = 200


@dataclass
class FitConfig:
    lut_size: int = 33
    iterations: int = 40
    hue_bins: int = 24
    chroma_floor: float = 0.03
    lambda_smooth: float = 1e-3
    lambda_identity: float = 1e-4
    strength: float = 1.0
    seed: int = 0


@dataclass
class FitResult:
    lut: LUT3D
    transported_lab: np.ndarray
    target_weights: np.ndarray


def fit(
    source_pool: PixelPool,
    target_pool: PixelPool,
    cfg: FitConfig,
    progress: Callable[[str], None] | None = None,
) -> FitResult:
    say = progress or (lambda _m: None)
    rng = np.random.default_rng(cfg.seed)
    src_lab, tgt_lab = source_pool.lab, target_pool.lab

    say("reweighting target hues to match the source hue histogram")
    weights = hue_weights(src_lab, tgt_lab, cfg.hue_bins, cfg.chroma_floor)

    say(f"iterative distribution transfer, {cfg.iterations} rounds")
    moved = iterative_distribution_transfer(src_lab, tgt_lab, weights, cfg.iterations, rng)
    partner_lab = src_lab + cfg.strength * (moved - src_lab)

    say(f"fitting {cfg.lut_size}^3 LUT by regularised least squares")
    lut = fit_lut(
        source_pool.srgb,
        np.clip(oklab_to_srgb(partner_lab), 0.0, 1.0),
        n=cfg.lut_size,
        lambda_smooth=cfg.lambda_smooth,
        lambda_identity=cfg.lambda_identity,
    )
    return FitResult(lut=lut, transported_lab=partner_lab.astype(np.float32), target_weights=weights)


def train(
    source_dir: str | Path,
    target_dir: str | Path,
    out_dir: str | Path,
    cfg: FitConfig,
    sample_cfg: SampleConfig,
    grain: GrainParams | None,
    proxy_source: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict:
    say = progress or (lambda _m: None)
    source_dir, target_dir, out_dir = Path(source_dir), Path(target_dir), Path(out_dir)
    source_paths, target_paths = list_images(source_dir), list_images(target_dir)
    if len(source_paths) < MIN_SOURCE_IMAGES:
        say(f"warning: only {len(source_paths)} source images; {MIN_SOURCE_IMAGES}+ recommended")
    if len(target_paths) < MIN_TARGET_IMAGES:
        say(f"warning: only {len(target_paths)} target images; {MIN_TARGET_IMAGES}+ recommended")

    source_normalize = NormalizeParams()
    target_normalize = NormalizeParams(white_balance=False)
    say(f"sampling {len(source_paths)} source images")
    source_pool = build_pool(source_paths, source_normalize, sample_cfg, say)
    say(f"sampling {len(target_paths)} target images")
    target_pool = build_pool(target_paths, target_normalize, sample_cfg, say)

    t0 = time.perf_counter()
    result = fit(source_pool, target_pool, cfg, say)
    fit_seconds = time.perf_counter() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    write_cube(result.lut, out_dir / "kodachrome.cube", title=f"kodachrome strength={cfg.strength}")
    say("writing report")
    metrics = write_report(
        out_dir / "report",
        result.lut,
        source_pool,
        target_pool,
        result.target_weights,
        result.transported_lab,
        source_paths,
        target_paths,
        source_normalize,
        target_normalize,
        sample_cfg,
        cfg.hue_bins,
        cfg.chroma_floor,
    )
    training = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_dir": str(target_dir),
        "n_target_images": len(target_paths),
        "n_target_pixels": int(len(target_pool.srgb)),
        "source_dir": str(source_dir),
        "source_dir_sha1": dir_fingerprint(source_paths),
        "n_source_images": len(source_paths),
        "n_source_pixels": int(len(source_pool.srgb)),
        "proxy_source": proxy_source,
        **asdict(cfg),
        "sample": asdict(sample_cfg),
        "fit_seconds": round(fit_seconds, 1),
        "metrics": {k: v for k, v in metrics.items() if k != "hue_bins"},
    }
    write_params(out_dir, source_normalize, grain or GrainParams(), training=training)
    say(f"wrote {out_dir / 'kodachrome.cube'} and params.json; report in {out_dir / 'report'}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kodachrome-train", description="Fit the Kodachrome LUT.")
    parser.add_argument("--source", type=Path, required=True, help="folder of camera photos")
    parser.add_argument("--target", type=Path, default=Path("data/kodachrome"), help="folder of Kodachrome scans")
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--lut-size", type=int, default=33)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--hue-bins", type=int, default=24)
    parser.add_argument("--lambda-smooth", type=float, default=1e-3)
    parser.add_argument("--lambda-identity", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grain-strength", type=float, default=0.025)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--pixels-per-image", type=int, default=3000)
    parser.add_argument("--max-pixels", type=int, default=400_000)
    parser.add_argument("--proxy-source", action="store_true", help="mark the source as stand-in photos, not U20CAM shots")
    args = parser.parse_args(argv)

    for label, path in (("source", args.source), ("target", args.target)):
        if not path.is_dir() or not list_images(path):
            print(f"error: {label} folder {path} does not exist or has no images", file=sys.stderr)
            return 1
    if args.source.resolve() == args.target.resolve():
        print("error: source and target are the same folder", file=sys.stderr)
        return 1
    if not 2 <= args.lut_size <= 65:
        print("error: --lut-size must be in 2..65", file=sys.stderr)
        return 1

    cfg = FitConfig(
        lut_size=args.lut_size,
        iterations=args.iterations,
        hue_bins=args.hue_bins,
        lambda_smooth=args.lambda_smooth,
        lambda_identity=args.lambda_identity,
        strength=args.strength,
        seed=args.seed,
    )
    sample_cfg = SampleConfig(
        max_side=args.max_side,
        pixels_per_image=args.pixels_per_image,
        max_pixels=args.max_pixels,
        seed=args.seed,
    )
    metrics = train(
        args.source,
        args.target,
        args.out,
        cfg,
        sample_cfg,
        GrainParams(strength=args.grain_strength),
        proxy_source=args.proxy_source,
        progress=print,
    )
    print(
        f"distance to Kodachrome: {metrics['swd_before']:.4f} before -> {metrics['swd_after']:.4f} after; "
        f"LUT fit RMS dE {metrics['lut_fit_rms_deltaE']:.4f}; grey axis monotone: {metrics['grey_axis_monotone']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_fit.py -q` → all pass. `test_fit_recovers_a_convex_transform` is the acceptance test for the whole method. If `delta_e` lands between 0.02 and 0.04, first try `iterations=50` in the test and check whether it improves (transport not converged) before touching the lambdas; record whatever changes in `docs/decisions.md`. Do not loosen the bound above 0.03 without writing down why.

- [ ] **Step 5: Update the spec tree and the README, then commit**

In the spec, section 4 tree: replace the `fit.py` line with three lines:
```
      transport.py          # hue reweighting, IDT, sliced Wasserstein
      lutfit.py             # trilinear design matrix, smoothness, CG solve
      fit.py                # orchestration and params writing  (CLI: kodachrome-train)
```
and in section 6.4 change the heading `### 6.4 \`fit.py\`` to `### 6.4 Fitting (\`transport.py\`, \`lutfit.py\`, \`fit.py\`)`. In section 6.2 add after the proxy paragraph: "The trainer cannot detect a proxy corpus; pass `--proxy-source` to mark it."

In `README.md`, Training section, add:

````markdown
### 2. Collect camera samples

Take 50 or more shots with the U20CAM across varied scenes: indoors and out,
sky, foliage, skin, neutral walls, mixed lighting. `kodachrome-capture` saves
originals under `~/Pictures/kodachrome/<date>/`; copy those `*_original.jpg`
files into one folder, for example `data/source/`.

### 3. Fit the LUT

```bash
.venv/bin/kodachrome-train --source data/source --target data/kodachrome
```

Runs in a minute or two. Writes `artifacts/kodachrome.cube`,
`artifacts/params.json` and `artifacts/report/`. Useful knobs:

| Flag | Default | Effect |
|---|---|---|
| `--strength` | 1.0 | 0 = no change, 1 = full Kodachrome; 0.7 for a lighter touch |
| `--lambda-smooth` | 1e-3 | raise if the ramps show banding or the fit looks noisy |
| `--lambda-identity` | 1e-4 | raise if colours the camera never produced go strange |
| `--grain-strength` | 0.025 | film grain, in luminance units at mid-grey |
| `--proxy-source` | off | mark the source folder as stand-in photos, not U20CAM shots |

### 4. Read the report

- `report/contact_sheet.png`: normalised source, graded source, and real
  scans. The graded row should sit naturally next to the scans.
- `report/ramps.png`: grey ramp and hue sweeps, before over after. Expect
  deeper shadows and stronger reds; do not accept banding or a grey ramp
  that goes backwards.
- `report/metrics.json`: `swd_after` must be well below `swd_before`
  (distance to the Kodachrome colour cloud). `lut_fit_rms_deltaE` around
  0.01 to 0.02 means the smooth LUT captured the transport; much higher
  means the transport asked for something a smooth LUT cannot do.
  `hue_bins` lists the learned lightness, chroma and hue change per hue.

Commit `artifacts/` when you are happy; the Pi just pulls the repo.
````

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/train/fit.py tests/test_fit.py README.md docs/superpowers/specs/2026-09-03-kodachrome-film-design.md
git commit -m "feat: kodachrome-train orchestration writing LUT, params and report

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 17: Fetch the data and train the default artifact

Implements spec 6.2 (proxy corpus) and replaces the identity placeholder from Task 6. This task uses the network and takes 10 to 20 minutes, mostly downloading.

**Files:**
- Modify: `artifacts/kodachrome.cube`, `artifacts/params.json`, `docs/decisions.md`, `README.md`
- Create (git-ignored): `data/kodachrome/`, `data/proxy-source/`

**Interfaces:**
- Consumes: `kodachrome-fetch`, `kodachrome-train`
- Produces: a trained default artifact with `training.proxy_source == true`

- [ ] **Step 1: Fetch the Kodachrome scans**

```bash
.venv/bin/kodachrome-fetch --out data/kodachrome
```
Expected: progress lines, then `done: N files` with N around 900 to 1,000 and `manifest.json` present. If N is below 200 the command exits 1; check network and the category name before retrying. The command is resumable.

- [ ] **Step 2: Choose a proxy source category that meets the spec's criteria**

Criteria (spec 6.2): public domain or CC0, modern digital cameras, varied everyday subjects, at least 60 images. Probe candidates with the Commons API; take the first that exists with at least 200 files:

```bash
for cat in "Category:Photographs by Lance Cheung" "Category:Photographs by Preston Keres" "Category:Photographs by the United States Department of Agriculture"; do
  .venv/bin/python - "$cat" <<'EOF'
import sys, requests
from kodachrome.train.fetch import API_URL, USER_AGENT
cat = sys.argv[1]
r = requests.get(API_URL, params={"action": "query", "prop": "categoryinfo", "titles": cat, "format": "json"},
                 headers={"User-Agent": USER_AGENT}, timeout=60).json()
page = next(iter(r["query"]["pages"].values()))
print(cat, "->", page.get("categoryinfo", "MISSING"))
EOF
done
```
Pick the first with `files >= 200`. If none qualifies, search Commons for another US federal photographer category (USDA, NPS, NASA "Johnson Space Center" everyday photos) and apply the same probe. Record the chosen category and its file count.

- [ ] **Step 3: Fetch a seeded sample of the proxy corpus**

```bash
.venv/bin/kodachrome-fetch --category "<chosen category>" --out data/proxy-source --sample 80 --seed 0 --min-files 60
```
Expected: 60 to 80 files. Open a dozen thumbnails (`open data/proxy-source`) and confirm they are photographs of everyday scenes, not diagrams or documents. If more than a handful are not photographs, delete them and re-check the count stays at or above 60.

- [ ] **Step 4: Train the default artifact**

```bash
.venv/bin/kodachrome-train --source data/proxy-source --target data/kodachrome --out artifacts --proxy-source
```
Expected: warnings about source count are acceptable if 60 to 80 images; a summary line with `swd_after` clearly below `swd_before` and `grey axis monotone: True`.

- [ ] **Step 5: Inspect the report and tune if needed**

```bash
open artifacts/report/contact_sheet.png artifacts/report/ramps.png
.venv/bin/python -c "import json; m=json.load(open('artifacts/report/metrics.json')); print({k:v for k,v in m.items() if k!='hue_bins'}); [print(b) for b in m['hue_bins'] if b['count']>0]"
```
Acceptance:
- grey ramp after is smooth and darker in the shadows than before, no banding;
- hue sweeps show no discontinuities;
- `grey_axis_monotone` is true;
- `lut_fit_rms_deltaE` below 0.03;
- graded row on the contact sheet reads as the same family as the scans.

If banding appears, rerun with `--lambda-smooth 3e-3`. If any hue sweep wobbles, rerun with `--lambda-smooth 1e-2`. If the look is too strong for taste, that is a `--strength` question for the user, not a fitting problem; keep 1.0 for the default. Record the final flags used.

- [ ] **Step 6: Run the whole test suite and try the artifact end to end**

```bash
.venv/bin/pytest -q
.venv/bin/kodachrome-process data/proxy-source /tmp/proxy-graded --artifacts artifacts
open /tmp/proxy-graded
```
Expected: all tests pass (the committed-artifact test now loads the trained LUT); regraded proxy images look like the contact sheet's graded row.

- [ ] **Step 7: Document provenance and commit**

Append to `docs/decisions.md`:

```markdown
## <today's date>: Default artifact trained on a proxy source corpus

**Decided:** the committed `artifacts/` were fitted with `--source` =
<chosen category> (<N> files, seed 0, sampled via `kodachrome-fetch --sample 80`)
and `--target` = <M> FSA Kodachrome scans, flags: <final flags>.
**Why:** the repository must work on a fresh Pi before anyone has taken
U20CAM shots. The proxy photos are public domain, modern digital and varied,
but they are not the U20CAM's rendering; `params.json` carries
`"proxy_source": true` and the README tells users to retrain with 50+ of
their own shots.
```

In `README.md`, add under "How it works":

```markdown
The committed `artifacts/` were trained against a stand-in source corpus
(see `docs/decisions.md`), so `params.json` says `"proxy_source": true`.
Retrain with your own U20CAM shots for the best match; see Training.
```

```bash
git add artifacts docs/decisions.md README.md
git commit -m "feat: default Kodachrome LUT trained on FSA scans with a proxy source corpus

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 18: Pi deployment documentation and on-hardware measurement

Implements spec 7.4 and the performance numbers promised in 7.2. Steps 1 and 2 run on the Mac; steps 3 to 5 need the Pi and the camera, so they are written for the user (or an agent with SSH access) to run.

**Files:**
- Modify: `README.md` (Pi setup, measured performance, known limitations)

- [ ] **Step 1: Write the Pi setup section**

Replace "Written in a later step." in `README.md` with:

````markdown
On Raspberry Pi OS (Bookworm or newer, 64-bit):

```bash
sudo apt update
sudo apt install -y git python3-venv python3-opencv python3-numpy python3-pil
git clone <this repo> ~/kodachrome-film && cd ~/kodachrome-film
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -e .
.venv/bin/kodachrome-capture --fake     # smoke test without the camera
.venv/bin/kodachrome-capture            # real camera
```

Why `--system-site-packages`: OpenCV comes from apt so the preview window
works. Pip's `opencv-python-headless` wheel has no GUI support and would
silently force headless mode. Do not `pip install opencv-python` on the Pi.

Plug the U20CAM into a USB port directly on the Pi 400, not through an
unpowered hub; the vendor FAQ blames hubs for dropped frames. Check it is
seen with `ls /dev/video*` (the first node is usually the capture stream).
````

- [ ] **Step 2: Write the known-limitations section**

Append to `README.md`:

```markdown
## Known limitations

- The look is learned from Library of Congress scans of 1940s Kodachrome,
  including their scanner and colour management. Later Kodachrome (K-14,
  1974 onwards) looks different. Point `--target` at your own scans to
  change the reference.
- Unpaired matching cannot learn hue rotations larger than about one hue
  bin (15 degrees at 24 bins); the hue reweighting that prevents content
  bias also damps them. Saturation, lightness and tone curve per hue are
  learned fully.
- White balance is grey-world with clamped gains. A scene that is
  legitimately all one colour will be partially neutralised.
- No lens, halation or vignette modelling; the camera's 121-degree lens
  distortion is left as is.
```

- [ ] **Step 3 (on the Pi): install and run the smoke test**

Follow the Pi setup section exactly. Expected: `kodachrome-capture --fake` opens a window with the synthetic frame (or reports headless mode over SSH), SPACE writes two JPEGs under `~/Pictures/kodachrome/<date>/`.

- [ ] **Step 4 (on the Pi): capture five real shots and read the timings**

```bash
.venv/bin/kodachrome-capture
# press SPACE five times, then Q
python3 -c "import json,glob; rows=[json.loads(l) for f in glob.glob('$HOME/Pictures/kodachrome/*/captures.jsonl') for l in open(f)]; print([r['processing_ms'] for r in rows[-5:]])"
```
Expected: five numbers, each under 1000 ms (spec target), typically 300 to 600 ms.

- [ ] **Step 5: Record the numbers and commit**

Add to `README.md` under "How it works":

```markdown
Measured on a Raspberry Pi 400 at 1920x1080: about <median> ms per capture
(five shots, <min> to <max> ms), of which the LUT and grain dominate. Preview
runs at camera frame rate at 640x360.
```

```bash
git add README.md
git commit -m "docs: Pi setup, measured performance, known limitations

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

If the Pi is not available when the rest of the plan is done, commit steps 1 and 2 with the sentence "Performance on the Pi 400 has not been measured yet" in place of the numbers, and leave steps 3 to 5 unchecked for the user.

---

## Plan self-review notes

- **Spec coverage:** sections 3 and 6.1 → Task 11; 4 → Tasks 1, 7; 5.1 to 5.6 → Tasks 2 to 6; 6.2, 6.3 → Tasks 12, 17; 6.4 → Tasks 13, 14, 16; 6.5 → Task 15; 7.1 → Task 8; 7.2 → Task 10; 7.3 → Task 9; 7.4 → Task 18; 8 (error table) → Tasks 6, 8, 9, 10, 11, 16; 9 (tests) → each task's Step 1; 10 (docs) → every task's final step; 11 and 12 → Task 18 limitations.
- **Deviation from spec, recorded:** `fit.py` split into `transport.py`, `lutfit.py`, `fit.py` (Task 16 updates the spec tree); `--proxy-source` is an explicit flag (Task 16 updates spec 6.2).
- **Type consistency checked:** `Gains.combined`, `normalize_u8` return order `(image, gains)`, `LUT3D.apply_pillow(rgb_u8, filt)`, `PixelPool.lab`, `FitResult` field names, `write_report` argument order all match between defining and consuming tasks.

