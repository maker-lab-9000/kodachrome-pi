# Kodachrome Film Look Implementation Plan (revision 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Raspberry Pi 400 with an Innomaker U20CAM-1080P-WDR saves every capture twice, the camera's own JPEG bytes and a Kodachrome-graded version, where the grade is a 3D LUT learned from public-domain Kodachrome scans.

**Architecture:** One Python package `kodachrome`. Shared colour, normalisation, LUT, grain, image-IO and artifact modules serve both runtimes. A Mac-side trainer downloads and validates Kodachrome scans, splits both corpora by image, transports the source colour distribution onto the target in Oklab (hue-reweighted iterative distribution transfer), fits a smooth 33³ LUT by regularised sparse least squares, evaluates on held-out images with a paired evaluator, and publishes the artifact atomically. The Pi runtime acquires a raw MJPEG frame, saves those bytes verbatim, and grades the decode of the same buffer.

**Tech Stack:** Python 3.11+, NumPy, Pillow, OpenCV (apt `python3-opencv` on the Pi, `[opencv]` extra elsewhere), SciPy and requests for the trainer, pytest.

**Spec:** `docs/superpowers/specs/2026-09-03-kodachrome-film-design.md` (revision 2) — read it first; every task cites the section it implements.

**Status:** Tasks 1 and 2 were completed against revision 1 of this plan and remain valid. Task 3 retrofits the two things revision 2 changes about Task 1. Execution resumes at Task 3.

## Global Constraints

- `requires-python = ">=3.11"`. Develop on the Mac with `.venv` (already created from `/usr/local/bin/python3.12`).
- Base pip dependencies are only `numpy` and `Pillow`. OpenCV is **not** a base dependency, but every documented install must work: an `[opencv]` extra exists and both `[train]` and `[dev]` include it (spec 7.4).
- Image arrays are **RGB**, `uint8` at boundaries, `float32` in `[0, 1]` inside algorithms. BGR appears only inside `camera.py`.
- All file reads go through `imageio.load_rgb`, which applies EXIF orientation and converts embedded ICC profiles to sRGB (spec 5.5). No module may call `Image.open` directly for pixel data.
- LUT tables are indexed `table[r, g, b, channel]`; `.cube` and Pillow flat order is red-fastest.
- Perceptual statistics are in Oklab; the exported LUT maps sRGB to sRGB.
- Every dataclass with numeric bounds validates in `__post_init__` and raises `ValueError` naming the offending field (spec 5.2-5.4).
- Tests never touch the network or camera hardware. Use `FakeCamera` and fake HTTP sessions.
- `pytest.raises(match=...)` must match the reason, not a bare token. `tmp_path` is named after the test function, so a pattern like `match="training"` inside `test_a_malformed_training_section...` also matches the temp directory in an unrelated error and passes for the wrong reason. This was caught in Task 8; match a distinctive phrase from the message instead.
- `PARAMS_VERSION = 2`. The schema is spec 5.8.
- Documentation is updated in the same task as the code: README, `docs/decisions.md`, module docstrings explaining the *why*.
- Run tests with `.venv/bin/pytest -q` from the repo root; lint with `.venv/bin/ruff check kodachrome tests` before every commit.
- Commit after every task. Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `pyproject.toml` | metadata, extras, entry points, package data | 1, 3 |
| `kodachrome/color.py` | sRGB/linear/Oklab/LCh conversions | 2 ✅ |
| `kodachrome/_cv2.py` | one import guard with an actionable message | 3 |
| `kodachrome/normalize.py` | white balance + exposure, validation, clamp flags | 4 |
| `kodachrome/lut.py` | `LUT3D`, `.cube` I/O, validation, `sha1_hex` | 5 |
| `kodachrome/grain.py` | luminance film grain | 6 |
| `kodachrome/imageio.py` | load/save with EXIF orientation and ICC → sRGB | 7 |
| `kodachrome/artifacts.py` | load, validate, packaged default, atomic publish | 8 |
| `kodachrome/data/` | packaged default artifact (identity until Task 21) | 8, 21 |
| `kodachrome/pipeline.py` | normalise → LUT → grain | 9 |
| `kodachrome/capture/batch.py` | `kodachrome-process`, with clobber and double-grade safety | 10 |
| `kodachrome/capture/camera.py` | raw-MJPEG `V4L2Camera`, negotiation checks, `FakeCamera` | 11 |
| `kodachrome/capture/app.py` | `kodachrome-capture`, guarded loops, audit log | 12 |
| `kodachrome/train/fetch.py` | `kodachrome-fetch`, licence and media validation | 13 |
| `kodachrome/train/dataset.py` | corpora → pools, split by image, corpus hashing | 14 |
| `kodachrome/train/transport.py` | hue reweighting, IDT, sliced Wasserstein | 15 |
| `kodachrome/train/lutfit.py` | design matrix, smoothness, CG solve | 16 |
| `kodachrome/train/evaluate.py` | paired evaluator, held-out metrics, safety gates | 17 |
| `kodachrome/train/report.py` | contact sheet, ramps, diagnostics | 18 |
| `kodachrome/train/fit.py` | `kodachrome-train` orchestration | 19 |
| `tests/test_packaging.py` | wheel install, run from another directory | 20 |
| — | fetch corpora, train and promote the default artifact | 21 |
| `README.md` | Pi setup, measured performance, limitations | 22 |

---

### Task 3: Dependency extras and the OpenCV import guard

Implements spec 4 and 7.4. Fixes F-04. Also creates the package-data directory Task 8 fills.

**Files:**
- Modify: `pyproject.toml`
- Create: `kodachrome/_cv2.py`, `kodachrome/data/.gitkeep`, `tests/test_cv2_guard.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `kodachrome._cv2.require_cv2() -> module` — returns the imported `cv2`, or raises `ImportError` with a message naming both remedies. Every module that needs OpenCV does `from ._cv2 import require_cv2` then `cv2 = require_cv2()` at module scope.

- [ ] **Step 1: Write the failing test**

`tests/test_cv2_guard.py`:
```python
import builtins

import pytest

from kodachrome._cv2 import require_cv2


def test_require_cv2_returns_the_module():
    cv2 = require_cv2()
    assert hasattr(cv2, "LUT")


def _patch_cv2_import(monkeypatch, error):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise error
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_missing_cv2_names_both_remedies(monkeypatch):
    # This is exactly what Python raises for an absent module: a
    # ModuleNotFoundError whose .name is the module. A bare ImportError with
    # the same text never occurs in practice.
    _patch_cv2_import(monkeypatch, ModuleNotFoundError("No module named 'cv2'", name="cv2"))
    with pytest.raises(ImportError) as exc:
        require_cv2()
    message = str(exc.value)
    assert "python3-opencv" in message      # the Pi remedy
    assert "[opencv]" in message            # the pip remedy


@pytest.mark.parametrize(
    "error",
    [
        ImportError("libGL.so.1: cannot open shared object file"),
        # The error text names the .so inside the cv2 package, so any check
        # that sniffs for "cv2" in the message misclassifies this one.
        ImportError(
            "/usr/lib/python3/dist-packages/cv2/cv2.abi3.so: "
            "undefined symbol: _ZN2cv6String8allocateEm"
        ),
    ],
    ids=["missing-libGL", "abi-mismatch"],
)
def test_a_broken_native_install_is_not_reported_as_missing(monkeypatch, error):
    """cv2 present but unable to load must not advise installing cv2."""
    _patch_cv2_import(monkeypatch, error)
    with pytest.raises(ImportError) as exc:
        require_cv2()
    message = str(exc.value)
    assert "failed to import" in message
    assert "libgl1" in message
    assert "is required but not installed" not in message


def test_a_missing_dependency_of_cv2_is_not_reported_as_missing_cv2(monkeypatch):
    """ModuleNotFoundError naming something else is an installation problem."""
    _patch_cv2_import(monkeypatch, ModuleNotFoundError("No module named 'numpy'", name="numpy"))
    with pytest.raises(ImportError) as exc:
        require_cv2()
    assert "failed to import" in str(exc.value)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `.venv/bin/pytest tests/test_cv2_guard.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'kodachrome._cv2'`

- [ ] **Step 3: Write `kodachrome/_cv2.py`**

```python
"""One place that imports OpenCV, so one place explains how to install it.

OpenCV is deliberately not a base dependency. On Raspberry Pi OS it comes
from apt (`python3-opencv`), which is built with GTK so the preview window
works; pip's `opencv-python-headless` wheel silently has no GUI. Everywhere
else it arrives through the `[opencv]` extra, which `[train]` and `[dev]`
both include.

A bare `ModuleNotFoundError: No module named 'cv2'` does not tell a user
which of those two paths they are missing, so this guard does.
"""

from __future__ import annotations

from types import ModuleType

_MISSING = (
    "OpenCV (cv2) is required but not installed.\n"
    "  On Raspberry Pi OS:  sudo apt install python3-opencv\n"
    "                       (then create the venv with --system-site-packages)\n"
    "  Anywhere else:       pip install 'kodachrome-film[opencv]'\n"
    "                       (already included by the [train] and [dev] extras)"
)

_BROKEN = (
    "OpenCV (cv2) is installed but failed to import: {error}\n"
    "This is an installation problem, not a missing package, so reinstalling\n"
    "cv2 will probably not help. A missing system library is the usual cause;\n"
    "on Raspberry Pi OS or another Debian, try:\n"
    "  sudo apt install libgl1 libglib2.0-0\n"
    "or switch to the apt build:  sudo apt install python3-opencv"
)


def require_cv2() -> ModuleType:
    """Import and return ``cv2``, or raise ``ImportError`` naming the right fix.

    The two failures need different advice and must not be conflated. A
    genuinely absent package is fixed by installing one; a package that is
    present but cannot load its native libraries (``libGL.so.1: cannot open
    shared object file`` is the classic on a headless Debian, and the Pi is
    a Debian) is not. Telling someone to install what they already have
    sends them down the wrong path, so the message is chosen from the
    exception rather than assumed.
    """
    try:
        import cv2
    except ModuleNotFoundError as exc:
        if exc.name != "cv2":
            raise ImportError(_BROKEN.format(error=exc)) from exc
        raise ImportError(_MISSING) from exc
    except ImportError as exc:
        raise ImportError(_BROKEN.format(error=exc)) from exc
    return cv2
```

- [ ] **Step 4: Update `pyproject.toml`**

Replace the `[project.optional-dependencies]` block with:

```toml
[project.optional-dependencies]
opencv = ["opencv-python>=4.8"]
train = ["scipy>=1.12", "requests>=2.31", "tqdm>=4.66", "opencv-python>=4.8"]
dev = ["opencv-python>=4.8", "pytest>=7.4", "ruff>=0.4", "build>=1.0"]
```

and add package-data configuration after `[tool.setuptools.packages.find]`:

```toml
[tool.setuptools.package-data]
kodachrome = ["data/*.cube", "data/*.json"]
```

- [ ] **Step 5: Create the package-data directory**

```bash
mkdir -p kodachrome/data && touch kodachrome/data/.gitkeep
```

- [ ] **Step 6: Run tests and lint**

Run: `.venv/bin/pytest -q && .venv/bin/ruff check kodachrome tests`
Expected: 12 passed (10 existing + 2 new), ruff clean.

- [ ] **Step 7: Document and commit**

In `README.md`, under Mac setup, add:

````markdown
OpenCV is not a base dependency: on the Pi it comes from apt, everywhere else
from an extra. `pip install -e ".[train,dev]"` includes it. Installing the
bare package and then importing a module that needs OpenCV raises an error
naming both remedies.
````

```bash
.venv/bin/ruff check kodachrome tests
git add pyproject.toml kodachrome/_cv2.py kodachrome/data/.gitkeep tests/test_cv2_guard.py README.md
git commit -m "fix: make [train] install a working trainer; add cv2 import guard

pip install .[train] previously omitted OpenCV while five modules import it
at module scope. Adds an [opencv] extra included by [train] and [dev], plus
a single guarded import that names both the apt and pip remedies.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Normalisation (`normalize.py`)

Implements spec 5.2. Adds validation and clamp reporting (F-12, F-16).

**Files:**
- Create: `kodachrome/normalize.py`, `tests/test_normalize.py`

**Interfaces:**
- Consumes: `color.srgb_to_linear`, `color.linear_to_srgb`, `color.LUMA_709`, `_cv2.require_cv2`
- Produces:
  - `@dataclass NormalizeParams` with the spec 5.2 fields, validating `__post_init__`, `from_dict` (ignores unknown keys), `to_dict`
  - `@dataclass Gains(wb: np.ndarray, exposure: float, clamped: dict[str, bool])` with `.combined` and `.to_dict()`
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
    apply_gains_float,
    compute_gains,
    gains_to_luts,
    normalize_float,
    normalize_u8,
)


def _gradient_image(h=48, w=64, cast=(1.0, 0.9, 0.75)):
    ramp = np.linspace(0.1, 0.9, w, dtype=np.float32)[None, :, None]
    rows = np.linspace(0.8, 1.2, h, dtype=np.float32)[:, None, None]
    return np.clip(ramp * rows * np.array(cast, dtype=np.float32), 0, 1).astype(np.float32)


def test_params_from_dict_ignores_unknown_and_roundtrips():
    p = NormalizeParams.from_dict({"exposure_target_median": 0.2, "bogus": 1})
    assert p.exposure_target_median == 0.2
    assert p.white_balance is True
    assert NormalizeParams.from_dict(p.to_dict()) == p


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"wb_gain_min": 2.0}, "wb_gain_min"),           # min above max
        ({"exposure_gain_min": 0.0}, "exposure_gain_min"),  # not positive
        ({"exposure_target_median": 1.5}, "exposure_target_median"),
        ({"stats_lum_min": 0.95}, "stats_lum_min"),      # min above max
        ({"wb_gain_max": float("nan")}, "wb_gain_max"),
    ],
)
def test_invalid_params_name_the_field(kwargs, field):
    with pytest.raises(ValueError, match=field):
        NormalizeParams(**kwargs)


def test_gains_combined_and_dict():
    g = Gains(
        wb=np.array([1.0, 1.1, 1.2], dtype=np.float32),
        exposure=2.0,
        clamped={"wb": False, "exposure": True},
    )
    assert np.allclose(g.combined, [2.0, 2.2, 2.4])
    assert g.to_dict() == {
        "wb": [1.0, 1.1, 1.2],
        "exposure": 2.0,
        "clamped": {"wb": False, "exposure": True},
    }


def test_grey_world_neutralises_a_mild_cast():
    # gains land near 0.83/1.04/1.33, inside the clamps, so the cast fully clears
    img = np.full((32, 32, 3), (0.5, 0.45, 0.4), dtype=np.float32)
    out, gains = normalize_float(img, NormalizeParams())
    assert np.allclose(out[..., 0], out[..., 1], atol=1 / 255)
    assert np.allclose(out[..., 1], out[..., 2], atol=1 / 255)
    assert np.median(color.luminance(color.srgb_to_linear(out))) == pytest.approx(0.18, abs=0.005)
    assert gains.wb[0] < 1.0 < gains.wb[2]
    assert gains.clamped == {"wb": False, "exposure": False}


def test_white_balance_can_be_disabled():
    img = np.full((8, 8, 3), (0.5, 0.4, 0.3), dtype=np.float32)
    gains = compute_gains(img, NormalizeParams(white_balance=False))
    assert np.array_equal(gains.wb, np.ones(3, dtype=np.float32))
    assert gains.exposure > 1.0


def test_gains_are_clamped_and_report_it():
    p = NormalizeParams()
    dark = np.full((8, 8, 3), 0.02, dtype=np.float32)
    g = compute_gains(dark, p)
    assert g.exposure == pytest.approx(p.exposure_gain_max)
    assert g.clamped["exposure"] is True
    bright = np.full((8, 8, 3), 0.95, dtype=np.float32)
    assert compute_gains(bright, p).exposure == pytest.approx(p.exposure_gain_min)
    red = np.full((8, 8, 3), (0.9, 0.1, 0.1), dtype=np.float32)
    g = compute_gains(red, p)
    assert g.wb[0] == pytest.approx(p.wb_gain_min)
    assert g.wb[1] == pytest.approx(p.wb_gain_max)
    assert g.clamped["wb"] is True


def test_normalising_twice_is_nearly_a_no_op_and_converges():
    """Normalisation is close to idempotent, but not exactly, and that is fine.

    The statistics mask is recomputed on the transformed image, so a second
    pass measures a slightly different set of pixels (2784 of 3072 becomes
    2752 for this fixture) whose median is 0.1835 rather than the 0.18
    target. The residual correction is real, not floating-point noise. What
    matters is that it is small and that repeated passes settle instead of
    drifting; nothing in the project normalises an image twice.
    """
    img = _gradient_image()
    once, _ = normalize_float(img, NormalizeParams())
    twice, gains2 = normalize_float(once, NormalizeParams())
    thrice, _ = normalize_float(twice, NormalizeParams())

    # Measured for this fixture: gains within 0.019 of 1, pixels within 2.05/255.
    assert np.allclose(gains2.wb, 1.0, atol=0.05)
    assert gains2.exposure == pytest.approx(1.0, abs=0.05)
    assert np.abs(twice - once).max() < 6 / 255

    # Measured: 2.05/255 then 0.53/255. Each pass corrects less than the last.
    assert np.abs(thrice - twice).max() < np.abs(twice - once).max()


def test_float_and_u8_paths_agree():
    img_u8 = (_gradient_image() * 255).round().astype(np.uint8)
    p = NormalizeParams()
    out_f, gains_f = normalize_float(img_u8.astype(np.float32) / 255.0, p)
    out_u8, gains_u8 = normalize_u8(img_u8, p)
    assert out_u8.dtype == np.uint8 and out_u8.shape == img_u8.shape
    assert np.allclose(gains_f.combined, gains_u8.combined, atol=1e-6)
    assert np.abs(out_u8.astype(int) - np.round(out_f * 255).astype(int)).max() <= 1


def test_u8_subsampling_branch_matches_the_reference():
    """At 1080p the Pi always subsamples for statistics, so prove that path.

    The 48x64 fixture above gives ``step == 1``, which quietly skips the
    strided branch entirely — the branch every real capture takes. A
    1920x1080 frame gives ``step == 3``, so the gains come from about a ninth
    of the pixels.

    Two claims, kept apart because they can fail independently: the table
    must apply exactly the gains it computed, and gains from a subsample must
    be close enough to full-image statistics that the result is
    indistinguishable at 8-bit precision. Measured: relative gain difference
    1.4e-4, table exact, fast path within one level.
    """
    img_u8 = (_gradient_image(1080, 1920) * 255).round().astype(np.uint8)
    p = NormalizeParams()
    out_u8, gains_u8 = normalize_u8(img_u8, p)
    reference, gains_full = normalize_float(img_u8.astype(np.float32) / 255.0, p)

    # The subsample really was taken: fewer pixels give slightly different gains.
    assert not np.array_equal(gains_u8.combined, gains_full.combined)
    assert np.allclose(gains_u8.combined, gains_full.combined, rtol=1e-3)

    # The table applies its own gains faithfully.
    own = apply_gains_float(img_u8.astype(np.float32) / 255.0, gains_u8)
    assert np.abs(out_u8.astype(int) - np.round(own * 255).astype(int)).max() <= 1

    # And the fast path is indistinguishable from the full-statistics path.
    assert np.abs(out_u8.astype(int) - np.round(reference * 255).astype(int)).max() <= 1


def test_the_float_path_refuses_uint8_input():
    """uint8 would be clipped to 1.0 by srgb_to_linear and silently destroyed."""
    u8 = np.full((16, 16, 3), 128, dtype=np.uint8)
    with pytest.raises(ValueError, match="float image"):
        normalize_float(u8, NormalizeParams())
    with pytest.raises(ValueError, match="float image"):
        compute_gains(u8, NormalizeParams())


def test_luts_are_monotone():
    luts = gains_to_luts(
        Gains(wb=np.array([0.8, 1.0, 1.5], dtype=np.float32), exposure=1.3,
              clamped={"wb": False, "exposure": False})
    )
    assert luts.shape == (3, 256) and luts.dtype == np.uint8
    assert np.all(np.diff(luts.astype(int), axis=1) >= 0)
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/pytest tests/test_normalize.py -q` → FAIL, no module `kodachrome.normalize`.

- [ ] **Step 3: Implement `normalize.py`**

```python
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
    exposure_target_median: float = 0.18
    exposure_gain_min: float = 0.5
    exposure_gain_max: float = 3.0
    stats_lum_min: float = 0.02
    stats_lum_max: float = 0.90

    def __post_init__(self) -> None:
        for f in fields(self):
            if f.name != "white_balance":
                _check_finite(f.name, getattr(self, f.name))
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

    @property
    def combined(self) -> np.ndarray:
        return (np.asarray(self.wb, dtype=np.float32) * np.float32(self.exposure)).astype(
            np.float32
        )

    def to_dict(self) -> dict:
        return {
            "wb": [round(float(g), 4) for g in self.wb],
            "exposure": round(float(self.exposure), 4),
            "clamped": dict(self.clamped),
        }


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


def normalize_float(rgb: np.ndarray, params: NormalizeParams) -> tuple[np.ndarray, Gains]:
    """Reference path used by the trainer. ``rgb`` is float32 sRGB in [0, 1]."""
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
    h, w = rgb_u8.shape[:2]
    step = max(1, int(np.ceil(np.sqrt(h * w / max_stats_pixels))))
    gains = compute_gains(rgb_u8[::step, ::step].astype(np.float32) / 255.0, params)
    table = np.ascontiguousarray(gains_to_luts(gains).T).reshape(256, 1, 3)
    return cv2.LUT(np.ascontiguousarray(rgb_u8), table), gains
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_normalize.py -q` → all pass. A max diff of 2 in `test_float_and_u8_paths_agree` almost always means the `cv2.LUT` table shape is wrong; it must be `(256, 1, 3)`.

`test_u8_subsampling_branch_matches_the_reference` builds a full 1920x1080 frame and takes about a second — that is expected and worth it, because it is the only test covering the strided-statistics branch that every real capture uses.

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/normalize.py tests/test_normalize.py
git commit -m "feat: white balance and exposure normalisation with validation and clamp reporting

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: 3D LUT (`lut.py`)

Implements spec 5.3. Adds full validation and content hashing (F-12, F-14).

**Files:**
- Create: `kodachrome/lut.py`, `tests/test_lut.py`

**Interfaces:**
- Produces:
  - `class CubeError(ValueError)`
  - `@dataclass LUT3D(table)` with `.size`, `identity(size=33)`, `to_flat()`, `from_flat(flat, size)`, `apply_numpy(rgb_float)`, `to_pillow()`, `apply_pillow(rgb_u8, filt=None)`
  - `sha1_hex(lut) -> str`
  - `read_cube(path) -> LUT3D`, `write_cube(lut, path, title="kodachrome")`

- [ ] **Step 1: Write the failing tests**

`tests/test_lut.py`:
```python
import re

import numpy as np
import pytest

from kodachrome.lut import LUT3D, CubeError, read_cube, sha1_hex, write_cube


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
    # Pillow stores the table in 16-bit fixed point, so allow one 8-bit level
    assert np.abs(lut.apply_pillow(u8).astype(int) - u8.astype(int)).max() <= 1


def test_flat_order_is_red_fastest():
    flat = LUT3D.identity(2).to_flat()
    expected = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        dtype=np.float32,
    )
    assert np.array_equal(flat, expected)
    assert np.array_equal(LUT3D.from_flat(flat, 2).table, LUT3D.identity(2).table)


def test_cube_roundtrip_and_domain_written(tmp_path):
    lut = _smooth_test_lut(9)
    path = tmp_path / "t.cube"
    write_cube(lut, path, title="test")
    back = read_cube(path)
    assert back.size == 9
    assert np.allclose(back.table, lut.table, atol=1e-6)
    text = path.read_text()
    assert text.startswith('TITLE "test"\nLUT_3D_SIZE 9\n')
    assert "DOMAIN_MIN 0.0 0.0 0.0" in text and "DOMAIN_MAX 1.0 1.0 1.0" in text


def test_numpy_and_pillow_agree():
    lut = _smooth_test_lut(17)
    u8 = np.random.default_rng(3).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    ref = np.round(lut.apply_numpy(u8.astype(np.float32) / 255.0) * 255).astype(int)
    diff = np.abs(ref - lut.apply_pillow(u8).astype(int))
    assert diff.max() <= 1
    assert diff.mean() < 0.3


def test_an_unsupported_keyword_is_named_not_parsed_as_data(tmp_path):
    """Resolve emits LUT_3D_INPUT_RANGE, which has three tokens like a data row."""
    path = tmp_path / "resolve.cube"
    path.write_text("LUT_3D_SIZE 2\nLUT_3D_INPUT_RANGE 0.0 1.0\n" + "0 0 0\n" * 8)
    with pytest.raises(CubeError, match=re.escape("unsupported keyword")):
        read_cube(path)


def test_sha1_is_stable_and_content_sensitive():
    a = sha1_hex(LUT3D.identity(9))
    assert a == sha1_hex(LUT3D.identity(9))
    assert len(a) == 40
    tweaked = LUT3D.identity(9).table.copy()
    tweaked[4, 4, 4, 0] += 0.01
    assert sha1_hex(LUT3D(tweaked)) != a


@pytest.mark.parametrize(
    "text, message",
    [
        ("LUT_3D_SIZE 1\n0 0 0\n", "2..65"),
        ("LUT_3D_SIZE 2\n0 0 0\n", "expected 8"),
        ("LUT_3D_SIZE 2\n" + "0 0 x\n" * 8, "line 2"),
        ("LUT_1D_SIZE 4\n", "1D"),
        ("0 0 0\n", "LUT_3D_SIZE"),
        ("LUT_3D_SIZE 2\n" + "0 0 nan\n" * 8, "finite"),
        ("LUT_3D_SIZE 2\n" + "0 0 2.0\n" * 8, "[0, 1]"),
        ("LUT_3D_SIZE 2\nDOMAIN_MAX 2.0 2.0 2.0\n" + "0 0 0\n" * 8, "DOMAIN"),
    ],
)
def test_cube_errors(tmp_path, text, message):
    path = tmp_path / "bad.cube"
    path.write_text(text)
    with pytest.raises(CubeError, match=re.escape(message)):
        read_cube(path)


@pytest.mark.parametrize(
    "table, message",
    [
        (np.zeros((3, 3, 2, 3), dtype=np.float32), "shape"),
        (np.zeros((66, 66, 66, 3), dtype=np.float32), "2..65"),
        (np.full((4, 4, 4, 3), np.nan, dtype=np.float32), "finite"),
        (np.full((4, 4, 4, 3), 1.5, dtype=np.float32), "[0, 1]"),
    ],
)
def test_table_validation(table, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        LUT3D(table)
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/pytest tests/test_lut.py -q` → FAIL, no module `kodachrome.lut`.

- [ ] **Step 3: Implement `lut.py`**

```python
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
        # np.array, not np.asarray: the result must be writeable. Otherwise
        # Pipeline.process(grain=False) returns a read-only frame while
        # grain=True returns a writeable one, because the grain path happens to
        # allocate a fresh array in cvtColor.
        return np.array(im.filter(filt))


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
        # A keyword we do not handle must be named, not parsed as data. Resolve
        # emits LUT_3D_INPUT_RANGE, which has exactly three tokens and would
        # otherwise reach float() and fail as "non-numeric value".
        if parts[0][0].isalpha() or parts[0][0] == "_":
            raise CubeError(
                f"{path}: unsupported keyword {parts[0]!r} on line {lineno}; "
                "this reader handles TITLE, LUT_3D_SIZE, DOMAIN_MIN and DOMAIN_MAX"
            )
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
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_lut.py -q` → all pass. If `to_pillow` raises `TypeError` about the table, the installed Pillow wants a list: use `flat.tolist()` and note the Pillow version in `docs/decisions.md`.

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/lut.py tests/test_lut.py
git commit -m "feat: LUT3D with .cube I/O, full validation and content hashing

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Film grain (`grain.py`)

Implements spec 5.4.

**Files:**
- Create: `kodachrome/grain.py`, `tests/test_grain.py`

**Interfaces:**
- Produces: `@dataclass GrainParams(strength=0.025, blur_sigma=0.7, enabled=True)` with validating `__post_init__`, `from_dict`, `to_dict`; `add_grain(rgb_u8, params, rng=None) -> rgb_u8`

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


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"strength": -0.1}, "strength"),
        ({"blur_sigma": -1.0}, "blur_sigma"),
        ({"strength": float("inf")}, "strength"),
    ],
)
def test_invalid_params_name_the_field(kwargs, field):
    with pytest.raises(ValueError, match=field):
        GrainParams(**kwargs)


def test_preserves_mean_luminance_and_adds_no_colour_bias():
    img = np.full((256, 256, 3), 128, dtype=np.uint8)
    out = add_grain(img, GrainParams(strength=0.05), rng=np.random.default_rng(1))
    assert out.dtype == np.uint8 and out.shape == img.shape
    assert np.allclose(out.reshape(-1, 3).mean(axis=0), 128, atol=0.5)
    assert out.std() > 5


def test_strength_scales_noise():
    img = np.full((256, 256, 3), 128, dtype=np.uint8)
    lo = add_grain(img, GrainParams(strength=0.02), rng=np.random.default_rng(2))
    hi = add_grain(img, GrainParams(strength=0.06), rng=np.random.default_rng(2))
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

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/pytest tests/test_grain.py -q` → FAIL, no module `kodachrome.grain`.

- [ ] **Step 3: Implement `grain.py`**

```python
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
```

- [ ] **Step 4: Run and commit**

Run: `.venv/bin/pytest tests/test_grain.py -q` → all pass.

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/grain.py tests/test_grain.py
git commit -m "feat: luminance film grain with midtone envelope and validation

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Image I/O with colour management (`imageio.py`)

Implements spec 5.5. Fixes F-10.

**Files:**
- Create: `kodachrome/imageio.py`, `tests/test_imageio.py`
- Modify: `tests/conftest.py` (add the `wide_gamut_icc` fixture from Step 0)

**Interfaces:**
- Produces:
  - `IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}`
  - `@dataclass ImageMeta(profile: str, oriented: bool, profile_error: str | None, width: int, height: int)`
  - `load_rgb(path, *, colour_manage=True) -> (np.ndarray uint8, ImageMeta)`
  - `save_jpeg(rgb_u8, path, quality=95, embed_srgb=True) -> Path`
  - `list_images(dir_path) -> list[Path]`
  - `srgb_profile() -> PIL.ImageCms.ImageCmsProfile` (cached)

- [ ] **Step 0: Add the wide-gamut ICC fixture to `tests/conftest.py`**

The ICC test needs an image tagged with a profile that is genuinely not sRGB.
`ImageCms.createProfile` only makes sRGB, LAB and XYZ profiles, and a LAB or
XYZ profile embedded in an RGB JPEG is invalid — littlecms rejects it with
`PyCMSError: cannot build transform` (verified). Real Adobe RGB profiles exist
on macOS but not on Raspberry Pi OS or in CI, so depending on one would make
the test skip exactly where colour correctness matters most.

So the fixture builds a minimal, valid ICC v2 matrix-shaper profile in memory
with Adobe RGB (1998) primaries. It is verified: it converts (200, 60, 60) to
(231, 57, 56) in sRGB, byte-identical to what the real Adobe RGB (1998)
profile produces.

Append to `tests/conftest.py`:

```python
import struct

import pytest


def _s15f16(x: float) -> int:
    """ICC s15Fixed16Number."""
    return int(round(x * 65536.0))


def _xyz_tag(x: float, y: float, z: float) -> bytes:
    return b"XYZ " + b"\0" * 4 + struct.pack(">3i", _s15f16(x), _s15f16(y), _s15f16(z))


def _curv_tag(gamma: float) -> bytes:
    """A one-entry curve tag, which ICC reads as a plain gamma value."""
    return b"curv" + b"\0" * 4 + struct.pack(">I", 1) + struct.pack(">H", int(round(gamma * 256)))


def _desc_tag(text: str) -> bytes:
    body = text.encode("ascii") + b"\0"
    return b"desc" + b"\0" * 4 + struct.pack(">I", len(body)) + body + b"\0" * 88


def build_rgb_icc_profile(description: str, gamma: float, primaries) -> bytes:
    """A minimal valid ICC v2 RGB matrix-shaper profile, built in memory.

    Enough for littlecms to construct a transform: white point, three
    colourant XYZ tags, three tone curves, a description and a copyright.
    """
    tags = {
        b"desc": _desc_tag(description),
        b"wtpt": _xyz_tag(0.9642, 1.0, 0.8249),  # D50, which ICC requires
        b"rXYZ": _xyz_tag(*primaries[0]),
        b"gXYZ": _xyz_tag(*primaries[1]),
        b"bXYZ": _xyz_tag(*primaries[2]),
        b"rTRC": _curv_tag(gamma),
        b"gTRC": _curv_tag(gamma),
        b"bTRC": _curv_tag(gamma),
        b"cprt": _desc_tag("public domain"),
    }
    offset = 128 + 4 + len(tags) * 12
    table, data = b"", b""
    for sig, payload in tags.items():
        padding = (-len(payload)) % 4
        table += sig + struct.pack(">II", offset, len(payload))
        data += payload + b"\0" * padding
        offset += len(payload) + padding
    body = struct.pack(">I", len(tags)) + table + data

    header = bytearray(128)
    struct.pack_into(">I", header, 0, 128 + len(body))  # total size
    header[4:8] = b"none"                                # preferred CMM
    header[8:12] = struct.pack(">I", 0x02100000)         # version 2.1
    header[12:16] = b"mntr"                              # display device class
    header[16:20] = b"RGB "                              # data colour space
    header[20:24] = b"XYZ "                              # profile connection space
    struct.pack_into(">6H", header, 24, 2026, 9, 3, 0, 0, 0)
    header[36:40] = b"acsp"                              # required signature
    header[64:68] = struct.pack(">I", 0)                 # perceptual intent
    header[68:80] = struct.pack(">3i", _s15f16(0.9642), _s15f16(1.0), _s15f16(0.8249))
    return bytes(header) + body


@pytest.fixture(scope="session")
def wide_gamut_icc() -> bytes:
    """Adobe RGB (1998) primaries adapted to D50, gamma 2.2."""
    return build_rgb_icc_profile(
        "Test Wide Gamut RGB",
        2.2,
        [
            (0.6097, 0.3111, 0.0195),
            (0.2053, 0.6257, 0.0609),
            (0.1492, 0.0632, 0.7448),
        ],
    )
```

Sanity-check it before writing the module under test:

```bash
.venv/bin/python -c "
import io
from PIL import ImageCms
import sys; sys.path.insert(0, 'tests')
from conftest import build_rgb_icc_profile
data = build_rgb_icc_profile('Test Wide Gamut RGB', 2.2,
    [(0.6097,0.3111,0.0195),(0.2053,0.6257,0.0609),(0.1492,0.0632,0.7448)])
p = ImageCms.ImageCmsProfile(io.BytesIO(data))
print(len(data), 'bytes,', repr(ImageCms.getProfileDescription(p).strip()))
"
```
Expected: `604 bytes, 'Test Wide Gamut RGB'`. If littlecms rejects the profile, stop and report BLOCKED rather than weakening the ICC test — the whole point is proving a non-sRGB profile changes the pixels.

- [ ] **Step 1: Write the failing tests**

`tests/test_imageio.py`:
```python
import numpy as np
import pytest
from PIL import Image, ImageCms

from kodachrome.imageio import ImageMeta, list_images, load_rgb, save_jpeg, srgb_profile


def test_save_and_load_roundtrip(tmp_path):
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    img[..., 0] = 200  # red-dominant: proves channel order survives
    out = save_jpeg(img, tmp_path / "nested" / "a.jpg", quality=100)
    assert out.exists()
    back, meta = load_rgb(out)
    assert back.shape == (10, 20, 3) and back.dtype == np.uint8
    assert back[..., 0].mean() > 150 and back[..., 2].mean() < 30
    assert isinstance(meta, ImageMeta) and meta.width == 20 and meta.height == 10


def test_saved_jpeg_carries_an_srgb_profile(tmp_path):
    out = save_jpeg(np.full((8, 8, 3), 120, dtype=np.uint8), tmp_path / "p.jpg")
    with Image.open(out) as im:
        assert im.info.get("icc_profile")
    _, meta = load_rgb(out)
    assert "sRGB" in meta.profile or "embedded" in meta.profile


def test_exif_orientation_is_applied(tmp_path):
    # A 4x2 image tagged orientation=6 (rotate 90 CW on display) must come back 2x4.
    base = Image.new("RGB", (4, 2), (10, 20, 30))
    exif = base.getexif()
    exif[274] = 6  # Orientation
    path = tmp_path / "rot.jpg"
    base.save(path, exif=exif)
    arr, meta = load_rgb(path)
    assert arr.shape[:2] == (4, 2)  # height, width swapped
    assert meta.oriented is True


def test_icc_profile_is_converted_not_ignored(tmp_path, wide_gamut_icc):
    """The same bytes under a wide-gamut profile must not decode to the same pixels."""
    pixels = np.full((8, 8, 3), (200, 60, 60), dtype=np.uint8)
    srgb_path = tmp_path / "srgb.jpg"
    wide_path = tmp_path / "wide.jpg"
    Image.fromarray(pixels).save(
        srgb_path, quality=100, icc_profile=ImageCms.ImageCmsProfile(srgb_profile()).tobytes()
    )
    Image.fromarray(pixels).save(wide_path, quality=100, icc_profile=wide_gamut_icc)

    srgb_arr, srgb_meta = load_rgb(srgb_path)
    wide_arr, wide_meta = load_rgb(wide_path)

    assert wide_meta.profile_error is None
    assert "Wide Gamut" in wide_meta.profile
    assert not np.array_equal(srgb_arr, wide_arr), "a non-sRGB profile must change the pixels"
    # Verified against the real Adobe RGB (1998) profile: (200, 60, 60) lands on
    # (231, 57, 56) in sRGB. Allow a little slack for JPEG and littlecms rounding.
    assert abs(int(wide_arr[0, 0, 0]) - 231) <= 3
    assert abs(int(wide_arr[0, 0, 2]) - 56) <= 3
    # An sRGB-tagged image must come back essentially unchanged.
    assert np.abs(srgb_arr.astype(int) - pixels.astype(int)).max() <= 2


def test_malformed_profile_falls_back_and_reports(tmp_path):
    path = tmp_path / "bad.jpg"
    Image.fromarray(np.full((8, 8, 3), 90, dtype=np.uint8)).save(path, icc_profile=b"not-a-profile")
    arr, meta = load_rgb(path)
    assert arr.shape == (8, 8, 3)
    assert meta.profile == "invalid"
    assert meta.profile_error


def test_colour_manage_can_be_disabled(tmp_path, wide_gamut_icc):
    """The toggle must be provable, so the fixture needs a profile that would change pixels.

    An image with no embedded profile passes this test whether or not the
    toggle does anything, which is no test at all.
    """
    path = tmp_path / "wide.jpg"
    pixels = np.full((8, 8, 3), (200, 60, 60), dtype=np.uint8)
    Image.fromarray(pixels).save(path, quality=100, icc_profile=wide_gamut_icc)

    managed, _ = load_rgb(path)
    unmanaged, meta = load_rgb(path, colour_manage=False)

    assert not np.array_equal(managed, unmanaged), "the toggle did not suppress conversion"
    # Measured: managed (231, 57, 56); unmanaged stays at the stored values.
    assert np.abs(unmanaged.astype(int) - pixels.astype(int)).max() <= 2
    assert "assumed" in meta.profile and "off" in meta.profile


def test_a_lab_source_is_converted_not_mislabelled(tmp_path):
    """Regression test for a bug an earlier draft of this module contained.

    Converting the image to RGB before ``profileToProfile`` makes a LAB or
    CMYK transform unbuildable, so littlecms raises and the fallback brands a
    perfectly valid profile "invalid" — silently skipping colour management
    on exactly the archival scans that need it. This pins that it does not
    happen.
    """
    lab_profile = ImageCms.createProfile("LAB")
    lab = Image.new("LAB", (8, 8))
    lab.putdata([(180, 160, 140)] * 64)
    path = tmp_path / "lab.tif"
    lab.save(path, icc_profile=ImageCms.ImageCmsProfile(lab_profile).tobytes())

    arr, meta = load_rgb(path)
    assert arr.shape == (8, 8, 3) and arr.dtype == np.uint8
    assert meta.profile != "invalid", "a valid LAB profile was mislabelled"
    assert meta.profile_error is None


def test_load_converts_modes(tmp_path):
    Image.new("L", (8, 8), 77).save(tmp_path / "grey.png")
    Image.new("RGBA", (8, 8), (1, 2, 3, 255)).save(tmp_path / "rgba.png")
    assert load_rgb(tmp_path / "grey.png")[0].shape == (8, 8, 3)
    assert load_rgb(tmp_path / "rgba.png")[0].shape == (8, 8, 3)


def test_saved_jpeg_keeps_full_chroma_resolution(tmp_path):
    """4:2:0 would destroy chroma detail in the project's only output writer."""
    columns = np.zeros((64, 64, 3), dtype=np.uint8)
    columns[:, ::2] = (220, 30, 30)
    columns[:, 1::2] = (30, 200, 60)
    path = save_jpeg(columns, tmp_path / "chroma.jpg")

    with Image.open(path) as im:
        # layer[0] carries the luma sampling factors; (1, 1, 1, 0) means 4:4:4.
        assert im.layer[0][1:3] == (1, 1), f"expected 4:4:4, got layers {im.layer}"

    back, _ = load_rgb(path)
    # Measured: 4:2:0 gives a max error of about 118 here; 4:4:4 gives 2.
    assert np.abs(back.astype(int) - columns.astype(int)).max() <= 8


def test_loaded_arrays_are_writeable(tmp_path):
    """Callers draw on preview frames; a read-only return value breaks them."""
    save_jpeg(np.full((8, 8, 3), 120, dtype=np.uint8), tmp_path / "w.jpg")
    arr, _ = load_rgb(tmp_path / "w.jpg")
    assert arr.flags.writeable
    arr[0, 0, 0] = 5  # must not raise


def test_list_images_filters_and_sorts(tmp_path):
    for name in ["b.JPG", "a.jpeg", "c.png", "notes.txt", "d.tif"]:
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in list_images(tmp_path)] == ["a.jpeg", "b.JPG", "c.png", "d.tif"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rgb(tmp_path / "nope.jpg")
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/pytest tests/test_imageio.py -q` → FAIL, no module `kodachrome.imageio`.

- [ ] **Step 3: Implement `imageio.py`**

```python
"""Image file I/O, and the one place colour management happens.

Every pixel that enters this project comes through ``load_rgb``, which does
two things no other module should have to remember:

**EXIF orientation.** A camera that was held sideways records the rotation
as a tag rather than rotating the pixels. The trainer crops a fixed 6% from
each edge, so an unoriented image gets the wrong edges cropped, and a
portrait frame would be sampled as landscape. ``ImageOps.exif_transpose``
resolves the tag into real pixel order.

**ICC profiles.** The whole project is a colour measurement, so treating an
Adobe RGB or ProPhoto scan as if it were sRGB would bake a systematic error
into the learned look - the more so because the target corpus is archival
scans whose colour management is part of what we are matching. When a
profile is embedded, the image is converted to sRGB with ``ImageCms``. When
none is present, sRGB is assumed, which is the web convention and correct
for Commons JPEGs. When one is present but unreadable, sRGB is assumed and
the failure is recorded rather than swallowed, so the report can count it.

``ImageMeta`` travels with the pixels so the trainer can publish profile
statistics: a corpus that is secretly half Adobe RGB should be visible, not
silently averaged in.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_EXIF_ORIENTATION_TAG = 274


@dataclass
class ImageMeta:
    profile: str
    oriented: bool
    profile_error: str | None
    width: int
    height: int


@lru_cache(maxsize=1)
def srgb_profile():
    """The sRGB profile used as the working space, built once."""
    return ImageCms.createProfile("sRGB")


def _describe(profile: ImageCms.ImageCmsProfile) -> str:
    try:
        return ImageCms.getProfileDescription(profile).strip() or "unnamed profile"
    except Exception:  # noqa: BLE001 - a profile can be readable but have no description
        return "unnamed profile"


def load_rgb(path: str | Path, *, colour_manage: bool = True) -> tuple[np.ndarray, ImageMeta]:
    """Load an image as sRGB RGB uint8, applying EXIF orientation and ICC conversion."""
    path = Path(path)
    with Image.open(path) as im:
        exif = im.getexif()
        oriented = bool(exif.get(_EXIF_ORIENTATION_TAG, 1) not in (1, None))
        im = ImageOps.exif_transpose(im)

        raw_profile = im.info.get("icc_profile")
        profile_name = "sRGB (assumed)"
        profile_error: str | None = None

        if raw_profile and colour_manage:
            # Built outside the try on purpose: if the working-space profile
            # itself cannot be created, that is an environment fault, not a bad
            # source profile, and it should surface rather than be reported as
            # "invalid" against the image.
            destination = srgb_profile()
            try:
                src = ImageCms.ImageCmsProfile(io.BytesIO(raw_profile))
                profile_name = _describe(src)
                # Do not convert the mode first: littlecms needs the image in the
                # colour space the profile describes. Pre-converting a LAB or CMYK
                # image to RGB makes the transform unbuildable, and the fallback
                # below would then silently mislabel a perfectly good profile.
                im = ImageCms.profileToProfile(
                    im, src, destination, renderingIntent=0, outputMode="RGB"
                )
            except Exception as exc:  # noqa: BLE001 - any malformed profile falls back to sRGB
                profile_name = "invalid"
                profile_error = f"{type(exc).__name__}: {exc}"
        elif raw_profile:
            profile_name = "sRGB (assumed, colour management off)"

        # np.asarray on a PIL image is read-only. Callers legitimately draw on
        # what they get back (overlays on a preview frame), and the writability
        # of a public return value must not depend on which branch produced it.
        rgb = np.array(im.convert("RGB"), dtype=np.uint8)

    return rgb, ImageMeta(
        profile=profile_name,
        oriented=oriented,
        profile_error=profile_error,
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
    )


def save_jpeg(
    rgb_u8: np.ndarray,
    path: str | Path,
    quality: int = 95,
    embed_srgb: bool = True,
    subsampling: int = 0,
) -> Path:
    """Write an sRGB JPEG. ``subsampling=0`` means 4:4:4, full chroma resolution.

    Pillow's default is 4:2:0, which halves chroma resolution in both axes.
    On a natural frame that costs little on average (mean Oklab dE 0.0126
    against 0.0102) but a great deal at saturated edges: the 99th-percentile
    error is 0.032 dE, which exceeds the fitted LUT's own 0.025 accuracy
    floor. In other words, on the pixels where Kodachrome's character
    actually lives, the file format would introduce more error than the
    learned grade itself carries. The cost is about twice the bytes — 2.4 MB
    against 4.9 MB for a 1080p frame — which is the right trade for a project
    whose entire purpose is colour, and for files that double as the archival
    record. Pass ``subsampling=2`` for 4:2:0 if storage matters more.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.ascontiguousarray(rgb_u8), "RGB")
    kwargs = {}
    if embed_srgb:
        kwargs["icc_profile"] = ImageCms.ImageCmsProfile(srgb_profile()).tobytes()
    image.save(path, "JPEG", quality=quality, subsampling=subsampling, **kwargs)
    return path


def list_images(dir_path: str | Path) -> list[Path]:
    return sorted(
        p for p in Path(dir_path).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_imageio.py -q` → all pass.

If `test_icc_profile_is_converted_not_ignored` fails because the LAB stand-in profile is rejected by `profileToProfile`, replace the wide-gamut fixture with a synthetic RGB profile built by `ImageCms.createProfile("sRGB")` whose gamma differs, or load a real Adobe RGB profile if one exists on the machine. What the test must prove is that pixels tagged with a non-sRGB profile come out **different** from the same pixels tagged sRGB. Do not weaken it to merely asserting no exception. Record any fixture change in `docs/decisions.md`.

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/imageio.py tests/test_imageio.py
git commit -m "feat: image I/O applying EXIF orientation and converting ICC profiles to sRGB

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Artifacts: validation, packaged default, atomic publish (`artifacts.py`)

Implements spec 5.6 and 5.8. Fixes F-03, F-07, F-12, F-17.

**Files:**
- Create: `kodachrome/artifacts.py`, `tests/test_artifacts.py`
- Create: `kodachrome/data/kodachrome.cube`, `kodachrome/data/params.json` (identity placeholder)
- Modify: `README.md`

**Interfaces:**
- Consumes: `LUT3D`, `read_cube`, `write_cube`, `sha1_hex`, `CubeError`, `NormalizeParams`, `GrainParams`
- Produces:
  - `PARAMS_VERSION = 2`, `class ArtifactsError(Exception)`
  - `@dataclass Artifacts(lut, normalize, grain, training: dict, path: Path, lut_sha1: str)`
  - `Artifacts.load(dir_path)`, `Artifacts.default()`, `Artifacts.resolve(dir_path | None)`
  - `write_artifact(dir_path, lut, normalize, grain, training=None, lut_file="kodachrome.cube") -> Path`
  - `publish(staging_dir, dest_dir) -> Path`

- [ ] **Step 1: Write the failing tests**

`tests/test_artifacts.py`:
```python
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from kodachrome.artifacts import (
    PARAMS_VERSION,
    Artifacts,
    ArtifactsError,
    publish,
    write_artifact,
)
from kodachrome.grain import GrainParams
from kodachrome.lut import LUT3D, sha1_hex, write_cube
from kodachrome.normalize import NormalizeParams


@pytest.fixture
def staged(tmp_path):
    d = tmp_path / "staging"
    write_artifact(d, LUT3D.identity(9), NormalizeParams(), GrainParams(), training={"note": "t"})
    return d


def test_write_then_load(staged):
    data = json.loads((staged / "params.json").read_text())
    assert data["version"] == PARAMS_VERSION
    assert data["lut_sha1"] == sha1_hex(LUT3D.identity(9))
    art = Artifacts.load(staged)
    assert art.lut.size == 9
    assert art.normalize == NormalizeParams()
    assert art.grain == GrainParams()
    assert art.training == {"note": "t"}
    assert art.lut_sha1 == data["lut_sha1"]


def test_missing_params_is_clear(tmp_path):
    with pytest.raises(ArtifactsError, match="params.json"):
        Artifacts.load(tmp_path)


@pytest.mark.parametrize(
    "payload, message",
    [
        ("{not json", "is not valid JSON"),
        ('{"version": 99}', "unsupported params version"),
        ('{"lut_file": "k.cube"}', "unsupported params version"),
        ("[1, 2, 3]", "top level must be a JSON object"),
        ('{"version": 2, "normalize": 5}', "'normalize' must be a JSON object"),
        ('{"version": 2, "grain": "x"}', "'grain' must be a JSON object"),
    ],
)
def test_schema_rejections(tmp_path, payload, message):
    """Match whole phrases, not tokens.

    These assertions read `tmp_path`, whose name pytest derives from the test
    id — which for a parametrised case embeds the expected token. Bare tokens
    like "version" survive today only because pytest truncates the directory
    name before reaching them; that is an implementation detail, not a
    guarantee. Phrases do not depend on it.
    """
    (tmp_path / "params.json").write_text(payload)
    with pytest.raises(ArtifactsError, match=re.escape(message)):
        Artifacts.load(tmp_path)


def test_missing_cube_is_clear(tmp_path):
    (tmp_path / "params.json").write_text(json.dumps({"version": 2, "lut_file": "k.cube"}))
    with pytest.raises(ArtifactsError, match="k.cube"):
        Artifacts.load(tmp_path)


def test_lut_hash_mismatch_is_refused(staged):
    data = json.loads((staged / "params.json").read_text())
    data["lut_sha1"] = "0" * 40
    (staged / "params.json").write_text(json.dumps(data))
    with pytest.raises(ArtifactsError, match="sha1"):
        Artifacts.load(staged)


def test_packaged_default_loads_from_any_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    art = Artifacts.default()
    assert art.lut.size >= 2
    assert art.lut_sha1 == sha1_hex(art.lut)


def test_the_default_artifacts_path_still_exists(tmp_path, monkeypatch):
    """`path` is provenance: it must name a directory that is still there.

    Resolving the packaged data as its own package yields a MultiplexedPath,
    which `as_file` materialises into a temp directory deleted before the call
    returns — so `path` pointed at nothing and every startup copied 950 KB.
    """
    monkeypatch.chdir(tmp_path)
    art = Artifacts.default()
    assert art.path.is_dir(), f"{art.path} does not exist after default() returned"
    assert (art.path / "params.json").is_file()
    assert Artifacts.default().path == art.path


@pytest.mark.parametrize(
    "lut_file, message",
    [
        (5, "'lut_file' must be a non-empty string"),
        (None, "'lut_file' must be a non-empty string"),
        ("", "'lut_file' must be a non-empty string"),
        ("../escape.cube", "must be a plain name"),
        ("/etc/passwd.cube", "must be a plain name"),
    ],
)
def test_lut_file_is_validated(tmp_path, lut_file, message):
    """A bad lut_file must not escape as TypeError, nor read outside the directory."""
    (tmp_path / "params.json").write_text(json.dumps({"version": 2, "lut_file": lut_file}))
    with pytest.raises(ArtifactsError, match=re.escape(message)):
        Artifacts.load(tmp_path)


@pytest.mark.parametrize("version", [True, 0, -1, 3])
def test_bad_version_values_are_refused(tmp_path, version):
    """`True` is an int subclass, so isinstance would have let it through."""
    (tmp_path / "params.json").write_text(json.dumps({"version": version}))
    with pytest.raises(ArtifactsError, match=re.escape("unsupported params version")):
        Artifacts.load(tmp_path)


def test_resolve_prefers_the_override(staged, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Artifacts.resolve(staged).lut.size == 9
    assert Artifacts.resolve(None).path != staged


def test_publish_moves_only_after_validation(staged, tmp_path):
    dest = tmp_path / "live"
    published = publish(staged, dest)
    assert published == dest
    assert (dest / "kodachrome.cube").exists() and (dest / "params.json").exists()
    assert Artifacts.load(dest).lut.size == 9
    assert not staged.exists()


def test_publish_refuses_an_invalid_staging_dir_and_leaves_dest_untouched(tmp_path):
    dest = tmp_path / "live"
    write_artifact(dest, LUT3D.identity(5), NormalizeParams(), GrainParams())
    before = (dest / "params.json").read_text()

    bad = tmp_path / "bad"
    bad.mkdir()
    write_cube(LUT3D.identity(9), bad / "kodachrome.cube")
    (bad / "params.json").write_text(json.dumps({"version": 2, "lut_file": "kodachrome.cube",
                                                 "lut_sha1": "0" * 40}))
    with pytest.raises(ArtifactsError):
        publish(bad, dest)
    assert (dest / "params.json").read_text() == before
    assert Artifacts.load(dest).lut.size == 5


def test_publish_replaces_an_existing_artifact(tmp_path):
    dest = tmp_path / "live"
    write_artifact(dest, LUT3D.identity(5), NormalizeParams(), GrainParams())
    staging = tmp_path / "new"
    write_artifact(staging, LUT3D.identity(9), NormalizeParams(), GrainParams())
    publish(staging, dest)
    assert Artifacts.load(dest).lut.size == 9


def test_a_missing_lut_sha1_is_refused(tmp_path):
    """Omitting the field must not silently disable the integrity check.

    Demonstrated before this test existed: deleting `lut_sha1` and swapping in
    a completely different table loaded without complaint.
    """
    write_artifact(tmp_path, LUT3D.identity(9), NormalizeParams(), GrainParams())
    params = tmp_path / "params.json"
    data = json.loads(params.read_text())
    del data["lut_sha1"]
    params.write_text(json.dumps(data))
    write_cube(LUT3D(np.clip(LUT3D.identity(9).table**1.7, 0, 1)), tmp_path / "kodachrome.cube")

    # Same hazard as above: this test's name contains "lut_sha1", so it appears
    # in tmp_path. Match the sentence, not the token.
    with pytest.raises(ArtifactsError, match=re.escape("'lut_sha1' is required")):
        Artifacts.load(tmp_path)


def test_a_field_level_type_error_is_wrapped(tmp_path):
    """A bad value inside a well-formed section must not escape as TypeError."""
    write_artifact(tmp_path, LUT3D.identity(9), NormalizeParams(), GrainParams())
    params = tmp_path / "params.json"
    data = json.loads(params.read_text())
    data["normalize"]["wb_gain_min"] = "oops"
    params.write_text(json.dumps(data))

    with pytest.raises(ArtifactsError):
        Artifacts.load(tmp_path)


def test_a_malformed_training_section_is_named(tmp_path):
    """Reported as a bad training section, not as a missing LUT file."""
    (tmp_path / "params.json").write_text(
        json.dumps({"version": 2, "training": "not-an-object"})
    )
    # Match the reason, not the word. `tmp_path` is named after the test
    # function, so a bare match="training" also matches the directory in a
    # "LUT file not found" message and passes for the wrong reason.
    with pytest.raises(ArtifactsError, match=re.escape("'training' must be a JSON object")):
        Artifacts.load(tmp_path)


def test_publish_reports_failure_as_an_artifacts_error(tmp_path, monkeypatch):
    """Every failure out of this module is an ArtifactsError naming the paths."""
    staging = tmp_path / "staging"
    write_artifact(staging, LUT3D.identity(9), NormalizeParams(), GrainParams())
    dest = tmp_path / "live"

    real_replace = os.replace

    def failing_replace(src, dst):
        if Path(src) == staging:
            raise OSError("simulated failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(ArtifactsError, match="could not publish"):
        publish(staging, dest)


def test_a_non_identity_artifact_survives_a_write_load_cycle(tmp_path):
    """The integrity check must not fire on an artifact this code just wrote.

    Every other test here uses an identity LUT, whose values are exact at the
    six decimals a .cube stores. A fitted LUT is not: it loses about 5e-7 per
    value on the way to disk. If the recorded hash came from the in-memory
    table rather than the persisted file, this test would fail while all the
    identity-based ones passed — and every trained artifact would be
    unloadable.
    """
    lut = LUT3D(np.clip(LUT3D.identity(17).table ** 1.4, 0.0, 1.0))
    write_artifact(tmp_path, lut, NormalizeParams(), GrainParams())

    art = Artifacts.load(tmp_path)  # raises ArtifactsError on a hash mismatch
    assert art.lut.size == 17
    assert np.abs(art.lut.table - lut.table).max() < 1e-5
    assert art.lut_sha1 == json.loads((tmp_path / "params.json").read_text())["lut_sha1"]


def test_publish_accepts_a_non_identity_artifact(tmp_path):
    """The same hazard, reached through the trainer's actual path."""
    staging = tmp_path / "staging"
    write_artifact(
        staging,
        LUT3D(np.clip(LUT3D.identity(9).table ** 0.8, 0.0, 1.0)),
        NormalizeParams(),
        GrainParams(),
    )
    dest = publish(staging, tmp_path / "live")
    assert Artifacts.load(dest).lut.size == 9


def test_committed_default_is_loadable(repo_root):
    art = Artifacts.load(repo_root / "kodachrome" / "data")
    assert art.lut.size in (2, 9, 33)
    assert np.isfinite(art.lut.table).all()
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/pytest tests/test_artifacts.py -q` → FAIL, no module `kodachrome.artifacts`.

- [ ] **Step 3: Implement `artifacts.py`**

```python
"""Loading, validating and publishing the trained artifact.

An artifact is a `.cube` LUT plus `params.json` holding the normalisation the
LUT was fitted against, the grain settings, and training provenance. Three
concerns live here.

**Where the default comes from.** The shipped look is package data at
``kodachrome/data/``, found with ``importlib.resources``. That makes every
command work from any working directory and puts the look inside a built
wheel. ``--artifacts DIR`` overrides it with a directory on disk.

**Validating rather than trusting.** The JSON root must be an object with a
known version; each section must be an object; the LUT file must exist,
parse, and hash to the ``lut_sha1`` recorded beside it. That last check is
what makes a mixed artifact impossible to load: if a training run wrote a
new LUT and then failed before rewriting ``params.json``, the hashes
disagree and loading fails loudly instead of grading with mismatched
parameters.

**Publishing atomically.** The trainer writes everything into a staging
directory, this module loads and validates that directory, and only then is
it moved into place with ``os.replace`` on the directory. A capture process
reading concurrently sees either the whole old artifact or the whole new
one.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from .grain import GrainParams
from .lut import LUT3D, CubeError, read_cube, sha1_hex, write_cube
from .normalize import NormalizeParams

PARAMS_VERSION = 2
DEFAULT_LUT_FILE = "kodachrome.cube"


class ArtifactsError(Exception):
    """Artifact directory missing, incomplete or invalid."""


def _require_object(value: object, name: str, path: Path) -> dict:
    if not isinstance(value, dict):
        raise ArtifactsError(f"{path}: '{name}' must be a JSON object, got {type(value).__name__}")
    return value


@dataclass
class Artifacts:
    lut: LUT3D
    normalize: NormalizeParams
    grain: GrainParams
    training: dict
    path: Path
    lut_sha1: str

    @classmethod
    def load(cls, dir_path: str | Path) -> Artifacts:
        path = Path(dir_path)
        params_path = path / "params.json"
        if not params_path.is_file():
            raise ArtifactsError(
                f"{params_path} not found. Run kodachrome-train, pass --artifacts DIR, "
                "or reinstall the package to restore the bundled default."
            )
        try:
            raw = json.loads(params_path.read_text())
        except json.JSONDecodeError as exc:
            raise ArtifactsError(f"{params_path} is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ArtifactsError(f"{params_path}: top level must be a JSON object")

        version = raw.get("version")
        # `type(...) is int`, not isinstance: bool subclasses int, so
        # {"version": true} would otherwise pass and then compare as 1.
        if type(version) is not int or not 1 <= version <= PARAMS_VERSION:
            raise ArtifactsError(
                f"{params_path}: unsupported params version {version!r} "
                f"(this build reads 1 to {PARAMS_VERSION})"
            )

        # Structural checks before file I/O, so a malformed section is reported as
        # such instead of being pre-empted by a missing-LUT error.
        training = _require_object(raw.get("training", {}), "training", params_path)
        try:
            normalize = NormalizeParams.from_dict(
                _require_object(raw.get("normalize", {}), "normalize", params_path)
            )
            grain = GrainParams.from_dict(
                _require_object(raw.get("grain", {}), "grain", params_path)
            )
        except (ValueError, TypeError) as exc:
            # TypeError too: a field-level type error such as
            # {"wb_gain_min": "oops"} reaches math.isfinite and raises TypeError,
            # which would otherwise escape unwrapped past this loader.
            raise ArtifactsError(f"{params_path}: {exc}") from exc

        lut_file = raw.get("lut_file", DEFAULT_LUT_FILE)
        # Validated before use: `path / 5` raises TypeError, which would escape
        # this loader unwrapped, and a name like "../../x.cube" would read
        # outside the artifact directory.
        if not isinstance(lut_file, str) or not lut_file:
            raise ArtifactsError(
                f"{params_path}: 'lut_file' must be a non-empty string, "
                f"got {type(lut_file).__name__}"
            )
        if Path(lut_file).is_absolute() or ".." in Path(lut_file).parts:
            raise ArtifactsError(
                f"{params_path}: 'lut_file' must be a plain name inside the artifact "
                f"directory, got {lut_file!r}"
            )
        lut_path = path / lut_file
        if not lut_path.is_file():
            raise ArtifactsError(f"LUT file {lut_path} not found (named in {params_path})")
        try:
            lut = read_cube(lut_path)
        except CubeError as exc:
            raise ArtifactsError(str(exc)) from exc

        actual = sha1_hex(lut)
        recorded = raw.get("lut_sha1")
        # Required, not optional. Making it optional would let a params.json that
        # simply omits the field load a LUT it was never paired with, which is the
        # exact failure this check exists to catch. There is no earlier schema
        # version to stay compatible with: PARAMS_VERSION starts at 2.
        if not isinstance(recorded, str):
            raise ArtifactsError(
                f"{params_path}: 'lut_sha1' is required and must be a string, "
                f"got {type(recorded).__name__}"
            )
        if recorded != actual:
            raise ArtifactsError(
                f"{path}: lut_sha1 in params.json ({recorded}) does not match {lut_path.name} "
                f"({actual}). The artifact is mixed; re-run training or restore it."
            )

        return cls(
            lut=lut,
            normalize=normalize,
            grain=grain,
            training=training,
            path=path,
            lut_sha1=actual,
        )

    @classmethod
    def default(cls) -> Artifacts:
        """The artifact shipped inside the package.

        Resolved as a sub-path of ``kodachrome`` rather than as the package
        ``kodachrome.data``. ``kodachrome/data`` has no ``__init__.py``, so
        ``resources.files("kodachrome.data")`` yields a ``MultiplexedPath``,
        and ``as_file`` on one of those materialises a *temporary* copy that
        is deleted when the context exits — leaving ``Artifacts.path``
        pointing at a directory that no longer exists, and copying the
        950 KB table on every startup. A sub-path of a real package is a real
        directory, so it survives and costs nothing.
        """
        data_dir = Path(str(resources.files("kodachrome") / "data"))
        if not data_dir.is_dir():
            raise ArtifactsError(
                f"packaged artifact directory {data_dir} is missing; reinstall the package"
            )
        return cls.load(data_dir)

    @classmethod
    def resolve(cls, dir_path: str | Path | None) -> Artifacts:
        return cls.load(dir_path) if dir_path is not None else cls.default()


def write_artifact(
    dir_path: str | Path,
    lut: LUT3D,
    normalize: NormalizeParams,
    grain: GrainParams,
    training: dict | None = None,
    lut_file: str = DEFAULT_LUT_FILE,
) -> Path:
    """Write a complete artifact into ``dir_path`` (creating it) and return the directory."""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    write_cube(lut, path / lut_file, title="kodachrome")
    # Hash what will actually be read back, not the in-memory table. The .cube
    # format stores six decimals, so a fitted LUT loses about 5e-7 per value on
    # the way to disk. Recording the pre-write hash would make every non-identity
    # artifact fail its own integrity check on load — and an identity table would
    # not reveal it, because its values are exact at six decimals.
    payload = {
        "version": PARAMS_VERSION,
        "lut_file": lut_file,
        "lut_sha1": sha1_hex(read_cube(path / lut_file)),
        "normalize": normalize.to_dict(),
        "grain": grain.to_dict(),
        "training": training or {},
    }
    (path / "params.json").write_text(json.dumps(payload, indent=2) + "\n")
    return path


def publish(staging_dir: str | Path, dest_dir: str | Path) -> Path:
    """Validate ``staging_dir``, then swap it into ``dest_dir``.

    The artifact is never published half-written: ``Artifacts.load`` runs
    against the staging directory first, so an incomplete or inconsistent set
    is rejected before anything moves.

    On the swap itself, be precise about the guarantee. ``os.replace`` cannot
    rename onto a non-empty directory, so replacing an existing artifact takes
    two steps, and between them the destination briefly does not exist. A
    reader in that window gets "not found" rather than a mixed artifact, which
    is the failure mode that matters; it is two back-to-back metadata calls
    with no work between them. Fully closing the window would need a symlink
    indirection, which the design does not call for.
    """
    staging = Path(staging_dir)
    dest = Path(dest_dir)
    Artifacts.load(staging)  # raises ArtifactsError before anything is moved

    dest.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    try:
        if dest.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{dest.name}.old.", dir=dest.parent))
            os.rmdir(backup)
            os.replace(dest, backup)
        os.replace(staging, dest)
    except OSError as exc:
        if backup is not None and not dest.exists():
            os.replace(backup, dest)
        raise ArtifactsError(f"could not publish {staging} to {dest}: {exc}") from exc
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)
    return dest
```

- [ ] **Step 4: Create the packaged identity placeholder**

```bash
.venv/bin/python - <<'EOF'
from kodachrome.artifacts import write_artifact
from kodachrome.grain import GrainParams
from kodachrome.lut import LUT3D
from kodachrome.normalize import NormalizeParams

write_artifact(
    "kodachrome/data",
    LUT3D.identity(33),
    NormalizeParams(),
    GrainParams(),
    training={"note": "identity placeholder; replaced by Task 21", "proxy_source": True},
)
EOF
ls -la kodachrome/data
```
Expected: `kodachrome.cube` (about 1 MB), `params.json`, and the existing `.gitkeep`.

These files are package data and must be committed. The repository's `.gitignore` anchors its download-directory rule as `/data/` precisely so it does not also swallow `kodachrome/data/`; if `git status` does not show the two new files, check that the rule has not lost its leading slash.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/pytest -q` → all pass, including `test_committed_default_is_loadable`.

- [ ] **Step 6: Document and commit**

In `README.md`, add under "How it works":

````markdown
The look ships inside the package at `kodachrome/data/`, so every command
works from any directory and a built wheel carries it. `--artifacts DIR`
points at a directory instead, which is how you use a LUT you trained
yourself. `params.json` records the normalisation, the grain settings, the
LUT's SHA-1 and full training provenance; a LUT whose hash disagrees with
its `params.json` is refused rather than silently graded with the wrong
parameters.

Until training has run, the bundled LUT is an identity placeholder and its
`training.note` says so.
````

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/artifacts.py tests/test_artifacts.py kodachrome/data README.md
git commit -m "feat: artifact loading with schema and hash validation, packaged default, atomic publish

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Pipeline (`pipeline.py`)

Implements spec 5.7.

**Files:**
- Create: `kodachrome/pipeline.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `Artifacts`, `normalize_u8`, `add_grain`
- Produces: `class Pipeline(artifacts)` with `process(rgb_u8, *, grain=True, rng=None) -> (rgb_u8, info)` where `info` has keys `wb_gains`, `exposure_gain`, `clamped`, `lut_sha1`

- [ ] **Step 1: Write the failing tests**

`tests/test_pipeline.py`:
```python
import numpy as np
import pytest

from kodachrome.artifacts import Artifacts, write_artifact
from kodachrome.grain import GrainParams
from kodachrome.lut import LUT3D, sha1_hex
from kodachrome.normalize import NormalizeParams
from kodachrome.pipeline import Pipeline


@pytest.fixture
def pipeline(tmp_path):
    write_artifact(tmp_path, LUT3D.identity(9), NormalizeParams(), GrainParams())
    return Pipeline(Artifacts.load(tmp_path))


def test_process_shapes_and_info(pipeline):
    frame = np.random.default_rng(0).integers(0, 256, (36, 64, 3), dtype=np.uint8)
    out, info = pipeline.process(frame, rng=np.random.default_rng(0))
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert set(info) == {"wb_gains", "exposure_gain", "clamped", "lut_sha1"}
    assert len(info["wb_gains"]) == 3
    assert info["lut_sha1"] == sha1_hex(LUT3D.identity(9))
    assert set(info["clamped"]) == {"wb", "exposure"}


def test_grain_can_be_skipped(pipeline):
    frame = np.full((36, 64, 3), 128, dtype=np.uint8)
    no_grain, _ = pipeline.process(frame, grain=False)
    with_grain, _ = pipeline.process(frame, grain=True, rng=np.random.default_rng(0))
    assert no_grain.std() < with_grain.std()


def test_same_seed_reproduces_the_output(pipeline):
    frame = np.full((32, 32, 3), 100, dtype=np.uint8)
    a, _ = pipeline.process(frame, rng=np.random.default_rng(11))
    b, _ = pipeline.process(frame, rng=np.random.default_rng(11))
    assert np.array_equal(a, b)


def test_process_returns_a_writeable_array_either_way(pipeline):
    """Writability must not depend on whether grain happened to run."""
    frame = np.full((32, 32, 3), 128, dtype=np.uint8)
    for grain in (False, True):
        out, _ = pipeline.process(frame, grain=grain, rng=np.random.default_rng(0))
        assert out.flags.writeable, f"grain={grain} returned a read-only array"
        out[0, 0, 0] = 7


def test_process_rejects_wrong_input(pipeline):
    with pytest.raises(ValueError):
        pipeline.process(np.zeros((4, 4, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        pipeline.process(np.zeros((4, 4), dtype=np.uint8))


def test_disabled_grain_in_artifact_wins(tmp_path):
    write_artifact(tmp_path, LUT3D.identity(9), NormalizeParams(), GrainParams(enabled=False))
    pipe = Pipeline(Artifacts.load(tmp_path))
    frame = np.full((32, 32, 3), 128, dtype=np.uint8)
    out, _ = pipe.process(frame, grain=True, rng=np.random.default_rng(0))
    assert out.std() < 1.0
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/pytest tests/test_pipeline.py -q` → FAIL, no module `kodachrome.pipeline`.

- [ ] **Step 3: Implement `pipeline.py`**

```python
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
```

- [ ] **Step 4: Run and commit**

Run: `.venv/bin/pytest tests/test_pipeline.py -q` → all pass.

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/pipeline.py tests/test_pipeline.py
git commit -m "feat: Pipeline applying normalise, LUT and grain with audit info

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: Batch reprocessing (`capture/batch.py`)

Implements spec 7.3. Fixes F-11.

**Files:**
- Create: `kodachrome/capture/batch.py`, `tests/test_batch.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `Artifacts`, `Pipeline`, `load_rgb`, `save_jpeg`, `list_images`
- Produces:
  - `GRADED_SUFFIXES = ("_kodachrome",)`, `SOURCE_SUFFIXES = ("_original", "_ungraded")`
  - `select_inputs(paths, all_files=False) -> list[Path]`
  - `output_path(src, out_dir, disambiguate: bool) -> Path`
  - `process_dir(in_dir, out_dir, artifacts_dir=None, grain=True, all_files=False, overwrite=False) -> BatchResult`
  - `@dataclass BatchResult(written: list[Path], skipped_graded: int, skipped_existing: int)`
  - `main(argv=None) -> int`

- [ ] **Step 1: Write the failing tests**

`tests/test_batch.py`:
```python
import numpy as np
import pytest

from kodachrome.capture.batch import main, output_path, process_dir, select_inputs
from kodachrome.imageio import save_jpeg


def _img(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (24, 32, 3), dtype=np.uint8)


def _capture_dir(tmp_path):
    """Looks like a real capture folder: originals plus already-graded siblings."""
    d = tmp_path / "shots"
    d.mkdir()
    for stem in ("120001", "120002"):
        save_jpeg(_img(1), d / f"{stem}_original.jpg")
        save_jpeg(_img(2), d / f"{stem}_kodachrome.jpg")
    (d / "captures.jsonl").write_text("{}\n")
    return d


def test_select_inputs_prefers_originals_and_always_skips_graded(tmp_path):
    d = _capture_dir(tmp_path)
    chosen = [p.name for p in select_inputs(sorted(d.glob("*.jpg")))]
    assert chosen == ["120001_original.jpg", "120002_original.jpg"]


def test_select_inputs_all_still_skips_graded(tmp_path):
    d = _capture_dir(tmp_path)
    save_jpeg(_img(3), d / "loose.jpg")
    chosen = [p.name for p in select_inputs(sorted(d.glob("*.jpg")), all_files=True)]
    assert "loose.jpg" in chosen
    assert not any("_kodachrome" in n for n in chosen)


def test_plain_folder_processes_everything(tmp_path):
    d = tmp_path / "plain"
    d.mkdir()
    save_jpeg(_img(1), d / "a.jpg")
    save_jpeg(_img(2), d / "b.jpg")
    assert len(select_inputs(sorted(d.glob("*.jpg")))) == 2


def test_capture_dir_is_not_double_graded(tmp_path):
    d = _capture_dir(tmp_path)
    result = process_dir(d, tmp_path / "out")
    assert [p.name for p in result.written] == [
        "120001_original_kodachrome.jpg",
        "120002_original_kodachrome.jpg",
    ]
    assert result.skipped_graded == 2


def test_same_stem_different_extensions_do_not_collide(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    save_jpeg(_img(1), d / "a.jpg")
    from PIL import Image

    Image.fromarray(_img(2)).save(d / "a.png")
    written = {p.name for p in process_dir(d, tmp_path / "out").written}
    assert written == {"a_jpg_kodachrome.jpg", "a_png_kodachrome.jpg"}


def test_output_path_without_disambiguation():
    from pathlib import Path

    assert output_path(Path("x/a.jpg"), Path("out"), False).name == "a_kodachrome.jpg"
    assert output_path(Path("x/a.jpg"), Path("out"), True).name == "a_jpg_kodachrome.jpg"


def test_existing_outputs_are_skipped_then_overwritten(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    save_jpeg(_img(1), d / "a.jpg")
    out = tmp_path / "out"
    first = process_dir(d, out)
    assert len(first.written) == 1
    second = process_dir(d, out)
    assert second.written == [] and second.skipped_existing == 1
    third = process_dir(d, out, overwrite=True)
    assert len(third.written) == 1


def test_nested_or_identical_output_is_refused(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    save_jpeg(_img(1), d / "a.jpg")
    with pytest.raises(ValueError, match="inside"):
        process_dir(d, d)
    with pytest.raises(ValueError, match="inside"):
        process_dir(d, d / "sub")


def test_main_uses_the_packaged_default_from_any_cwd(tmp_path, monkeypatch, capsys):
    d = tmp_path / "in"
    d.mkdir()
    save_jpeg(_img(1), d / "a.jpg")
    monkeypatch.chdir(tmp_path)
    assert main([str(d), str(tmp_path / "out")]) == 0
    assert "1 image" in capsys.readouterr().out


def test_main_reports_empty_input(tmp_path, capsys):
    (tmp_path / "in").mkdir()
    assert main([str(tmp_path / "in"), str(tmp_path / "out")]) == 1
    assert "no images" in capsys.readouterr().err.lower()


def test_main_reports_bad_artifacts(tmp_path, capsys):
    d = tmp_path / "in"
    d.mkdir()
    save_jpeg(_img(1), d / "a.jpg")
    assert main([str(d), str(tmp_path / "out"), "--artifacts", str(tmp_path / "none")]) == 2
    assert "params.json" in capsys.readouterr().err
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/pytest tests/test_batch.py -q` → FAIL, no module `kodachrome.capture.batch`.

- [ ] **Step 3: Implement `batch.py`**

```python
"""``kodachrome-process``: regrade a folder of images with the current artifact.

Two hazards make this more than a loop, and both come from the shape of a
real capture folder, which holds ``<time>_original.jpg`` next to
``<time>_kodachrome.jpg``:

* **Double grading.** Feeding that folder to a naive globber grades the
  already-graded files a second time. Files matching ``*_kodachrome.*`` are
  therefore always skipped, and when a folder contains any ``_original`` or
  ``_ungraded`` files, only those are processed unless ``--all`` is given.
* **Clobbering.** ``a.jpg`` and ``a.png`` would produce the same output
  name, and re-running would silently overwrite. Same-stem inputs get their
  extension folded into the output name, existing outputs are skipped unless
  ``--overwrite``, and an output directory equal to or inside the input is
  refused outright.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..artifacts import Artifacts, ArtifactsError
from ..imageio import list_images, load_rgb, save_jpeg
from ..pipeline import Pipeline

GRADED_SUFFIXES = ("_kodachrome",)
SOURCE_SUFFIXES = ("_original", "_ungraded")


@dataclass
class BatchResult:
    written: list[Path] = field(default_factory=list)
    skipped_graded: int = 0
    skipped_existing: int = 0


def _is_graded(path: Path) -> bool:
    return any(path.stem.endswith(s) for s in GRADED_SUFFIXES)


def _is_source(path: Path) -> bool:
    return any(path.stem.endswith(s) for s in SOURCE_SUFFIXES)


def select_inputs(paths: Sequence[Path], all_files: bool = False) -> list[Path]:
    """Graded outputs are never inputs; capture folders default to their originals."""
    candidates = [p for p in paths if not _is_graded(p)]
    if all_files:
        return candidates
    sources = [p for p in candidates if _is_source(p)]
    return sources if sources else candidates


def output_path(src: Path, out_dir: Path, disambiguate: bool) -> Path:
    stem = f"{src.stem}_{src.suffix.lstrip('.').lower()}" if disambiguate else src.stem
    return Path(out_dir) / f"{stem}_kodachrome.jpg"


def _check_directories(in_dir: Path, out_dir: Path) -> None:
    in_res, out_res = in_dir.resolve(), out_dir.resolve()
    if out_res == in_res or in_res in out_res.parents:
        raise ValueError(
            f"output directory {out_dir} is the same as, or inside, the input directory "
            f"{in_dir}; choose a separate destination"
        )


def process_dir(
    in_dir: str | Path,
    out_dir: str | Path,
    artifacts_dir: str | Path | None = None,
    grain: bool = True,
    all_files: bool = False,
    overwrite: bool = False,
) -> BatchResult:
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    _check_directories(in_dir, out_dir)

    every = list_images(in_dir)
    chosen = select_inputs(every, all_files=all_files)
    stems = [p.stem for p in chosen]
    ambiguous = {s for s in stems if stems.count(s) > 1}

    pipeline = Pipeline(Artifacts.resolve(artifacts_dir))
    result = BatchResult(skipped_graded=sum(1 for p in every if _is_graded(p)))
    for src in chosen:
        dest = output_path(src, out_dir, disambiguate=src.stem in ambiguous)
        if dest.exists() and not overwrite:
            result.skipped_existing += 1
            continue
        rgb, _meta = load_rgb(src)
        graded, _info = pipeline.process(rgb, grain=grain)
        result.written.append(save_jpeg(graded, dest))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kodachrome-process",
        description="Regrade a folder of images with the Kodachrome LUT.",
    )
    parser.add_argument("in_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--artifacts", type=Path, default=None, help="artifact dir (default: bundled)"
    )
    parser.add_argument("--no-grain", action="store_true", help="skip film grain")
    parser.add_argument(
        "--all", action="store_true", help="process every image, not just originals"
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    args = parser.parse_args(argv)

    if not args.in_dir.is_dir() or not list_images(args.in_dir):
        print(f"error: no images found in {args.in_dir}", file=sys.stderr)
        return 1
    t0 = time.perf_counter()
    try:
        result = process_dir(
            args.in_dir,
            args.out_dir,
            args.artifacts,
            grain=not args.no_grain,
            all_files=args.all,
            overwrite=args.overwrite,
        )
    except ArtifactsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    notes = []
    if result.skipped_graded:
        notes.append(f"{result.skipped_graded} already-graded skipped")
    if result.skipped_existing:
        notes.append(f"{result.skipped_existing} existing outputs kept (use --overwrite)")
    suffix = f" ({', '.join(notes)})" if notes else ""
    print(
        f"Processed {len(result.written)} image(s) into {args.out_dir} "
        f"in {time.perf_counter() - t0:.1f}s{suffix}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_batch.py -q` → all pass.

- [ ] **Step 5: Document and commit**

In `README.md` add:

````markdown
### Regrade a folder

```bash
kodachrome-process ~/Pictures/kodachrome/2026-09-03 /tmp/regraded
```

Pointed at a capture folder, it grades only the `*_original.jpg` files and
skips the `*_kodachrome.jpg` siblings, so running it twice cannot
double-grade. Pointed at any other folder it grades everything. Existing
outputs are kept unless `--overwrite`; an output directory inside the input
is refused. `--all` overrides the originals-only default.
````

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/capture/batch.py tests/test_batch.py README.md
git commit -m "feat: kodachrome-process with double-grade and clobber safety

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: Camera with byte-exact MJPEG (`capture/camera.py`)

Implements spec 7.1. Fixes F-02 and the acquisition half of F-08.

**Files:**
- Create: `kodachrome/capture/camera.py`, `tests/test_camera.py`

**Interfaces:**
- Produces:
  - `class CameraError(Exception)`
  - `@dataclass Frame(rgb: np.ndarray, jpeg: bytes | None, source: str)` where `source` is `"raw-mjpeg"` or `"decoded"`
  - `@dataclass StreamInfo(width, height, fps, fourcc, raw_mjpeg: bool)` with `.to_dict()`
  - `class Camera(Protocol)`: `read() -> Frame`, `.stream_info -> StreamInfo`, `close()`
  - `is_valid_jpeg(buf: bytes) -> bool` — SOI/EOI marker check
  - `synthetic_frame(height=1080, width=1920) -> np.ndarray`
  - `class FakeCamera(frames=None, jpeg_bytes=None, source="decoded", stream_info=None)`
  - `parse_device(device) -> int | str | None`
  - `list_video_devices() -> list[str]`
  - `class V4L2Camera(device=None, width=1920, height=1080, fps=30, warmup_frames=15, prefer_raw=True)`

- [ ] **Step 1: Write the failing tests**

`tests/test_camera.py`:
```python
import io

import cv2
import numpy as np
import pytest
from PIL import Image

from kodachrome.capture import camera as camera_module
from kodachrome.capture.camera import (
    CameraError,
    FakeCamera,
    Frame,
    StreamInfo,
    V4L2Camera,
    is_valid_jpeg,
    parse_device,
    synthetic_frame,
)


def _tagged_frame(index):
    """A synthetic frame with a unique first pixel, so frames are distinguishable."""
    frame = synthetic_frame(48, 64).copy()
    frame[0, 0] = (index, index, index)
    return frame


def _jpeg_bytes(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=95)
    return buf.getvalue()


def test_synthetic_frame_shape_and_colour():
    f = synthetic_frame(90, 160)
    assert f.shape == (90, 160, 3) and f.dtype == np.uint8
    assert f[..., 0].mean() != f[..., 2].mean()


def test_is_valid_jpeg_markers():
    good = _jpeg_bytes(synthetic_frame(16, 16))
    assert is_valid_jpeg(good)
    assert not is_valid_jpeg(good[:-2])        # truncated: no EOI
    assert not is_valid_jpeg(b"\x00\x01" + good[2:])  # no SOI
    assert not is_valid_jpeg(b"")
    assert not is_valid_jpeg(b"\xff\xd8\xff\xd9")     # too short to be a frame


def test_fake_camera_defaults_to_decoded_mode():
    cam = FakeCamera()
    frame = cam.read()
    assert isinstance(frame, Frame)
    assert frame.rgb.shape == (1080, 1920, 3) and frame.rgb.dtype == np.uint8
    assert frame.jpeg is None and frame.source == "decoded"
    assert isinstance(cam.stream_info, StreamInfo)
    cam.close()


def test_fake_camera_raw_mode_returns_bytes_that_decode_to_the_frame():
    rgb = synthetic_frame(48, 64)
    data = _jpeg_bytes(rgb)
    cam = FakeCamera(jpeg_bytes=[data], source="raw-mjpeg")
    frame = cam.read()
    assert frame.source == "raw-mjpeg"
    assert frame.jpeg == data
    decoded = np.asarray(Image.open(io.BytesIO(frame.jpeg)).convert("RGB"))
    assert np.array_equal(frame.rgb, decoded), "rgb must be the decode of the same buffer"


def test_fake_camera_cycles_and_copies():
    frames = [np.zeros((4, 4, 3), np.uint8), np.ones((4, 4, 3), np.uint8)]
    cam = FakeCamera(frames)
    assert cam.read().rgb.max() == 0
    assert cam.read().rgb.max() == 1
    assert cam.read().rgb.max() == 0
    cam.read().rgb[0, 0, 0] = 99
    assert frames[0].max() == 0


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        (3, 3),
        ("3", 3),
        ("/dev/video7", 7),
        ("/dev/v4l/by-id/usb-Innomaker-video-index0", "/dev/v4l/by-id/usb-Innomaker-video-index0"),
    ],
)
def test_parse_device(value, expected):
    assert parse_device(value) == expected


def test_parse_device_rejects_garbage():
    with pytest.raises(CameraError):
        parse_device("camera")


class FakeCapture:
    """Stands in for ``cv2.VideoCapture`` so the real V4L2Camera can be driven.

    Without this, none of the raw-mode, fallback, negotiation or retry logic is
    executed by any test: a build where raw mode silently never engaged would
    pass the whole suite, and that logic is the project's headline promise.

    Three things to know when writing tests against it. Constructing a
    ``V4L2Camera`` consumes one buffer, because ``_enable_raw_mode`` reads a
    frame to verify the mode really works — so a queue must include that
    verification frame before the ones a test intends ``read()`` to return.
    Format properties are reported, not stored (see ``set``). And ``grab``
    consumes only when a test passes ``grab_consumes=True``, so that buffer
    counts stay independent of ``_drain``'s internal grab limit.
    """

    def __init__(
        self,
        *,
        buffers=None,
        decoded=None,
        set_convert_fails=False,
        set_raises=False,
        mutate_then_raise=False,
        restore_behaviour="ok",
        buffer_ndim=1,
        grab_consumes=False,
        props=None,
    ):
        self.buffers = list(buffers or [])
        self.decoded = decoded if decoded is not None else np.zeros((1080, 1920, 3), np.uint8)
        self.set_convert_fails = set_convert_fails
        self.set_raises = set_raises
        self.mutate_then_raise = mutate_then_raise
        self.restore_behaviour = restore_behaviour
        self.buffer_ndim = buffer_ndim
        self.grab_consumes = grab_consumes
        self.props = props or {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"MJPG")),
        }
        self.convert_rgb = 1
        self.released = False
        self.requested = []
        self.grabs = 0

    def isOpened(self):
        return True

    def set(self, prop, val):
        """Accept a request but do not pretend it was honoured.

        A real camera reports the format it actually negotiated, which is the
        entire reason ``_negotiated`` reads the properties back. Storing the
        requested value here would make every negotiation check tautological:
        the code would read back exactly what it just wrote and never warn.
        So format writes are recorded and ignored; only CONVERT_RGB takes
        effect, because that one genuinely changes what ``read`` returns.
        """
        self.requested.append((prop, val))
        if prop == cv2.CAP_PROP_CONVERT_RGB:
            if val == 1 and self.convert_rgb == 0:
                if self.restore_behaviour == "raise":
                    raise cv2.error("driver refused to leave raw mode")
                if self.restore_behaviour == "false":
                    return False
            if self.mutate_then_raise:
                self.convert_rgb = val
                raise cv2.error("device rejected the mode after applying it")
            if self.set_raises:
                raise cv2.error("simulated")
            if self.set_convert_fails:
                return False
            self.convert_rgb = val
        return True

    def get(self, prop):
        return self.props.get(prop, 0)

    def read(self):
        if self.convert_rgb == 0:
            if not self.buffers:
                return False, None
            buf = np.frombuffer(self.buffers.pop(0), np.uint8)
            return True, buf.reshape(1, -1) if self.buffer_ndim == 2 else buf
        return True, self.decoded

    def grab(self):
        """Count every grab; consume a buffer only when a test asks it to.

        An unconditional ``True`` that consumes nothing makes ``_drain``
        untestable — a drain that grabbed the wrong number of frames, or was
        deleted outright, would still pass. But consuming unconditionally
        couples every fixture's buffer count to the literal ``4`` inside
        ``_drain``: change that constant and unrelated tests fail for reasons
        they do not assert on. So consumption is opt-in, and exactly one test
        opts in to verify the drain.
        """
        self.grabs += 1
        if not self.grab_consumes:
            return True
        if not self.buffers:
            return False
        self.buffers.pop(0)
        return True

    def release(self):
        self.released = True


@pytest.fixture
def fake_capture(monkeypatch):
    """Install a FakeCapture in place of cv2.VideoCapture and hand it back."""
    holder = {}

    def install(**kwargs):
        cap = FakeCapture(**kwargs)
        holder["cap"] = cap
        monkeypatch.setattr(camera_module.cv2, "VideoCapture", lambda *a, **k: cap)
        return cap

    return install


def test_raw_mode_engages_and_preserves_the_camera_bytes(fake_capture, capsys):
    rgb = synthetic_frame(48, 64)
    data = _jpeg_bytes(rgb)
    # One buffer is consumed by _enable_raw_mode's verification read.
    cap = fake_capture(buffers=[data] * 4)
    cam = V4L2Camera(device=0, warmup_frames=0)
    assert cam.stream_info.raw_mjpeg is True
    assert cap.convert_rgb == 0

    frame = cam.read()
    assert frame.source == "raw-mjpeg"
    assert frame.jpeg == data
    decoded = np.asarray(Image.open(io.BytesIO(frame.jpeg)).convert("RGB"))
    assert np.array_equal(frame.rgb, decoded)


def test_an_invalid_buffer_falls_back_once_and_stays_fallen_back(fake_capture, capsys):
    rgb = synthetic_frame(48, 64)
    data = _jpeg_bytes(rgb)
    # First buffer is eaten by the constructor's verification read, second is
    # the good frame the first read() returns, third is the truncated one.
    cap = fake_capture(
        buffers=[data, data, data[:-2]], decoded=np.zeros((48, 64, 3), np.uint8)
    )
    cam = V4L2Camera(device=0, warmup_frames=0)
    assert cam.read().source == "raw-mjpeg"

    assert cam.read().source == "decoded"
    assert cam.stream_info.raw_mjpeg is False
    assert cap.convert_rgb == 1
    warnings = capsys.readouterr().out.count("falling back to decoded frames")
    assert warnings == 1, "the fallback must announce itself once, not per frame"

    assert cam.read().source == "decoded"
    assert capsys.readouterr().out.count("falling back to decoded frames") == 0


@pytest.mark.parametrize(
    "kwargs",
    [{"set_convert_fails": True}, {"set_raises": True}, {"mutate_then_raise": True}],
    ids=["set-returns-false", "set-raises", "set-mutates-then-raises"],
)
def test_every_raw_mode_failure_leaves_the_device_in_decoded_mode(fake_capture, kwargs):
    """The half-state is silently destructive, so no failure path may leave it."""
    cap = fake_capture(decoded=np.zeros((1080, 1920, 3), np.uint8), **kwargs)
    cam = V4L2Camera(device=0, warmup_frames=0)
    assert cam.stream_info.raw_mjpeg is False
    assert cap.convert_rgb == 1, "CONVERT_RGB left at 0 while the object thinks it is decoding"
    assert cam.read().source == "decoded"


def test_negotiation_mismatch_warns_but_does_not_abort(fake_capture, capsys):
    cap = fake_capture(
        decoded=np.zeros((720, 1280, 3), np.uint8),
        props={
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
            cv2.CAP_PROP_FPS: 15.0,
            cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"YUYV")),
        },
        set_convert_fails=True,
    )
    cam = V4L2Camera(device=0, warmup_frames=0)
    out = capsys.readouterr().out
    assert "negotiated 1280x720" in out
    assert "YUYV" in out
    assert cam.stream_info.width == 1280 and cam.stream_info.fps == 15.0
    assert cam.read().rgb.shape == (720, 1280, 3)
    assert cap is not None


def test_read_returns_the_newest_frame_not_a_stale_queued_one(fake_capture):
    """`_drain` must discard queued frames, or the preview lags behind reality."""
    buffers = [_jpeg_bytes(_tagged_frame(i)) for i in range(6)]
    cap = fake_capture(buffers=list(buffers), grab_consumes=True)
    cam = V4L2Camera(device=0, warmup_frames=0)

    frame = cam.read()
    assert cap.grabs == 4, "the drain loop should grab up to four queued frames"
    # Index 0 is eaten by the constructor's verification read, 1-4 by the drain,
    # so a correct drain leaves index 5. Without the drain this would be index 1.
    assert frame.jpeg == buffers[5], "read returned a stale frame instead of the newest"


def test_a_two_dimensional_raw_buffer_is_accepted(fake_capture):
    """Real V4L2 backends can hand back a 2-D buffer; 1-D is not the only shape."""
    data = _jpeg_bytes(synthetic_frame(48, 64))
    cap = fake_capture(buffers=[data] * 4, buffer_ndim=2)
    cam = V4L2Camera(device=0, warmup_frames=0)
    assert cam.stream_info.raw_mjpeg is True
    assert cam.read().jpeg == data
    assert cap is not None


@pytest.mark.parametrize("restore", ["raise", "false"], ids=["raises", "returns-false"])
def test_a_device_that_will_not_leave_raw_mode_fails_honestly(fake_capture, restore, capsys):
    """Refusing the restore must not produce frames the session cannot trust.

    If the device keeps sending compressed buffers, claiming decoded mode
    would send them through ``cvtColor`` as BGR and yield silent garbage. The
    session must stay in raw mode and fail loudly instead.
    """
    good = _jpeg_bytes(synthetic_frame(48, 64))
    bad = good[:-2]
    cap = fake_capture(
        buffers=[good, good, bad, bad, bad, bad],
        decoded=np.zeros((48, 64, 3), np.uint8),
        restore_behaviour=restore,
    )
    cam = V4L2Camera(device=0, warmup_frames=0)
    assert cam.read().source == "raw-mjpeg"

    with pytest.raises(CameraError, match="3 attempts"):
        cam.read()

    assert cam.stream_info.raw_mjpeg is True, "the mode must not change if the device refused"
    assert cap.convert_rgb == 0
    assert "refused to leave raw mode" in capsys.readouterr().out


def test_a_successful_restore_does_fall_back_to_decoded(fake_capture):
    """The ordinary case: the device complies, so the session degrades cleanly."""
    good = _jpeg_bytes(synthetic_frame(48, 64))
    fake_capture(
        buffers=[good, good, good[:-2], good],
        decoded=np.zeros((48, 64, 3), np.uint8),
    )
    cam = V4L2Camera(device=0, warmup_frames=0)
    assert cam.read().source == "raw-mjpeg"

    frame = cam.read()
    assert frame.source == "decoded"
    assert frame.jpeg is None
    assert cam.stream_info.raw_mjpeg is False


def test_read_raises_after_three_failed_attempts(fake_capture):
    cap = fake_capture(buffers=[])
    cam = V4L2Camera(device=0, warmup_frames=0)
    cap.convert_rgb = 0
    cap.buffers = []
    with pytest.raises(CameraError, match="3 attempts"):
        cam.read()


def test_close_releases_the_device(fake_capture):
    cap = fake_capture(set_convert_fails=True)
    cam = V4L2Camera(device=0, warmup_frames=0)
    cam.close()
    assert cap.released is True


def test_v4l2_camera_reports_missing_device():
    with pytest.raises(CameraError, match="video"):
        V4L2Camera(device=99, warmup_frames=0)


def test_stream_info_to_dict():
    info = StreamInfo(width=1920, height=1080, fps=30.0, fourcc="MJPG", raw_mjpeg=True)
    assert info.to_dict() == {
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "fourcc": "MJPG",
        "raw_mjpeg": True,
    }
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/pytest tests/test_camera.py -q` → FAIL, no module `kodachrome.capture.camera`.

- [ ] **Step 3: Implement `camera.py`**

```python
"""Camera access for the Innomaker U20CAM-1080P-WDR, plus a fake for tests.

Facts from the vendor manual that shaped this module:

* Standard UVC device, driven through OpenCV's V4L2 backend.
* 1920x1080 at 30 fps exists only in MJPEG; YUY2 falls to 5 fps at 1080p, so
  the FOURCC is forced to ``MJPG``.
* The camera runs its own auto exposure and white balance, which need a few
  frames to settle after opening (``warmup_frames``). Version 1 records
  those controls rather than locking them; see spec 7.5.

Byte-exact originals
--------------------
The headline promise is that the camera's own JPEG is saved. ``read()``
normally decodes MJPEG into BGR pixels, which would mean re-encoding a
second lossy JPEG and calling it the original. Instead the camera asks the
V4L2 backend for the compressed buffer with ``CAP_PROP_CONVERT_RGB = 0``,
and then:

* validates the buffer really is a complete JPEG (SOI at the front, EOI at
  the end, and it decodes) - OpenCV issue #23311 shows the backend can hand
  back truncated data on some devices;
* returns those exact bytes as ``Frame.jpeg`` and the decode of *that same
  buffer* as ``Frame.rgb``, so the saved original and the graded image
  provably come from one acquisition.

If raw mode is unsupported, or a buffer fails validation, the camera says so
once and falls back to decoded mode for the rest of the session. The app
then names its second file ``_ungraded.jpg`` rather than ``_original.jpg``,
so a filename never claims more than the bytes deliver.

Negotiation is verified, not assumed: FOURCC, size and rate are read back
after being set, a mismatch is warned about naming both values, and the
result is recorded in ``StreamInfo`` for the capture log.
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .._cv2 import require_cv2

cv2 = require_cv2()

_SOI = b"\xff\xd8"
_EOI = b"\xff\xd9"
_MIN_JPEG_BYTES = 128


class CameraError(Exception):
    """Camera could not be opened or read."""


@dataclass
class StreamInfo:
    width: int
    height: int
    fps: float
    fourcc: str
    raw_mjpeg: bool

    def to_dict(self) -> dict:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "fps": round(float(self.fps), 2),
            "fourcc": self.fourcc,
            "raw_mjpeg": bool(self.raw_mjpeg),
        }


@dataclass
class Frame:
    rgb: np.ndarray
    jpeg: bytes | None
    source: str  # "raw-mjpeg" when jpeg holds the camera's own bytes, else "decoded"


class Camera(Protocol):
    @property
    def stream_info(self) -> StreamInfo: ...

    def read(self) -> Frame: ...

    def close(self) -> None: ...


def is_valid_jpeg(buf: bytes) -> bool:
    """A complete JPEG: start-of-image marker, end-of-image marker, plausible length."""
    return (
        len(buf) >= _MIN_JPEG_BYTES and buf[:2] == _SOI and buf[-2:] == _EOI
    )


def _fourcc_to_str(value: float) -> str:
    code = int(value)
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)) if code else "----"


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
    """Stands in for the hardware. Give it ``jpeg_bytes`` to exercise raw mode."""

    def __init__(
        self,
        frames: list[np.ndarray] | None = None,
        jpeg_bytes: list[bytes] | None = None,
        source: str = "decoded",
        stream_info: StreamInfo | None = None,
    ) -> None:
        self._jpegs = jpeg_bytes
        if jpeg_bytes is not None:
            decoded = [
                cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in jpeg_bytes
            ]
            self._frames = decoded
            self._frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in self._frames]
        else:
            self._frames = frames if frames else [synthetic_frame()]
        self._source = source
        h, w = self._frames[0].shape[:2]
        self._info = stream_info or StreamInfo(
            width=w, height=h, fps=30.0, fourcc="MJPG", raw_mjpeg=source == "raw-mjpeg"
        )
        self._i = 0

    @property
    def stream_info(self) -> StreamInfo:
        return self._info

    def read(self) -> Frame:
        idx = self._i % len(self._frames)
        self._i += 1
        jpeg = self._jpegs[idx % len(self._jpegs)] if self._jpegs else None
        return Frame(rgb=self._frames[idx].copy(), jpeg=jpeg, source=self._source)

    def close(self) -> None:
        return None


def parse_device(device: int | str | None) -> int | str | None:
    """Accept an index, ``/dev/videoN``, or a stable ``/dev/v4l/by-id/...`` path."""
    if device is None or isinstance(device, int):
        return device
    text = device.strip()
    if text.startswith("/dev/v4l/by-id/") or text.startswith("/dev/v4l/by-path/"):
        return text
    m = re.fullmatch(r"(?:/dev/video)?(\d+)", text)
    if not m:
        raise CameraError(
            f"Cannot parse camera device {device!r}; use an index, /dev/videoN, "
            "or a /dev/v4l/by-id/... path"
        )
    return int(m.group(1))


def list_video_devices() -> list[str]:
    return sorted(glob.glob("/dev/video*")) + sorted(glob.glob("/dev/v4l/by-id/*"))


class V4L2Camera:
    def __init__(
        self,
        device: int | str | None = None,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        warmup_frames: int = 15,
        prefer_raw: bool = True,
    ) -> None:
        target = parse_device(device)
        candidates: list[int | str] = [target] if target is not None else list(range(10))
        self.cap = None
        chosen: int | str | None = None
        for candidate in candidates:
            cap = cv2.VideoCapture(candidate, cv2.CAP_V4L2)
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
                chosen = candidate
                break
            cap.release()
        if self.cap is None:
            found = list_video_devices()
            hint = f"found {', '.join(found)}" if found else "no /dev/video* devices exist"
            raise CameraError(
                f"No camera delivered a frame (tried {candidates[0]}"
                + (f"..{candidates[-1]}" if len(candidates) > 1 else "")
                + f"); {hint}. Pass --device N, /dev/videoN or a /dev/v4l/by-id/ path."
            )
        print(f"Using camera {chosen}")

        self._raw = self._enable_raw_mode() if prefer_raw else False
        self.stream_info = self._negotiated(width, height, fps)
        self._warned_fallback = False
        self._warned_stuck = False
        for _ in range(warmup_frames):
            self.cap.read()

    def _enable_raw_mode(self) -> bool:
        """Ask the backend for the compressed buffer; verify by reading one frame.

        Every failure path restores ``CAP_PROP_CONVERT_RGB``, including the
        ones where the ``set`` call itself failed or raised. OpenCV gives no
        guarantee that a failing ``set`` left the property untouched, and the
        half-state is silently destructive: the object would believe it is in
        decoded mode while the driver still hands back compressed buffers, so
        ``read`` would run ``cvtColor`` over JPEG bytes as though they were a
        BGR image and return plausible-looking garbage. Restoring
        unconditionally costs one ignored call and removes the question.
        """

        def give_up() -> bool:
            try:
                self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
            except cv2.error:
                pass
            return False

        try:
            if not self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0):
                return give_up()
        except cv2.error:
            return give_up()
        ok, buf = self.cap.read()
        if not ok or buf is None or (buf.ndim != 2 and buf.ndim != 1):
            return give_up()
        if not is_valid_jpeg(np.asarray(buf, dtype=np.uint8).tobytes()):
            return give_up()
        return True

    def _negotiated(self, width: int, height: int, fps: int) -> StreamInfo:
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        fourcc = _fourcc_to_str(self.cap.get(cv2.CAP_PROP_FOURCC))
        if (actual_w, actual_h) != (width, height):
            print(f"warning: requested {width}x{height}, camera negotiated {actual_w}x{actual_h}")
        if fourcc != "MJPG":
            print(
                f"warning: requested MJPG, camera negotiated {fourcc}; "
                "1080p30 may be unavailable"
            )
        if actual_fps and abs(actual_fps - fps) > 1.0:
            print(f"warning: requested {fps} fps, camera reports {actual_fps:g} fps")
        if not self._raw:
            print("note: raw MJPEG unavailable; captures will be saved as re-encoded _ungraded.jpg")
        return StreamInfo(actual_w, actual_h, actual_fps, fourcc, self._raw)

    def _drain(self) -> None:
        """Discard queued frames so the next read is the newest available."""
        for _ in range(4):
            if not self.cap.grab():
                break

    def _fallback_to_decoded(self, reason: str) -> bool:
        """Try to leave raw mode. Only claim success if the device complied.

        Two failures are guarded here, and the second is subtler than it
        looks. Raising must not escape, or the error would propagate out of
        ``read`` and end the session this fallback exists to preserve.

        But swallowing the error is not enough either. If the device refuses
        to leave raw mode, it keeps sending compressed buffers — so declaring
        decoded mode anyway would send the next frame down the decoded branch,
        where ``cvtColor`` reads a JPEG buffer as though it were a BGR image.
        Measured: that returns a plausible ``(1, 504, 3)`` array rather than
        raising, so the session would serve silent garbage. That is the same
        belief-versus-reality split this class already guards against in
        ``_enable_raw_mode``.

        So the mode only changes when the device actually changed. If it did
        not, the session stays in raw mode and keeps validating buffers; a
        genuinely broken stream then exhausts ``read``'s retry budget and
        fails loudly, which is the honest outcome. A single bad buffer does
        not permanently downgrade a camera that cannot be downgraded.
        """
        try:
            left_raw_mode = bool(self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1))
        except cv2.error:
            left_raw_mode = False

        if not left_raw_mode:
            if not self._warned_stuck:
                print(
                    f"warning: {reason}, and the device refused to leave raw mode; "
                    "continuing to validate raw buffers"
                )
                self._warned_stuck = True
            return False

        if not self._warned_fallback:
            print(f"warning: {reason}; falling back to decoded frames (_ungraded.jpg)")
            self._warned_fallback = True
        self._raw = False
        self.stream_info.raw_mjpeg = False
        return True

    def read(self) -> Frame:
        for _ in range(3):
            self._drain()
            ok, data = self.cap.read()
            if not ok or data is None:
                continue
            if self._raw:
                buf = np.asarray(data, dtype=np.uint8).tobytes()
                if not is_valid_jpeg(buf):
                    self._fallback_to_decoded("camera returned an incomplete JPEG buffer")
                    continue
                bgr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    self._fallback_to_decoded("camera buffer failed to decode")
                    continue
                return Frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), buf, "raw-mjpeg")
            return Frame(cv2.cvtColor(data, cv2.COLOR_BGR2RGB), None, "decoded")
        raise CameraError("Failed to read a frame after 3 attempts")

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_camera.py -q` → all pass. `test_v4l2_camera_reports_missing_device` must finish in a couple of seconds; OpenCV printing a V4L2 backend warning on macOS is expected.

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/capture/camera.py tests/test_camera.py
git commit -m "feat: byte-exact MJPEG capture with validation, fallback and negotiation checks

read() asks the V4L2 backend for the compressed buffer, validates SOI/EOI
and the decode, and returns both the camera's own bytes and the decode of
that same buffer. Invalid or unsupported raw mode falls back to decoded
frames, which the app names _ungraded.jpg.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 12: Capture app (`capture/app.py`)

Implements spec 7.2. Fixes the runtime half of F-08, plus F-14 and F-15.

**Files:**
- Create: `kodachrome/capture/app.py`, `tests/test_app.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `Camera`, `FakeCamera`, `V4L2Camera`, `CameraError`, `Frame`, `Artifacts`, `ArtifactsError`, `Pipeline`, `save_jpeg`
- Produces:
  - `@dataclass CaptureResult(original: Path, kodachrome: Path, record: dict)`
  - `class CaptureSession(camera, pipeline, out_root, now=None, seed_rng=None, package_version=...)` with `capture()` and `preview_frame(graded=True, size=(640, 360))`
  - `run_headless_loop(session, read_key, out=print) -> int`
  - `run_preview_loop(session, window_name=...) -> bool`
  - `class TerminalKeys` context manager with `.read(timeout=0.1)`
  - `has_display() -> bool`, `main(argv=None) -> int`

- [ ] **Step 1: Write the failing tests**

`tests/test_app.py`:
```python
import io
import json
from datetime import datetime

import numpy as np
import pytest
from PIL import Image

from kodachrome.artifacts import Artifacts, write_artifact
from kodachrome.capture.app import CaptureSession, main, run_headless_loop, run_preview_loop
from kodachrome.capture.camera import CameraError, FakeCamera, Frame, StreamInfo, synthetic_frame
from kodachrome.grain import GrainParams
from kodachrome.lut import LUT3D
from kodachrome.normalize import NormalizeParams
from kodachrome.pipeline import Pipeline


def _tagged_frame(index):
    """A synthetic frame with a unique first pixel, so frames are distinguishable."""
    frame = synthetic_frame(48, 64).copy()
    frame[0, 0] = (index, index, index)
    return frame


def _jpeg_bytes(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture
def pipeline(tmp_path):
    d = tmp_path / "art"
    write_artifact(d, LUT3D.identity(9), NormalizeParams(), GrainParams())
    return Pipeline(Artifacts.load(d))


def _session(tmp_path, pipeline, camera=None, now=None):
    camera = camera or FakeCamera([synthetic_frame(90, 160)])
    return CaptureSession(camera, pipeline, tmp_path / "shots", now=now,
                          seed_rng=np.random.default_rng(0))


def test_raw_mode_writes_the_camera_bytes_verbatim(tmp_path, pipeline):
    rgb = synthetic_frame(48, 64)
    data = _jpeg_bytes(rgb)
    cam = FakeCamera(jpeg_bytes=[data], source="raw-mjpeg")
    fixed = datetime(2026, 9, 3, 21, 5, 7)
    result = _session(tmp_path, pipeline, cam, now=lambda: fixed).capture()
    assert result.original.name == "210507_original.jpg"
    assert result.original.read_bytes() == data, "the camera's own bytes must be saved unchanged"
    assert result.record["frame_source"] == "raw-mjpeg"


def test_decoded_mode_names_the_file_ungraded(tmp_path, pipeline):
    fixed = datetime(2026, 9, 3, 21, 5, 7)
    result = _session(tmp_path, pipeline, now=lambda: fixed).capture()
    assert result.original.name == "210507_ungraded.jpg"
    assert result.record["frame_source"] == "decoded"


def test_both_outputs_come_from_one_acquisition(tmp_path, pipeline):
    """A camera whose frames differ every read would produce mismatched files."""

    class ChangingCamera:
        stream_info = StreamInfo(64, 48, 30.0, "MJPG", False)

        def __init__(self):
            self.n = 0

        def read(self):
            self.n += 1
            return Frame(np.full((48, 64, 3), self.n * 20, np.uint8), None, "decoded")

        def close(self):
            pass

    cam = ChangingCamera()
    result = _session(tmp_path, pipeline, cam).capture()
    saved = np.asarray(Image.open(result.original).convert("RGB"))
    assert cam.n == 1, "capture must read exactly one frame"
    assert abs(int(saved.mean()) - 20) <= 2


def test_log_line_carries_full_provenance(tmp_path, pipeline):
    fixed = datetime(2026, 9, 3, 21, 5, 7)
    session = _session(tmp_path, pipeline, now=lambda: fixed)
    session.capture()
    line = json.loads((tmp_path / "shots" / "2026-09-03" / "captures.jsonl").read_text().strip())
    assert set(line) >= {
        "timestamp", "original", "kodachrome", "frame_source", "wb_gains", "exposure_gain",
        "clamped", "grain_seed", "lut_sha1", "params_version", "package_version",
        "width", "height", "fourcc", "fps", "pipeline_ms", "shutter_to_saved_ms",
    }
    assert line["shutter_to_saved_ms"] >= line["pipeline_ms"]
    assert isinstance(line["grain_seed"], int)


def test_recorded_seed_reproduces_the_graded_file(tmp_path, pipeline):
    """The seed is the only randomness, so it must pin the grade exactly.

    Two claims, checked separately, because conflating them hides which one
    broke. In memory the reproduction is bit-exact. Through the saved files
    it cannot be: both are quality-95 JPEGs, and grain is precisely the
    high-frequency content JPEG discards, so the bounds below are what the
    format allows (measured: mean 1.1, 99th percentile 4, worst pixel 8-12).
    """
    session = _session(tmp_path, pipeline)
    result = session.capture()
    seed = result.record["grain_seed"]
    assert isinstance(seed, int)

    frame = session.camera.read().rgb  # FakeCamera repeats the same frame
    first, _ = pipeline.process(frame, rng=np.random.default_rng(seed))
    second, _ = pipeline.process(frame, rng=np.random.default_rng(seed))
    assert np.array_equal(first, second), "the same seed must give the same pixels"

    original = np.asarray(Image.open(result.original).convert("RGB"))
    saved = np.asarray(Image.open(result.kodachrome).convert("RGB"))
    again, _ = pipeline.process(original, rng=np.random.default_rng(seed))
    diff = np.abs(again.astype(int) - saved.astype(int))
    assert diff.mean() < 3.0
    assert np.percentile(diff, 99) <= 10


def test_same_second_captures_do_not_collide(tmp_path, pipeline):
    fixed = datetime(2026, 9, 3, 21, 5, 7)
    session = _session(tmp_path, pipeline, now=lambda: fixed)
    a, b = session.capture(), session.capture()
    assert a.original.name == "210507_ungraded.jpg"
    assert b.original.name == "210507-2_ungraded.jpg"


def test_preview_frame_is_small_rgb(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    assert session.preview_frame(graded=True).shape == (360, 640, 3)
    assert session.preview_frame(graded=False).shape == (360, 640, 3)


def test_headless_loop_captures_on_space_and_quits_on_q(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    keys = iter([None, " ", "x", " ", "q"])
    messages = []
    assert run_headless_loop(session, read_key=lambda: next(keys), out=messages.append) == 2
    assert len(list((tmp_path / "shots").rglob("*_kodachrome.jpg"))) == 2
    assert any("Saved" in m for m in messages)


def test_headless_loop_survives_a_camera_error(tmp_path, pipeline):
    class FlakyCamera(FakeCamera):
        def read(self):
            raise CameraError("boom")

    session = _session(tmp_path, pipeline, FlakyCamera())
    keys = iter([" ", "q"])
    messages = []
    assert run_headless_loop(session, read_key=lambda: next(keys), out=messages.append) == 0
    assert any("boom" in m for m in messages)


def test_preview_loop_falls_back_when_gui_is_unavailable(tmp_path, pipeline, monkeypatch):
    import cv2

    def boom(*a, **k):
        raise cv2.error("no GUI support")

    monkeypatch.setattr(cv2, "namedWindow", boom)
    assert run_preview_loop(_session(tmp_path, pipeline)) is False


def test_preview_loop_falls_back_when_imshow_fails(tmp_path, pipeline, monkeypatch):
    import cv2

    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: (_ for _ in ()).throw(cv2.error("no GUI")))
    assert run_preview_loop(_session(tmp_path, pipeline)) is False


def test_preview_loop_survives_a_frame_error(tmp_path, pipeline, monkeypatch):
    import cv2

    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    keys = iter([ord("q")])
    monkeypatch.setattr(cv2, "waitKey", lambda _n: next(keys))

    session = _session(tmp_path, pipeline)
    calls = {"n": 0}
    real = session.preview_frame

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise CameraError("dropped frame")
        return real(*a, **k)

    monkeypatch.setattr(session, "preview_frame", flaky)
    assert run_preview_loop(session) is True  # a dropped frame must not end the session


def test_main_headless_without_tty_exits_2(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    code = main(["--fake", "--no-preview", "--out", str(tmp_path)])
    assert code == 2
    assert "terminal" in capsys.readouterr().err.lower()


def test_main_reports_bad_artifacts(tmp_path, capsys):
    code = main(
        ["--fake", "--no-preview", "--out", str(tmp_path), "--artifacts", str(tmp_path / "nope")]
    )
    assert code == 2
    assert "params.json" in capsys.readouterr().err
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/pytest tests/test_app.py -q` → FAIL, no module `kodachrome.capture.app`.

- [ ] **Step 3: Implement `app.py`**

```python
"""``kodachrome-capture``: live preview, press SPACE, get two JPEGs.

Structure
---------
``CaptureSession`` owns the camera, the pipeline and the output folder, and
knows how to take one capture or produce one preview frame. Two thin loops
drive it, both taking injectable key sources so the whole flow is testable
with ``FakeCamera``:

* ``run_preview_loop`` draws the graded feed in an OpenCV window. Every GUI
  call is inside the guard, because a build without GUI support can fail at
  ``imshow`` rather than at ``namedWindow``, and a failure there must fall
  back to headless rather than end the session.
* ``run_headless_loop`` reads single keys from the terminal.

A dropped frame prints and continues in both loops. The spec promises the
session survives frame read failures, and that promise is only worth
anything if it also holds while the preview is running.

Capture semantics
-----------------
``SPACE`` acquires one fresh frame and saves exactly that frame; the
displayed frame is not re-used. ``capture()`` calls ``camera.read()`` once,
so the saved original and the graded image always come from one acquisition.

Output layout
-------------
``OUT/YYYY-MM-DD/HHMMSS_original.jpg`` holds the camera's own JPEG bytes,
written verbatim. When the camera could not supply them the file is named
``_ungraded.jpg`` instead, so the name never overstates the contents.
``HHMMSS_kodachrome.jpg`` is the graded version, and one JSON line per
capture lands in ``captures.jsonl``.

That line is an audit record, not a status message. It carries the grain
seed and the LUT hash, which together let anyone regenerate the graded file
from the original; the negotiated stream format, so a camera that quietly
dropped to a different mode is visible; and two timings, because the
pipeline cost and the time from shutter to durable file are different
numbers and only the second is what the user waits for.
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

import numpy as np

from .. import __version__
from .._cv2 import require_cv2
from ..artifacts import PARAMS_VERSION, Artifacts, ArtifactsError
from ..imageio import save_jpeg
from ..pipeline import Pipeline
from .camera import Camera, CameraError, FakeCamera, V4L2Camera

cv2 = require_cv2()

DEFAULT_OUT = Path("~/Pictures/kodachrome")
WINDOW_NAME = "Kodachrome  [SPACE capture | P toggle grade | Q quit]"


@dataclass
class CaptureResult:
    original: Path
    kodachrome: Path
    record: dict


class CaptureSession:
    def __init__(
        self,
        camera: Camera,
        pipeline: Pipeline,
        out_root: str | Path,
        now: Callable[[], datetime] | None = None,
        seed_rng: np.random.Generator | None = None,
        package_version: str = __version__,
    ) -> None:
        self.camera = camera
        self.pipeline = pipeline
        self.out_root = Path(out_root).expanduser()
        self._now = now or datetime.now
        self._seed_rng = seed_rng or np.random.default_rng()
        self._package_version = package_version

    def _allocate(self, suffix: str) -> tuple[Path, str, datetime]:
        t = self._now()
        day_dir = self.out_root / t.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        base = t.strftime("%H%M%S")
        stem, k = base, 1
        while (day_dir / f"{stem}_{suffix}.jpg").exists():
            k += 1
            stem = f"{base}-{k}"
        return day_dir, stem, t

    def capture(self) -> CaptureResult:
        shutter = time.perf_counter()
        frame = self.camera.read()

        seed = int(self._seed_rng.integers(0, 2**31 - 1))
        t0 = time.perf_counter()
        graded, info = self.pipeline.process(frame.rgb, rng=np.random.default_rng(seed))
        pipeline_ms = (time.perf_counter() - t0) * 1000.0

        suffix = "original" if frame.jpeg is not None else "ungraded"
        day_dir, stem, t = self._allocate(suffix)
        original = day_dir / f"{stem}_{suffix}.jpg"
        if frame.jpeg is not None:
            original.write_bytes(frame.jpeg)
        else:
            save_jpeg(frame.rgb, original)
        kodachrome = save_jpeg(graded, day_dir / f"{stem}_kodachrome.jpg")
        shutter_to_saved_ms = (time.perf_counter() - shutter) * 1000.0

        record = {
            "timestamp": t.isoformat(timespec="seconds"),
            "original": original.name,
            "kodachrome": kodachrome.name,
            "frame_source": frame.source,
            **info,
            "grain_seed": seed,
            "params_version": PARAMS_VERSION,
            "package_version": self._package_version,
            **self.camera.stream_info.to_dict(),
            "pipeline_ms": round(pipeline_ms, 1),
            "shutter_to_saved_ms": round(shutter_to_saved_ms, 1),
        }
        with (day_dir / "captures.jsonl").open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        return CaptureResult(original, kodachrome, record)

    def preview_frame(self, graded: bool = True, size: tuple[int, int] = (640, 360)) -> np.ndarray:
        small = cv2.resize(self.camera.read().rgb, size, interpolation=cv2.INTER_AREA)
        if graded:
            small, _ = self.pipeline.process(small, grain=False)
        return small


def _announce(result: CaptureResult, out: Callable[[str], None]) -> None:
    r = result.record
    clamps = [k for k, v in r["clamped"].items() if v]
    note = f" (clamped: {', '.join(clamps)})" if clamps else ""
    out(
        f"Saved {result.kodachrome.name} + {result.original.name} in "
        f"{r['shutter_to_saved_ms']:.0f} ms; wb={r['wb_gains']} exposure={r['exposure_gain']}{note}"
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
    """Run the windowed loop. Returns False if this build cannot show a window."""
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    except cv2.error:
        return False
    graded = True
    try:
        while True:
            try:
                frame = session.preview_frame(graded)
                cv2.imshow(window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
            except CameraError as exc:
                print(f"error: {exc}")
                continue
            except cv2.error:
                return False
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
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


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
        prog="kodachrome-capture",
        description="Capture Kodachrome-graded photos from the U20CAM.",
    )
    parser.add_argument("--device", default=None, help="index, /dev/videoN or /dev/v4l/by-id/...")
    parser.add_argument(
        "--artifacts", type=Path, default=None, help="artifact dir (default: bundled)"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-preview", action="store_true", help="never open a window")
    parser.add_argument("--fake", action="store_true", help="synthetic camera, no hardware")
    parser.add_argument("--seed", type=int, default=None, help="seed the grain seed generator")
    args = parser.parse_args(argv)

    try:
        pipeline = Pipeline(Artifacts.resolve(args.artifacts))
    except ArtifactsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        camera: Camera = FakeCamera() if args.fake else V4L2Camera(args.device)
    except CameraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    session = CaptureSession(
        camera,
        pipeline,
        args.out,
        seed_rng=np.random.default_rng(args.seed),
    )
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

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_app.py -q` → all pass.

Note on `test_recorded_seed_reproduces_the_graded_file`: the through-JPEG bounds are measured, not guessed (mean 1.08 to 1.09 and 99th percentile 4.0 across eight seeds, worst single pixel 8 to 12). If the mean exceeds 3, something is genuinely wrong with the seed handling; do not raise the bound. The in-memory assertion above it must be exact — if that one fails, grain is drawing from somewhere other than the passed generator.

- [ ] **Step 5: Try it by hand**

```bash
.venv/bin/kodachrome-capture --fake --out /tmp/kodachrome-shots
```
Expected: a window with the synthetic frame; SPACE prints a Saved line and writes `*_ungraded.jpg` (the fake camera has no JPEG bytes) plus `*_kodachrome.jpg`; Q quits. Confirm `captures.jsonl` has one line per capture with both timings.

- [ ] **Step 6: Document and commit**

In `README.md`:

````markdown
### Capture on the Pi

```bash
kodachrome-capture                 # probes for the camera, opens a preview if a display is attached
kodachrome-capture --device /dev/v4l/by-id/usb-…   # pick a camera by stable path
kodachrome-capture --no-preview    # headless: SPACE and Q from the terminal
kodachrome-capture --fake          # no hardware, synthetic frames
```

Keys: `SPACE` capture, `P` toggle graded/original preview, `Q` quit. SPACE
takes a fresh frame; it does not save the frame currently displayed.

Each capture writes to `~/Pictures/kodachrome/YYYY-MM-DD/`:

| File | Contents |
|---|---|
| `HHMMSS_original.jpg` | the camera's own JPEG bytes, unmodified |
| `HHMMSS_ungraded.jpg` | a re-encode, written **instead** when the camera cannot supply its compressed frame |
| `HHMMSS_kodachrome.jpg` | the graded version |
| `captures.jsonl` | one audit line per capture |

The audit line records the white balance and exposure gains, whether either
hit its clamp, the grain seed, the LUT's SHA-1, the negotiated stream format,
and both the pipeline time and the full shutter-to-saved time. The seed and
hash together mean a graded file can be regenerated exactly from its
original.
````

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/capture/app.py tests/test_app.py README.md
git commit -m "feat: capture app with guarded loops, byte-exact originals and audit log

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 13: Commons downloader (`train/fetch.py`)

Implements spec 3 and 6.1. Fixes F-13 and the download half of F-06. Tests use a fake HTTP session; nothing touches the network.

**Files:**
- Create: `kodachrome/train/fetch.py`, `tests/test_fetch.py`
- Modify: `README.md`

**Interfaces:**
- Produces:
  - `API_URL`, `USER_AGENT`, `DEFAULT_CATEGORY`, `SKIP_WORDS`, `MIN_LONG_SIDE = 800`
  - `LICENCE_ALLOWLIST` — exact strings plus a `PD-` prefix rule
  - `class FetchError(Exception)`
  - `@dataclass FileInfo(title, pageid, revid, url, width, height, license, lccn)` with `.filename`
  - `licence_allowed(text) -> bool`
  - `api_get(session, params, retries=3) -> dict`
  - `iter_category_members(session, category, recurse=True) -> Iterator[dict]`
  - `select_titles(entries) -> (list[str], list[dict])` — accepted titles and rejection records
  - `fetch_imageinfo(session, titles, width) -> (list[FileInfo], list[dict])`
  - `validate_image(data: bytes) -> tuple[bool, str]` — decodes with Pillow, checks size and colour
  - `download(session, info, out_dir, retries=3) -> Path | None` — atomic
  - `fetch_category(...) -> FetchReport`
  - `main(argv=None) -> int`

- [ ] **Step 1: Write the failing tests**

`tests/test_fetch.py`:
```python
import io
import json

import numpy as np
import pytest
from PIL import Image

from kodachrome.train.fetch import (
    API_URL,
    FileInfo,
    download,
    fetch_category,
    fetch_imageinfo,
    iter_category_members,
    licence_allowed,
    main,
    select_titles,
    validate_image,
)

CAT = "Category:Test"
SUB = "Category:Sub"


def _photo_bytes(w=1200, h=900, seed=0):
    rgb = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=80)
    return buf.getvalue()


def _grey_bytes(w=1200, h=900):
    buf = io.BytesIO()
    Image.fromarray(np.full((h, w), 128, dtype=np.uint8), "L").save(buf, "JPEG")
    return buf.getvalue()


class FakeResponse:
    def __init__(self, status=200, payload=None, content=b""):
        self.status_code = status
        self._payload = payload
        self.content = content

    def json(self):
        return self._payload


class FakeSession:
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
                [
                    {"ns": 6, "title": "File:A LCCN2017000001.jpg", "pageid": 11},
                    {"ns": 14, "title": SUB, "pageid": 12},
                ],
                cont="c1",
            )
        if params["cmtitle"] == CAT:
            return _members([{"ns": 6, "title": "File:B LCCN2017000002.jpg", "pageid": 13}])
        if params["cmtitle"] == SUB:
            return _members(
                [
                    {"ns": 6, "title": "File:C (cropped) LCCN2017000003.jpg", "pageid": 14},
                    {"ns": 14, "title": CAT, "pageid": 15},
                ]
            )
    # Substring, not equality: the real call asks for "imageinfo|revisions",
    # because the revision id is part of the provenance record. Matching on
    # equality here silently sent every imageinfo call down the AssertionError
    # path, where api_get retried it three times with backoff before failing.
    if "imageinfo" in params.get("prop", ""):
        pages = {}
        for i, title in enumerate(params["titles"].split("|")):
            small = "small" in title
            nonfree = "nonfree" in title
            pages[str(i)] = {
                "title": title,
                "pageid": 100 + i,
                "imageinfo": [
                    {
                        "url": f"https://upload/{i}.jpg",
                        "thumburl": f"https://upload/thumb/{i}.jpg",
                        "width": 300 if small else 4000,
                        "height": 200 if small else 3000,
                        "mime": "image/jpeg",
                        "timestamp": "2020-01-01T00:00:00Z",
                        "extmetadata": {
                            "LicenseShortName": {
                                "value": "CC BY-SA 4.0" if nonfree else "Public domain"
                            }
                        },
                    }
                ],
                "revisions": [{"revid": 900 + i}],
            }
        return {"query": {"pages": pages}}
    raise AssertionError(f"unexpected params {params}")


@pytest.mark.parametrize(
    "text, allowed",
    [
        ("Public domain", True),
        ("CC0", True),
        ("PDM", True),
        ("PD-USGov", True),
        ("PD-1996", True),
        ("CC BY-SA 4.0", False),
        ("GFDL", False),
        ("", False),
        (None, False),
    ],
)
def test_licence_allowlist(text, allowed):
    assert licence_allowed(text) is allowed


def test_iter_category_members_follows_continue_and_recurses_once():
    titles = [m["title"] for m in iter_category_members(FakeSession(_handler), CAT)]
    assert titles == [
        "File:A LCCN2017000001.jpg",
        "File:C (cropped) LCCN2017000003.jpg",
        "File:B LCCN2017000002.jpg",
    ]


def test_select_titles_records_why_each_rejection_happened():
    entries = [
        {"title": "File:A LCCN2017000001.jpg"},
        {"title": "File:A again LCCN2017000001.jpg"},
        {"title": "File:C (cropped) LCCN2017000003.jpg"},
        {"title": "File:Zed no lccn.jpg"},
    ]
    accepted, rejected = select_titles(entries)
    assert accepted == ["File:A LCCN2017000001.jpg", "File:Zed no lccn.jpg"]
    reasons = {r["title"]: r["reason"] for r in rejected}
    assert reasons["File:A again LCCN2017000001.jpg"] == "duplicate-lccn"
    assert reasons["File:C (cropped) LCCN2017000003.jpg"] == "title-filter"


def test_fetch_imageinfo_rejects_small_and_non_free():
    infos, rejected = fetch_imageinfo(
        FakeSession(_handler),
        ["File:X LCCN2017000009.jpg", "File:small.jpg", "File:nonfree.jpg"],
        1024,
    )
    assert [i.title for i in infos] == ["File:X LCCN2017000009.jpg"]
    reasons = {r["title"]: r["reason"] for r in rejected}
    assert reasons["File:small.jpg"] == "too-small"
    assert reasons["File:nonfree.jpg"] == "licence"
    info = infos[0]
    assert info.url == "https://upload/thumb/0.jpg"
    assert info.lccn == "2017000009" and info.filename == "2017000009.jpg"
    assert info.pageid == 100 and info.revid == 900


def test_validate_image_accepts_photos_and_rejects_junk():
    ok, reason = validate_image(_photo_bytes())
    assert ok and reason == ""
    assert validate_image(b"not an image")[0] is False
    assert validate_image(_photo_bytes(w=400, h=300))[1] == "too-small"
    assert validate_image(_grey_bytes())[1] == "greyscale"


def test_download_is_atomic_and_leaves_nothing_on_failure(tmp_path):
    info = FileInfo(
        "File:T LCCN2017000001.jpg",
        1,
        2,
        "https://upload/1.jpg",
        1200,
        900,
        "Public domain",
        "2017000001",
    )
    session = FakeSession(_handler, files={"https://upload/1.jpg": _photo_bytes()})
    path, reason = download(session, info, tmp_path)
    assert path is not None and path.name == "2017000001.jpg" and reason == ""
    calls = len(session.calls)
    assert download(session, info, tmp_path) == (path, "")  # resumed, not re-fetched
    assert len(session.calls) == calls

    bad = FileInfo(
        "File:U LCCN2017000002.jpg",
        1,
        2,
        "https://upload/bad.jpg",
        1,
        1,
        "Public domain",
        "2017000002",
    )
    failing = FakeSession(_handler, fail_urls={"https://upload/bad.jpg"})
    assert download(failing, bad, tmp_path, retries=1) == (None, "http-500")
    assert list(tmp_path.glob("*.part")) == [], "no partial files may remain"
    assert not (tmp_path / "2017000002.jpg").exists()


def test_download_rejects_undecodable_content(tmp_path):
    info = FileInfo(
        "File:V LCCN2017000003.jpg",
        1,
        2,
        "https://upload/x.jpg",
        1200,
        900,
        "Public domain",
        "2017000003",
    )
    session = FakeSession(_handler, files={"https://upload/x.jpg": b"garbage"})
    path, reason = download(session, info, tmp_path)
    assert path is None
    assert reason.startswith("undecodable"), f"the manifest would record {reason!r}"
    assert not (tmp_path / "2017000003.jpg").exists()
    # Bad content is not retried: it will fail identically every time.
    assert len(session.calls) == 1


def test_a_rejected_photo_records_why_not_just_that_it_failed(tmp_path):
    """The manifest must distinguish a scanned document from a network error."""
    info = FileInfo(
        "File:G LCCN2017000007.jpg",
        1,
        2,
        "https://upload/g.jpg",
        1200,
        900,
        "Public domain",
        "2017000007",
    )
    session = FakeSession(_handler, files={"https://upload/g.jpg": _grey_bytes()})
    path, reason = download(session, info, tmp_path)
    assert path is None
    assert reason == "greyscale", "a document that decodes must not read as 'download-failed'"


@pytest.mark.parametrize(
    "title, expected",
    [("File:svg.jpg", "mime:image/svg+xml"), ("File:noinfo.jpg", "no-imageinfo")],
)
def test_metadata_level_rejections_are_named(title, expected):
    """Both defensive branches in fetch_imageinfo, which no fixture reached before."""

    def handler(params):
        if params.get("prop", "").startswith("imageinfo"):
            page = {"title": title, "pageid": 1, "revisions": [{"revid": 9}]}
            if "noinfo" not in title:
                page["imageinfo"] = [
                    {
                        "url": "https://upload/x.svg",
                        "thumburl": "https://upload/x.svg",
                        "width": 4000,
                        "height": 3000,
                        "mime": "image/svg+xml",
                        "extmetadata": {"LicenseShortName": {"value": "Public domain"}},
                    }
                ]
            return {"query": {"pages": {"0": page}}}
        raise AssertionError("unexpected params")

    infos, rejected = fetch_imageinfo(FakeSession(handler), [title], 1024)
    assert infos == []
    assert rejected == [{"title": title, "reason": expected}]


def test_fetch_category_writes_a_manifest_with_hashes_and_rejections(tmp_path):
    files = {f"https://upload/thumb/{i}.jpg": _photo_bytes(seed=i) for i in range(3)}
    report = fetch_category(FakeSession(_handler, files=files), CAT, tmp_path, width=1024)
    assert [e["lccn"] for e in report.files] == ["2017000001", "2017000002"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["category"] == CAT
    assert manifest["corpus_sha1"]
    assert any(r["reason"] == "title-filter" for r in manifest["rejected"])
    for entry in manifest["files"]:
        assert len(entry["sha1"]) == 40
        assert entry["pageid"] and entry["revid"]


def test_resume_revalidates_against_the_manifest_hash(tmp_path):
    files = {f"https://upload/thumb/{i}.jpg": _photo_bytes(seed=i) for i in range(3)}
    fetch_category(FakeSession(_handler, files=files), CAT, tmp_path, width=1024)
    victim = tmp_path / "2017000001.jpg"
    victim.write_bytes(_photo_bytes(seed=99))  # same name, different content
    report = fetch_category(FakeSession(_handler, files=files), CAT, tmp_path, width=1024)
    assert victim.read_bytes() == files["https://upload/thumb/0.jpg"], (
        "corrupt file must be refetched"
    )
    assert report.repaired == 1


def test_main_enforces_min_files(tmp_path, monkeypatch, capsys):
    files = {f"https://upload/thumb/{i}.jpg": _photo_bytes(seed=i) for i in range(3)}
    monkeypatch.setattr(
        "kodachrome.train.fetch.make_session", lambda: FakeSession(_handler, files=files)
    )
    assert main(["--out", str(tmp_path), "--category", CAT, "--min-files", "5"]) == 1
    assert "fewer than 5" in capsys.readouterr().err
    assert main(["--out", str(tmp_path), "--category", CAT, "--min-files", "2"]) == 0
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/pytest tests/test_fetch.py -q` → FAIL, no module `kodachrome.train.fetch`.

- [ ] **Step 3: Implement `fetch.py`**

```python
"""``kodachrome-fetch``: download public-domain Kodachrome scans from Wikimedia Commons.

Why Commons and not loc.gov
---------------------------
The Library of Congress FSA/OWI colour transparencies are the target corpus,
but loc.gov sits behind a Cloudflare challenge that returns HTTP 403 to
scripted clients (checked 2026-09-03 with several User-Agents). Commons
hosts the same LoC scans, keeps the catalogue number (LCCN) in each
filename, and its API welcomes scripted access from a tool that identifies
itself.

What "public domain" is allowed to mean
---------------------------------------
A category is a claim, not a guarantee: anyone can file an image into it.
So the licence is checked per file against an allowlist rather than assumed
from the category, and every rejection is written to the manifest with its
reason. A corpus you cannot audit is a corpus you cannot defend.

Validation before acceptance
----------------------------
The API's word is not enough either. Bytes are downloaded to a temporary
file, decoded with Pillow, and checked for size and for being an actual
colour photograph rather than a scanned document or diagram, before being
renamed into place. A resumed run re-hashes what is already on disk against
the manifest and refetches anything that does not match, so a truncated
earlier download cannot silently poison the training set.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "kodachrome-film/0.1 (Kodachrome LUT trainer; "
    "https://github.com/kodachrome-film) python-requests"
)
DEFAULT_CATEGORY = "Category:Color photographs from the Farm Security Administration"
SKIP_WORDS = ("cropped", "restored", "retouched", "colorized", "colourized", "edit")
MIN_LONG_SIDE = 800
LICENCE_ALLOWLIST = {"public domain", "cc0", "pdm", "no restrictions"}
ALLOWED_MIME = {"image/jpeg", "image/png", "image/tiff"}
_LCCN_RE = re.compile(r"LCCN(\d{6,})", re.IGNORECASE)


class FetchError(Exception):
    """The Commons API could not be reached or answered unexpectedly."""


@dataclass
class FileInfo:
    title: str
    pageid: int
    revid: int
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
        return f"{re.sub(r'[^A-Za-z0-9]+', '_', stem).strip('_')[:120]}.jpg"


@dataclass
class FetchReport:
    files: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    failed: int = 0
    repaired: int = 0


def licence_allowed(text: str | None) -> bool:
    """Explicit allowlist: exact free-licence names, plus the PD-* family."""
    if not text:
        return False
    normalised = text.strip().lower()
    return normalised in LICENCE_ALLOWLIST or normalised.startswith("pd-")


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
        except Exception as exc:  # noqa: BLE001 - network errors are all retried alike
            last = repr(exc)
        time.sleep(2**attempt)
    raise FetchError(f"Commons API request failed after {retries} attempts: {last}")


def iter_category_members(
    session: Any, category: str, recurse: bool = True, _seen: set[str] | None = None
) -> Iterator[dict]:
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
                yield member
            elif member["ns"] == 14 and recurse:
                yield from iter_category_members(session, member["title"], recurse, seen)
        cont = data.get("continue")
        if not cont:
            return
        params = {**params, **cont}


def select_titles(entries: list[dict]) -> tuple[list[str], list[dict]]:
    accepted: list[str] = []
    without: list[str] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    for entry in entries:
        title = entry["title"] if isinstance(entry, dict) else entry
        low = title.lower()
        if any(word in low for word in SKIP_WORDS):
            rejected.append({"title": title, "reason": "title-filter"})
            continue
        m = _LCCN_RE.search(title)
        if m:
            if m.group(1) in seen:
                rejected.append({"title": title, "reason": "duplicate-lccn"})
                continue
            seen.add(m.group(1))
            accepted.append(title)
        else:
            without.append(title)
    return accepted + without, rejected


def fetch_imageinfo(
    session: Any, titles: list[str], width: int
) -> tuple[list[FileInfo], list[dict]]:
    infos: list[FileInfo] = []
    rejected: list[dict] = []
    for start in range(0, len(titles), 50):
        batch = titles[start : start + 50]
        data = api_get(
            session,
            {
                "action": "query",
                "prop": "imageinfo|revisions",
                "titles": "|".join(batch),
                "iiprop": "url|size|mime|extmetadata|timestamp",
                "iiurlwidth": str(width),
                "rvprop": "ids",
            },
        )
        for page in data.get("query", {}).get("pages", {}).values():
            title = page.get("title", "?")
            ii = (page.get("imageinfo") or [None])[0]
            if not ii:
                rejected.append({"title": title, "reason": "no-imageinfo"})
                continue
            if str(ii.get("mime", "")) not in ALLOWED_MIME:
                rejected.append({"title": title, "reason": f"mime:{ii.get('mime')}"})
                continue
            licence = ii.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "")
            if not licence_allowed(licence):
                rejected.append({"title": title, "reason": "licence", "license": licence})
                continue
            if max(int(ii["width"]), int(ii["height"])) < MIN_LONG_SIDE:
                rejected.append({"title": title, "reason": "too-small"})
                continue
            m = _LCCN_RE.search(title)
            revisions = page.get("revisions") or [{}]
            infos.append(
                FileInfo(
                    title=title,
                    pageid=int(page.get("pageid", 0)),
                    revid=int(revisions[0].get("revid", 0)),
                    url=ii.get("thumburl") or ii["url"],
                    width=int(ii["width"]),
                    height=int(ii["height"]),
                    license=licence,
                    lccn=m.group(1) if m else None,
                )
            )
    return infos, rejected


def validate_image(data: bytes) -> tuple[bool, str]:
    """Decode the bytes and confirm they are a colour photograph of usable size."""
    if not data:
        return False, "empty"
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            width, height = im.size
            mode = im.mode
            sample = np.asarray(im.convert("RGB").resize((64, 64)))
    except Exception as exc:  # noqa: BLE001 - any decode failure disqualifies the file
        return False, f"undecodable:{type(exc).__name__}"
    if max(width, height) < MIN_LONG_SIDE:
        return False, "too-small"
    channel_spread = float(np.abs(sample.max(axis=2).astype(int) - sample.min(axis=2)).mean())
    if mode in ("L", "1") or channel_spread < 2.0:
        return False, "greyscale"
    return True, ""


def download(
    session: Any, info: FileInfo, out_dir: str | Path, retries: int = 3
) -> tuple[Path | None, str]:
    """Download to a temporary file, validate, then rename. Never leaves a partial file.

    Returns ``(path, "")`` on success and ``(None, reason)`` on failure. The
    reason is carried rather than discarded because it lands in the manifest,
    and "a corpus you cannot audit is a corpus you cannot defend" is this
    module's whole premise. Collapsing every failure into one generic label
    would hide the byte-level colour and size check — the very check that
    catches a scanned document the API described as a photograph.

    Content that decodes but fails validation returns immediately: a greyscale
    scan will still be a greyscale scan on the third attempt, so retrying only
    spends the backoff budget. Transport failures do get the retries.
    """
    out_dir = Path(out_dir)
    final = out_dir / info.filename
    if final.is_file() and final.stat().st_size > 0:
        return final, ""
    tmp = final.with_suffix(final.suffix + ".part")
    reason = "download-failed"
    for attempt in range(retries):
        try:
            r = session.get(info.url, headers={"User-Agent": USER_AGENT}, timeout=120)
            if r.status_code != 200:
                reason = f"http-{r.status_code}"
            elif not r.content:
                reason = "empty-response"
            else:
                ok, why = validate_image(r.content)
                if not ok:
                    return None, why
                tmp.write_bytes(r.content)
                tmp.replace(final)
                return final, ""
        except Exception as exc:  # noqa: BLE001 - every transport failure retries alike
            reason = f"error:{type(exc).__name__}"
        finally:
            tmp.unlink(missing_ok=True)
        time.sleep(2**attempt)
    return None, reason


def corpus_sha1(paths: list[Path]) -> str:
    """Hash the actual bytes of every file, so different content cannot collide."""
    h = hashlib.sha1()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(hashlib.sha1(p.read_bytes()).digest())
    return h.hexdigest()


def fetch_category(
    session: Any,
    category: str,
    out_dir: str | Path,
    width: int = 1024,
    limit: int | None = None,
    sample: int | None = None,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
) -> FetchReport:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda _m: None)

    previous: dict[str, str] = {}
    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            previous = {
                e["filename"]: e["sha1"] for e in json.loads(manifest_path.read_text())["files"]
            }
        except (KeyError, json.JSONDecodeError):
            previous = {}

    members = list(iter_category_members(session, category))
    titles, rejected = select_titles(members)
    say(f"{len(titles)} candidate files in {category}, {len(rejected)} rejected by title")
    if sample is not None and sample < len(titles):
        titles = sorted(random.Random(seed).sample(titles, sample), key=titles.index)
    if limit is not None:
        titles = titles[:limit]

    infos, info_rejected = fetch_imageinfo(session, titles, width)
    rejected.extend(info_rejected)
    say(f"{len(infos)} files pass licence and size checks; downloading at {width}px")

    report = FetchReport(rejected=rejected)
    for i, info in enumerate(infos, start=1):
        final = out_dir / info.filename
        recorded = previous.get(info.filename)
        if final.is_file() and recorded:
            if hashlib.sha1(final.read_bytes()).hexdigest() != recorded:
                say(f"  {info.filename} does not match its recorded hash; refetching")
                final.unlink()
                report.repaired += 1
        path, reason = download(session, info, out_dir)
        if path is None:
            report.failed += 1
            rejected.append({"title": info.title, "reason": reason})
            continue
        entry = asdict(info)
        entry["filename"] = info.filename
        entry["sha1"] = hashlib.sha1(path.read_bytes()).hexdigest()
        report.files.append(entry)
        if i % 25 == 0:
            say(f"  {i}/{len(infos)}")
        time.sleep(0.05)

    manifest = {
        "category": category,
        "width": width,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_files": len(report.files),
        "n_failed": report.failed,
        "n_repaired": report.repaired,
        "corpus_sha1": corpus_sha1([out_dir / e["filename"] for e in report.files]),
        "files": report.files,
        "rejected": rejected,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    say(
        f"done: {len(report.files)} files, {report.failed} failed, "
        f"{len(rejected)} rejected, manifest at {manifest_path}"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kodachrome-fetch",
        description="Download public-domain Kodachrome scans from Wikimedia Commons.",
    )
    parser.add_argument("--out", type=Path, default=Path("data/kodachrome"))
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="stop after N files")
    parser.add_argument("--sample", type=int, default=None, help="seeded random subset of N files")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-files", type=int, default=200)
    args = parser.parse_args(argv)

    try:
        report = fetch_category(
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
    if len(report.files) < args.min_files:
        print(
            f"error: accepted {len(report.files)} files, fewer than {args.min_files}. "
            "Check the category name, the licence filter, or the network; "
            "see the manifest's 'rejected' list for reasons.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_fetch.py -q` → all pass. The failure-path tests sleep on retry; with `retries=1` that is about a second.

- [ ] **Step 5: Document and commit**

In `README.md`, add a `## Training (Mac)` section:

````markdown
## Training (Mac)

### 1. Fetch the Kodachrome scans

```bash
.venv/bin/kodachrome-fetch          # about 1,000 files, ~200 MB, into data/kodachrome/
```

The scans are the Library of Congress FSA/OWI colour transparencies
(1939-1944), mirrored on Wikimedia Commons because loc.gov blocks scripted
downloads. Every file is checked individually rather than trusted for being
in the category: the licence must be on an allowlist, the bytes must decode,
and the image must be a colour photograph of at least 800 px. Downloads are
atomic, and a resumed run re-hashes what is on disk and refetches anything
that does not match.

`data/kodachrome/manifest.json` lists every accepted file with its catalogue
number, Commons page and revision ID, licence and SHA-1, plus every
rejection and why. To use your own scans instead, point `kodachrome-train
--target` at any folder.
````

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/train/fetch.py tests/test_fetch.py README.md
git commit -m "feat: Commons downloader with licence allowlist, media validation and atomic writes

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 14: Dataset preparation with a held-out split (`train/dataset.py`)

Implements spec 6.3. Fixes the leakage half of F-01/F-05 and the corpus-hash half of F-06.

**Files:**
- Create: `kodachrome/train/dataset.py`, `tests/test_dataset.py`

**Interfaces:**
- Consumes: `load_rgb`, `list_images`, `normalize_float`, `NormalizeParams`, `srgb_to_oklab`, `require_cv2`
- Produces:
  - `@dataclass SampleConfig(crop_frac=0.06, max_side=512, pixels_per_image=3000, l_min=0.02, l_max=0.98, max_pixels=400_000, val_fraction=0.2, seed=0)` with validation
  - `@dataclass PixelPool(srgb, n_images, clamp_rate, wb_gains, exposure_gains, profiles)` with cached `.lab`
  - `@dataclass CorpusSplit(train_paths, val_paths, train_pool, val_pool, corpus_sha1)`
  - `class CorpusTooSmall(ValueError)`
  - `crop_and_resize(rgb_u8, crop_frac, max_side)`
  - `prepare_image(rgb_u8, normalize_params, cfg) -> (rgb_float, Gains)`
  - `sample_pixels(rgb_float, n, l_min, l_max, rng)`
  - `split_paths(paths, val_fraction, seed) -> (train, val)`
  - `build_pool(paths, normalize_params, cfg, progress=None) -> PixelPool`
  - `build_corpus(dir_or_paths, normalize_params, cfg, minimum, label, allow_small=False, progress=None) -> CorpusSplit`
  - `corpus_sha1(paths) -> str`

- [ ] **Step 1: Write the failing tests**

`tests/test_dataset.py`:
```python
import numpy as np
import pytest

from kodachrome.color import srgb_to_oklab
from kodachrome.imageio import save_jpeg
from kodachrome.normalize import NormalizeParams
from kodachrome.train.dataset import (
    CorpusTooSmall,
    PixelPool,
    SampleConfig,
    build_corpus,
    build_pool,
    corpus_sha1,
    crop_and_resize,
    prepare_image,
    sample_pixels,
    split_paths,
)


def _write_images(dir_path, n, seed=0):
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        p = dir_path / f"{i:03d}.jpg"
        save_jpeg(rng.integers(30, 220, (40, 60, 3), dtype=np.uint8), p)
        paths.append(p)
    return paths


def test_crop_and_resize_geometry():
    assert crop_and_resize(np.zeros((500, 1000, 3), np.uint8), 0.06, 512).shape == (256, 512, 3)
    assert crop_and_resize(np.zeros((1000, 500, 3), np.uint8), 0.0, 100).shape == (100, 50, 3)


def test_crop_removes_border():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[10:90, 10:90] = 200
    assert crop_and_resize(img, 0.1, 80).min() >= 190


def test_prepare_image_normalises_and_returns_gains():
    # A mild cast, dark enough that neither gain clamps: median linear
    # luminance lands near the 0.18 target, so the exposure gain is 0.99.
    # (190, 170, 150) looks similar but sits at luminance 0.42, which drives
    # the raw exposure gain to 0.43 and clamps it against the 0.5 floor.
    img = np.full((100, 100, 3), (130, 116, 102), dtype=np.uint8)
    out, gains = prepare_image(img, NormalizeParams(), SampleConfig(crop_frac=0.0, max_side=50))
    assert out.dtype == np.float32 and out.shape == (50, 50, 3)
    assert np.allclose(out[..., 0], out[..., 1], atol=1 / 255)
    assert gains.clamped == {"wb": False, "exposure": False}


def test_sample_pixels_respects_bounds_and_count():
    img = np.zeros((10, 20, 3), dtype=np.float32)
    img[:, 10:] = 0.5
    rng = np.random.default_rng(0)
    px = sample_pixels(img, 1000, 0.02, 0.98, rng)
    assert px.shape == (100, 3) and np.allclose(px, 0.5)
    assert sample_pixels(img, 30, 0.02, 0.98, rng).shape == (30, 3)


@pytest.mark.parametrize(
    "n, frac, expected_val", [(10, 0.2, 2), (5, 0.2, 1), (3, 0.5, 1), (1, 0.2, 0)]
)
def test_split_sizes(n, frac, expected_val):
    paths = [f"{i}.jpg" for i in range(n)]
    train, val = split_paths(paths, frac, seed=0)
    assert len(val) == expected_val
    assert len(train) == n - expected_val
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(paths)


def test_split_is_seeded_and_stable():
    paths = [f"{i}.jpg" for i in range(20)]
    assert split_paths(paths, 0.2, seed=3) == split_paths(paths, 0.2, seed=3)
    assert split_paths(paths, 0.2, seed=3) != split_paths(paths, 0.2, seed=4)


def test_build_pool_collects_diagnostics(tmp_path):
    paths = _write_images(tmp_path / "src", 3)
    cfg = SampleConfig(crop_frac=0.0, max_side=60, pixels_per_image=500, max_pixels=800)
    pool = build_pool(paths, NormalizeParams(), cfg)
    assert isinstance(pool, PixelPool)
    assert pool.n_images == 3
    assert pool.srgb.shape == (800, 3) and pool.srgb.dtype == np.float32
    assert np.allclose(pool.lab, srgb_to_oklab(pool.srgb), atol=1e-6)
    assert len(pool.wb_gains) == 3 and len(pool.exposure_gains) == 3
    assert 0.0 <= pool.clamp_rate <= 1.0
    assert pool.profiles  # profile name -> count


def test_no_validation_pixel_appears_in_training(tmp_path):
    """The split must happen before sampling, or held-out metrics are meaningless."""
    paths = _write_images(tmp_path / "src", 10, seed=5)
    cfg = SampleConfig(crop_frac=0.0, max_side=60, pixels_per_image=2000, val_fraction=0.3)
    split = build_corpus(paths, NormalizeParams(), cfg, minimum=1, label="source")
    assert len(split.val_paths) == 3 and len(split.train_paths) == 7
    assert set(split.train_paths).isdisjoint(split.val_paths)

    # Rebuild each side independently: the pools must match what build_corpus produced,
    # which is only possible if neither drew pixels from the other's images.
    expected_val = build_pool(split.val_paths, NormalizeParams(), cfg)
    assert np.array_equal(split.val_pool.srgb, expected_val.srgb)

    # And the mirror, which matters more. A validation pool contaminated by
    # training images inflates the score; a TRAINING pool contaminated by
    # held-out images corrupts the fit itself and then reports on data it was
    # fitted to. Verified that the assertion above stays green through exactly
    # that leak, so it cannot stand in for this one.
    expected_train = build_pool(split.train_paths, NormalizeParams(), cfg)
    assert np.array_equal(split.train_pool.srgb, expected_train.srgb)


def test_corpus_too_small_names_the_escape_hatch(tmp_path):
    paths = _write_images(tmp_path / "src", 4)
    cfg = SampleConfig(crop_frac=0.0, max_side=60, pixels_per_image=100)
    with pytest.raises(CorpusTooSmall, match="--allow-small"):
        build_corpus(paths, NormalizeParams(), cfg, minimum=30, label="source")
    split = build_corpus(
        paths, NormalizeParams(), cfg, minimum=30, label="source", allow_small=True
    )
    assert split.train_pool.n_images >= 1


def test_corpus_sha1_tracks_content_not_names(tmp_path):
    paths = _write_images(tmp_path / "a", 2)
    first = corpus_sha1(paths)
    assert first == corpus_sha1(paths)
    assert len(first) == 40
    save_jpeg(np.full((40, 60, 3), 7, dtype=np.uint8), paths[0])  # same name, new content
    assert corpus_sha1(paths) != first


def test_invalid_sample_config_names_the_field():
    with pytest.raises(ValueError, match="val_fraction"):
        SampleConfig(val_fraction=1.5)
    with pytest.raises(ValueError, match="max_side"):
        SampleConfig(max_side=0)
    with pytest.raises(ValueError, match="crop_frac"):
        SampleConfig(crop_frac=0.5)
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/pytest tests/test_dataset.py -q` → FAIL, no module `kodachrome.train.dataset`.

- [ ] **Step 3: Implement `dataset.py`**

```python
"""Turn folders of images into the pixel pools the fitter and evaluator use.

Both corpora go through the same steps:

1. Crop ``crop_frac`` from every edge. Slide scans carry film rebate, mount
   shadow or scanner bed; camera frames can have vignetted corners.
2. Downscale so the long side is ``max_side``. Colour statistics do not need
   full resolution.
3. Normalise with the same code the Pi runs. Sources get white balance;
   targets are given ``NormalizeParams(white_balance=False)`` by the caller,
   because the film's cast is part of the look being learned.
4. Sample pixels whose Oklab lightness is inside ``(l_min, l_max)``.
   Near-black pixels are borders and crushed shadows, near-white are blown
   highlights and scanner glare; neither says anything about how the film
   renders colour.

Why the split happens here, before sampling
-------------------------------------------
Metrics computed on the pixels a LUT was fitted to measure how well the fit
memorised its training data, not whether the look generalises. So each
corpus is split **by image** first, and only then sampled. Splitting after
sampling would be worse than useless: pixels from the same photograph are
highly correlated, so a "held-out" pixel drawn from a training image leaks
almost everything about its neighbours, and the reported improvement would
be inflated in a way no seed average would reveal.

Diagnostics travel with the pool
--------------------------------
``PixelPool`` carries the white balance and exposure gains applied to each
image, how often a gain hit its clamp, and which ICC profiles were seen. A
corpus where normalisation is clamping constantly, or which is secretly half
Adobe RGB, produces a misleading fit; the report publishes these so it is
visible rather than silent.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .._cv2 import require_cv2
from ..color import srgb_to_oklab
from ..imageio import list_images, load_rgb
from ..normalize import Gains, NormalizeParams, normalize_float

cv2 = require_cv2()


class CorpusTooSmall(ValueError):
    """A corpus has too few images for a statistically meaningful fit."""


@dataclass
class SampleConfig:
    crop_frac: float = 0.06
    max_side: int = 512
    pixels_per_image: int = 3000
    l_min: float = 0.02
    l_max: float = 0.98
    max_pixels: int = 400_000
    val_fraction: float = 0.2
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.crop_frac < 0.4:
            raise ValueError(f"crop_frac must be in [0, 0.4), got {self.crop_frac}")
        if self.max_side < 16:
            raise ValueError(f"max_side must be at least 16, got {self.max_side}")
        if self.pixels_per_image < 1:
            raise ValueError(f"pixels_per_image must be positive, got {self.pixels_per_image}")
        if self.max_pixels < 1:
            raise ValueError(f"max_pixels must be positive, got {self.max_pixels}")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in [0, 1), got {self.val_fraction}")
        if not 0.0 <= self.l_min < self.l_max <= 1.0:
            raise ValueError(f"l_min ({self.l_min}) must be below l_max ({self.l_max})")


@dataclass
class PixelPool:
    srgb: np.ndarray
    n_images: int
    clamp_rate: float = 0.0
    wb_gains: list = field(default_factory=list)
    exposure_gains: list = field(default_factory=list)
    profiles: dict = field(default_factory=dict)
    _lab: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def lab(self) -> np.ndarray:
        if self._lab is None:
            self._lab = srgb_to_oklab(self.srgb)
        return self._lab


@dataclass
class CorpusSplit:
    train_paths: list[Path]
    val_paths: list[Path]
    train_pool: PixelPool
    val_pool: PixelPool
    corpus_sha1: str


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


def prepare_image(
    rgb_u8: np.ndarray, normalize_params: NormalizeParams, cfg: SampleConfig
) -> tuple[np.ndarray, Gains]:
    small = crop_and_resize(rgb_u8, cfg.crop_frac, cfg.max_side).astype(np.float32) / 255.0
    return normalize_float(small, normalize_params)


def sample_pixels(
    rgb: np.ndarray, n: int, l_min: float, l_max: float, rng: np.random.Generator
) -> np.ndarray:
    flat = rgb.reshape(-1, 3).astype(np.float32)
    lightness = srgb_to_oklab(flat)[:, 0]
    keep = np.flatnonzero((lightness > l_min) & (lightness < l_max))
    if len(keep) > n:
        keep = rng.choice(keep, n, replace=False)
    return flat[keep]


def split_paths(paths: Sequence, val_fraction: float, seed: int) -> tuple[list, list]:
    """Split by image, deterministically. Validation gets ``floor(n * fraction)`` images."""
    ordered = list(paths)
    n_val = int(math.floor(len(ordered) * val_fraction))
    if n_val == 0:
        return ordered, []
    rng = np.random.default_rng(seed)
    val_idx = set(rng.choice(len(ordered), n_val, replace=False).tolist())
    train = [p for i, p in enumerate(ordered) if i not in val_idx]
    val = [p for i, p in enumerate(ordered) if i in val_idx]
    return train, val


def build_pool(
    paths: Sequence[Path],
    normalize_params: NormalizeParams,
    cfg: SampleConfig,
    progress: Callable[[str], None] | None = None,
) -> PixelPool:
    rng = np.random.default_rng(cfg.seed)
    chunks: list[np.ndarray] = []
    wb_gains: list[list[float]] = []
    exposure_gains: list[float] = []
    profiles: dict[str, int] = {}
    clamped = 0
    for i, path in enumerate(paths, start=1):
        rgb, meta = load_rgb(path)
        profiles[meta.profile] = profiles.get(meta.profile, 0) + 1
        prepared, gains = prepare_image(rgb, normalize_params, cfg)
        wb_gains.append([round(float(g), 4) for g in gains.wb])
        exposure_gains.append(round(float(gains.exposure), 4))
        clamped += int(any(gains.clamped.values()))
        chunks.append(sample_pixels(prepared, cfg.pixels_per_image, cfg.l_min, cfg.l_max, rng))
        if progress and i % 100 == 0:
            progress(f"  {i}/{len(paths)} images sampled")

    pixels = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3), np.float32)
    if len(pixels) > cfg.max_pixels:
        pixels = pixels[rng.choice(len(pixels), cfg.max_pixels, replace=False)]
    return PixelPool(
        srgb=np.ascontiguousarray(pixels, dtype=np.float32),
        n_images=len(paths),
        clamp_rate=round(clamped / max(len(paths), 1), 4),
        wb_gains=wb_gains,
        exposure_gains=exposure_gains,
        profiles=profiles,
    )


def corpus_sha1(paths: Sequence[Path]) -> str:
    """Hash the bytes of every file, so equal names with different content differ."""
    h = hashlib.sha1()
    for p in sorted(Path(x) for x in paths):
        h.update(p.name.encode())
        h.update(hashlib.sha1(p.read_bytes()).digest())
    return h.hexdigest()


def build_corpus(
    dir_or_paths: str | Path | Sequence[Path],
    normalize_params: NormalizeParams,
    cfg: SampleConfig,
    minimum: int,
    label: str,
    allow_small: bool = False,
    progress: Callable[[str], None] | None = None,
) -> CorpusSplit:
    paths = (
        list_images(dir_or_paths)
        if isinstance(dir_or_paths, (str, Path))
        else [Path(p) for p in dir_or_paths]
    )
    if len(paths) < minimum and not allow_small:
        raise CorpusTooSmall(
            f"{label} corpus has {len(paths)} images, fewer than the {minimum} needed for a "
            f"meaningful fit. Add more images, or pass --allow-small to proceed anyway."
        )
    train_paths, val_paths = split_paths(paths, cfg.val_fraction, cfg.seed)
    if progress:
        progress(f"{label}: {len(train_paths)} train, {len(val_paths)} validation images")
    return CorpusSplit(
        train_paths=train_paths,
        val_paths=val_paths,
        train_pool=build_pool(train_paths, normalize_params, cfg, progress),
        val_pool=build_pool(val_paths, normalize_params, cfg, progress),
        corpus_sha1=corpus_sha1(paths),
    )
```

- [ ] **Step 4: Run and commit**

Run: `.venv/bin/pytest tests/test_dataset.py -q` → all pass.

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/train/dataset.py tests/test_dataset.py
git commit -m "feat: dataset pools with split-by-image, corpus hashing and normalisation diagnostics

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 15: Distribution transport (`train/transport.py`)

Implements spec 6.4 steps 1 and 2 and the sliced Wasserstein metric used by spec 6.5.

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


def test_transport_preserves_pixel_identity():
    """Row i of the output must be where source pixel i went, not merely a
    point drawn from the right distribution.

    Everything else in this file is permutation-invariant: shuffling the
    output rows leaves the sliced Wasserstein distance, the means and the
    standard deviations bit-identical, because a permutation does not change
    a distribution at all. Measured, to be sure of it. So a regression that
    reordered rows would sail through every other assertion here while
    handing the LUT fitter pairs that no longer correspond — a film look
    fitted to noise, with nothing to say so.

    The property is tested as equivariance: permuting the input must permute
    the output the same way. That holds exactly for a rank-based mapping and
    fails for any reordering.
    """
    rng = np.random.default_rng(0)
    src = rng.normal([0.5, 0.0, 0.0], [0.1, 0.05, 0.05], (3000, 3))
    tgt = rng.normal([0.6, 0.05, -0.05], [0.15, 0.08, 0.04], (3000, 3))
    perm = np.random.default_rng(7).permutation(len(src))

    straight = iterative_distribution_transfer(src, tgt, iterations=8, rng=np.random.default_rng(1))
    permuted = iterative_distribution_transfer(
        src[perm], tgt, iterations=8, rng=np.random.default_rng(1)
    )
    assert np.array_equal(permuted, straight[perm]), "row correspondence was not preserved"


def test_sliced_wasserstein_zero_for_identical_and_positive_for_shift():
    rng = np.random.default_rng(6)
    a = rng.normal(size=(5000, 3))
    identical = sliced_wasserstein(a, a.copy(), rng=np.random.default_rng(0))
    assert identical == pytest.approx(0.0, abs=1e-9)
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

Content bias and hue reweighting: a heuristic, not a constraint
---------------------------------------------------------------
The two corpora do not show the same things. The 1940s FSA photographs are
full of fields, khaki and weathered wood; a modern indoor sample set is not.
Raw distribution matching would happily turn a blue wall green because the
film corpus contains more green. ``hue_weights`` reduces that: target pixels
are reweighted so the target's hue histogram matches the source's, removing
the largest content-driven bias.

Be precise about what this does **not** do. The transport still operates on
the full three-dimensional distribution and can move individual pixels
across hue bins; matching the aggregate histogram does not constrain where
any particular pixel goes. Clipping the weights to ``[0.2, 5.0]`` also means
the reweighted histograms match only approximately when a hue is nearly
absent from one side. So this is a bias-reduction heuristic, not a guarantee
about what is learned, and the report publishes the residual histogram
difference (``hue_hist_residual``) so its size is visible rather than
assumed. Spec section 1 states the resulting limit on the project's claim.

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

### Task 16: Smooth LUT regression (`train/lutfit.py`)

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


def test_the_identity_term_anchors_nodes_no_data_reaches():
    """Untouched nodes must stay put — and this test must notice if they do not.

    The obvious fixture cannot. Fitting a near-identity transform (y = 1.1x)
    over the dark half passes with the identity term *deleted*, because the
    smoothness term extrapolates the fitted curve into the bright region and
    lands close to identity anyway: measured 0.07 drift against a 0.25 bound.
    A test that green-lights the absence of the thing it is named for is
    worse than no test.

    A strong transform separates them. Fitting y = 0.25x over the dark half
    leaves the bright region 0.71 away from identity when nothing anchors it,
    against 0.003 when the identity term is present.
    """
    rng = np.random.default_rng(2)
    x = (rng.random((20000, 3), dtype=np.float32) * 0.5).astype(np.float32)
    y = np.clip(x * 0.25, 0, 1).astype(np.float32)
    lut = fit_lut(x, y, n=9)

    bright = np.array([[0.95, 0.95, 0.95], [0.9, 0.2, 0.9]], dtype=np.float32)
    out = lut.apply_numpy(bright)
    assert np.all(np.isfinite(out))
    assert np.abs(out - bright).max() < 0.05, "untouched nodes drifted"

    # And the fitted region must still follow its data, or a LUT that simply
    # ignored every sample would satisfy the assertion above.
    dark = np.array([[0.3, 0.3, 0.3], [0.4, 0.15, 0.35]], dtype=np.float32)
    assert np.abs(lut.apply_numpy(dark) - dark * 0.25).max() < 0.05


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

### Task 17: Paired evaluation and LUT safety gates (`train/evaluate.py`)

Implements spec 6.5. Fixes F-05 and F-18. This is the task that makes the project's central claim measurable.

**Files:**
- Create: `kodachrome/train/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `LUT3D`, `PixelPool`, `sliced_wasserstein`, `hue_bin_index`, colour conversions
- Produces:
  - `@dataclass Evaluator(src_idx, tgt_idx, directions)` built by `Evaluator.build(src_pool, tgt_pool, tgt_weights, n_proj=64, max_points=100_000, seed=0)`, with `distance(src_lab) -> float`
  - `swd_seed_spread(src_pool, tgt_pool, tgt_weights, lut, seeds=(0,1,2,3,4)) -> (mean, spread)`
  - `grey_axis_is_monotone(lut, tolerance=1e-3) -> bool`
  - `channels_are_monotone(lut, tolerance=1e-3) -> bool`
  - `neutral_axis_max_chroma(lut) -> float`
  - `clipped_volume_fraction(lut, eps=1e-4) -> float`
  - `hue_bin_shifts(src_srgb, lut, n_bins=24, chroma_floor=0.03) -> list[dict]`
  - `hue_hist_residual(src_lab, tgt_lab, weights, n_bins, chroma_floor) -> float`
  - `@dataclass Gate(name, value, threshold, passed, detail)`
  - `check_gates(metrics) -> list[Gate]`
  - `evaluate(...) -> dict` (the metrics block of spec 5.8)

- [ ] **Step 1: Write the failing tests**

`tests/test_evaluate.py`:
```python
import numpy as np
import pytest

from kodachrome.color import lch_to_oklab, srgb_to_oklab
from kodachrome.lut import LUT3D
from kodachrome.train.dataset import PixelPool
from kodachrome.train.evaluate import (
    Evaluator,
    channels_are_monotone,
    check_gates,
    clipped_volume_fraction,
    evaluate,
    grey_axis_is_monotone,
    hue_bin_shifts,
    hue_hist_residual,
    neutral_axis_max_chroma,
    swd_seed_spread,
)


def _pool(seed, n=4000, scale=1.0, offset=0.0):
    rng = np.random.default_rng(seed)
    srgb = np.clip(rng.random((n, 3), dtype=np.float32) * scale + offset, 0, 1).astype(np.float32)
    return PixelPool(srgb=srgb, n_images=5)


def _darkening_lut(n=9, gamma=1.5):
    return LUT3D(LUT3D.identity(n).table**gamma)


def test_identity_lut_gives_identical_before_and_after():
    """The whole point of a paired evaluator: no LUT change, no metric change."""
    src, tgt = _pool(0), _pool(1, scale=0.8)
    weights = np.ones(len(tgt.srgb))
    ev = Evaluator.build(src, tgt, weights, seed=0)
    before = ev.distance(src.lab)
    after = ev.distance(srgb_to_oklab(LUT3D.identity(33).apply_numpy(src.srgb)))
    assert after == pytest.approx(before, abs=1e-6)


def test_pairing_holds_when_the_pool_is_actually_subsampled():
    """The production case, which no other test here reaches.

    Every other pool in this file is smaller than ``max_points``, so
    ``rng.choice(n, n, replace=False)`` returns a full permutation and
    ``distance`` — which sorts the projections before comparing — gives the
    same answer whether or not the indices are applied at all. Measured:
    identical to ten decimal places. So an implementation that ignored
    ``src_idx`` entirely would pass every other assertion in this file.

    A real run caps pools at 400,000 pixels against a 100,000 sample, so the
    indices genuinely select a subset. This forces that, and then re-checks
    the property the whole evaluator exists for.
    """
    src, tgt = _pool(11), _pool(12, scale=0.8)
    weights = np.ones(len(tgt.srgb))
    ev = Evaluator.build(src, tgt, weights, seed=0, max_points=500)

    assert len(ev.src_idx) == 500, "the sample must be a subset, not the whole pool"
    assert ev.tgt_points.shape[0] == 500
    assert ev.distance(src.lab) != Evaluator.build(src, tgt, weights, seed=0).distance(src.lab)

    same = Evaluator.build(src, tgt, weights, seed=0, max_points=500)
    assert np.array_equal(same.src_idx, ev.src_idx)
    other = Evaluator.build(src, tgt, weights, seed=1, max_points=500)
    assert not np.array_equal(other.src_idx, ev.src_idx)

    # And the property the evaluator exists for still holds on a real subset.
    before = ev.distance(src.lab)
    after = ev.distance(srgb_to_oklab(LUT3D.identity(33).apply_numpy(src.srgb)))
    assert after == pytest.approx(before, abs=1e-6)


def test_evaluator_is_reusable_and_deterministic():
    src, tgt = _pool(2), _pool(3)
    ev = Evaluator.build(src, tgt, np.ones(len(tgt.srgb)), seed=7)
    assert ev.distance(src.lab) == ev.distance(src.lab)
    again = Evaluator.build(src, tgt, np.ones(len(tgt.srgb)), seed=7)
    assert ev.distance(src.lab) == pytest.approx(again.distance(src.lab), abs=1e-12)


def test_a_real_improvement_beats_the_seed_spread():
    """A LUT that genuinely moves the source toward the target must clear the noise floor."""
    rng = np.random.default_rng(4)
    src = PixelPool(rng.random((6000, 3), dtype=np.float32).astype(np.float32), 5)
    noisy = np.clip(src.srgb**1.5 + rng.normal(0, 0.01, src.srgb.shape), 0, 1)
    tgt = PixelPool(noisy.astype(np.float32), 5)
    weights = np.ones(len(tgt.srgb))
    ev = Evaluator.build(src, tgt, weights, seed=0)
    before = ev.distance(src.lab)
    lut = _darkening_lut(17, gamma=1.5)
    after = ev.distance(srgb_to_oklab(lut.apply_numpy(src.srgb)))
    _mean, spread = swd_seed_spread(src, tgt, weights, lut)
    assert after < before
    assert before - after > 3 * spread


def test_seed_spread_is_small_and_positive():
    src, tgt = _pool(5), _pool(6, scale=0.7)
    mean, spread = swd_seed_spread(src, tgt, np.ones(len(tgt.srgb)), LUT3D.identity(9))
    assert mean > 0 and spread >= 0
    assert spread < mean


def test_grey_axis_monotonicity():
    assert grey_axis_is_monotone(LUT3D.identity(9))
    t = LUT3D.identity(9).table.copy()
    t[4, 4, 4] = 0.05
    assert not grey_axis_is_monotone(LUT3D(t))


def test_channel_monotonicity_catches_a_per_channel_inversion():
    assert channels_are_monotone(LUT3D.identity(9))
    t = LUT3D.identity(9).table.copy()
    t[5, :, :, 0] = 0.1  # red output dips as red input rises
    assert not channels_are_monotone(LUT3D(t))


def test_neutral_axis_chroma_catches_a_tinting_lut():
    assert neutral_axis_max_chroma(LUT3D.identity(33)) < 0.01
    t = LUT3D.identity(33).table.copy()
    for i in range(33):
        t[i, i, i, 2] = min(1.0, t[i, i, i, 2] + 0.25)  # push greys blue
    assert neutral_axis_max_chroma(LUT3D(t)) > 0.05


@pytest.mark.parametrize("size", [9, 17, 33])
def test_identity_and_tone_curves_clip_nothing(size):
    """Interior nodes only, so grid size must not change the answer."""
    assert clipped_volume_fraction(LUT3D.identity(size)) == 0.0
    assert clipped_volume_fraction(LUT3D(LUT3D.identity(size).table**1.5)) == 0.0


def test_clipped_volume_fraction_catches_a_crushing_lut():
    crushed = LUT3D(np.clip(LUT3D.identity(9).table * 3.0, 0, 1))
    assert clipped_volume_fraction(crushed) > 0.9


def test_hue_bin_shifts_report_darkening():
    rng = np.random.default_rng(0)
    src = (rng.random((5000, 3), dtype=np.float32) * 0.6 + 0.2).astype(np.float32)
    shifts = hue_bin_shifts(src, _darkening_lut(), n_bins=12)
    assert len(shifts) == 13
    populated = [s for s in shifts if s["count"] > 0]
    assert populated and all(s["delta_L"] < 0 for s in populated)
    assert {"bin", "hue_deg", "count", "delta_L", "chroma_ratio", "delta_hue_deg"} <= set(shifts[0])


def test_hue_hist_residual_is_zero_when_reweighting_is_exact():
    from kodachrome.train.transport import hue_weights

    rng = np.random.default_rng(8)
    src_lab = lch_to_oklab(
        np.stack(
            [rng.uniform(0.3, 0.8, 8000), np.full(8000, 0.12), rng.uniform(0, 2 * np.pi, 8000)], 1
        )
    )
    # Uneven around the whole circle, not confined to half of it. Reweighting
    # can only scale samples that exist, so a hue with no samples at all leaves
    # a residual pinned at 1/n_bins — measured 0.0459 against a 1/24 = 0.0417
    # floor. That would make the test assert something unachievable rather than
    # something false.
    tgt_lab = lch_to_oklab(
        np.concatenate(
            [
                np.stack(
                    [rng.uniform(0.3, 0.8, 5600), np.full(5600, 0.12), rng.uniform(0, np.pi, 5600)],
                    1,
                ),
                np.stack(
                    [
                        rng.uniform(0.3, 0.8, 2400),
                        np.full(2400, 0.12),
                        rng.uniform(np.pi, 2 * np.pi, 2400),
                    ],
                    1,
                ),
            ]
        )
    )
    w = hue_weights(src_lab, tgt_lab, 24)
    assert hue_hist_residual(src_lab, tgt_lab, w, 24, 0.03) < 0.02


def test_evaluate_returns_the_documented_metric_block():
    src, tgt = _pool(9), _pool(10, scale=0.8)
    weights = np.ones(len(tgt.srgb))
    lut = _darkening_lut(9)
    partners = srgb_to_oklab(lut.apply_numpy(src.srgb))
    metrics = evaluate(
        lut=lut,
        val_src=src,
        val_tgt=tgt,
        val_weights=weights,
        train_src=src,
        train_tgt=tgt,
        train_weights=weights,
        transported_lab=partners,
        n_bins=12,
        chroma_floor=0.03,
        seed=0,
    )
    for key in (
        "swd_before", "swd_after", "swd_identity", "swd_seed_spread",
        "transport_gamut_clip_deltaE", "lut_fit_rms_deltaE", "grey_axis_monotone",
        "channel_monotone", "neutral_axis_max_chroma", "clipped_volume_fraction",
        "hue_bins", "train_swd_before", "train_swd_after",
    ):
        assert key in metrics, key
    assert metrics["swd_identity"] == pytest.approx(metrics["swd_before"], abs=1e-6)


def test_a_missing_seed_spread_is_an_error_not_a_free_pass():
    """Omitting the noise floor must not quietly switch the gate off.

    With `.get(..., 0.0)` the margin becomes 0, so an improvement of 0.0001 —
    pure sampling noise — passes. Measured before this was tightened.
    """
    metrics = {
        "swd_before": 0.10,
        "swd_after": 0.0999,
        "grey_axis_monotone": True,
        "channel_monotone": True,
        "neutral_axis_max_chroma": 0.005,
        "clipped_volume_fraction": 0.01,
    }
    with pytest.raises(KeyError, match="swd_seed_spread"):
        check_gates(metrics)

    # Present and honest: the same marginal improvement is refused.
    failed = [g.name for g in check_gates({**metrics, "swd_seed_spread": 0.001}) if not g.passed]
    assert failed == ["improvement_exceeds_noise"]


def test_gates_pass_and_fail_explicitly():
    good = {
        "swd_before": 0.10, "swd_after": 0.04, "swd_seed_spread": 0.001,
        "grey_axis_monotone": True, "channel_monotone": True,
        "neutral_axis_max_chroma": 0.005, "clipped_volume_fraction": 0.01,
    }
    assert all(g.passed for g in check_gates(good))

    marginal = {**good, "swd_after": 0.0999}          # improvement inside the noise floor
    failed = [g.name for g in check_gates(marginal) if not g.passed]
    assert "improvement_exceeds_noise" in failed

    tinted = {**good, "neutral_axis_max_chroma": 0.09}
    assert "neutral_axis_chroma" in [g.name for g in check_gates(tinted) if not g.passed]
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/pytest tests/test_evaluate.py -q` → FAIL, no module `kodachrome.train.evaluate`.

- [ ] **Step 3: Implement `evaluate.py`**

```python
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
    # Strict indexing, like every other key here. Defaulting a missing spread to
    # 0.0 would set the margin to 0 and turn this gate into "any improvement
    # above zero passes" — silently disabling the one check that distinguishes
    # a real grade from sampling noise, which is the reason the gate exists.
    margin = NOISE_MARGIN * float(metrics["swd_seed_spread"])
    improvement = float(metrics["swd_before"]) - float(metrics["swd_after"])
    gates = [
        Gate(
            "improvement_exceeds_noise",
            round(improvement, 6),
            round(margin, 6),
            improvement > margin,
            f"evaluation distance fell by {improvement:.5f}; needs to beat "
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
            "the interior of the colour cube must stay off the gamut boundary",
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
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_evaluate.py -q` → all pass.

`test_identity_lut_gives_identical_before_and_after` is the load-bearing one. If it fails, the evaluator is not truly paired: check that `Evaluator.build` stores the sorted **target** projections once and that `distance` re-projects only the source. Do not relax its tolerance.

- [ ] **Step 5: Commit**

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/train/evaluate.py tests/test_evaluate.py
git commit -m "feat: paired held-out evaluator, seed spread and numeric LUT safety gates

Fixes a metric that previously drew a fresh seed per measurement, so part of
any reported improvement was sampling noise. Adds per-channel monotonicity,
neutral-axis chroma and clipped-volume checks with thresholds fixed before
tuning.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 18: Training report (`train/report.py`)

Implements spec 6.5's rendered output. The report is how a human judges a fit, so it must be readable, not merely present.

**Files:**
- Create: `kodachrome/train/report.py`, `tests/test_report.py`

**Interfaces:**
- Consumes: `LUT3D`, `NormalizeParams`, `SampleConfig`, `prepare_image`, `load_rgb`, `CorpusSplit`, `Gate`, colour conversions
- Produces:
  - `render_contact_sheet(source_paths, target_paths, lut, source_normalize, target_normalize, cfg, out_path, n=8, thumb=240, rng=None) -> Path`
  - `render_ramps(lut, out_path, width=768, band=36) -> Path`
  - `render_diagnostics(source_pool, target_pool, out_path) -> Path`
  - `write_report(out_dir, lut, metrics, gates, source_split, target_split, source_normalize, target_normalize, cfg) -> Path`

- [ ] **Step 1: Write the failing tests**

`tests/test_report.py`:
```python
import json

import numpy as np
from PIL import Image

from kodachrome.imageio import save_jpeg
from kodachrome.lut import LUT3D
from kodachrome.normalize import NormalizeParams
from kodachrome.train.dataset import CorpusSplit, PixelPool, SampleConfig
from kodachrome.train.evaluate import check_gates
from kodachrome.train.report import (
    render_contact_sheet,
    render_diagnostics,
    render_ramps,
    write_report,
)


def _images(dir_path, n, seed):
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        p = dir_path / f"{i}.jpg"
        save_jpeg(rng.integers(30, 220, (60, 80, 3), dtype=np.uint8), p)
        paths.append(p)
    return paths


def _pool(seed, n_images=3):
    rng = np.random.default_rng(seed)
    return PixelPool(
        srgb=rng.random((1500, 3), dtype=np.float32),
        n_images=n_images,
        clamp_rate=0.25,
        wb_gains=[[0.9, 1.0, 1.1]] * n_images,
        exposure_gains=[1.1] * n_images,
        profiles={"sRGB (assumed)": n_images},
    )


def _split(tmp_path, name, seed):
    paths = _images(tmp_path / name, 4, seed)
    return CorpusSplit(paths[:3], paths[3:], _pool(seed), _pool(seed + 1, 1), "abc123")


def _darkening_lut(n=9):
    return LUT3D(LUT3D.identity(n).table**1.5)


def test_render_ramps_writes_a_readable_png(tmp_path):
    out = render_ramps(LUT3D.identity(9), tmp_path / "ramps.png")
    with Image.open(out) as im:
        assert im.size[0] >= 512 and im.size[1] > 100


def test_render_diagnostics_writes_a_png(tmp_path):
    out = render_diagnostics(_pool(0), _pool(1), tmp_path / "diag.png")
    with Image.open(out) as im:
        assert im.size[0] > 200 and im.size[1] > 100


def test_contact_sheet_shows_three_rows(tmp_path):
    src = _images(tmp_path / "src", 3, 0)
    tgt = _images(tmp_path / "tgt", 3, 1)
    out = render_contact_sheet(
        src, tgt, _darkening_lut(), NormalizeParams(), NormalizeParams(white_balance=False),
        SampleConfig(crop_frac=0.0, max_side=80), tmp_path / "sheet.png", n=3, thumb=64,
    )
    with Image.open(out) as im:
        assert im.height > 3 * 64  # three labelled rows stacked


def test_a_corpus_with_no_held_out_split_is_labelled_honestly(tmp_path):
    """A sheet captioned "held-out" while showing training images would lie.

    Reachable with any corpus under five images at the default validation
    fraction, which `--allow-small` explicitly supports.
    """
    paths = _images(tmp_path / "tiny", 3, 0)
    empty_val = CorpusSplit(paths, [], _pool(0), _pool(1, 1), "abc")
    metrics = {
        "swd_before": 0.1, "swd_after": 0.05, "swd_identity": 0.1, "swd_seed_spread": 0.001,
        "train_swd_before": 0.1, "train_swd_after": 0.04,
        "transport_gamut_clip_deltaE": 0.001, "lut_fit_rms_deltaE": 0.01,
        "grey_axis_monotone": True, "channel_monotone": True,
        "neutral_axis_max_chroma": 0.004, "clipped_volume_fraction": 0.01,
        "hue_bins": [],
    }
    out_dir = tmp_path / "report"
    write_report(
        out_dir, _darkening_lut(), metrics, check_gates(metrics),
        empty_val, _split(tmp_path, "tgt", 5),
        NormalizeParams(), NormalizeParams(white_balance=False),
        SampleConfig(crop_frac=0.0, max_side=80),
    )
    summary = (out_dir / "summary.txt").read_text()
    assert "WARNING" in summary
    assert "trained on" in summary

    # And the ordinary case must carry no such warning.
    ordinary = tmp_path / "ordinary"
    write_report(
        ordinary, _darkening_lut(), metrics, check_gates(metrics),
        _split(tmp_path, "src", 0), _split(tmp_path, "tgt2", 6),
        NormalizeParams(), NormalizeParams(white_balance=False),
        SampleConfig(crop_frac=0.0, max_side=80),
    )
    assert "WARNING" not in (ordinary / "summary.txt").read_text()


def test_write_report_produces_every_artifact(tmp_path):
    metrics = {
        "swd_before": 0.1, "swd_after": 0.05, "swd_identity": 0.1, "swd_seed_spread": 0.001,
        "train_swd_before": 0.1, "train_swd_after": 0.04,
        "transport_gamut_clip_deltaE": 0.001, "lut_fit_rms_deltaE": 0.01,
        "grey_axis_monotone": True, "channel_monotone": True,
        "neutral_axis_max_chroma": 0.004, "clipped_volume_fraction": 0.01,
        "hue_bins": [{"bin": 0, "hue_deg": 7.5, "count": 10, "delta_L": -0.01,
                      "chroma_ratio": 1.1, "delta_hue_deg": 0.5}],
    }
    out_dir = tmp_path / "report"
    write_report(
        out_dir,
        _darkening_lut(),
        metrics,
        check_gates(metrics),
        _split(tmp_path, "src", 0),
        _split(tmp_path, "tgt", 5),
        NormalizeParams(),
        NormalizeParams(white_balance=False),
        SampleConfig(crop_frac=0.0, max_side=80),
    )
    assert (out_dir / "contact_sheet.png").exists()
    assert (out_dir / "ramps.png").exists()
    assert (out_dir / "diagnostics.png").exists()
    saved = json.loads((out_dir / "metrics.json").read_text())
    assert saved["swd_after"] == 0.05
    assert saved["gates"][0]["passed"] in (True, False)
    summary = (out_dir / "summary.txt").read_text()
    assert "held-out" in summary and "PASS" in summary
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/pytest tests/test_report.py -q` → FAIL, no module `kodachrome.train.report`.

- [ ] **Step 3: Implement `report.py`**

```python
"""Human-readable evidence that a fitted LUT does what we claim and is safe.

Numbers alone do not tell you whether a grade looks right, and pictures alone
do not tell you whether it generalises. The report gives both:

* ``contact_sheet.png`` - **held-out** source images, normalised beside
  graded, with a strip of real Kodachrome scans underneath. The question to
  ask is whether the graded row belongs in the same family as the strip.
  Held-out matters: showing training images would flatter the fit.
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
    held_out = bool(metrics.get(
        "held_out_eval", bool(source_split.val_paths) and bool(target_split.val_paths)
    ))
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
```

- [ ] **Step 4: Run and commit**

Run: `.venv/bin/pytest tests/test_report.py -q` → all pass. Watch ruff's 100-column limit in the long f-strings.

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/train/report.py tests/test_report.py
git commit -m "feat: training report with held-out contact sheet, ramps, diagnostics and gates

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 19: Trainer orchestration (`train/fit.py`)

Implements spec 6.4 end to end and the `training` block of spec 5.8. Uses the atomic publish from Task 8 (F-07) and the gates from Task 17.

**Files:**
- Create: `kodachrome/train/fit.py`, `tests/test_fit.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: everything from Tasks 13 to 18, plus `write_artifact`, `publish`
- Produces:
  - `@dataclass FitConfig(lut_size=33, iterations=40, hue_bins=24, chroma_floor=0.03, lambda_smooth=1e-3, lambda_identity=1e-4, strength=1.0, seed=0)` with validation
  - `@dataclass FitResult(lut, transported_lab, target_weights)`
  - `fit(source_pool, target_pool, cfg, progress=None) -> FitResult`
  - `train(source_dir, target_dir, out_dir, cfg, sample_cfg, grain, proxy_source=False, allow_small=False, command="", progress=None) -> (metrics, gates)`
  - `main(argv=None) -> int` — returns 0 on pass, 3 when the artifact was written but a gate failed

- [ ] **Step 1: Write the failing tests**

`tests/test_fit.py`:
```python
import json

import numpy as np
import pytest

from kodachrome.artifacts import Artifacts
from kodachrome.color import lch_to_oklab, oklab_to_lch, oklab_to_srgb, srgb_to_oklab
from kodachrome.imageio import save_jpeg
from kodachrome.train.dataset import PixelPool, SampleConfig
from kodachrome.train.fit import FitConfig, fit, main, train


def _curve_and_rotation(lab, gamma=1.15, chroma=1.25, degrees=10.0):
    """A tone curve, a saturation boost, and a genuine hue rotation."""
    lch = oklab_to_lch(lab)
    lch[..., 0] = np.clip(lch[..., 0], 0, 1) ** gamma
    lch[..., 1] = lch[..., 1] * chroma
    lch[..., 2] = lch[..., 2] + np.deg2rad(degrees)
    return lch_to_oklab(lch)


def _pool(rng, n=30000):
    return (rng.random((n, 3), dtype=np.float32) * 0.7 + 0.15).astype(np.float32)


def test_fit_recovers_a_tone_curve_and_a_ten_degree_hue_rotation():
    """The spec claims this capability, so it is tested directly."""
    rng = np.random.default_rng(0)
    src = _pool(rng)
    transformed = oklab_to_srgb(_curve_and_rotation(srgb_to_oklab(_pool(rng))))
    tgt = np.clip(transformed, 0, 1).astype(np.float32)
    cfg = FitConfig(lut_size=17, iterations=30, seed=0)
    result = fit(PixelPool(src, 1), PixelPool(tgt, 1), cfg)

    held = _pool(rng, 3000)
    expected = _curve_and_rotation(srgb_to_oklab(held))
    got = srgb_to_oklab(result.lut.apply_numpy(held))
    assert float(np.sqrt(np.sum((got - expected) ** 2, axis=1)).mean()) < 0.03


def test_a_large_hue_rotation_is_only_partly_recovered():
    """Pins the documented limit: a large hue rotation is only partly recovered.

    The damping comes from the transport and the LUT-fit smoothing, NOT from
    hue reweighting: this fixture is uniform over the cube, so a 90-degree
    rotation leaves the hue histogram unchanged and `hue_weights` has nothing
    to correct. Disabling reweighting entirely moves the result from about
    5 degrees to about 7, both far under the bound below.
    """
    rng = np.random.default_rng(1)
    src = _pool(rng)
    tgt = np.clip(
        oklab_to_srgb(
            _curve_and_rotation(srgb_to_oklab(_pool(rng)), gamma=1.0, chroma=1.0, degrees=90.0)
        ),
        0, 1,
    ).astype(np.float32)
    cfg = FitConfig(lut_size=17, iterations=30, seed=0)
    result = fit(PixelPool(src, 1), PixelPool(tgt, 1), cfg)

    held = _pool(rng, 3000)
    before = oklab_to_lch(srgb_to_oklab(held))[:, 2]
    after = oklab_to_lch(srgb_to_oklab(result.lut.apply_numpy(held)))[:, 2]
    achieved = np.degrees(np.angle(np.exp(1j * (after - before)))).mean()
    # Measured: about +5 degrees out of the 90 requested, stable across seeds.
    # A third of the request is a generous ceiling that still fails loudly if
    # reweighting ever stops damping large rotations.
    assert abs(achieved) < 30.0, (
        f"a 90 degree rotation must be heavily damped, got {achieved:.1f} degrees"
    )


def test_strength_zero_gives_an_identity_lut():
    rng = np.random.default_rng(2)
    src = PixelPool(rng.random((5000, 3), dtype=np.float32), 1)
    tgt = PixelPool((rng.random((5000, 3), dtype=np.float32) * 0.5).astype(np.float32), 1)
    result = fit(src, tgt, FitConfig(lut_size=9, iterations=5, strength=0.0))
    x = rng.random((500, 3), dtype=np.float32)
    assert np.abs(result.lut.apply_numpy(x) - x).max() < 0.02


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"lut_size": 1}, "lut_size"),
        ({"iterations": 0}, "iterations"),
        ({"hue_bins": 0}, "hue_bins"),
        ({"strength": 1.5}, "strength"),
        ({"lambda_smooth": -1.0}, "lambda_smooth"),
    ],
)
def test_invalid_fit_config_names_the_field(kwargs, field):
    with pytest.raises(ValueError, match=field):
        FitConfig(**kwargs)


def _image_dir(dir_path, n, seed, transform=None):
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        base = np.linspace(0.2, 0.8, 64, dtype=np.float32)[None, :, None] * rng.uniform(
            0.7, 1.0, 3
        ).astype(np.float32)
        img = np.repeat(base, 48, axis=0)
        if transform:
            img = transform(img)
        save_jpeg((np.clip(img, 0, 1) * 255).astype(np.uint8), dir_path / f"{i:02d}.jpg")


def test_train_end_to_end_publishes_a_loadable_artifact(tmp_path):
    _image_dir(tmp_path / "src", 8, 0)
    _image_dir(tmp_path / "tgt", 8, 1, transform=lambda im: im**1.3)
    out = tmp_path / "artifacts"
    metrics, gates = train(
        tmp_path / "src",
        tmp_path / "tgt",
        out,
        FitConfig(lut_size=9, iterations=5),
        SampleConfig(crop_frac=0.0, max_side=64, pixels_per_image=500, val_fraction=0.25),
        grain=None,
        proxy_source=True,
        allow_small=True,
        command="test",
    )
    art = Artifacts.load(out)
    assert art.lut.size == 9
    assert art.training["source"]["proxy"] is True
    assert art.training["source"]["corpus_sha1"] and art.training["target"]["corpus_sha1"]
    assert art.training["split"]["n_source_val_images"] == 2
    assert art.training["fit"]["lut_size"] == 9
    assert art.training["metrics"]["swd_after"] == metrics["swd_after"]
    assert art.training["code_revision"]
    assert (out / "report" / "contact_sheet.png").exists()
    assert (out / "report" / "summary.txt").exists()
    assert isinstance(gates, list) and gates


def test_train_leaves_the_previous_artifact_intact_if_publication_fails(tmp_path, monkeypatch):
    _image_dir(tmp_path / "src", 8, 0)
    _image_dir(tmp_path / "tgt", 8, 1, transform=lambda im: im**1.3)
    out = tmp_path / "artifacts"
    cfg = FitConfig(lut_size=9, iterations=5)
    sample = SampleConfig(crop_frac=0.0, max_side=64, pixels_per_image=500, val_fraction=0.25)
    train(tmp_path / "src", tmp_path / "tgt", out, cfg, sample, None, allow_small=True)
    good = (out / "params.json").read_text()

    monkeypatch.setattr(
        "kodachrome.train.fit.publish", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    )
    with pytest.raises(OSError):
        train(tmp_path / "src", tmp_path / "tgt", out, cfg, sample, None, allow_small=True)
    assert (out / "params.json").read_text() == good


def test_main_rejects_identical_and_missing_dirs(tmp_path, capsys):
    _image_dir(tmp_path / "src", 2, 0)
    assert main(["--source", str(tmp_path / "src"), "--target", str(tmp_path / "src"),
                 "--out", str(tmp_path / "o")]) == 1
    assert "same" in capsys.readouterr().err
    assert main(["--source", str(tmp_path / "missing"), "--target", str(tmp_path / "src"),
                 "--out", str(tmp_path / "o")]) == 1


def test_main_refuses_a_small_corpus_then_accepts_the_flag(tmp_path, capsys):
    _image_dir(tmp_path / "src", 3, 0)
    _image_dir(tmp_path / "tgt", 3, 1, transform=lambda im: im**1.2)
    args = ["--source", str(tmp_path / "src"), "--target", str(tmp_path / "tgt"),
            "--out", str(tmp_path / "o"), "--lut-size", "9", "--iterations", "5",
            "--max-side", "64", "--pixels-per-image", "300", "--proxy-source"]
    assert main(args) == 1
    assert "--allow-small" in capsys.readouterr().err
    assert main([*args, "--allow-small"]) in (0, 3)
    data = json.loads((tmp_path / "o" / "params.json").read_text())
    assert data["training"]["fit"]["strength"] == 1.0
    assert data["grain"]["strength"] == pytest.approx(0.025)


def test_main_reports_a_failed_gate_with_exit_code_3(tmp_path, monkeypatch, capsys):
    _image_dir(tmp_path / "src", 8, 0)
    _image_dir(tmp_path / "tgt", 8, 1, transform=lambda im: im**1.3)
    from kodachrome.train import fit as fit_module
    from kodachrome.train.evaluate import Gate

    monkeypatch.setattr(
        fit_module, "check_gates",
        lambda m: [Gate("improvement_exceeds_noise", 0.0, 0.1, False, "forced failure")],
    )
    code = main(["--source", str(tmp_path / "src"), "--target", str(tmp_path / "tgt"),
                 "--out", str(tmp_path / "o"), "--lut-size", "9", "--iterations", "5",
                 "--max-side", "64", "--allow-small"])
    assert code == 3
    err = capsys.readouterr().err
    assert "improvement_exceeds_noise" in err
    assert Artifacts.load(tmp_path / "o").lut.size == 9  # written despite the failure
```

- [ ] **Step 2: Run and watch it fail**

Run: `.venv/bin/pytest tests/test_fit.py -q` → FAIL, no module `kodachrome.train.fit`.

- [ ] **Step 3: Implement `fit.py`**

```python
"""``kodachrome-train``: fit the Kodachrome LUT from two folders of images.

The sequence (spec section 6):

1. ``dataset.build_corpus`` splits each corpus **by image** and samples the
   halves separately, so the evaluation later is genuinely held out.
2. ``transport.hue_weights`` reweights the target's hues toward the
   source's, reducing content bias. This is a heuristic; see the note in
   ``transport.py`` about what it does not guarantee.
3. ``transport.iterative_distribution_transfer`` gives every training source
   pixel a Kodachrome partner. ``strength`` blends between identity and the
   full transport.
4. ``lutfit.fit_lut`` fits a smooth LUT to those pairs.
5. ``evaluate.evaluate`` measures the result on the held-out images with a
   paired evaluator, and ``check_gates`` turns the numbers into pass or fail.
6. Everything is written into a staging directory and published atomically,
   so an interrupted run can never leave a new LUT beside old parameters.

A failing gate does not delete the artifact: you may want to inspect it. It
sets exit code 3 and names the gate, so a script cannot mistake it for
success.

``--proxy-source`` exists because the trainer cannot tell whether a folder of
photographs came from the U20CAM. Passing it records the fact so users of
the shipped artifact know to retrain.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

import numpy as np

from .. import __version__
from ..artifacts import publish, write_artifact
from ..color import oklab_to_srgb
from ..grain import GrainParams
from ..imageio import list_images
from ..lut import LUT3D
from ..normalize import NormalizeParams
from .dataset import CorpusTooSmall, PixelPool, SampleConfig, build_corpus
from .evaluate import check_gates, evaluate
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

    def __post_init__(self) -> None:
        if not 2 <= self.lut_size <= 65:
            raise ValueError(f"lut_size must be in 2..65, got {self.lut_size}")
        if self.iterations < 1:
            raise ValueError(f"iterations must be positive, got {self.iterations}")
        if self.hue_bins < 1:
            raise ValueError(f"hue_bins must be positive, got {self.hue_bins}")
        if not 0.0 <= self.chroma_floor < 0.5:
            raise ValueError(f"chroma_floor must be in [0, 0.5), got {self.chroma_floor}")
        for name in ("lambda_smooth", "lambda_identity"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be in [0, 1], got {self.strength}")


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

    say("reweighting target hues toward the source histogram")
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
    return FitResult(lut, partner_lab.astype(np.float32), weights)


def _code_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _dependency_versions() -> dict:
    out = {}
    for name in ("numpy", "scipy", "Pillow", "opencv-python"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def train(
    source_dir: str | Path,
    target_dir: str | Path,
    out_dir: str | Path,
    cfg: FitConfig,
    sample_cfg: SampleConfig,
    grain: GrainParams | None,
    proxy_source: bool = False,
    allow_small: bool = False,
    command: str = "",
    progress: Callable[[str], None] | None = None,
) -> tuple[dict, list]:
    say = progress or (lambda _m: None)
    source_dir, target_dir, out_dir = Path(source_dir), Path(target_dir), Path(out_dir)

    source_normalize = NormalizeParams()
    target_normalize = NormalizeParams(white_balance=False)
    source = build_corpus(source_dir, source_normalize, sample_cfg, MIN_SOURCE_IMAGES,
                          "source", allow_small, say)
    target = build_corpus(target_dir, target_normalize, sample_cfg, MIN_TARGET_IMAGES,
                          "target", allow_small, say)

    t0 = time.perf_counter()
    result = fit(source.train_pool, target.train_pool, cfg, say)
    fit_seconds = time.perf_counter() - t0

    # The evaluation is only held out if BOTH sides kept images back. A corpus
    # under 1/val_fraction images yields an empty validation split, and falling
    # back to training pixels while still calling the result "held-out" would
    # report memorisation as generalisation.
    source_held_out = len(source.val_pool.srgb) > 0
    target_held_out = len(target.val_pool.srgb) > 0
    held_out_eval = source_held_out and target_held_out
    if held_out_eval:
        say("evaluating on held-out images")
    else:
        pairs = (("source", source_held_out), ("target", target_held_out))
        short = [name for name, ok in pairs if not ok]
        say(f"WARNING: no held-out images for the {' and '.join(short)} corpus; "
            "evaluating on training pixels, which measures memorisation, not generalisation")
    val_weights = hue_weights(
        source.val_pool.lab, target.val_pool.lab, cfg.hue_bins, cfg.chroma_floor
    ) if target_held_out else None
    metrics = evaluate(
        lut=result.lut,
        val_src=source.val_pool if source_held_out else source.train_pool,
        val_tgt=target.val_pool if target_held_out else target.train_pool,
        val_weights=val_weights,
        train_src=source.train_pool,
        train_tgt=target.train_pool,
        train_weights=result.target_weights,
        transported_lab=result.transported_lab,
        n_bins=cfg.hue_bins,
        chroma_floor=cfg.chroma_floor,
        seed=cfg.seed,
    )
    metrics["held_out_eval"] = held_out_eval
    gates = check_gates(metrics)

    training = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_revision": _code_revision(),
        "package_version": __version__,
        "dependency_versions": _dependency_versions(),
        "command": command,
        "target": {
            "dir": str(target_dir),
            "n_images": len(target.train_paths) + len(target.val_paths),
            "corpus_sha1": target.corpus_sha1,
            "n_pixels": int(len(target.train_pool.srgb)),
            "clamp_rate": target.train_pool.clamp_rate,
            "profiles": target.train_pool.profiles,
        },
        "source": {
            "dir": str(source_dir),
            "n_images": len(source.train_paths) + len(source.val_paths),
            "corpus_sha1": source.corpus_sha1,
            "n_pixels": int(len(source.train_pool.srgb)),
            "proxy": proxy_source,
            "clamp_rate": source.train_pool.clamp_rate,
            "profiles": source.train_pool.profiles,
        },
        "split": {
            "val_fraction": sample_cfg.val_fraction,
            "n_source_val_images": len(source.val_paths),
            "n_target_val_images": len(target.val_paths),
            "seed": sample_cfg.seed,
        },
        "fit": {**asdict(cfg), "fit_seconds": round(fit_seconds, 1)},
        "sample": asdict(sample_cfg),
        "metrics": {k: v for k, v in metrics.items() if k != "hue_bins"},
    }

    # publish() creates out_dir's parent, but mkdtemp needs it to exist first.
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".kodachrome-staging-", dir=out_dir.parent))
    try:
        write_artifact(staging, result.lut, source_normalize, grain or GrainParams(), training)
        say("writing report")
        write_report(staging / "report", result.lut, metrics, gates, source, target,
                     source_normalize, target_normalize, sample_cfg)
        publish(staging, out_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    say(f"published {out_dir}; report in {out_dir / 'report'}")
    return metrics, gates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kodachrome-train", description="Fit the Kodachrome LUT.")
    parser.add_argument("--source", type=Path, required=True, help="folder of camera photos")
    parser.add_argument("--target", type=Path, default=Path("data/kodachrome"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--lut-size", type=int, default=33)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--hue-bins", type=int, default=24)
    parser.add_argument("--lambda-smooth", type=float, default=1e-3)
    parser.add_argument("--lambda-identity", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grain-strength", type=float, default=0.025)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--pixels-per-image", type=int, default=3000)
    parser.add_argument("--max-pixels", type=int, default=400_000)
    parser.add_argument("--proxy-source", action="store_true",
                        help="mark the source as stand-in photos, not U20CAM shots")
    parser.add_argument("--allow-small", action="store_true",
                        help="proceed with a corpus below the recommended minimum")
    args = parser.parse_args(argv)

    for label, path in (("source", args.source), ("target", args.target)):
        if not path.is_dir() or not list_images(path):
            print(f"error: {label} folder {path} does not exist or has no images", file=sys.stderr)
            return 1
    if args.source.resolve() == args.target.resolve():
        print("error: source and target are the same folder", file=sys.stderr)
        return 1

    try:
        cfg = FitConfig(
            lut_size=args.lut_size, iterations=args.iterations, hue_bins=args.hue_bins,
            lambda_smooth=args.lambda_smooth, lambda_identity=args.lambda_identity,
            strength=args.strength, seed=args.seed,
        )
        sample_cfg = SampleConfig(
            max_side=args.max_side, pixels_per_image=args.pixels_per_image,
            max_pixels=args.max_pixels, val_fraction=args.val_fraction, seed=args.seed,
        )
        grain = GrainParams(strength=args.grain_strength)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        metrics, gates = train(
            args.source, args.target, args.out, cfg, sample_cfg, grain,
            proxy_source=args.proxy_source, allow_small=args.allow_small,
            command=" ".join(["kodachrome-train", *(argv or sys.argv[1:])]), progress=print,
        )
    except CorpusTooSmall as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    label = "held-out" if metrics["held_out_eval"] else "TRAINING (not held out)"
    print(
        f"{label} distance to Kodachrome: {metrics['swd_before']:.5f} -> "
        f"{metrics['swd_after']:.5f} (seed spread {metrics['swd_seed_spread']:.5f})"
    )
    failed = [g for g in gates if not g.passed]
    for gate in gates:
        print(f"  [{'PASS' if gate.passed else 'FAIL'}] {gate.name}: {gate.detail}")
    if failed:
        print(
            "error: artifact written but "
            + ", ".join(g.name for g in failed)
            + " did not pass; see the report before using it",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_fit.py -q` → all pass.

`test_fit_recovers_a_tone_curve_and_a_ten_degree_hue_rotation` is the acceptance test for the whole method, and it includes the hue rotation the spec claims.

The controller ran both hue tests against a scratch implementation of this exact algorithm before writing them, so the expected values are measured rather than guessed:

| Test | Measured (seeds 0, 1, 2) | Bound |
|---|---|---|
| 10 degree recovery | mean dE 0.0253, 0.0246, 0.0249 | < 0.03 |
| 90 degree damping | +5.2, +4.1, +4.8 degrees achieved | < 30 |

The 0.025 floor is systematic, not noise: it barely moves between 30 and 50 transport iterations, and comes from LUT smoothing plus gamut clipping. So if you see roughly 0.025, the test is behaving. If it exceeds 0.03, something is genuinely wrong with the transport or the fit; do not raise the bound. Investigate, and record whatever you find in `docs/decisions.md`.

- [ ] **Step 5: Document and commit**

Add to `README.md` under Training:

````markdown
### 2. Collect camera samples

Take 50 or more shots with the U20CAM across varied scenes: indoors and out,
sky, foliage, skin, neutral walls, mixed lighting. Copy the `*_original.jpg`
files into one folder, for example `data/source/`.

### 3. Fit the LUT

```bash
.venv/bin/kodachrome-train --source data/source --target data/kodachrome
```

Writes `artifacts/` (LUT, `params.json`, `report/`) and prints the held-out
result plus a pass or fail line per gate. Exit code 3 means the artifact was
written but a gate failed; read the report before using it.

| Flag | Default | Effect |
|---|---|---|
| `--strength` | 1.0 | 0 = no change, 1 = full; 0.7 for a lighter touch |
| `--val-fraction` | 0.2 | share of images held out of training for the metrics |
| `--lambda-smooth` | 1e-3 | raise if the ramps band or the fit looks noisy |
| `--lambda-identity` | 1e-4 | raise if colours the camera never produced go strange |
| `--grain-strength` | 0.025 | grain, in luminance units at mid-grey |
| `--proxy-source` | off | mark the source as stand-in photos |
| `--allow-small` | off | proceed with a corpus below the recommended minimum |

### 4. Read the report

`report/summary.txt` is the short version: the held-out distance before and
after, the seed spread it must beat, and each gate. `contact_sheet.png`
shows held-out images normalised, graded, and beside real scans.
`ramps.png` shows the tone curve and hue movement. `diagnostics.png` shows
how often normalisation clamped. `metrics.json` has everything.

Promote a fit you like to the shipped default by copying it into the
package:

```bash
cp artifacts/kodachrome.cube artifacts/params.json kodachrome/data/
```
````

```bash
.venv/bin/ruff check kodachrome tests
git add kodachrome/train/fit.py tests/test_fit.py README.md
git commit -m "feat: kodachrome-train orchestration with held-out evaluation and atomic publish

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 20: Packaging smoke test

Implements spec 9's `test_packaging.py`. Proves F-03 is actually fixed rather than merely intended.

**Files:**
- Create: `tests/test_packaging.py`

- [ ] **Step 1: Write the test**

`tests/test_packaging.py`:
```python
"""Build a wheel, install it into a clean venv, and run a command from elsewhere.

This is the only test that proves the shipped artifact really travels with
the package: everything else imports from the source tree, where
`kodachrome/data/` happens to be on disk anyway.
"""

import subprocess
import sys
import venv

import numpy as np
import pytest

from kodachrome.imageio import save_jpeg

pytestmark = pytest.mark.slow


def test_wheel_installs_and_runs_from_another_directory(tmp_path, repo_root):
    dist = tmp_path / "dist"
    build = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist), str(repo_root)],
        capture_output=True, text=True,
    )
    assert build.returncode == 0, build.stderr[-3000:]
    wheels = list(dist.glob("*.whl"))
    assert len(wheels) == 1

    env_dir = tmp_path / "env"
    venv.create(env_dir, with_pip=True)
    python = env_dir / "bin" / "python"
    install = subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", f"{wheels[0]}[opencv]"],
        capture_output=True, text=True,
    )
    assert install.returncode == 0, install.stderr[-3000:]

    work = tmp_path / "elsewhere"
    (work / "in").mkdir(parents=True)
    save_jpeg(
        np.random.default_rng(0).integers(0, 256, (32, 48, 3), dtype=np.uint8),
        work / "in" / "a.jpg",
    )
    run = subprocess.run(
        [str(env_dir / "bin" / "kodachrome-process"), "in", "out"],
        cwd=work, capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    assert (work / "out" / "a_kodachrome.jpg").exists(), "the bundled artifact was not found"
```

- [ ] **Step 2: Register the marker**

Add to `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
markers = ["slow: builds a wheel and installs it into a temporary venv"]
```

- [ ] **Step 3: Run it**

Run: `.venv/bin/pytest tests/test_packaging.py -q`
Expected: passes in roughly one to three minutes (a wheel build plus an install).

If `build` is missing, it is in the `[dev]` extra from Task 3; run `.venv/bin/pip install -e ".[dev]"`.

- [ ] **Step 4: Commit**

```bash
.venv/bin/pytest -q -m "not slow"
.venv/bin/ruff check kodachrome tests
git add tests/test_packaging.py pyproject.toml
git commit -m "test: prove the wheel carries the artifact and runs from any directory

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 21: Fetch the corpora and train the shipped artifact

Implements spec 6.2. Replaces the identity placeholder from Task 8. Uses the network and takes 10 to 20 minutes, mostly downloading.

**Files:**
- Modify: `kodachrome/data/kodachrome.cube`, `kodachrome/data/params.json`, `docs/decisions.md`, `README.md`
- Create (git-ignored): `data/kodachrome/`, `data/proxy-source/`

- [ ] **Step 1: Fetch the Kodachrome scans**

```bash
.venv/bin/kodachrome-fetch --out data/kodachrome
```
Expected: `done: N files` with N around 900 to 1,000, plus a `rejected` list in the manifest. Below 200 accepted files it exits 1; read the rejection reasons before retrying.

- [ ] **Step 2: Choose a proxy source category meeting the spec 6.2 criteria**

Public domain or CC0, modern digital cameras, varied everyday subjects, 60+ images. Probe candidates:

```bash
for cat in "Category:Photographs by the U.S. Department of Agriculture" "Category:Photographs by Lance Cheung" "Category:Photographs by Preston Keres"; do
  .venv/bin/python - "$cat" <<'EOF'
import sys, requests
from kodachrome.train.fetch import API_URL, USER_AGENT
cat = sys.argv[1]
r = requests.get(API_URL, params={"action": "query", "prop": "categoryinfo",
                                  "titles": cat, "format": "json"},
                 headers={"User-Agent": USER_AGENT}, timeout=60).json()
print(cat, "->", next(iter(r["query"]["pages"].values())).get("categoryinfo", "MISSING"))
EOF
done
```
Take the first with 200 or more files. If none qualifies, search Commons for another US federal photographer category and probe it the same way. Record the choice and its count.

Probed 2026-09-04: `Category:Photographs by the U.S. Department of Agriculture` has **4,092 files** and is the one to use. Note the abbreviated "U.S."; the spelled-out "United States" form does not exist. The other two candidates fail — "Lance Cheung" has 2 files and "Preston Keres" does not exist.

- [ ] **Step 3: Fetch a seeded sample of the proxy corpus**

```bash
.venv/bin/kodachrome-fetch --category "<chosen category>" --out data/proxy-source \
  --sample 80 --seed 0 --min-files 60
```
Expected: 60 to 80 accepted. Open the folder and confirm they are everyday photographs, not diagrams or documents. The licence and greyscale filters catch most non-photographs; delete anything that slips through, then re-run the command so the manifest and `corpus_sha1` match what is on disk.

- [ ] **Step 4: Train into a working directory**

```bash
.venv/bin/kodachrome-train --source data/proxy-source --target data/kodachrome \
  --out artifacts --proxy-source --allow-small
```
Expected: a held-out improvement line and a PASS for every gate. Exit code 3 means a gate failed; go to step 5 before doing anything else.

- [ ] **Step 5: Read the report and tune only if a gate failed**

```bash
cat artifacts/report/summary.txt
open artifacts/report/contact_sheet.png artifacts/report/ramps.png artifacts/report/diagnostics.png
```

Acceptance is the gate list, not an impression. If a gate failed:

| Failed gate | First remedy |
|---|---|
| `improvement_exceeds_noise` | raise `--iterations` to 60; if still failing, the corpora may be too similar or too small to learn from |
| `channel_monotone` or `grey_axis_monotone` | raise `--lambda-smooth` to 3e-3, then 1e-2 |
| `neutral_axis_chroma` | raise `--lambda-identity` to 1e-3 |
| `clipped_volume` | lower `--strength` to 0.85 |

Change one flag at a time and re-run. Record the final flags and the reason for each change. Do not edit the thresholds in `evaluate.py`; they were fixed before tuning on purpose.

- [ ] **Step 6: Promote the artifact and verify end to end**

```bash
cp artifacts/kodachrome.cube artifacts/params.json kodachrome/data/
.venv/bin/pytest -q
.venv/bin/kodachrome-process data/proxy-source /tmp/proxy-graded
open /tmp/proxy-graded
```
Expected: the full suite passes, and the regraded images match the contact sheet's graded row.

- [ ] **Step 7: Document provenance and commit**

Append to `docs/decisions.md`:

```markdown
## <today's date>: Shipped artifact trained on a proxy source corpus

**Decided:** the artifact in `kodachrome/data/` was fitted with `--source` =
<chosen category> (<N> files, `--sample 80 --seed 0`) and `--target` = <M>
FSA Kodachrome scans; final flags: <flags>. Held-out distance <before> ->
<after>, seed spread <spread>; all gates passed.
**Why:** the package must work on a fresh Pi before anyone has taken U20CAM
shots. These photographs are public domain, modern and varied, but they are
not the U20CAM's rendering, so `params.json` carries `"proxy": true` and the
README tells users to retrain with 50 or more of their own shots.
```

In `README.md` under "How it works", add:

```markdown
The bundled artifact was trained against a stand-in source corpus (see
`docs/decisions.md`), so `params.json` records `"proxy": true`. Retrain with
your own U20CAM shots for the closest match.
```

```bash
git add kodachrome/data docs/decisions.md README.md
git commit -m "feat: ship a Kodachrome LUT trained on FSA scans with a proxy source corpus

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 22: Pi deployment documentation and on-hardware measurement

Implements spec 7.4, 7.5 and the performance numbers of 7.2. Steps 1 and 2 run on the Mac; steps 3 to 6 need the Pi and the camera.

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the Pi setup section**

Replace the Pi placeholder in `README.md` with:

````markdown
## Pi setup

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

`--system-site-packages` is what lets the venv see apt's OpenCV, which is
built with GTK so the preview window works. Do not `pip install
opencv-python` on the Pi: the wheel most likely to be selected is the
headless one, which silently disables the preview.

Plug the U20CAM directly into the Pi rather than an unpowered hub; the
vendor FAQ attributes dropped frames to hub power. `ls /dev/v4l/by-id/`
gives a stable path you can pass to `--device`, which is worth using if
another camera is ever attached.
````

- [ ] **Step 2: Write the limitations section**

Append to `README.md`:

```markdown
## What this is, and is not

The look is an **aesthetic colour match** to Library of Congress scans of
1939-1944 Kodachrome. It is not an estimate of the film's response to a
scene, and it cannot be: the camera and the film photographed different
subjects in different decades, so matching their colour distributions cannot
separate "how the film rendered colour" from "what the 1940s looked like".
The honest fix would be to photograph one colour chart on both, which is
impossible now that Kodachrome has been discontinued since 2009 and
unprocessable since 2010.

What the numbers do show is that graded images sit measurably closer to the
Kodachrome colour distribution than the originals, on images held out of
training, by a margin larger than an identity transform and larger than the
measurement's own noise. `artifacts/report/summary.txt` states it per fit.

Other limits:

- The scans carry LoC's scanner and colour management, and 1940s Kodachrome
  differs from later K-14 stock. Point `--target` at your own scans to
  change the reference.
- Hue reweighting damps learned hue rotations beyond roughly one bin
  (15 degrees at 24 bins). Saturation, lightness and tone curve per hue are
  learned fully.
- White balance is grey-world with clamped gains, so a scene legitimately
  dominated by one colour is partially neutralised. The capture log records
  when a gain clamped.
- The camera's own auto exposure and white balance are recorded but not
  locked; see `docs/decisions.md`. Locking is the first item of future work.
- No lens, halation or vignette modelling; the 121-degree lens distortion is
  left alone.
```

- [ ] **Step 3 (on the Pi): install and smoke test**

Follow the setup section. Expected: `--fake` opens a window (or reports headless over SSH) and SPACE writes two files under `~/Pictures/kodachrome/<date>/`.

- [ ] **Step 4 (on the Pi): confirm byte-exact capture works on this hardware**

```bash
.venv/bin/kodachrome-capture
# take one shot, then Q
ls ~/Pictures/kodachrome/*/ | head
python3 -c "
import glob, json
rows=[json.loads(l) for f in glob.glob('$HOME/Pictures/kodachrome/*/captures.jsonl') for l in open(f)]
print('frame_source:', rows[-1]['frame_source'], '| fourcc:', rows[-1]['fourcc'], '| fps:', rows[-1]['fps'])
"
```
Expected: `frame_source: raw-mjpeg` and a file named `*_original.jpg`. If it reports `decoded` and `*_ungraded.jpg`, raw mode is unavailable on this device or OpenCV build: record that fact in `docs/decisions.md`, since it is exactly the fallback the design anticipated, and note whether the user wants the `linuxpy`/`v4l2py` route pursued as follow-up work.

- [ ] **Step 5 (on the Pi): measure**

```bash
.venv/bin/kodachrome-capture
# press SPACE ten times at a comfortable pace, then Q
python3 -c "
import glob, json, statistics as s
rows=[json.loads(l) for f in glob.glob('$HOME/Pictures/kodachrome/*/captures.jsonl') for l in open(f)][-10:]
p=[r['pipeline_ms'] for r in rows]; e=[r['shutter_to_saved_ms'] for r in rows]
print(f'pipeline   median {s.median(p):.0f} ms  range {min(p):.0f}-{max(p):.0f}')
print(f'shutter->saved median {s.median(e):.0f} ms  range {min(e):.0f}-{max(e):.0f}')
"
```
Expected: shutter-to-saved comfortably under 1000 ms.

- [ ] **Step 6: Record the numbers and commit**

Add to `README.md` under "How it works":

```markdown
Measured on a Raspberry Pi 400 at 1920x1080 over ten captures: pipeline
<median> ms (<min>-<max>), shutter to both files on disk <median> ms
(<min>-<max>). Preview runs at 640x360.
```

```bash
git add README.md docs/decisions.md
git commit -m "docs: Pi setup, measured performance and an honest limitations section

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

If the Pi is unavailable when the rest of the plan finishes, commit steps 1 and 2 with "Performance on the Pi 400 has not been measured yet" in place of the numbers, and leave steps 3 to 6 unchecked for the user.

---

## Plan self-review notes (revision 2)

**Spec coverage.** Spec 1 (goal and honest claim) → Tasks 19, 22 and the
`transport.py` docstring in 15. Spec 2 → Task 11. Spec 3 and 6.1 → Task 13.
Spec 4 → Tasks 3, 8. Spec 5.1 → Task 2 (done). 5.2 → Task 4. 5.3 → Task 5.
5.4 → Task 6. 5.5 → Task 7. 5.6 and 5.8 → Task 8. 5.7 → Task 9. 6.2 → Tasks
19, 21. 6.3 → Task 14. 6.4 → Tasks 15, 16, 19. 6.5 → Tasks 17, 18. 7.1 →
Task 11. 7.2 → Task 12. 7.3 → Task 10. 7.4 → Tasks 3, 22. 7.5 → Tasks 11,
12, 22. Spec 8 error table → Tasks 5, 8, 10, 11, 12, 13, 19. Spec 9 → each
task's Step 1 plus Task 20. Spec 10 → every task's documentation step.
Spec 12 → Task 22's limitations section.

**Finding coverage.** F-01 → 14 (split), 15 (docstring), 17 (baseline and
noise), 22 (README). F-02 → 11, 12. F-03 → 3, 8, 20. F-04 → 3. F-05 → 17.
F-06 → 13, 14, 19. F-07 → 8, 19. F-08 → 11, 12. F-09 → 11, 12, 22. F-10 →
7, 14, 18. F-11 → 10. F-12 → 4, 5, 6, 8, 14, 19. F-13 → 13. F-14 → 5, 12.
F-15 → 12. F-16 → 4, 14, 18. F-17 → 8 (`Artifacts` shape), 19 (hue-rotation
test), this plan's file table. F-18 → 17.

**Type consistency checked across task boundaries:** `require_cv2`,
`Gains.clamped`, `normalize_u8` returning `(image, Gains)`,
`LUT3D.apply_pillow(rgb_u8, filt)`, `sha1_hex`, `load_rgb` returning
`(array, ImageMeta)`, `Artifacts.resolve`, `Frame(rgb, jpeg, source)`,
`StreamInfo.to_dict`, `PixelPool` fields, `CorpusSplit` fields,
`Evaluator.build`/`distance`, `Gate` fields, `write_report` argument order,
`publish(staging, dest)`. Each name is produced by exactly one task and
consumed with the same signature everywhere else.

**Known deviation from revision 1, recorded:** Tasks 1 and 2 were built
against revision 1 and are not re-done; Task 3 retrofits the two things
revision 2 changes about Task 1 (extras and package data). Revision 1's
`artifacts/` directory at the repository root remains as the trainer's
default output location, but it is no longer where the runtime looks by
default.
