# Kodachrome Film Look for Raspberry Pi 400 + U20CAM: Design

Date: 2026-09-03
Status: approved in discussion, pending written review

## 1. Goal

Every photo taken on a Raspberry Pi 400 with an Innomaker U20CAM-1080P-WDR
USB camera is saved twice: the camera's original JPEG and a version graded to
look as close as possible to real Kodachrome film.

"Close as possible" is measured against real Kodachrome scans, not against a
hand-tuned recipe. A trainer running on a Mac learns a colour transform from
those scans; the Pi applies it.

### Non-goals (v1)

- No neural network on the Pi. A seam is kept so one could replace the LUT
  stage later.
- No lens correction, sharpening, halation, vignette or light leaks.
- No web UI. Capture is a terminal app with an optional preview window.
- No RAW capture. The camera is UVC and delivers 8-bit MJPEG/YUY2 only.

## 2. Hardware and platform facts

Verified 2026-09-03 from the vendor manual (`U20CAM-1080P-WDR UserManual
v1.0`) and live probing.

| Item | Fact | Consequence |
|---|---|---|
| Sensor | PixelPlus PS5268, 120 dB WDR, multi-exposure fusion | Output is already tone-mapped 8-bit; we grade it, not develop it |
| Interface | UVC 1.0, USB 2.0, plug and play on Linux | OpenCV V4L2 backend, no driver |
| 1080p rate | MJPEG 30 fps; YUY2 only 5 fps at 1080p | Capture must use MJPEG |
| Lens | 121 degree diagonal FOV | Wide; distortion accepted |
| UVC controls | auto white balance, auto exposure, gamma, gain, backlight comp | Camera does a first pass; our normalisation is a consistent second pass |
| Pi 400 | 4x Cortex-A72 1.8 GHz, 4 GB, Raspberry Pi OS Bookworm (Python 3.11) or newer | NumPy, OpenCV, Pillow only; target well under 1 s per frame |
| Trainer Mac | Intel x86_64, macOS 26, Python 3.12 (also usable on an Apple Silicon Mac) | PyTorch no longer ships Intel-Mac wheels after 2.2.2; trainer uses NumPy/SciPy |

## 3. Data source

The Library of Congress FSA/OWI colour collection (about 1,600 Kodachrome
transparencies, 1939 to 1944) is public domain. `loc.gov` sits behind a
Cloudflare bot challenge (HTTP 403 "Just a moment" to scripted clients,
confirmed), so the trainer fetches the same scans from Wikimedia Commons:

- Category: `Color photographs from the Farm Security Administration`
  (1,031 files plus 8 sub-categories at time of writing), all tagged public
  domain, filenames carry LoC catalogue numbers (`LCCN2017877392`).
- Commons API is automation-friendly given a descriptive `User-Agent`.
- Files are downloaded at 1024 px width via `imageinfo&iiurlwidth=1024`.
  About 200 MB total.
- Filters: skip titles containing `cropped`, `restored`, `retouched`,
  `colorized`, `edit`; skip originals under 800 px on the long side; prefer
  files with an LCCN in the name; de-duplicate by LCCN.

Caveat recorded for users: these scans carry LoC's scanner and colour
management, and 1940s Kodachrome differs from later K-14 stock. The fit learns
"LoC-scanned 1940s Kodachrome". The trainer accepts any folder of Kodachrome
scans as the target, so a user can substitute or add their own.

## 4. Architecture

One Python package, `kodachrome`, two runtimes sharing the same processing
code. The Pi applies exactly the normalisation the trainer fitted against.

```
kodachrome-film/
  pyproject.toml            # base: numpy, opencv-python-headless, Pillow
                            # extras: [train] scipy, requests, tqdm
                            #         [dev] pytest, ruff
  README.md                 # setup (Mac, Pi), usage, how it works, numbers
  docs/
    decisions.md            # dated log of non-obvious choices
    superpowers/specs/      # this document
  kodachrome/
    __init__.py
    color.py                # sRGB <-> linear <-> Oklab <-> LCh (NumPy, float32)
    normalize.py            # white balance + exposure; float reference and cv2.LUT fast path
    lut.py                  # LUT3D dataclass, .cube I/O, NumPy trilinear, Pillow fast path
    grain.py                # fine film grain on luminance
    pipeline.py             # Artifacts loader; normalise -> LUT -> grain
    train/
      __init__.py
      fetch.py              # Commons downloader          (CLI: kodachrome-fetch)
      dataset.py            # crop, downscale, normalise, sample pixels
      fit.py                # hue reweighting, IDT, LUT regression (CLI: kodachrome-train)
      report.py             # contact sheet, ramps, metrics.json
    capture/
      __init__.py
      camera.py             # Camera protocol, V4L2Camera, FakeCamera
      app.py                # keypress loop, preview, saving  (CLI: kodachrome-capture)
      batch.py              # folder reprocessing            (CLI: kodachrome-process)
  artifacts/
    kodachrome.cube         # trained 33^3 LUT, committed
    params.json             # normalisation, grain, provenance, committed
    report/                 # trainer output, git-ignored
  data/                     # downloaded scans, git-ignored
  tests/
```

### Channel and value conventions

- Internal image arrays are RGB, `uint8` at the boundaries, `float32` in
  `[0, 1]` inside algorithms. OpenCV's BGR is converted at the camera and
  file boundaries only.
- The 3D LUT is indexed `table[r, g, b, channel]` in memory. For `.cube`
  and Pillow the flat order is red fastest, then green, then blue, so the
  table is transposed to `(b, g, r, c)` before flattening. A test pins this.
- Oklab is used for all perceptual distances and for the transport step.
  The LUT itself maps sRGB to sRGB because that is what `.cube` and Pillow
  expect.

## 5. Shared processing components

### 5.1 `color.py`

Pure functions on NumPy arrays of shape `(..., 3)`:
`srgb_to_linear`, `linear_to_srgb`, `linear_to_oklab`, `oklab_to_linear`,
`srgb_to_oklab`, `oklab_to_srgb`, `oklab_to_lch`, `lch_to_oklab`.
Uses the standard sRGB piecewise transfer and Björn Ottosson's Oklab
matrices. Docstrings state why Oklab (perceptual uniformity for hue-aware
statistics) over CIELAB (hue non-uniformity in blues).

### 5.2 `normalize.py`

The "dynamic" part. Brings every image to the white point and exposure the
LUT was fitted on so scene lighting does not fight the grade.

```python
@dataclass
class NormalizeParams:
    white_balance: bool = True          # False for Kodachrome targets
    wb_gain_min: float = 0.6
    wb_gain_max: float = 1.6
    exposure_target_median: float = 0.18   # linear luminance, "middle grey"
    exposure_gain_min: float = 0.5
    exposure_gain_max: float = 3.0
    stats_lum_min: float = 0.02          # pixels used for statistics
    stats_lum_max: float = 0.90
```

Algorithm, in linear light:
1. Mask pixels whose linear luminance (Rec. 709 weights) lies within
   `[stats_lum_min, stats_lum_max]`. If fewer than 1% of pixels qualify, use
   all pixels.
2. Grey-world white balance: per-channel gain `g_c = mean(Y) / mean(c)` over
   the mask, clamped to `[wb_gain_min, wb_gain_max]`. Skipped when
   `white_balance` is false.
3. Exposure: after white balance, scalar gain so the median luminance over
   the mask equals `exposure_target_median`, clamped to
   `[exposure_gain_min, exposure_gain_max]`.
4. Multiply, clip to `[0, 1]`, convert back to sRGB.

Because every step is a per-channel scalar gain in linear light, the whole
map is three monotone 1D functions. `compute_gains` returns the three
combined gains; `gains_to_luts` bakes them into a `(3, 256)` `uint8` table;
`normalize_u8` applies them with `cv2.LUT` in milliseconds. `normalize_float`
is the reference path used by the trainer. A test requires the two paths to
agree within 1/255.

### 5.3 `lut.py`

```python
@dataclass
class LUT3D:
    table: np.ndarray  # (N, N, N, 3) float32, domain and range [0, 1]
    @classmethod identity(N)
    apply_numpy(rgb_float) -> rgb_float     # trilinear, reference
    to_pillow() -> ImageFilter.Color3DLUT   # built once, reused
    apply_pillow(rgb_u8) -> rgb_u8          # fast path
read_cube(path) -> LUT3D
write_cube(lut, path, title=...)
```

`.cube` format: `TITLE`, `LUT_3D_SIZE N`, optional `DOMAIN_MIN/MAX 0 1`,
then `N^3` lines of `r g b` floats, red fastest. Pillow `Color3DLUT` accepts
sizes 2 to 65 and interpolates trilinearly (verified in Pillow docs).

### 5.4 `grain.py`

```python
@dataclass
class GrainParams:
    strength: float = 0.025   # noise std in luma units (0..1) at the midtone peak
    blur_sigma: float = 0.7   # pixels; makes it read as film, not sensor noise
    enabled: bool = True
```

Convert to YCrCb, add `noise * envelope(Y)` to Y where
`envelope(Y) = 4 * Y * (1 - Y)` (zero at black and white, one at mid-grey),
noise is Gaussian blurred by `blur_sigma`, convert back. Mean luminance is
preserved to within rounding; no chroma noise is added. Kodachrome 25/64 was
a fine-grained film, so the default is subtle. Takes an optional
`numpy.random.Generator` for reproducible tests.

### 5.5 `pipeline.py`

```python
@dataclass
class Artifacts:
    lut: LUT3D
    normalize: NormalizeParams
    grain: GrainParams
    meta: dict                 # training provenance, read-only
    @classmethod load(dir_path) -> Artifacts   # reads params.json + lut_file

class Pipeline:
    def __init__(self, artifacts: Artifacts): ...
    def process(self, rgb_u8, *, grain=True, rng=None) -> tuple[np.ndarray, dict]
        # returns (rgb_u8, info) where info = {"wb_gains": [...], "exposure_gain": x}
```

Order is fixed: normalise, LUT, grain. The preview calls `process(...,
grain=False)` on a 640x360 frame.

### 5.6 `params.json` schema (version 1)

```json
{
  "version": 1,
  "lut_file": "kodachrome.cube",
  "normalize": { "white_balance": true, "wb_gain_min": 0.6, "wb_gain_max": 1.6,
                 "exposure_target_median": 0.18, "exposure_gain_min": 0.5,
                 "exposure_gain_max": 3.0, "stats_lum_min": 0.02, "stats_lum_max": 0.9 },
  "grain": { "strength": 0.025, "blur_sigma": 0.7, "enabled": true },
  "training": {
    "date": "2026-09-03T20:00:00Z",
    "target_dir": "data/kodachrome", "n_target_images": 987, "n_target_pixels": 400000,
    "source_dir": "data/source", "source_dir_sha1": "...", "n_source_images": 62,
    "n_source_pixels": 400000, "proxy_source": false,
    "lut_size": 33, "strength": 1.0, "idt_iterations": 40,
    "hue_bins": 24, "lambda_smooth": 1e-3, "lambda_identity": 1e-4, "seed": 0,
    "metrics": { "swd_before": 0.0, "swd_after": 0.0, "lut_fit_rms_deltaE": 0.0 }
  }
}
```

Unknown keys are ignored on load; a missing `version` or a version above 1
is an error.

## 6. Trainer (Mac)

### 6.1 `kodachrome-fetch`

```
kodachrome-fetch [--out data/kodachrome] [--width 1024] [--limit N] [--category NAME]
```

- Walks the Commons category and its sub-categories with `list=categorymembers`
  (continuation handled), collects file titles, applies the filters in
  section 3, then fetches `imageinfo` in batches of 50 for URLs at the
  requested width.
- Downloads with `requests`, a descriptive `User-Agent`, 3 retries with
  exponential backoff, skipping files already on disk (resumable).
- Writes `data/kodachrome/manifest.json`: title, LCCN, Commons URL, licence
  string, width, height, sha1. This is the provenance record.
- Exit non-zero if fewer than 200 files were obtained.

### 6.2 Source corpus

A folder of photos from the U20CAM. The capture app always saves originals,
so ordinary use builds it. Guidance in the README: 50 or more shots, varied
scenes (indoor, outdoor, sky, foliage, skin, neutral walls), mixed lighting.

If no camera shots exist yet, any folder of digital photos is accepted. The
trainer then writes `"proxy_source": true` into `params.json` and warns that
the LUT is fitted to that camera's rendering, not the U20CAM's. The default
artifact committed to the repo is produced this way from public-domain modern
photos, and the README says so.

### 6.3 `dataset.py`

For each image in either corpus:
1. Load, convert to RGB float.
2. Crop 6% from every edge (slide mounts, film rebate, scanner borders).
3. Downscale so the long side is 512 px (area interpolation).
4. Normalise with `normalize_float`. Source: white balance on. Target:
   white balance off, exposure on. Rationale: the film's cast is the look;
   per-slide exposure variation is not.
5. Convert to Oklab, drop pixels with `L < 0.02` or `L > 0.98`, sample up to
   3,000 pixels uniformly at random (seeded).

Pools are capped at 400,000 pixels per domain by seeded subsampling. Warn if
fewer than 30 source images or 200 target images.

### 6.4 `fit.py`

```
kodachrome-train --source DIR [--target data/kodachrome] [--out artifacts]
                 [--strength 1.0] [--lut-size 33] [--iterations 40]
                 [--hue-bins 24] [--lambda-smooth 1e-3] [--lambda-identity 1e-4]
                 [--seed 0] [--grain-strength 0.025]
```

All in Oklab unless stated.

**Step 1, hue reweighting (content-bias control).** Bin pixels by hue into
`hue_bins` equal bins, with pixels of chroma below 0.03 in one achromatic
bin. Compute normalised histograms `h_src`, `h_tgt`. Each target pixel gets
weight `w = clip(h_src[bin] / h_tgt[bin], 0.2, 5.0)`, then weights are scaled
to mean 1. Effect: the target's hue marginal now matches the source's, so
the transport cannot move hue mass wholesale because 1940s scenes contain
more khaki and foliage than a modern room. What remains to learn is how
Kodachrome renders each hue: saturation, lightness, local hue shift, and the
tone curve. Known limitation: hue shifts larger than one bin (15 degrees at
24 bins) are damped.

**Step 2, iterative distribution transfer (Pitié, Kokaram, Dahyot 2005).**
`X` = source pool, `Y` = target pool with weights `w`. For `iterations`
rounds: draw a random 3x3 rotation `R` (QR of a Gaussian matrix, seeded);
project `X R`, `Y R`; along each of the three axes map source values to
target values by weighted quantile matching (target weighted quantile
function on 1,024 points, source values mapped by rank with linear
interpolation); rotate back. After convergence each source pixel `X0[i]` has
a partner `XT[i]` drawn from the Kodachrome distribution while preserving
the pixel correspondence. Apply strength: `Y_i = X0[i] + strength *
(XT[i] - X0[i])`.

**Step 3, smooth LUT regression.** Convert `X0` and `Y` to sRGB, clip to
`[0, 1]`. Build the sparse trilinear design matrix `A` (`M x N^3`, 8
non-zeros per row) for inputs `X0`. For each output channel `c` solve

```
minimise (1/M) ||A L_c - y_c||^2
        + lambda_smooth   * (1/|D|)  ||D L_c||^2
        + lambda_identity * (1/N^3)  ||L_c - I_c||^2
```

where `D` stacks second-difference operators along the three grid axes and
`I` is the identity LUT. Solve as one stacked sparse least-squares system
with `scipy.sparse.linalg.lsqr`, or by conjugate gradients on the normal
equations, whichever converges in seconds (decided during implementation and
recorded in `docs/decisions.md`). The identity term keeps grid nodes that no
source pixel touches (for example neon magenta) sane; the smoothness term
prevents banding. Defaults are starting points; the synthetic recovery test
in section 9 and the report ramps are the tuning tools.

**Outputs.** `artifacts/kodachrome.cube`, `artifacts/params.json` with the
`training` block filled, then `report.py` runs.

### 6.5 `report.py`

Written to `artifacts/report/`:

- `contact_sheet.png`: eight source images, each shown normalised and
  graded, plus a strip of eight random Kodachrome scans for eyeballing.
- `ramps.png`: a grey ramp and a hue sweep (at three lightness levels)
  before and after the LUT. Shows the learned tone curve and hue shifts;
  reveals banding or non-monotonic grey.
- `metrics.json`: sliced Wasserstein distance (64 projections, Oklab)
  source vs target and graded-source vs target; LUT fit residual RMS delta-E
  against the transport partners; per-hue-bin mean change in L, chroma and
  hue in degrees; a `grey_axis_monotone: true/false` flag.

The README explains how to read each and what "good" looks like.

## 7. Pi runtime

### 7.1 `camera.py`

```python
class Camera(Protocol):
    def read(self) -> np.ndarray          # RGB uint8 (H, W, 3)
    def close(self) -> None

class V4L2Camera(Camera):
    def __init__(self, device: int | str | None, width=1920, height=1080, fps=30)

class FakeCamera(Camera):
    def __init__(self, frames: list[np.ndarray] | None = None)  # synthetic gradient frames by default
```

`V4L2Camera` opens `cv2.VideoCapture(index, cv2.CAP_V4L2)`, sets FOURCC
`MJPG`, size and fps, verifies the negotiated size, discards 15 warm-up
frames so the camera's own AE/AWB settle, and on each `read` grabs three
frames and returns the last (flushes the driver queue). With `device=None`
it tries `/dev/video0` through `/dev/video9` and keeps the first that
delivers a frame. Frame read failures raise `CameraError` after three
attempts.

### 7.2 `app.py`

```
kodachrome-capture [--device N|/dev/videoN] [--artifacts artifacts]
                   [--out ~/Pictures/kodachrome] [--no-preview] [--fake]
```

- Loads `Artifacts`, builds `Pipeline`, opens the camera.
- Preview (default when `DISPLAY` is set and `--no-preview` absent): an
  OpenCV window shows the live feed resized to 640x360 with
  `process(grain=False)` applied, at camera frame rate. Keys: `SPACE`
  capture, `P` toggle graded/original preview, `Q` quit.
- Headless: no window; `SPACE` and `Q` read from the terminal in cbreak
  mode; one status line per capture.
- Capture: full-resolution frame, `process()`, write
  `OUT/YYYY-MM-DD/HHMMSS_original.jpg` and `HHMMSS_kodachrome.jpg` (JPEG
  quality 95), append one JSON line to `OUT/YYYY-MM-DD/captures.jsonl`:
  timestamp, both filenames, white balance gains, exposure gain, processing
  time in ms.
- `--fake` uses `FakeCamera`, so the whole app runs on the Mac.

Performance target on Pi 400 for 1080p: under 1 s per capture, expected
about 0.4 s (Pillow LUT and grain dominate). Measured numbers go in the
README after the first run on hardware.

### 7.3 `batch.py`

```
kodachrome-process IN_DIR OUT_DIR [--artifacts artifacts] [--no-grain]
```

Processes every `*.jpg`/`*.jpeg`/`*.png` in `IN_DIR` through the same
`Pipeline`, writes `<stem>_kodachrome.jpg` to `OUT_DIR`, prints a summary.
Used to regrade old originals after retraining and to test on the Mac.

## 8. Error handling

| Situation | Behaviour |
|---|---|
| No camera opens | List existing `/dev/video*`, print how to pass `--device`, exit 2 |
| Negotiated size differs from requested | Warn, continue with what the camera gave |
| Frame read fails | Retry 3 times; on failure print error, keep the app running |
| `artifacts/` missing or `params.json` invalid | Exit 2 with the expected path and "run kodachrome-train or restore the committed files" |
| `.cube` size not in 2..65 or malformed | Exit 2 with line number of the problem |
| Disk write fails | Print error with path, keep session alive |
| Fetch: individual file fails | Retry with backoff, then skip and count |
| Fetch: fewer than 200 files | Exit 1, do not proceed |
| Train: fewer than 30 source or 200 target images | Warn, continue |
| Train: source and target folders identical | Exit 1 |
| Preview requested but no display | Fall back to headless with a notice |

## 9. Testing

All tests run on the Mac with no hardware, via `pytest`.

- `test_color.py`: sRGB/linear/Oklab round-trips within 1e-4; a few
  published Oklab reference values (white, pure red, pure blue) match.
- `test_normalize.py`: float and `cv2.LUT` paths agree within 1/255; gains
  are clamped; normalising twice moves the median by under 1/255; a grey
  image with a colour cast comes out neutral.
- `test_lut.py`: identity LUT leaves an image unchanged; `.cube` write then
  read is exact; NumPy trilinear and Pillow agree within 1/255 on random
  images; flat ordering (red fastest) pinned by a hand-built 2x2x2 case.
- `test_grain.py`: mean luminance preserved within 0.5/255; no chroma bias;
  reproducible with a seeded generator; `enabled=False` is identity.
- `test_fit.py`: source is a random colour cloud; target is the same cloud
  through a known tone curve plus a 10 degree hue rotation. The fitted LUT
  (size 17, fewer iterations) reproduces the transform within a mean
  delta-E of 0.02 on held-out samples. Also: hue reweighting makes the
  target hue histogram match the source's.
- `test_pipeline.py`: `Artifacts.load` on the committed files; `process`
  on a synthetic frame returns the right shape, dtype and info keys;
  `params.json` schema validation errors are clear.
- `test_app.py`: capture app with `FakeCamera` and a temp output dir writes
  both JPEGs and one JSONL line per capture; headless mode works without a
  display.
- `test_fetch.py`: category walking and filtering against recorded API
  responses (no network in tests).

Manual verification: inspect `artifacts/report/`; time a capture on the Pi.

## 10. Documentation plan

Kept current with each implementation step, not written at the end:

- `README.md`: what it does, Mac setup, Pi setup (`apt` packages, venv with
  `--system-site-packages`), the three commands, how to build a source
  corpus, how to read the report, measured performance, known limitations.
- `docs/decisions.md`: dated entries for non-obvious choices. Initial
  entries: Commons instead of loc.gov; NumPy/SciPy instead of PyTorch;
  transport plus regression instead of gradient descent; white balance
  off for targets; LUT in sRGB but distances in Oklab.
- Module docstrings explain the colour science and the "why", not only
  parameters.
- This spec is updated if implementation forces a change, with the change
  noted in `decisions.md`.

## 11. Risks and mitigations

- **Content bias in unpaired matching.** Mitigated by hue reweighting,
  smoothness and identity regularisation, and the report's hue-bin metrics
  which make any wholesale shift visible.
- **Scans are not the film.** Scanner and LoC colour management are baked
  in. Accepted; documented; target folder is user-replaceable.
- **Proxy source corpus for the default artifact.** Marked in
  `params.json`, README tells users to retrain with their own shots.
- **Pi performance.** Pillow's C LUT and `cv2.LUT` normalisation keep the
  budget; grain is the fallback knob (`enabled: false`) if the Pi is slow.
- **Commons category changes.** The manifest records exactly what was used;
  `--limit` and `--category` allow control.

## 12. Future work (out of scope)

- Small CNN replacing the LUT stage for context-aware grading (skin vs sky),
  trainable on an Apple Silicon Mac, exported to ONNX for the Pi.
- Scene-type LUT variants (daylight, tungsten) selected per capture.
- Halation and vignette models.
