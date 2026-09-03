# Kodachrome Film Look for Raspberry Pi 400 + U20CAM: Design

Date: 2026-09-03
Status: revision 2, approved in discussion

Revision 2 responds to an external review (`findings.md`, F-01 to F-18).
Changes are summarised in section 13 and recorded in `docs/decisions.md`.

## 1. Goal

Every photo taken on a Raspberry Pi 400 with an Innomaker U20CAM-1080P-WDR
USB camera is saved twice: the camera's own JPEG bytes, and a version graded
to look like Kodachrome.

### What "like Kodachrome" means here

This project produces an **aesthetic colour match to Library of Congress
scans of 1939-1944 Kodachrome transparencies**. It does not estimate the
film's spectral response, and it does not claim that a given scene would
have looked exactly this way on Kodachrome.

That limit is a property of the method, not a shortfall in effort. The two
image sets are unpaired: the camera photographs today's scenes, the film
photographed 1940s scenes, and no image in one set corresponds to any image
in the other. Matching their colour distributions therefore cannot separate
"how the film renders colour" from "what the 1940s looked like". Hue
reweighting (section 6.4) reduces the largest such confound but does not
remove it, because the transport still moves individual pixels freely in
three dimensions.

The honest alternative would be paired evidence: photograph one colour chart
on the U20CAM and on Kodachrome, then fit to the pairs. That is impossible.
Kodachrome was discontinued in 2009 and the last laboratory able to process
it stopped at the end of 2010.

So the measurable goal is: **graded camera images should sit closer to the
Kodachrome colour distribution than the ungraded originals do, measured on
images held out of training, and by a margin larger than an identity
transform and larger than seed-to-seed noise.** Section 6.5 defines the
measurement.

### Non-goals (v1)

- No neural network on the Pi. A seam is kept so one could replace the LUT
  stage later.
- No lens correction, sharpening, halation, vignette or light leaks.
- No web UI. Capture is a terminal app with an optional preview window.
- No RAW capture. The camera is UVC and delivers 8-bit MJPEG/YUY2 only.
- No locking of the camera's internal auto-exposure and auto-white-balance.
  Section 7.5 explains why this is deferred and what is recorded instead.

## 2. Hardware and platform facts

Verified 2026-09-03 from the vendor manual (`U20CAM-1080P-WDR UserManual
v1.0`) and live probing.

| Item | Fact | Consequence |
|---|---|---|
| Sensor | PixelPlus PS5268, 120 dB WDR, multi-exposure fusion | Output is already tone-mapped 8-bit; we grade it, not develop it |
| Interface | UVC 1.0, USB 2.0, plug and play on Linux | OpenCV V4L2 backend, no driver |
| 1080p rate | MJPEG 30 fps; YUY2 only 5 fps at 1080p | Capture must use MJPEG |
| Lens | 121 degree diagonal FOV | Wide; distortion accepted |
| UVC controls | auto white balance, auto exposure, gamma, gain, backlight comp | Camera does a first pass; recorded but not locked (7.5) |
| Raw frame access | V4L2 backend honours `CAP_PROP_CONVERT_RGB = 0`, handing back the compressed MJPEG buffer; OpenCV issue #23311 reports truncated buffers on some devices | Byte-exact originals are possible but must be validated per frame, with a fallback (7.1) |
| Pi 400 | 4x Cortex-A72 1.8 GHz, 4 GB, Raspberry Pi OS Bookworm (Python 3.11) or newer | NumPy, OpenCV, Pillow only; target well under 1 s per frame |
| Trainer Mac | Intel x86_64, macOS 26, Python 3.12 (also usable on an Apple Silicon Mac) | PyTorch no longer ships Intel-Mac wheels after 2.2.2; trainer uses NumPy/SciPy |

## 3. Data source

The Library of Congress FSA/OWI colour collection (about 1,600 Kodachrome
transparencies, 1939 to 1944) is public domain. `loc.gov` sits behind a
Cloudflare bot challenge (HTTP 403 "Just a moment" to scripted clients,
confirmed), so the trainer fetches the same scans from Wikimedia Commons:

- Category: `Color photographs from the Farm Security Administration`
  (1,031 files plus 8 sub-categories at time of writing), filenames carry
  LoC catalogue numbers (`LCCN2017877392`).
- Commons API is automation-friendly given a descriptive `User-Agent`.
- Files are downloaded at 1024 px width via `imageinfo&iiurlwidth=1024`.
  About 200 MB total.

**Selection rules, enforced not assumed:**

- **Licence allowlist.** A file is accepted only if its
  `LicenseShortName` matches an explicit allowlist (`Public domain`,
  `CC0`, `PDM`, and the `PD-*` family). Anything else is rejected and
  counted, whatever category it sits in.
- **Media validation.** The API-reported MIME type must be `image/jpeg`,
  `image/png` or `image/tiff`; the downloaded bytes must decode with
  Pillow; the decoded image must be at least 800 px on its long side and
  must not be greyscale (a scanned document or diagram usually is).
- **Title filter.** Skip `cropped`, `restored`, `retouched`, `colorized`,
  `colourized`, `edit`: these are Commons editors' derivatives whose
  colours have already been changed.
- De-duplicate by LCCN; files with an LCCN are preferred over those without.
- **Atomic download.** Fetch to a temporary file, validate status, content
  type, decode and size, hash the bytes, then rename into place. A resumed
  run re-validates existing files against the manifest hash rather than
  trusting any non-empty file.

`manifest.json` records, per file: Commons title, page ID and revision ID,
LCCN, URL, dimensions, licence string, and the SHA-1 of the bytes on disk.
Rejections are recorded too, with a reason, so the corpus is auditable.

Caveat recorded for users: these scans carry LoC's scanner and colour
management, and 1940s Kodachrome differs from later K-14 stock. The fit
learns "LoC-scanned 1940s Kodachrome". The trainer accepts any folder of
Kodachrome scans as the target, so a user can substitute or add their own.

## 4. Architecture

One Python package, `kodachrome`, two runtimes sharing the same processing
code. The Pi applies exactly the normalisation the trainer fitted against.

```
kodachrome-film/
  pyproject.toml            # base: numpy, Pillow
                            # extras: [opencv] opencv-python
                            #         [train]  scipy, requests, tqdm + opencv
                            #         [dev]    pytest, ruff + train
  README.md                 # setup (Mac, Pi), usage, how it works, numbers
  docs/
    decisions.md            # dated log of non-obvious choices
    superpowers/specs/      # this document
  kodachrome/
    __init__.py
    color.py                # sRGB <-> linear <-> Oklab <-> LCh (NumPy, float32)
    normalize.py            # white balance + exposure; float and cv2.LUT paths
    lut.py                  # LUT3D dataclass, .cube I/O, NumPy trilinear, Pillow fast path
    grain.py                # fine film grain on luminance
    imageio.py              # load/save; EXIF orientation, ICC -> sRGB
    artifacts.py            # Artifacts load/validate/publish, importlib.resources default
    pipeline.py             # normalise -> LUT -> grain
    data/                   # PACKAGED default artifact (ships in the wheel)
      kodachrome.cube
      params.json
    train/
      __init__.py
      fetch.py              # Commons downloader          (CLI: kodachrome-fetch)
      dataset.py            # crop, orient, colour-manage, normalise, sample; train/val split
      transport.py          # hue reweighting, IDT, sliced Wasserstein
      lutfit.py             # trilinear design matrix, smoothness, CG solve
      evaluate.py           # paired evaluator, held-out metrics, LUT safety checks
      report.py             # contact sheet, ramps, diagnostics, metrics.json
      fit.py                # orchestration and params writing (CLI: kodachrome-train)
    capture/
      __init__.py
      camera.py             # Camera protocol, V4L2Camera (raw MJPEG), FakeCamera
      app.py                # keypress loop, preview, saving  (CLI: kodachrome-capture)
      batch.py              # folder reprocessing            (CLI: kodachrome-process)
  artifacts/                # working directory for training output (git-ignored except .gitkeep)
  data/                     # downloaded corpora, git-ignored
  tests/
```

**Artifact location (F-03).** The default LUT and parameters live inside the
package at `kodachrome/data/` and are located with `importlib.resources`, so
every command works from any working directory and a built wheel carries the
look. `--artifacts DIR` overrides with a directory on disk. `kodachrome-train`
writes to `./artifacts` by default; promoting a trained artifact to the
default means copying it into `kodachrome/data/`, which the README documents.

### Channel and value conventions

- Internal image arrays are RGB, `uint8` at the boundaries, `float32` in
  `[0, 1]` inside algorithms. OpenCV's BGR is converted at the camera and
  file boundaries only.
- The 3D LUT is indexed `table[r, g, b, channel]` in memory. For `.cube`
  and Pillow the flat order is red fastest, then green, then blue.
- Oklab is used for all perceptual distances and for the transport step.
  The LUT itself maps sRGB to sRGB.
- All image files are decoded to sRGB: EXIF orientation applied, embedded
  ICC profile converted to sRGB (section 5.5).

## 5. Shared processing components

### 5.1 `color.py`

Pure functions on NumPy arrays of shape `(..., 3)`:
`srgb_to_linear`, `linear_to_srgb`, `linear_to_oklab`, `oklab_to_linear`,
`srgb_to_oklab`, `oklab_to_srgb`, `oklab_to_lch`, `lch_to_oklab`,
`luminance`, and `LUMA_709`. Uses the standard sRGB piecewise transfer and
Björn Ottosson's Oklab matrices.

**Implemented in Task 2; unchanged by revision 2.**

### 5.2 `normalize.py`

The "dynamic" per-shot step. Brings every image to the white point and
exposure the LUT was fitted on so scene lighting does not fight the grade.

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

Validated at construction: every bound finite, each `*_min < *_max`, gains
positive, `stats_lum_min < stats_lum_max`, target median inside
`(0, 1)`. Invalid values raise `ValueError` naming the field.

Algorithm, in linear light: mask pixels by luminance, grey-world per-channel
gains clamped to the wb range, then a scalar exposure gain so the masked
median hits the target, clamped to the exposure range. Because every step is
a per-channel scalar gain in linear light, the composite map is three
monotone 1D functions, applied on the Pi with `cv2.LUT`.

`compute_gains` returns `Gains(wb, exposure, clamped: dict[str, bool])`. The
`clamped` flags feed the diagnostics in 6.5 and the capture log in 7.2, so
that scenes where normalisation hit a limit are visible rather than silent
(F-16, reduced scope: diagnostics, not an ablation study).

### 5.3 `lut.py`

```python
@dataclass
class LUT3D:
    table: np.ndarray  # (N, N, N, 3) float32, [r, g, b, channel], values in [0, 1]
```

with `identity`, `to_flat`, `from_flat`, `apply_numpy`, `to_pillow`,
`apply_pillow`, plus `read_cube` and `write_cube`.

**Validation (F-12).** `LUT3D.__post_init__` rejects a non-cubic shape, a
size outside 2 to 65, non-finite values, and values outside `[0, 1]`.
`read_cube` additionally rejects a missing `LUT_3D_SIZE`, a wrong row count,
non-numeric or non-finite entries, and a `DOMAIN_MIN`/`DOMAIN_MAX` other
than `0 0 0` / `1 1 1` (the in-memory contract assumes the unit domain; a
different domain is refused with a clear message rather than silently
misapplied). `write_cube` writes the domain explicitly.

`sha1_hex(lut)` returns the SHA-1 of the canonical flat table bytes; it
identifies an artifact in logs and in `params.json` (F-14).

### 5.4 `grain.py`

```python
@dataclass
class GrainParams:
    strength: float = 0.025   # noise std in luma units at the midtone peak
    blur_sigma: float = 0.7   # pixels
    enabled: bool = True
```

Validated at construction: `strength >= 0`, `blur_sigma >= 0`, both finite.

Gaussian noise on luminance only, blurred then renormalised to unit
variance, scaled by an envelope `4Y(1-Y)` that vanishes at black and white.
`add_grain(rgb_u8, params, rng)` takes an explicit `numpy.random.Generator`.
Callers that need reproducibility pass a seeded generator and record the
seed (7.2).

### 5.5 `imageio.py`

All file reads go through one function so colour management cannot vary by
call site (F-10).

```python
load_rgb(path, *, colour_manage=True) -> (np.ndarray uint8, ImageMeta)
```

1. Open with Pillow, apply `ImageOps.exif_transpose`. Orientation matters
   because the trainer crops a fixed 6% from each edge; on a rotated image
   an unoriented crop removes the wrong edges.
2. If an ICC profile is embedded, convert to sRGB with `ImageCms`
   (perceptual intent). If the profile is missing, assume sRGB. If it is
   present but malformed, assume sRGB and record the failure.
3. Convert to RGB, return with `ImageMeta(profile: str, oriented: bool,
   profile_error: str | None, width, height)`.

`ImageMeta.profile` is one of `"sRGB (assumed)"`, `"sRGB (embedded)"`, the
profile description, or `"invalid"`. The trainer aggregates these counts
into the report so a corpus that is secretly half Adobe RGB is visible.

`save_jpeg(rgb_u8, path, quality=95, icc=SRGB)` embeds an sRGB profile.
`list_images(dir)` lists supported extensions, sorted.

### 5.6 `artifacts.py`

Loading, validating and publishing the trained artifact.

```python
PARAMS_VERSION = 2

@dataclass
class Artifacts:
    lut: LUT3D
    normalize: NormalizeParams
    grain: GrainParams
    training: dict          # provenance, read-only
    path: Path              # where it was loaded from
    lut_sha1: str
```

- `Artifacts.load(dir_path)` reads and validates.
- `Artifacts.default()` loads the packaged artifact via
  `importlib.resources.files("kodachrome.data")`.
- `Artifacts.resolve(dir_path | None)` returns the override or the default.
- `publish(tmp_dir, dest_dir)` publishes atomically (F-07): the trainer
  writes the LUT, `params.json` and the report into a temporary directory,
  `Artifacts.load` validates that directory, and only then is it moved into
  place. A LUT whose SHA-1 disagrees with `params.json` fails validation, so
  a half-written pair can never be loaded.

Every failure raises `ArtifactsError` naming the path and the problem. The
JSON is validated structurally: the root must be an object, `version` must
be present and `<= PARAMS_VERSION`, and `normalize`, `grain` and `training`
must be objects, with values of the declared types.

### 5.7 `pipeline.py`

```python
class Pipeline:
    def __init__(self, artifacts: Artifacts): ...
    def process(self, rgb_u8, *, grain=True, rng=None) -> tuple[np.ndarray, dict]
```

Order is fixed: normalise, LUT, grain. `info` carries `wb_gains`,
`exposure_gain`, `clamped`, and `lut_sha1`. The preview calls
`process(..., grain=False)` on a 640x360 frame.

### 5.8 `params.json` schema (version 2)

```json
{
  "version": 2,
  "lut_file": "kodachrome.cube",
  "lut_sha1": "…",
  "normalize": { "white_balance": true, "...": "as 5.2" },
  "grain": { "strength": 0.025, "blur_sigma": 0.7, "enabled": true },
  "training": {
    "date": "2026-09-03T20:00:00Z",
    "code_revision": "798ac95",
    "package_version": "0.1.0",
    "dependency_versions": { "numpy": "…", "scipy": "…", "Pillow": "…", "opencv": "…" },
    "command": "kodachrome-train --source … --target …",
    "target": { "dir": "data/kodachrome", "n_images": 987, "manifest_sha1": "…",
                "corpus_sha1": "…", "n_pixels": 400000 },
    "source": { "dir": "data/source", "n_images": 62, "manifest_sha1": null,
                "corpus_sha1": "…", "n_pixels": 400000, "proxy": false },
    "split": { "val_fraction": 0.2, "n_source_val_images": 12, "n_target_val_images": 197,
               "seed": 0 },
    "fit": { "lut_size": 33, "strength": 1.0, "idt_iterations": 40, "hue_bins": 24,
             "chroma_floor": 0.03, "lambda_smooth": 1e-3, "lambda_identity": 1e-4,
             "seed": 0, "fit_seconds": 12.4 },
    "metrics": { "…": "see 6.5" }
  }
}
```

`corpus_sha1` hashes the actual bytes of every file in the corpus, sorted by
name, not just names and sizes (F-06). Unknown keys are ignored on load; a
missing `version`, or one above `PARAMS_VERSION`, is an error.

## 6. Trainer (Mac)

### 6.1 `kodachrome-fetch`

```
kodachrome-fetch [--out data/kodachrome] [--width 1024] [--limit N]
                 [--sample N] [--seed 0] [--category NAME] [--min-files 200]
```

Walks the category and sub-categories, applies the section 3 rules, fetches
`imageinfo` in batches of 50, downloads atomically with retries, writes
`manifest.json` including rejections and their reasons. Exits non-zero if
fewer than `--min-files` files were accepted.

### 6.2 Source corpus

A folder of photos from the U20CAM; `kodachrome-capture` saves originals, so
ordinary use builds it. Guidance: 50 or more shots across varied scenes and
lighting.

If no camera shots exist, any folder of digital photos is accepted with
`--proxy-source`, which records `"proxy": true` in `params.json` and prints
a warning. The trainer cannot detect a proxy corpus, so the flag is
explicit.

**Default artifact corpus criteria.** Public domain or CC0, modern digital
cameras, varied everyday subjects, at least 60 images, fetched by the same
reproducible script with its manifest committed. The chosen category is
recorded in `docs/decisions.md` and in `params.json`.

### 6.3 `dataset.py`

Per image: `load_rgb` (orientation and ICC applied), crop 6% from each edge,
downscale so the long side is 512 px, normalise with `normalize_float`
(source: white balance on; target: white balance off, because the film's
cast is the look being learned), convert to Oklab, drop pixels outside
`(0.02, 0.98)` lightness, sample up to 3,000 pixels.

**Train/validation split (F-01, F-05).** The split happens **by image,
before any pixel is sampled**, so no pixel from a validation image can
appear in training. Default `--val-fraction 0.2`, seeded. Both corpora are
split. `build_pools` returns train and validation `PixelPool`s for each
side, plus the image lists, so the report can render held-out examples.

Pools are capped at `max_pixels` by seeded subsampling. Corpora smaller than
30 source or 200 target images fail with a message naming `--allow-small`,
rather than warning and continuing into a statistically weak fit (F-12).

### 6.4 Fitting (`transport.py`, `lutfit.py`, `fit.py`)

```
kodachrome-train --source DIR [--target data/kodachrome] [--out artifacts]
                 [--strength 1.0] [--lut-size 33] [--iterations 40]
                 [--hue-bins 24] [--lambda-smooth 1e-3] [--lambda-identity 1e-4]
                 [--val-fraction 0.2] [--seed 0] [--grain-strength 0.025]
                 [--proxy-source] [--allow-small]
```

All numeric options are validated: positive where they must be positive,
finite, and inside documented ranges. Out-of-range values exit 1 naming the
flag.

**Step 1, hue reweighting — a heuristic, not a constraint (F-01).** Target
pixels are weighted so the target's hue histogram matches the source's,
which removes the largest content-driven bias: 1940s scenes contain more
khaki and foliage than a modern room. It does **not** prevent the transport
from moving individual pixels across hue bins, because the transport
operates on the full 3D distribution; and clipping the weights to
`[0.2, 5.0]` means the reweighted histograms match only approximately when a
hue is nearly absent from one side. The report publishes the residual
histogram difference so the size of this effect is visible.

**Step 2, iterative distribution transfer** (Pitié, Kokaram, Dahyot 2005).
Random 3D rotation, weighted quantile matching along each axis, rotate back,
repeat. Preserves pixel identity, so each source pixel gains a partner.
`strength` blends between identity (0) and full transport (1).

**Step 3, smooth LUT regression.** Fit a trilinear LUT to the pairs by
regularised sparse least squares: data term, second-difference smoothness,
and a weak pull toward identity for untouched grid nodes. Solved per channel
by conjugate gradients on the normal equations.

### 6.5 `evaluate.py` and `report.py`

**Paired evaluation (F-05).** One `Evaluator` object is built once from a
fixed seed and holds the sample indices and projection directions. Both
`swd_before` and `swd_after` are computed with that same object, so the two
numbers differ only because the LUT changed the pixels. A test asserts that
an identity LUT yields `swd_before == swd_after` exactly.

Metrics are reported on **held-out images** (primary) and on the training
pool (secondary, marked as such). Reported:

| Metric | Meaning |
|---|---|
| `swd_before`, `swd_after` | distance to the Kodachrome cloud, held-out, paired evaluator |
| `swd_identity` | identity-LUT baseline; equals `swd_before` by construction, asserted in tests |
| `swd_seed_spread` | spread of `swd_after` across 5 evaluator seeds, so improvement can be compared against noise |
| `transport_gamut_clip_deltaE` | RMS Oklab distance from transported partners to their in-gamut clipped versions |
| `lut_fit_rms_deltaE` | RMS Oklab distance from LUT output to the **clipped** partners |
| `grey_axis_monotone` | luminance non-decreasing along the neutral axis |
| `channel_monotone` | each output channel non-decreasing along its own input axis |
| `neutral_axis_max_chroma` | largest Oklab chroma the LUT gives a neutral input |
| `clipped_volume_fraction` | fraction of cube nodes mapping onto a gamut boundary |
| `hue_bins` | per-hue-bin mean change in L, chroma and hue degrees |
| `hue_hist_residual` | max per-bin difference remaining after reweighting |
| `wb_gain_histogram`, `exposure_gain_histogram`, `clamp_rate` | normalisation diagnostics per corpus |
| `icc_profile_counts` | how many images carried which colour profile |

Separating gamut clipping from LUT fit error (F-05) matters because a large
`lut_fit_rms_deltaE` caused by clipping means "the transport asked for
colours outside sRGB", which is a different problem from "the smooth LUT
could not express the transport".

`report.py` writes `contact_sheet.png` (held-out source images, normalised
and graded, plus real scans), `ramps.png` (grey ramp and hue sweeps, before
over after), `diagnostics.png` (gain histograms and clamp rates), and
`metrics.json`.

**Numeric acceptance gates**, checked by the trainer and printed as pass or
fail: `swd_after < swd_before - 3 * swd_seed_spread`; `grey_axis_monotone`
and `channel_monotone` true; `neutral_axis_max_chroma < 0.02`;
`clipped_volume_fraction < 0.05`. Failing gates do not delete the artifact,
but the trainer exits non-zero and the report says which gate failed.

## 7. Pi runtime

### 7.1 `camera.py`

```python
@dataclass
class Frame:
    rgb: np.ndarray            # decoded RGB uint8
    jpeg: bytes | None         # camera's own bytes, when raw mode is active
    source: str                # "raw-mjpeg" | "decoded"
```

`V4L2Camera` opens the device with the V4L2 backend, sets FOURCC `MJPG`,
size and rate, then **verifies what it actually got** (F-08): FOURCC,
dimensions and frame rate are read back and reported; a mismatch is a
warning naming both requested and negotiated values, and the negotiated
values are recorded in the capture log.

**Byte-exact originals (F-02).** The camera first tries raw mode
(`CAP_PROP_CONVERT_RGB = 0`). A returned buffer is accepted only if it
starts with `FF D8` and ends with `FF D9` and decodes with `cv2.imdecode`.
Then `Frame.jpeg` is the camera's own bytes and `Frame.rgb` is the decode of
**those same bytes**, so both saved files come from one acquisition. If raw
mode is unavailable, or a buffer fails validation, the camera falls back to
decoded mode for the rest of the session, sets `source="decoded"`, and the
app names its second file `_ungraded.jpg` instead of `_original.jpg` so the
filename never lies about what it contains.

Device selection prefers a stable `/dev/v4l/by-id/*` path when one matches;
`--device` accepts an index, a `/dev/videoN` path or a by-id path. Probing
picks the first node that both opens and delivers a valid frame, and prints
which node it chose.

`read()` returns the newest frame. The driver queue is drained by grabbing
until no further frame is immediately available, rather than a fixed triple
grab, so preview genuinely runs near camera rate. `FakeCamera` implements
the same protocol and can be given real JPEG bytes so raw-mode logic is
testable without hardware.

### 7.2 `app.py`

```
kodachrome-capture [--device N|/dev/videoN|by-id-path] [--artifacts DIR]
                   [--out ~/Pictures/kodachrome] [--no-preview] [--fake]
                   [--seed N]
```

Preview shows the graded feed at 640x360; `SPACE` captures, `P` toggles
graded and original, `Q` quits. Headless mode reads the same keys from the
terminal.

**Error handling (F-08).** The whole preview body is guarded: a
`CameraError` from a frame read prints and continues, per section 8; a
`cv2.error` from `namedWindow`, `imshow` or `waitKey` falls back to headless
mode with a notice rather than escaping.

**Capture semantics.** `SPACE` acquires a fresh frame and saves that frame;
the README says so explicitly, so nobody expects the displayed frame.

**Outputs.** `OUT/YYYY-MM-DD/HHMMSS_original.jpg` (camera bytes, written
verbatim) or `HHMMSS_ungraded.jpg` (re-encode, fallback mode), plus
`HHMMSS_kodachrome.jpg`, plus one JSON line in `captures.jsonl`:

```json
{"timestamp": "…", "original": "…", "kodachrome": "…", "frame_source": "raw-mjpeg",
 "wb_gains": [...], "exposure_gain": 1.02, "clamped": {"wb": false, "exposure": false},
 "grain_seed": 12345, "lut_sha1": "…", "params_version": 2, "package_version": "0.1.0",
 "width": 1920, "height": 1080, "fourcc": "MJPG", "fps": 30.0,
 "pipeline_ms": 380.2, "shutter_to_saved_ms": 610.7}
```

Recording `grain_seed` and `lut_sha1` makes the graded file reproducible
from the original (F-14). Reporting both `pipeline_ms` and
`shutter_to_saved_ms` stops the faster number being quoted as latency
(F-15).

### 7.3 `batch.py`

```
kodachrome-process IN_DIR OUT_DIR [--artifacts DIR] [--no-grain]
                   [--all] [--overwrite]
```

**Safety (F-11).** Files matching `*_kodachrome.*` are always skipped. If the
input directory contains any `*_original.*` or `*_ungraded.*`, only those are
processed unless `--all` is given. Output paths that would collide (`a.jpg`
and `a.png`) are disambiguated by appending the source extension. An
existing output is skipped unless `--overwrite`. `OUT_DIR` equal to, or
nested inside, `IN_DIR` is refused.

### 7.4 Dependencies

OpenCV is not a base dependency, because on Raspberry Pi OS it comes from
apt (`python3-opencv`, built with GTK so `cv2.imshow` works) through a venv
created with `--system-site-packages`. But every documented pip install must
produce a working program (F-04), so an `[opencv]` extra exists and both
`[train]` and `[dev]` include it. The Pi instructions install the base
package only, alongside apt OpenCV.

Importing `kodachrome.normalize`, `grain`, `dataset`, `camera` or `app`
without OpenCV raises a clear message naming both remedies rather than a
bare `ModuleNotFoundError`.

### 7.5 Camera image-processing state (F-09, reduced)

The U20CAM runs its own auto exposure and auto white balance, and the
PS5268 performs multi-exposure WDR fusion internally. Locking these, and
verifying the locks hold, needs the hardware in hand and may not be possible
through UVC for the fused modes. Version 1 therefore does **not** lock them.

What it does instead: after warm-up, read back every UVC control the driver
exposes plus the negotiated format, record them in the capture log and in
the source corpus metadata, and publish the distribution of software white
balance and exposure gains together with the clamp rate (6.5). If the camera
is drifting between shots, those numbers show it. Locking is listed in
section 12 as the first follow-up once the Pi is available.

## 8. Error handling

| Situation | Behaviour |
|---|---|
| No camera opens | List `/dev/video*` and `/dev/v4l/by-id/*`, name `--device`, exit 2 |
| Negotiated FOURCC, size or rate differs | Warn naming requested and actual; continue; record both |
| Raw MJPEG unsupported or buffer invalid | Warn once, fall back to decoded mode, name outputs `_ungraded.jpg` |
| Frame read fails | Retry 3 times; then print and keep the session alive |
| Preview window cannot be created or drawn | Fall back to headless with a notice |
| Headless and stdin is not a TTY | Exit 2: suggest a terminal or `kodachrome-process` |
| Artifacts missing or invalid | Exit 2 naming the path and the specific validation failure |
| LUT SHA-1 disagrees with `params.json` | Exit 2: artifact is mixed; re-run training or restore |
| `.cube` malformed, non-finite, out of range, or non-unit domain | Exit 2 naming the line |
| Disk write fails | Print with path, keep session alive |
| Fetch: file fails validation | Skip, record reason in manifest, continue |
| Fetch: fewer than `--min-files` accepted | Exit 1 |
| Train: corpus below minimum | Exit 1 naming `--allow-small` |
| Train: invalid numeric option | Exit 1 naming the flag and its valid range |
| Train: source and target identical | Exit 1 |
| Train: acceptance gate fails | Write the artifact and report, exit non-zero, name the gate |
| Batch: output dir equals or nests in input | Exit 1 |
| Batch: output exists | Skip unless `--overwrite` |

## 9. Testing

All tests run on the Mac with no hardware and no network.

- `test_color.py` — conversions, round trips, published Oklab values. **Done.**
- `test_normalize.py` — float and `cv2.LUT` agreement, clamping and the
  `clamped` flags, idempotence, neutralisation.
- `test_lut.py` — identity, `.cube` round trip, red-fastest order, NumPy
  against Pillow, and every validation rejection including non-finite,
  out-of-range and non-unit domain.
- `test_grain.py` — luminance preserved, no chroma bias, seeded
  reproducibility, black and white untouched.
- `test_imageio.py` — EXIF orientation applied (fixture rotated by tag), ICC
  profile converted (fixture tagged Adobe RGB gives different pixels than
  when tagged sRGB), malformed profile falls back and is reported, sRGB
  profile embedded on save.
- `test_artifacts.py` — load, validate, every rejection path, packaged
  default resolves via `importlib.resources`, override wins, atomic publish
  leaves no partial state when validation fails mid-publish, LUT hash
  mismatch refused.
- `test_pipeline.py` — order, `info` contents, `grain=False`.
- `test_camera.py` — raw mode accepted when the fake yields valid JPEG
  bytes; truncated or non-JPEG buffers trigger fallback; both outputs derive
  from one acquisition; negotiated-format mismatch warns; device parsing.
- `test_app.py` — both files written with the log line, `_ungraded` naming
  in fallback mode, same-second collision, headless loop, preview errors
  caught, seed recorded and reproducing the graded file.
- `test_batch.py` — `_kodachrome` skipped, `_original` preferred, `--all`,
  stem collisions, no-clobber, nested output refused.
- `test_fetch.py` — category walk against recorded responses, licence
  allowlist rejects a CC-BY-SA file, MIME and decode validation, atomic
  download leaves no partial file, manifest records rejections, resume
  re-validates.
- `test_dataset.py` — crop and resize, split by image before sampling with
  no leakage, pool caps, small-corpus failure and `--allow-small`.
- `test_transport.py` — hue binning and reweighting, weighted quantile map,
  rotation properties, IDT convergence, sliced Wasserstein.
- `test_lutfit.py` — design matrix, smoothness operator, curve recovery,
  identity pull where there is no data.
- `test_evaluate.py` — **identity LUT gives byte-identical
  `swd_before` and `swd_after`** under the paired evaluator; seed spread is
  computed; safety checks catch a channel-non-monotonic LUT, a
  neutral-tinting LUT and a heavily clipping LUT.
- `test_fit.py` — end-to-end on synthetic corpora; **recovers a known tone
  curve plus a 10 degree hue rotation** (the capability the spec claims) and
  a separate case asserting that a rotation far larger than one hue bin is
  only partially recovered, which pins the documented limitation (F-17).
- `test_packaging.py` — build a wheel into a temp venv, run
  `kodachrome-process` from a different working directory against the
  packaged default artifact.

Manual verification: the report images, and a timed run on the Pi.

## 10. Documentation plan

`README.md` (setup, usage, how it works, honest statement of what the match
is and is not, measured performance, limitations), `docs/decisions.md`
(dated entries), module docstrings explaining the why, and this spec kept in
step with the code.

## 11. Risks and mitigations

- **Unpaired matching cannot identify the film response.** Stated plainly in
  section 1 and the README; measured on held-out images against an identity
  baseline and seed noise.
- **Content bias.** Hue reweighting, regularisation, and a published
  residual histogram difference.
- **Scans are not the film.** Scanner and LoC colour management are baked
  in; documented, and the target corpus is user-replaceable.
- **Raw MJPEG may not work on this device.** Validated per frame with an
  honest fallback and a filename that reflects it.
- **Camera auto-exposure drift.** Not locked in v1; recorded and published
  as diagnostics so it is visible (7.5).
- **Proxy source corpus for the default artifact.** Flagged in
  `params.json`; README tells users to retrain.
- **Pi performance.** `cv2.LUT` normalisation and Pillow's C LUT keep the
  budget; grain is the fallback knob.

## 12. Future work (out of scope)

1. Lock and verify UVC exposure and white balance on real hardware; compare
   locked against automatic (the deferred half of F-09).
2. Small CNN replacing the LUT stage, trained on an Apple Silicon Mac,
   exported to ONNX.
3. Scene-type LUT variants selected per capture.
4. Halation and vignette models.

## 13. Revision 2 change log

| Finding | Disposition | Sections |
|---|---|---|
| F-01 identifiability | Accepted, reframed as an aesthetic match; paired evidence impossible (Kodachrome discontinued 2009, processing ended 2010); held-out split, identity baseline, seed spread | 1, 6.3, 6.5 |
| F-02 original JPEG | Accepted: byte-exact MJPEG with validated fallback and honest filenames | 7.1, 7.2 |
| F-03 artifact location | Accepted: packaged in `kodachrome/data/`, `importlib.resources` | 4, 5.6 |
| F-04 extras | Accepted: `[opencv]` extra included by `[train]` and `[dev]` | 4, 7.4 |
| F-05 metric | Accepted: paired evaluator, held-out, seed spread, clipping split out | 6.5 |
| F-06 provenance | Accepted (reduced): byte-level corpus hashes, manifests, atomic download; drift warns rather than hard-fails | 3, 5.8 |
| F-07 publication | Accepted: validate-then-move atomic publish | 5.6 |
| F-08 camera and preview | Accepted: verify FOURCC/size/rate, guard the whole preview, drain queue instead of triple grab | 7.1, 7.2 |
| F-09 camera state | Partially accepted: record and publish diagnostics; locking deferred to a hardware spike | 7.5, 12 |
| F-10 EXIF and ICC | Accepted: `exif_transpose`, ICC to sRGB, profile counts reported | 5.5, 6.5 |
| F-11 batch safety | Accepted in full | 7.3 |
| F-12 validation | Accepted (reduced): dataclass, schema, LUT and CLI numeric validation; small corpora fail with `--allow-small` | 5.2-5.6, 6.4, 8 |
| F-13 downloader | Accepted (reduced): licence allowlist, MIME and decode checks, page/revision IDs, rejections recorded | 3, 6.1 |
| F-14 audit | Accepted: grain seed, LUT hash, versions and stream settings in the log | 7.2 |
| F-15 performance | Accepted: pipeline and shutter-to-saved reported separately | 7.2 |
| F-16 normalisation | Partially accepted: diagnostics published; three-way ablation is out of scope | 5.2, 6.5 |
| F-17 drift | Accepted: `Artifacts` reconciled, hue-rotation test added, task numbering corrected | 5.6, 9 |
| F-18 LUT safety | Partially accepted: per-channel monotonicity, neutral chroma, clipped volume with numeric gates; Jacobian analysis out of scope | 6.5 |
