# Review findings: Kodachrome Film spec and implementation plan

Reviewed:

- `docs/superpowers/specs/2026-09-03-kodachrome-film-design.md`
- `docs/superpowers/plans/2026-09-03-kodachrome-film.md`
- `docs/decisions.md`

The design is thoughtfully decomposed and the plan is unusually executable, but a few
issues should be resolved before implementation. Severity means:

- **Blocker**: the implementation would not meet a stated top-level requirement, or the
  documented install/run path would be broken.
- **High**: likely to produce misleading results, corrupt/inconsistent artifacts, or a
  materially unreliable runtime.
- **Medium**: important robustness, reproducibility, or maintainability improvement.

## Blockers

### F-01 — The training method cannot establish the claimed “as close as possible” film response

**References:** spec lines 12–14, 306–346, 504–506; plan lines 3152–3163, 3950–3964,
4268, 4314–4319.

The source and target photographs are unpaired and depict different scenes. Matching their
pixel distributions does not identify how Kodachrome would render a given U20CAM scene; it
can also learn subject-matter, lighting, era, scanner, and corpus-selection differences. Hue
reweighting reduces one marginal bias but does not solve this identifiability problem.

The stronger claim that reweighting means transport “can then only learn how Kodachrome
renders each hue” is not guaranteed by the algorithm. IDT rotates the complete 3-D Oklab
space and performs unrestricted marginal mappings, so individual points can still move
across hue bins. Clipping the target weights also means the weighted hue histograms cannot
always match, especially when a hue bin is absent in the target.

**Recommendation:** either:

1. Reframe v1 as an *aesthetic distribution match to LoC-scanned 1940s Kodachrome*, not an
   estimate of the film response; or
2. Add paired/calibrated evidence: photograph the same colour chart and representative
   scenes with the U20CAM and on the chosen Kodachrome stock, scan through a controlled
   pipeline, and evaluate on held-out pairs.

For the unpaired approach, split train/validation by image before pixel sampling, compare
against identity and simple tone/saturation baselines, report multiple seeds/confidence
intervals, and describe hue reweighting as a heuristic rather than a constraint.

### F-02 — The saved “original JPEG” is not the camera’s original JPEG

**References:** spec lines 8–10 and 405–411; plan lines 1663–1675, 1692–1699, 2065–2074.

`cv2.VideoCapture.read()` decodes an MJPEG frame into BGR pixels. The capture session then
converts it to RGB and Pillow encodes a new quality-95 JPEG. That loses the camera’s original
compressed bytes and metadata and introduces a second lossy compression pass. This directly
conflicts with the headline promise that the camera’s original JPEG is saved.

**Recommendation:** decide what “original” means. If it means byte-preserved MJPEG, use a
V4L2 path that exposes the compressed frame buffer and decode that same buffer for grading.
If an ungraded re-encode is acceptable, rename it throughout the spec and README to
“ungraded capture” and document the decode/re-encode step. Add a test that both outputs come
from exactly the same acquired frame.

### F-03 — Installed commands depend on a repository-relative artifact directory

**References:** spec lines 94–97, 393–394, 418–419; plan lines 82–86, 1829, 2174, 4203–4204,
4457–4464.

The committed LUT and parameters live outside the Python package, while every CLI defaults
to `Path("artifacts")`. An editable install only works when the command is launched from the
repository root. A wheel would not contain the artifacts at all, and invoking the command
from another directory would fail.

**Recommendation:** package the default artifact under `kodachrome/` and locate it with
`importlib.resources`, while retaining `--artifacts` as an override. Add wheel/sdist smoke
tests that install into a clean environment and run the commands from a different working
directory. If the project intentionally supports only a checked-out repository, state that
explicitly and resolve the artifact path relative to the checkout rather than the caller’s
current directory.

### F-04 — The declared dependency extras do not install a working trainer or runtime

**References:** spec lines 68–72 and 427–433; plan lines 13–17, 73–80, 1444–1465,
2882–2890.

OpenCV is imported at module import time by normalisation, grain, dataset, camera, and the
capture app. However, the base dependencies contain only NumPy and Pillow, and `[train]`
does not include OpenCV. Consequently `pip install .[train]` does not install a usable
trainer, despite the package docstring recommending it. The Pi’s apt-provided OpenCV is a
reasonable deployment exception, but it does not make the Python dependency contract true
on other platforms.

**Recommendation:** introduce an `opencv` extra and include it from `train`/`dev`, or include
`opencv-python` in `train` directly. Keep the Pi instructions on apt OpenCV and add a clear
startup diagnostic when `cv2` is unavailable. Test the documented install combinations in
clean environments.

## High-priority findings

### F-05 — The before/after metric is not an apples-to-apples comparison

**References:** spec lines 352–361; plan lines 3260–3282 and 3848–3865.

`compute_metrics` gives `swd_before` and `swd_after` different RNG seeds. Each call therefore
uses different source samples, replacement samples from the target, and projection
directions. Some reported improvement can be sampling variance. Both values are also
calculated on the training pixel pool, so they measure optimization fit rather than
generalization. Finally, the LUT-fit error compares the in-gamut rendered LUT output with
unclipped transported Oklab partners; gamut clipping is folded into a metric described as
LUT regression error.

**Recommendation:** build one evaluation sample/projection object and reuse it for before
and after; evaluate on images held out before sampling; repeat across seeds or bootstrap a
confidence interval; and separately report transport-to-gamut clipping error and
LUT-to-clipped-target fit error. The test should prove that an identity LUT produces equal
before/after values under the paired evaluator.

### F-06 — Corpus and artifact provenance is not actually reproducible

**References:** spec lines 224–237, 258–260, 513–515; plan lines 2633–2643, 2677–2696,
2796, 2993–2998, 4180–4195, 4371–4376.

`source_dir_sha1` hashes only filenames and byte counts, so different content with the same
size has the same fingerprint. No target-corpus fingerprint is recorded. Training ignores
the download manifest and simply processes every recognized file in the directory. A
resumed download trusts any non-empty existing file without checking its recorded SHA-1 or
whether Pillow can decode it. Task 17 then instructs the implementer to delete unsuitable
files without rebuilding the manifest. These behaviors contradict the claim that the
manifest records exactly what was used.

**Recommendation:** train from a validated manifest (or generate a training manifest), hash
the actual bytes of every source and target file, record both corpus-manifest hashes, and
fail on missing/extra/hash-mismatched files unless an explicit repair flag is used. Record
the code revision, dependency versions, command/config, and target provenance in
`params.json`. Download to a temporary file, validate status/content type/decode/hash, then
atomically rename it.

### F-07 — Artifact publication can leave a mixed or unusable artifact set

**References:** plan lines 1294–1312 and 4158–4196.

Training overwrites `kodachrome.cube`, then generates the report, then overwrites
`params.json`. An interruption or report/write failure can leave a new LUT paired with old
parameters. Individual writes are also non-atomic. A concurrent capture can observe this
mixed state.

**Recommendation:** write LUT, parameters, and report into a temporary/versioned directory;
load and validate the complete artifact there; then publish it atomically (or update an
atomic `current` pointer). Include a LUT content hash in `params.json` and verify it on load.
Use the same temp-file/rename pattern for `params.json` and `.cube` writes.

### F-08 — Camera negotiation and preview failure handling do not meet the spec

**References:** spec lines 382–389 and 439–449; plan lines 1651–1704, 2085–2090,
2121–2143.

The implementation checks only negotiated dimensions, not FOURCC or FPS. A device can
silently fall back from MJPEG/30 fps to another mode. Auto-probing accepts the first video
node that emits a frame, which is risky on multi-node UVC devices or systems with another
webcam. The preview loop does not catch `CameraError` from `preview_frame`, despite the spec
saying frame failures keep the app running. It catches GUI errors only around
`namedWindow`; an error from `imshow`/`waitKey` escapes instead of triggering headless
fallback.

Two extra `grab()` calls on every `read()` also make the “preview at camera frame rate” claim
unlikely and mean pressing Space saves a later frame rather than the displayed frame.

**Recommendation:** verify and report actual FOURCC, dimensions, and FPS; select by stable
device path/vendor/product where possible; test all GUI calls and frame failures; and use a
bounded capture thread or driver buffer setting so preview consumes the newest frame
without triple-reading synchronously. Define whether capture freezes the displayed frame or
acquires a new one.

### F-09 — Camera image-processing state is neither controlled nor recorded

**References:** spec lines 25–33, 382–389; `docs/decisions.md` “White balance the source”.

The learned transform depends on the U20CAM’s rendering, but the plan only waits 15 frames
for AE/AWB. It does not set, lock, verify, or record exposure, white balance, gamma, gain,
backlight compensation, WDR mode, firmware, or negotiated stream format. Auto settings can
change across the source corpus and at runtime, undermining the assumption that the LUT sees
a consistent camera response. Applying grey-world WB after camera AWB does not make the
underlying tone/color processing reproducible.

**Recommendation:** define a supported UVC control profile, apply and read it back (for
example through V4L2 controls), record it in source/capture metadata, and warn on drift.
Evaluate locked versus automatic AE/AWB rather than assuming the 15-frame warm-up is enough.
At minimum, report the software gain distribution and clamp frequency so unsuitable source
images are visible.

### F-10 — Image loading ignores EXIF orientation and ICC color management

**References:** spec lines 41–61 and 280–291; plan lines 1444–1465.

`Image.open(...).convert("RGB")` neither applies EXIF orientation nor transforms an embedded
ICC profile into an explicit working sRGB space. Orientation affects the fixed 6% crop, and
color-profile handling is especially important when the target is a collection of archival
scans whose scanner/color management is acknowledged as part of the learned look.

**Recommendation:** apply `ImageOps.exif_transpose`, convert from embedded profiles to a
defined sRGB profile with `ImageCms`, specify behavior for missing/invalid profiles, and
record profile statistics in the manifest/report. Add oriented-image and non-sRGB-profile
fixtures. Decide whether output files should embed an sRGB profile and preserve selected
capture metadata.

### F-11 — Batch processing will double-grade captures and can overwrite outputs

**References:** spec lines 418–423; plan lines 1812–1819 and 1860–1868.

The batch command processes every supported image. A normal capture directory contains both
`*_original.jpg` and `*_kodachrome.jpg`, so the documented regrading workflow grades the
already graded files too. Files such as `a.jpg` and `a.png` map to the same output filename,
and an existing output is silently replaced. Using the same input/output directory makes
the collision behavior more dangerous.

**Recommendation:** default to `*_original.*` when that convention is detected, always skip
`*_kodachrome.*`, reject identical/nested output directories unless explicitly allowed, and
define a no-clobber policy. Preserve relative paths or disambiguate same-stem inputs. Add
tests for captured folders, repeated runs, same input/output, and same stems with different
extensions.

### F-12 — Configuration, artifact, and LUT validation is too shallow

**References:** spec lines 238–241 and 441–452; plan lines 768–789, 832–842, 905–939,
1259–1291, 4200–4258.

Only LUT size receives meaningful CLI validation. Zero/negative iterations, hue bins,
sample counts, or regularization; negative grain; non-finite values; inverted normalization
bounds; and arbitrary strength all pass through. Empty sampled pools and corrupt images are
not translated into actionable CLI errors. `Artifacts.load` assumes the JSON root and
nested values have the right types. The `.cube` reader ignores `DOMAIN_MIN/MAX` values and
accepts NaN/Inf or table values outside `[0,1]`, even though the in-memory contract requires
finite values in that range.

**Recommendation:** validate all dataclasses at construction, validate the complete versioned
JSON schema, require finite arrays/ranges, and either enforce a `[0,1]` cube domain or
correctly rescale it. Catch expected decode, schema, convergence, and write failures at CLI
boundaries. Make too-small corpora fail by default with an explicit `--allow-small` escape
hatch rather than warning and continuing into a statistically weak fit.

### F-13 — Downloader licensing and media selection are recorded but not enforced

**References:** spec lines 41–61 and 272–277; plan lines 2480–2490, 2581–2600,
2617–2629, 4352–4376.

The fetcher accepts every image returned by a category after filename filtering. It stores
`LicenseShortName` but never rejects a non-public-domain license, and it does not verify that
a file is a photograph rather than a scan, diagram, document, or derivative whose filename
lacks a skip word. Task 17 chooses a proxy category based first on file count and only
spot-checks twelve thumbnails. This does not establish the proxy corpus criteria.

**Recommendation:** use an explicit rights/license allowlist and record stable Commons page
and revision IDs; validate MIME/decode/dimensions; define auditable inclusion/exclusion
rules; and review a generated contact sheet of the full selected proxy set. Any manual
exclusions must be written back to the training manifest with reasons.

## Medium-priority findings and improvements

### F-14 — Output reproducibility and capture audit data are incomplete

**References:** spec lines 405–411; plan lines 2065–2082.

Film grain uses a fresh unrecorded RNG, and the capture log does not identify the LUT or
parameter file used. Reprocessing the original therefore cannot reproduce the saved graded
file, and two artifacts cannot be distinguished later.

**Recommendation:** record an artifact ID/LUT hash, params hash/schema version, grain seed,
camera/stream settings, image dimensions, and software version in each JSONL record. Consider
a sidecar per image or an append strategy with locking if more than one capture process can
run.

### F-15 — Performance measurement excludes important capture costs

**References:** spec lines 413–415 and 484; plan lines 2065–2074, 4501–4517.

`processing_ms` starts after camera acquisition and stops before encoding/writing either
JPEG or appending the log. Reporting it as “ms per capture” understates user-visible latency.
Five samples are also too few to characterize cold-start, sustained, or thermal behavior,
and preview FPS is asserted rather than measured.

**Recommendation:** report pipeline time and end-to-end shutter-to-durable-save time
separately, including p50/p95 over a longer warm and cold run. Measure preview FPS, dropped
frames, peak memory, negotiated stream mode, and behavior with grain disabled. Include disk
free-space handling in the Pi smoke test.

### F-16 — Normalization choices need ablation and diagnostics

**References:** spec lines 117–158 and 280–291; plan lines 534–672 and 4151–4156.

Per-image grey-world WB can remove intentional scene lighting, while forcing every target
scan to the same median can remove part of the film/scanner exposure distribution the LUT is
supposed to learn. The current report does not reveal how frequently gains clamp or how much
these steps alter each corpus.

**Recommendation:** make normalization modes explicit training/runtime profile choices and
compare at least: camera AWB only, locked camera plus software WB, and current double-WB.
Report WB/exposure gain histograms, clamp rates, clipped-pixel rates, and results with target
exposure normalization disabled. Store the selected policy in the artifact schema.

### F-17 — The acceptance tests and spec have drifted

**References:** spec lines 468–475; plan lines 35, 3950–3964, 4268–4278; spec lines 201–213
versus plan lines 1122–1128 and 1251–1257.

The spec promises a synthetic tone-curve-plus-10-degree-hue-rotation recovery test, while the
plan’s end-to-end test intentionally removes hue rotation and tests only lightness/chroma.
That avoids testing a stated capability and the documented hue limitation. The spec exposes
`Artifacts.meta`; the plan exposes `training` and `path`. The file table says the trained
artifact is produced in Task 15 although training occurs in Task 17. Task 16 updates only
some of these differences.

**Recommendation:** reconcile the spec before coding, keep one canonical public API/schema,
and add a traceability table that maps every behavioral requirement and error-table row to a
test. Add CI for supported Python versions plus clean package-install tests; retain Pi/U20CAM
checks as a separate hardware acceptance checklist.

### F-18 — Artifact safety checks should go beyond a luminance-only grey-axis flag

**References:** spec lines 352–361 and 504–506; plan lines 3806–3810 and 4391–4398.

`grey_axis_is_monotone` checks only luminance. A LUT may pass while turning neutral greys
colored, making an individual output channel non-monotonic, clipping large volumes of the
cube, or creating severe local folds away from the grey axis. Visual ramp inspection is
useful but subjective and covers only a few paths through the cube.

**Recommendation:** add neutral-axis chroma limits, per-channel monotonicity checks, dense
cube-grid Jacobian/fold diagnostics, clipping-volume percentages, maximum local gradient,
and continuity checks at the hue wrap. Establish numeric acceptance bounds before tuning the
default artifact.

## Decisions/questions to settle before implementation

1. Does “original JPEG” mean the byte-exact MJPEG frame emitted by the U20CAM, or is an
   ungraded quality-95 re-encode acceptable?
2. Is v1 intended to make a scientifically defensible camera-to-film response claim, or is
   an aesthetically convincing match to the LoC corpus sufficient? The current unpaired
   method supports only the latter claim.
3. Should the command-line tools work as a normally installed Python package from any
   directory, or is running from a repository checkout an intentional constraint?

## Suggested plan ordering changes

Before the current Task 1, resolve F-01/F-02/F-03 and update the goal, capture interface,
artifact location, and acceptance criteria. Then add schema/config validation and atomic
artifact publication before trainer orchestration. Build the real U20CAM acquisition spike
(including compressed-frame access, negotiated mode, and controls) before implementing the
full trainer; it is the highest-risk hardware assumption. Finally, train/accept the default
artifact only after corpus manifests, held-out evaluation, and numeric safety gates are in
place.
