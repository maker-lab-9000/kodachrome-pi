# Decisions

Dated log of non-obvious choices. Newest at the bottom. Each entry says what
was decided, what was rejected, and why, so the reasoning survives the code.

## 2026-09-03: Match real scans, not a hand-tuned recipe

**Decided:** the Kodachrome look is learned from real Kodachrome scans.
**Rejected:** a parametric filter built from documented traits (saturated
reds, cyan shadows, punchy contrast); a published `.cube` from a film
emulation pack.
**Why:** "as close as possible" needs a measurable target. A recipe is only
as good as the author's eye and a third-party LUT has unknown provenance.

## 2026-09-03: Fitted 3D LUT on the Pi, no neural network

**Decided:** the Pi applies a 33^3 LUT via Pillow plus grain. The "machine
learning" is fitting that LUT on the Mac.
**Rejected:** CycleGAN-style CNN with ONNX inference on the Pi.
**Why:** a colour grade is a low-dimensional mapping; a LUT captures it,
runs in about 150 ms on a Pi 400, is inspectable in any grading tool, and
cannot hallucinate detail. Unpaired GAN training on about 1,000 images is
finicky. A seam is kept in `pipeline.py` so a CNN stage could replace the
LUT later.

## 2026-09-03: Wikimedia Commons instead of loc.gov for the scans

**Decided:** `kodachrome-fetch` downloads the FSA/OWI colour transparencies
from the Commons category "Color photographs from the Farm Security
Administration".
**Rejected:** the loc.gov JSON API.
**Why:** loc.gov returned HTTP 403 with a Cloudflare "Just a moment"
challenge to every scripted request tried (default, browser-like and
descriptive User-Agents). Commons hosts the same LoC scans, keeps the LCCN in
the filename, tags each file public domain, and its API welcomes scripted
access with a descriptive User-Agent. The direct LoC tile server does serve
images, so it remains a fallback for known catalogue numbers.

## 2026-09-03: NumPy/SciPy fitter, not PyTorch gradient descent

**Decided:** fit the LUT by hue-reweighted iterative distribution transfer
followed by regularised sparse least squares.
**Rejected:** optimising LUT nodes by gradient descent on a sliced
Wasserstein loss in PyTorch.
**Why:** two reasons. PyTorch stopped publishing Intel-Mac wheels after
2.2.2 (checked on PyPI), which would pin the trainer to an old release on the
machine at hand. More importantly the problem decomposes: transport gives
each source pixel a Kodachrome partner, and fitting a trilinear LUT to pairs
is a linear problem. That is deterministic, converges without tuning a
learning rate, and every intermediate can be inspected. Running on an Apple
Silicon Mac would make PyTorch viable but would not change this reasoning.

## 2026-09-03: White balance the source, not the Kodachrome targets

**Decided:** source images get grey-world white balance and exposure
normalisation; Kodachrome scans get exposure normalisation only.
**Why:** Kodachrome was daylight-balanced and its colour cast is part of the
look we want to learn. Per-slide exposure variation is not, so that is
removed on both sides.

## 2026-09-03: LUT in sRGB, statistics in Oklab

**Decided:** the transport and all metrics run in Oklab; the exported LUT
maps sRGB to sRGB.
**Why:** Oklab is perceptually uniform enough that hue bins and distances
mean what they say. `.cube` files and Pillow's `Color3DLUT` expect an sRGB
domain, and every grading tool can then open the artifact.

## 2026-09-03: Normalisation as three 1D lookups on the Pi

**Decided:** white balance and exposure are applied with `cv2.LUT` on three
256-entry tables.
**Why:** both are per-channel scalar gains in linear light, so the composite
sRGB-to-sRGB map is exactly three monotone 1D functions. The float reference
path stays in the trainer; a test keeps the two within 1/255.

## 2026-09-03: Revision 2, responding to external review

An external review (`findings.md`, F-01 to F-18) was run against revision 1
of the spec and plan. The controller verified each checkable claim against
the documents before accepting it. Ten findings were accepted in full, six
accepted with reduced scope, one partially rejected, and one reframed
because the reviewer's preferred remedy is impossible. Section 13 of the
spec tabulates the dispositions. The entries below record the reasoning that
does not fit in that table.

## 2026-09-03: The Kodachrome claim is aesthetic, not photometric

**Decided:** the project documents itself as producing an aesthetic colour
match to Library of Congress scans of 1939-1944 Kodachrome, measured on
held-out images against an identity baseline and seed noise.
**Rejected:** claiming to reproduce the film's response to a scene.
**Why:** the corpora are unpaired and depict different subjects in different
eras, so distribution matching cannot separate "how the film renders colour"
from "what the 1940s looked like". The reviewer's alternative, shooting one
colour chart on both the U20CAM and Kodachrome and fitting to the pairs,
cannot be done: Kodachrome was discontinued in 2009 and the last laboratory
able to process it closed at the end of 2010. Reframing is the only honest
option, and it costs nothing but a paragraph.

Corollary recorded in the code: the claim that hue reweighting means the
transport "can only learn how Kodachrome renders each hue" was an
overstatement and has been removed. Reweighting matches the aggregate hue
histogram; the transport still moves individual pixels in three dimensions
and can cross hue bins.

## 2026-09-03: Byte-exact camera JPEG, with a fallback that renames the file

**Decided:** `V4L2Camera` requests the compressed MJPEG buffer with
`CAP_PROP_CONVERT_RGB = 0`, validates the SOI and EOI markers and the
decode, writes those exact bytes as `_original.jpg`, and grades the decode
of the same buffer.
**Rejected:** re-encoding the decoded frame and calling it the original.
**Why:** the user asked for the camera's real bytes. OpenCV's V4L2 backend
supports the raw path, but issue #23311 shows it can hand back truncated
buffers, so every frame is validated. When raw mode is unavailable the
session falls back to decoding, and the saved file is named `_ungraded.jpg`
so the filename never claims more than the contents deliver.

## 2026-09-03: Default artifact ships inside the package

**Decided:** `kodachrome/data/kodachrome.cube` and `params.json` are package
data, located with `importlib.resources`; `--artifacts` overrides.
**Rejected:** a repository-relative `artifacts/` directory as the default.
**Why:** the CLI defaulted to `Path("artifacts")`, which only resolves when
the command runs from the repository root, and a built wheel would contain
no look at all. Packaging the artifact makes every command work from any
directory.

## 2026-09-03: Camera controls recorded, not locked, in v1

**Decided:** read back and record the UVC controls and the negotiated
format, and publish software gain and clamp statistics; do not lock exposure
or white balance.
**Rejected:** defining and enforcing a UVC control profile now.
**Why:** the PS5268 performs multi-exposure WDR fusion internally, so the
meaning and availability of exposure locks cannot be determined without the
hardware. Recording is cheap and makes drift visible; locking is the first
item of future work once the Pi is in hand.

## 2026-09-03: Paired evaluator for the before/after metric

**Decided:** one evaluator object holds the sample indices and projection
directions and is reused for both measurements, on held-out images, with the
spread across five seeds reported alongside.
**Rejected:** drawing a fresh random seed per measurement.
**Why:** the previous code seeded `swd_before` and `swd_after`
independently, so part of any reported improvement was sampling variance,
and both were computed on training pixels.

Measured on 4,000 random colours: the old unpaired code reported a 12.2%
change in distance for an **identity LUT**, one that alters nothing. That is
the same order as the improvement a real grade would be expected to show, so
the metric could not have distinguished a working fit from a broken one. The
paired evaluator returns a bit-identical value in the same test, and a test
now asserts exactly that.

## 2026-09-03: Conjugate gradients on the normal equations for the LUT fit

**Decided:** `fit_lut` forms the normal equations of the regularised least
squares problem and solves them per channel with `scipy.sparse.linalg.cg`
and a Jacobi preconditioner.
**Rejected:** `lsqr` on the stacked system (slower for this shape); direct
`spsolve` (3D grid fill-in is memory-hungry at 33^3).
**Why:** the identity term makes the system positive definite, so CG is
safe, and each iteration is a sparse product over about a million
non-zeros. Fits finish in seconds on the Mac.

## 2026-09-04: A report may not claim "held-out" when it is not

**Decided:** one flag, `held_out_eval`, is computed where the fallback
happens and recorded in `params.json`. Every surface that reports the
number reads it: the contact sheet row labels, the `summary.txt` metric
line, the console line, and a warning in both the console and the summary.
The gate detail was reworded to "evaluation distance", which is true either
way.
**Rejected:** leaving the fallback silent; removing the fallback and
refusing to report at all on a small corpus.
**Why:** `split_paths` returns an empty validation list for any corpus under
`1 / val_fraction` images — under five at the default — and `--allow-small`
exists to permit exactly that. Both `report.write_report` and `fit` then
quietly substituted training pixels while four separate strings still said
"held-out". The contact sheet is the artifact a person uses to judge the
fit, and showing training images under a held-out caption is the single most
flattering thing the tool could do. Refusing to run would be worse: a small
corpus is a legitimate way to try the pipeline out. Saying plainly which
pixels were measured costs nothing and keeps the numbers interpretable.

Found by generating a report for a three-image corpus and reading it as a
user would. The external review caught the contact-sheet instance; the other
three turned up only by running it.
