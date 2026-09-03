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
