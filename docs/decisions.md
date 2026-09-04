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

## 2026-09-04: The first real fit fails `channel_monotone`, and the gate is right

**Decided:** do not promote the trained artifact to the packaged default.
Leave the gate thresholds exactly as they are.
**Rejected:** relaxing `channel_monotone`; shipping the artifact with a
failing gate; over-regularising until the gate passes.
**Why:** trained on 64 proxy source images and 1,140 Kodachrome scans with a
real held-out split, the fit reduces held-out distance from 0.02330 to
0.01346 — 42%, at seventeen times the seed-noise floor. The colour match
works. But the resulting LUT contains per-channel reversals of up to 0.366,
which is 93 levels out of 255: a gradient crossing one of those bands
posterises or inverts.

Two hypotheses were tested and the first was wrong. Unconstrained
extrapolation looked likely — 71% of the colour cube holds no source pixels,
and 82% of the *severe* reversals sat in empty cells. Anchoring those cells
with `--lambda-identity` fixed `grey_axis_monotone` and
`neutral_axis_chroma` and cut violations from 3,421 to 724. But at that
setting only 3% of the remaining violating nodes have zero support, against
47% of the cube; at `--lut-size 17` it is 0%. The reversals that survive are
in *well-supported* regions, so they are not extrapolation.

The gate was then checked against six obviously legitimate grades —
saturation, hue rotation at 5 and 10 degrees, warm channel gains, a contrast
S-curve, and identity. All six pass. The gate is not rejecting a plausible
film look; the fitted LUT is genuinely defective.

The cause is structural: iterative distribution transfer matches
distributions and makes no per-channel ordering promise, and the
least-squares fit that follows is unconstrained. Nothing in the pipeline
asks the learned map to be monotone. Adding that constraint is the real fix
and is a design decision, not a threshold to be quietly loosened —
`evaluate.py` says the thresholds were "agreed before tuning so the gates
cannot be quietly relaxed to fit whatever the fit produced", and a failing
gate on a correct gate means fix the fit.

## 2026-09-04: Resolved — a monotone projection, and defaults that reach the fit

**Decided:** keep the gate thresholds; fix the fit instead. Three changes.

1. `lutfit.enforce_monotone` projects each output channel onto the monotone
   cone along its own axis. The three constraints touch disjoint variables,
   so each is an exact 1-D isotonic regression along every fibre and one pass
   per channel is the optimal least-squares projection. `fit_lut` applies it
   by default. It costs nothing measurable: on the real fit it moved the
   held-out distance by less than 0.01%, because the reversals were not
   contributing to the match in the first place.
2. `lambda_identity` 1e-4 → 1.0 and `lambda_smooth` 1e-3 → 1e-2. Both were
   needed and neither alone sufficed: at 1e-4/1e-2 the grey-axis and
   neutral-tint gates still failed, and at 1.0/1e-3 they also failed. This
   was established by running the isolating pair rather than reasoning about
   it, after an earlier attribution turned out to be wrong.
3. The CLI's argparse defaults now derive from `FitConfig` instead of
   repeating the numbers. They had been duplicated, so raising the dataclass
   defaults changed nothing for anyone using `kodachrome-train` — the first
   run after the change came back byte-identical to the run before it.

**Result:** all five gates pass. Held-out distance 0.02330 → 0.01375, a 41%
reduction where the gate asks for 0.00161. Trained on 64 proxy-source images
and 1,140 Library of Congress Kodachrome scans, and promoted to the packaged
default.

**Caveat worth knowing:** the grade is warmer and richer but also *lifts
shadows*, which is flatter than Kodachrome's reputation for deep blacks. The
targets are scans of 70-year-old film, digitised flat; matching their
distribution reproduces the look of the scan, not of a slide on a light box.
That is consistent with the aesthetic claim this project already makes, but
it is a real limit on how "Kodachrome" the output can feel.

## 2026-09-04: The 150 ms Pi estimate is optimistic — measured 220 ms on a Mac

**Measured:** a 1920x1080 frame through `Pipeline.process` takes a median
220 ms on an Intel Core i5-1038NG7. The breakdown is grain 105 ms (47%),
the 3D LUT 56 ms (25%), normalisation 45 ms (20%).

**Why it matters:** the entry above estimates "about 150 ms on a Pi 400".
A Pi 400's Cortex-A72 at 1.8 GHz is several times slower than this laptop
for this kind of array work, so the realistic figure is closer to one
second per photo. The estimate was never measured and should not be quoted
until step 5 of Task 22 runs on the hardware.

**Not treated as a blocker:** capture is keypress-driven, one photo at a
time, not video. A second per shot is acceptable for that flow. But grain
is the dominant cost at nearly half the budget and is the obvious place to
look if it turns out to be too slow — it is currently the least optimised
stage of the three.

## 2026-09-04: Real captures showed the look was the scan, not the film

**Evidence:** six frames from the Pi 400 and a test chart. Whites went
0.987 → 0.847 in Oklab lightness, blacks lifted, mid grey darkened 16%,
every colour patch lost chroma (red −19%, skin patch −33%), neutrals carried
a blue-grey tint. Measured on 150 of the target scans as the trainer saw
them: **the white point sat at a median 0.72 linear luminance** (p10 0.43).
Archival scans are flat by design. The transport matched that flatness
faithfully and the LUT learned to compress tone into the middle.

**Decided, three changes, all on the training side except the exposure target:**

1. **Levels normalisation for target scans** (`NormalizeParams.levels`).
   Stretch p0.5–p99.5 of luminance to 0–1 in linear light, then a clamped
   gamma puts the median on the exposure target without moving either end.
   A gain cannot do that: it drags the white point again. One black point
   and one exponent for all channels, so neutrals stay neutral. The capture
   path refuses the flag.
2. **Exposure target 0.18 → 0.25.** Five of six Pi shots were dimmed 10–45%
   at 0.18; the camera's auto-exposure aims brighter than that.
3. **Grain 0.025 → 0.010.** 0.027 RMS on a flat grey patch, about 7 levels,
   and graded JPEGs 3–5× the originals because grain defeats compression.

**Result on the same chart:** whites 0.987 → 0.973, blacks 0.101 → 0.058,
mid grey unchanged, red −7%, green −2%, blue +3%. On the user's actual face
(not the chart's pale patch) skin chroma rose 5% where it had fallen 11%.
Held-out distance 0.02331 → 0.01326 before the neutral cap below.

**Rejected: a per-channel black point.** Tried because dark greys came out
olive; it made the fit *worse than identity* (0.01366 → 0.01750).
Subtracting a different offset per channel and clipping leaves 0.5% of
pixels at zero in one channel but not the others — saturated dark colours
the film never produced, fed straight into the target distribution. The
per-channel spread is still measured and reported, not corrected.

## 2026-09-04: Cap the neutral-axis tint at half the gate

**Decided:** `lutfit.cap_neutral_axis`, default cap 0.01 Oklab chroma,
applied after the fit and before the monotone projection. The grey ramp's
own tint is measured as a function of lightness, the excess over the cap is
subtracted from every node, and the subtraction tapers to zero above chroma
0.06 — just under skin, measured at 0.07 on a real face — so colour
rendering is untouched.
**Rejected:** relaxing `MAX_NEUTRAL_CHROMA` (the thresholds were agreed so
they cannot be quietly relaxed to fit what the fit produced); zeroing the
cast entirely.
**Why:** the levels-normalised fit left dark greys olive at 0.036 against
the 0.02 gate, and that cast is genuinely part of what the transport
matched, so removing it has a price. Measured on the held-out evaluator:

| cap | neutral max | held-out SWD | vs uncapped |
|---|---|---|---|
| none | 0.0363 | 0.01326 | — (fails gate) |
| 0.015 | 0.0151 | 0.01350 | +1.8% |
| **0.01** | **0.0101** | **0.01374** | **+3.6%** |
| 0.005 | 0.0051 | 0.01412 | +6.4% |
| 0 | 0.0007 | 0.01456 | +9.8% |

At 0.01 the match is within 0.00001 of the previously shipped artifact's
0.01375, with the tone problems gone and neutrals held below visibility.
`--neutral-axis-cap 0` gives fully neutral greys; a larger value keeps more
of the cast, up to the gate.

## 2026-09-04: The camera already out-saturates Kodachrome scans

**Measured** (colours with Oklab chroma above 0.06, all corpora through the
trainer's own normalisation): the U20CAM proxy source renders reds at 0.120
and greens at 0.103; the LoC 1940s scans 0.101 and 0.077; the K-14 era
scans 0.106 and 0.101. Overall colour chroma: source 0.090, LoC 0.083,
K-14 0.098. Modern ISPs push reds and greens hard; slide scans do not.

**Consequence:** an honest distribution match from this camera to any
Kodachrome scan lands near neutral or slightly desaturating on colours.
The first real fit's "subtlety" (colours −3%, deep reds +11%, greens −6 to
−10%) was the method working, not failing. The "Kodachrome look" in
popular memory is a palette — blue skies, muted greens, warm skin — plus
the density of a slide on a light table, which no scan carries.

## 2026-09-04: Reference changed to K-14 era slides, strength 1.4

**Decided:** the shipped default is trained on
`Category:Photographs taken on Kodachrome film` (761 files after
de-duplication: 106 public-domain family, 195 CC BY, 460 CC BY-SA; mostly
1970s–1990s slides, the K-14 process) at `--strength 1.4`. The LoC FSA corpus remains the tool's
default and the public-domain route.
**Rejected:** staying on LoC; LoC at 1.4; K-14 at 1.0.
**Why:** four candidates were measured identically on the user's own eight
Pi captures. All pass every gate. Against ungraded, effect sizes were LoC
0.031, LoC×1.4 0.035, K-14 0.034, K-14×1.4 0.041 Oklab; between any two
candidates the difference is 0.009–0.023, so the choice is close and was
made on character and era rather than on a decisive number. K-14 holds
skies and cyans (+2%) where LoC cut them (−7%), mutes greens half as much,
drives blacks to zero, and adds 5–7% contrast. Strength 1.4 extrapolates
the learned direction by 40% at no gate cost; above 1 the transport's
clipping starts to dominate, so the cap is 2.

**Licence.** CC BY and CC BY-SA must be asked for (`--licences`), the
policy and per-licence counts are in the artifact's provenance, and
`docs/reference-attribution.md` lists every file with author and licence.
A LUT reuses no pixels; attribution is provided anyway because it costs a
file and settles the question.

**Shipped:** held-out 0.03172 → 0.02029 (gate bar 0.00231), all five gates,
neutral max 0.0101. On the eight Pi captures: mean Oklab change 0.042,
contrast +8%, skin chroma +6%, colours −2%, whites 0.978, blacks 0.000.

## 2026-09-04: Fetcher double-listed files reachable through two categories

**Found:** the K-14 manifest claimed 835 files while 756 JPEGs were on
disk; the LoC manifest 1,140 against 1,133. First diagnosed as filename
collisions overwriting each other. That was wrong, and the correction came
from the guard written for it: the "colliding" names carried the same
Commons page id, so they were the *same file*, reached once through the
category and again through a subcategory. Nothing was lost; files were
listed, counted, and hashed into `corpus_sha1` twice.
**Fixed:** category traversal now skips a file title it has already seen;
a guard refuses to proceed if selected filenames still collide; non-LCCN
filenames carry the page id so genuinely different files with the same
stem cannot share one either. The earlier commit message that said
"79 files were lost" is superseded by this entry.

## 2026-09-04: Outdoor haze traced to an asymmetric normaliser; both sides now get levels

**Evidence:** the first outdoor captures came out flat. Stage by stage on
one street shot (Oklab L): the exposure gain of 0.79 pulled the 95th
percentile from 1.000 to 0.924, then the LUT lifted the 5th percentile
from 0.328 to 0.396 — above the ungraded 0.354 — and took 7% of the
colour. Range compressed 12%. White balance was not involved (gains
0.96–1.02 on every outdoor frame).

**Cause:** the pixel pools as the transport saw them. Source (WB + gain):
L quartiles 0.352 / 0.552 / 0.714, p95 0.912. K-14 target (levels): 0.396 /
0.582 / 0.747, p95 0.941. Same median by construction, but the target's
white was pinned at 1.0 by the stretch while the source's sat at gain ×
white, so the target was 0.03–0.045 L lighter from the first quartile up.
The transport faithfully learned "lift everything above the shadows" and
strength 1.4 amplified it. The earlier explanation offered for this — that
the target gamma lifts dense slides — was wrong in detail: the K-14 slides'
own median after the stretch is 0.21, close to the camera's 0.25. The gap
was in the highlights.

**Decided:** the source is normalised the same way — grey-world white
balance, then the black/white-point stretch and the gamma to the exposure
median — at training and on the Pi. All three steps are per-channel
monotone in linear light, so they bake into the same three 256-entry tables
and the float reference and the baked path agree within one 8-bit level
(tested, and falsified by dropping the gamma from the tables only).
Simulated before building: source quartiles move to 0.391 / 0.576 / 0.727,
within 0.005–0.02 of the target. The exposure gain had been clamping on 29%
of source images; levels clamps on none.

**What changes on a photo:** bright scenes keep their whites at white
instead of being dimmed; flat scenes get a clamped auto-contrast. The
street shot's 5th percentile goes from 0.396 (shipped) to 0.224 and its
range from 0.567 to 0.772, against 0.354 and 0.646 ungraded.

**Strength:** with both sides aligned the residual camera→Kodachrome
difference is small, and extrapolating it by 1.4 overshot: the ×1.4 fit was
worse than identity on held-out data and failed the grey-axis gate. The
shipped strength is set by what passes the gates, recorded below.

## 2026-09-04: Grey-axis projection, evaluator precision, strength back to 1.0

Three consequences of the symmetric normaliser, each measured before acting.

**Grey axis.** The first symmetric fit failed `grey_axis_monotone` by one
ramp step: −0.00108 at input 0.937, against a 0.001 tolerance. Per-channel
monotonicity does not imply it — along the diagonal all three inputs rise
together and cross terms can dip. Adjusting the diagonal nodes alone does
not fix it (the ramp between nodes blends all eight corners of the cell),
nor does a per-lightness scaling (a dip caused by corners at the same
lightness as the diagonal node survives it). `lutfit.enforce_grey_axis`
solves a small least-squares problem over the affected cells for the corner
luminance changes that put every ramp sample on its isotonic target,
repeated until the 256-point ramp is clean; on the real fit it moved 0.04%
of nodes. Runs between two per-channel passes.

**Evaluator precision.** The same fit failed `improvement_exceeds_noise` by
0.0003. Re-measured over five seeds, the improvement was 0.00237 ± 0.00033
— a seven-sigma effect — while the gate compared it to three times the
spread of a 64-projection distance that was itself noisier (0.0007 on a
0.0136 baseline) than the effect under judgement. The evaluator now uses
256 projections; the spread halves and the rule is unchanged. This is
instrument precision, documented here because it looks like a threshold
change from a distance.

**Strength.** With both sides normalised the same way the residual
camera→Kodachrome difference is small, and extrapolating it by 1.4 overshot:
the ×1.4 symmetric fit was *worse than identity* on held-out data. Shipped
strength is 1.0. The earlier choice of 1.4 amplified a tone artefact of the
asymmetric normaliser, which is why it looked like more Kodachrome.

**Shipped:** K-14, strength 1.0, symmetric levels. Held-out 0.01313 →
0.01134 (bar 0.00156), all five gates. On the seven captures that exposed
the problem: outdoor 5th-percentile lightness 0.19–0.30 (was 0.39–0.41,
ungraded 0.35–0.38), range +17–47% (was −6 to −10%), colours −8% to +3%.
On the earlier eight: mean change 0.065, contrast +25%, colours +10%.
