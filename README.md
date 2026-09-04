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

OpenCV is not a base dependency: on the Pi it comes from apt, everywhere else
from an extra. `pip install -e ".[train,dev]"` includes it. Installing the
bare package and then importing a module that needs OpenCV raises an error
naming both remedies.

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
| `--lambda-smooth` | 0.01 | raise if the ramps band or the fit looks noisy |
| `--lambda-identity` | 1.0 | raise if colours the camera never produced go strange |
| `--grain-strength` | 0.010 | grain, in luminance units at mid-grey |
| `--proxy-source` | off | mark the source as stand-in photos |
| `--allow-small` | off | proceed with a corpus below the recommended minimum |
| `--no-target-levels` | off | skip the black/white-point stretch on target scans (A/B only) |
| `--neutral-axis-cap` | 0.01 | most Oklab chroma a neutral input may gain; 0 = fully neutral greys |

### 4. Read the report

`report/summary.txt` is the short version: the held-out distance before and
after, the seed spread it must beat, and each gate. `contact_sheet.png`
shows held-out images normalised, graded, and beside real scans.
`ramps.png` shows the tone curve and hue movement. `diagnostics.png` shows
how often normalisation clamped. `metrics.json` has everything.

A corpus with fewer than `1 / --val-fraction` images — under five at the
default — cannot hold anything back, so the numbers are measured on the
pixels the fit was trained on. That is memorisation, not generalisation.
The trainer says so on the console, `summary.txt` opens with a WARNING, the
contact sheet labels its rows TRAINING rather than held-out, and
`params.json` records `metrics.held_out_eval: false`. Nothing silently
claims a held-out result it did not measure.

Promote a fit you like to the shipped default by copying it into the
package:

```bash
cp artifacts/kodachrome.cube artifacts/params.json kodachrome/data/
```

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

## How it works

The look ships inside the package at `kodachrome/data/`, so every command
works from any directory and a built wheel carries it. `--artifacts DIR`
points at a directory instead, which is how you use a LUT you trained
yourself. `params.json` records the normalisation, the grain settings, the
LUT's SHA-1 and full training provenance; a LUT whose hash disagrees with
its `params.json` is refused rather than silently graded with the wrong
parameters.

Until training has run, the bundled LUT is an identity placeholder and its
`training.note` says so.

## Commands

| Command | Where | What |
|---|---|---|
| `kodachrome-fetch` | Mac | download Kodachrome scans from Wikimedia Commons |
| `kodachrome-train` | Mac | fit the LUT and write `artifacts/` |
| `kodachrome-capture` | Pi | live preview, SPACE to capture |
| `kodachrome-process` | either | regrade a folder of originals |

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

### Regrade a folder

```bash
kodachrome-process ~/Pictures/kodachrome/2026-09-03 /tmp/regraded
```

Pointed at a capture folder, it grades only the `*_original.jpg` files and
skips the `*_kodachrome.jpg` siblings, so running it twice cannot
double-grade. Pointed at any other folder it grades everything. Existing
outputs are kept unless `--overwrite`; an output directory inside the input
is refused. `--all` overrides the originals-only default.

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
The shipped default was fit on 64 proxy-source photographs and 1,140 scans,
and moves the held-out distance from 0.02331 to 0.01374 where the gate asks
for 0.00158.

Other limits:

- The scans carry LoC's scanner and colour management, and 1940s Kodachrome
  differs from later K-14 stock. Point `--target` at your own scans to
  change the reference.
- **The scans are flat, and the trainer corrects for it.** The Library of
  Congress scans put white at about 0.72 linear luminance; an early fit
  learned that and pushed whites to 0.85. Target scans now get a
  black/white-point stretch before training (`--no-target-levels` turns it
  off). What remains is the film's colour, not the scanner's tone curve.
- **Neutral greys are held near neutral.** The learned cast on greys is
  capped at 0.01 Oklab chroma (`--neutral-axis-cap`); the film's warm
  shadows are real but stronger than looks good on a white wall.
- A large hue rotation is only partly recovered — roughly 5 degrees of a
  90-degree target. The damping comes from the transport and the LUT-fit
  smoothness, not from hue reweighting, which was measured and found to make
  almost no difference here. Saturation, lightness and tone curve are
  learned fully.
- 71% of the colour cube contains no source pixel at this corpus size, so
  those regions are held at identity by `--lambda-identity` rather than
  learned. More varied source photographs is the most direct improvement.
- White balance is grey-world with clamped gains, so a scene legitimately
  dominated by one colour is partially neutralised. The capture log records
  when a gain clamped.
- The camera's own auto exposure and white balance are recorded but not
  locked; see `docs/decisions.md`. Locking is the first item of future work.
- No lens, halation or vignette modelling; the 121-degree lens distortion is
  left alone.
