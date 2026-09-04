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

## Pi setup

Written in a later step.

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
