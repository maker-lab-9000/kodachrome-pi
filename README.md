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
