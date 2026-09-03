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

## Pi setup

Written in a later step.

## Commands

| Command | Where | What |
|---|---|---|
| `kodachrome-fetch` | Mac | download Kodachrome scans from Wikimedia Commons |
| `kodachrome-train` | Mac | fit the LUT and write `artifacts/` |
| `kodachrome-capture` | Pi | live preview, SPACE to capture |
| `kodachrome-process` | either | regrade a folder of originals |
