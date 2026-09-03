"""Kodachrome film look for Raspberry Pi 400 + Innomaker U20CAM-1080P-WDR.

The package has two halves that share code:

* ``kodachrome.color``, ``normalize``, ``lut``, ``grain``, ``pipeline`` are the
  processing core used on the Pi. They depend only on NumPy, Pillow and OpenCV.
* ``kodachrome.train`` fits the LUT on a Mac from real Kodachrome scans and
  needs SciPy and requests (``pip install -e ".[train]"``).

See ``docs/superpowers/specs/2026-09-03-kodachrome-film-design.md`` for the design.
"""

__version__ = "0.1.0"
