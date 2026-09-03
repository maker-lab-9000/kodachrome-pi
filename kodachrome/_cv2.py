"""One place that imports OpenCV, so one place explains how to install it.

OpenCV is deliberately not a base dependency. On Raspberry Pi OS it comes
from apt (`python3-opencv`), which is built with GTK so the preview window
works; pip's `opencv-python-headless` wheel silently has no GUI. Everywhere
else it arrives through the `[opencv]` extra, which `[train]` and `[dev]`
both include.

A bare `ModuleNotFoundError: No module named 'cv2'` does not tell a user
which of those two paths they are missing, so this guard does.
"""

from __future__ import annotations

from types import ModuleType

_MESSAGE = (
    "OpenCV (cv2) is required but not installed.\n"
    "  On Raspberry Pi OS:  sudo apt install python3-opencv\n"
    "                       (then create the venv with --system-site-packages)\n"
    "  Anywhere else:       pip install 'kodachrome-film[opencv]'\n"
    "                       (already included by the [train] and [dev] extras)"
)


def require_cv2() -> ModuleType:
    """Import and return ``cv2``, or raise ``ImportError`` explaining both fixes."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise ImportError(_MESSAGE) from exc
    return cv2
