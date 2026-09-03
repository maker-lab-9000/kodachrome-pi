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

_MISSING = (
    "OpenCV (cv2) is required but not installed.\n"
    "  On Raspberry Pi OS:  sudo apt install python3-opencv\n"
    "                       (then create the venv with --system-site-packages)\n"
    "  Anywhere else:       pip install 'kodachrome-film[opencv]'\n"
    "                       (already included by the [train] and [dev] extras)"
)

_BROKEN = (
    "OpenCV (cv2) is installed but failed to import: {error}\n"
    "This is an installation problem, not a missing package, so reinstalling\n"
    "cv2 will probably not help. A missing system library is the usual cause;\n"
    "on Raspberry Pi OS or another Debian, try:\n"
    "  sudo apt install libgl1 libglib2.0-0\n"
    "or switch to the apt build:  sudo apt install python3-opencv"
)


def require_cv2() -> ModuleType:
    """Import and return ``cv2``, or raise ``ImportError`` naming the right fix.

    The two failures need different advice and must not be conflated. A
    genuinely absent package is fixed by installing one; a package that is
    present but cannot load its native libraries (``libGL.so.1: cannot open
    shared object file`` is the classic on a headless Debian, and the Pi is
    a Debian) is not. Telling someone to install what they already have
    sends them down the wrong path, so the message is chosen from the
    exception rather than assumed.
    """
    try:
        import cv2
    except ModuleNotFoundError as exc:
        if exc.name != "cv2":
            raise ImportError(_BROKEN.format(error=exc)) from exc
        raise ImportError(_MISSING) from exc
    except ImportError as exc:
        raise ImportError(_BROKEN.format(error=exc)) from exc
    return cv2
