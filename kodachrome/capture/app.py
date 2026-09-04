"""``kodachrome-capture``: live preview, press SPACE, get two JPEGs.

Structure
---------
``CaptureSession`` owns the camera, the pipeline and the output folder, and
knows how to take one capture or produce one preview frame. Two thin loops
drive it, both taking injectable key sources so the whole flow is testable
with ``FakeCamera``:

* ``run_preview_loop`` draws the graded feed in an OpenCV window. Every GUI
  call is inside the guard, because a build without GUI support can fail at
  ``imshow`` rather than at ``namedWindow``, and a failure there must fall
  back to headless rather than end the session.
* ``run_headless_loop`` reads single keys from the terminal.

A dropped frame prints and continues in both loops. The spec promises the
session survives frame read failures, and that promise is only worth
anything if it also holds while the preview is running.

Capture semantics
-----------------
``SPACE`` acquires one fresh frame and saves exactly that frame; the
displayed frame is not re-used. ``capture()`` calls ``camera.read()`` once,
so the saved original and the graded image always come from one acquisition.

Output layout
-------------
``OUT/YYYY-MM-DD/HHMMSS_original.jpg`` holds the camera's own JPEG bytes,
written verbatim. When the camera could not supply them the file is named
``_ungraded.jpg`` instead, so the name never overstates the contents.
``HHMMSS_kodachrome.jpg`` is the graded version, and one JSON line per
capture lands in ``captures.jsonl``.

That line is an audit record, not a status message. It carries the grain
seed and the LUT hash, which together let anyone regenerate the graded file
from the original; the negotiated stream format, so a camera that quietly
dropped to a different mode is visible; and two timings, because the
pipeline cost and the time from shutter to durable file are different
numbers and only the second is what the user waits for.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import time
import tty
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import __version__
from .._cv2 import require_cv2
from ..artifacts import PARAMS_VERSION, Artifacts, ArtifactsError
from ..imageio import save_jpeg
from ..pipeline import Pipeline
from .camera import Camera, CameraError, FakeCamera, V4L2Camera

cv2 = require_cv2()

DEFAULT_OUT = Path("~/Pictures/kodachrome")
WINDOW_NAME = "Kodachrome  [SPACE capture | P toggle grade | Q quit]"


@dataclass
class CaptureResult:
    original: Path
    kodachrome: Path
    record: dict


class CaptureSession:
    def __init__(
        self,
        camera: Camera,
        pipeline: Pipeline,
        out_root: str | Path,
        now: Callable[[], datetime] | None = None,
        seed_rng: np.random.Generator | None = None,
        package_version: str = __version__,
    ) -> None:
        self.camera = camera
        self.pipeline = pipeline
        self.out_root = Path(out_root).expanduser()
        self._now = now or datetime.now
        self._seed_rng = seed_rng or np.random.default_rng()
        self._package_version = package_version

    def _allocate(self, suffix: str) -> tuple[Path, str, datetime]:
        t = self._now()
        day_dir = self.out_root / t.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        base = t.strftime("%H%M%S")
        stem, k = base, 1
        while (day_dir / f"{stem}_{suffix}.jpg").exists():
            k += 1
            stem = f"{base}-{k}"
        return day_dir, stem, t

    def capture(self) -> CaptureResult:
        shutter = time.perf_counter()
        frame = self.camera.read()

        seed = int(self._seed_rng.integers(0, 2**31 - 1))
        t0 = time.perf_counter()
        graded, info = self.pipeline.process(frame.rgb, rng=np.random.default_rng(seed))
        pipeline_ms = (time.perf_counter() - t0) * 1000.0

        suffix = "original" if frame.jpeg is not None else "ungraded"
        day_dir, stem, t = self._allocate(suffix)
        original = day_dir / f"{stem}_{suffix}.jpg"
        if frame.jpeg is not None:
            original.write_bytes(frame.jpeg)
        else:
            save_jpeg(frame.rgb, original)
        kodachrome = save_jpeg(graded, day_dir / f"{stem}_kodachrome.jpg")
        shutter_to_saved_ms = (time.perf_counter() - shutter) * 1000.0

        record = {
            "timestamp": t.isoformat(timespec="seconds"),
            "original": original.name,
            "kodachrome": kodachrome.name,
            "frame_source": frame.source,
            **info,
            "grain_seed": seed,
            "params_version": PARAMS_VERSION,
            "package_version": self._package_version,
            **self.camera.stream_info.to_dict(),
            "pipeline_ms": round(pipeline_ms, 1),
            "shutter_to_saved_ms": round(shutter_to_saved_ms, 1),
        }
        with (day_dir / "captures.jsonl").open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        return CaptureResult(original, kodachrome, record)

    def preview_frame(self, graded: bool = True, size: tuple[int, int] = (640, 360)) -> np.ndarray:
        small = cv2.resize(self.camera.read().rgb, size, interpolation=cv2.INTER_AREA)
        if graded:
            small, _ = self.pipeline.process(small, grain=False)
        return small


def _announce(result: CaptureResult, out: Callable[[str], None]) -> None:
    r = result.record
    clamps = [k for k, v in r["clamped"].items() if v]
    note = f" (clamped: {', '.join(clamps)})" if clamps else ""
    out(
        f"Saved {result.kodachrome.name} + {result.original.name} in "
        f"{r['shutter_to_saved_ms']:.0f} ms; wb={r['wb_gains']} "
        + (f"levels gamma={r['levels']['gamma']} stretch={r['levels']['stretch']}"
           if r.get("levels") else f"exposure={r['exposure_gain']}")
        + note
    )


def run_headless_loop(
    session: CaptureSession,
    read_key: Callable[[], str | None],
    out: Callable[[str], None] = print,
) -> int:
    out("Headless mode: SPACE to capture, Q to quit.")
    count = 0
    while True:
        key = read_key()
        if key is None:
            continue
        if key == " ":
            try:
                _announce(session.capture(), out)
                count += 1
            except (CameraError, OSError) as exc:
                out(f"error: {exc}")
        elif key.lower() == "q":
            return count


def run_preview_loop(session: CaptureSession, window_name: str = WINDOW_NAME) -> bool:
    """Run the windowed loop. Returns False if this build cannot show a window."""
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    except cv2.error:
        return False
    graded = True
    try:
        while True:
            try:
                frame = session.preview_frame(graded)
                cv2.imshow(window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1) & 0xFF
            except CameraError as exc:
                print(f"error: {exc}")
                continue
            except cv2.error:
                return False
            if key == ord(" "):
                try:
                    _announce(session.capture(), print)
                except (CameraError, OSError) as exc:
                    print(f"error: {exc}")
            elif key in (ord("p"), ord("P")):
                graded = not graded
            elif key in (ord("q"), ord("Q"), 27):
                return True
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


class TerminalKeys:
    """Put the terminal in cbreak mode and read single keys without Enter."""

    def __enter__(self) -> TerminalKeys:
        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc: object) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def read(self, timeout: float = 0.1) -> str | None:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if ready else None


def has_display() -> bool:
    return sys.platform == "darwin" or bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kodachrome-capture",
        description="Capture Kodachrome-graded photos from the U20CAM.",
    )
    parser.add_argument("--device", default=None, help="index, /dev/videoN or /dev/v4l/by-id/...")
    parser.add_argument(
        "--artifacts", type=Path, default=None, help="artifact dir (default: bundled)"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-preview", action="store_true", help="never open a window")
    parser.add_argument("--fake", action="store_true", help="synthetic camera, no hardware")
    parser.add_argument("--seed", type=int, default=None, help="seed the grain seed generator")
    args = parser.parse_args(argv)

    try:
        pipeline = Pipeline(Artifacts.resolve(args.artifacts))
    except ArtifactsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        camera: Camera = FakeCamera() if args.fake else V4L2Camera(args.device)
    except CameraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    session = CaptureSession(
        camera,
        pipeline,
        args.out,
        seed_rng=np.random.default_rng(args.seed),
    )
    try:
        if not args.no_preview and has_display():
            if run_preview_loop(session):
                return 0
            print("Preview unavailable (OpenCV built without GUI); falling back to headless mode.")
        if not sys.stdin.isatty():
            print(
                "error: stdin is not a terminal, so keys cannot be read. Run from a terminal, "
                "or use kodachrome-process for files.",
                file=sys.stderr,
            )
            return 2
        with TerminalKeys() as keys:
            run_headless_loop(session, keys.read)
        return 0
    finally:
        camera.close()


if __name__ == "__main__":
    sys.exit(main())
