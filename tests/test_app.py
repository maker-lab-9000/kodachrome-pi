import io
import json
from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from kodachrome.artifacts import Artifacts, write_artifact
from kodachrome.capture.app import CaptureSession, main, run_headless_loop, run_preview_loop
from kodachrome.capture.camera import CameraError, FakeCamera, Frame, StreamInfo, synthetic_frame
from kodachrome.grain import GrainParams
from kodachrome.imageio import load_rgb
from kodachrome.lut import LUT3D
from kodachrome.normalize import NormalizeParams
from kodachrome.pipeline import Pipeline


def _jpeg_bytes(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture
def pipeline(tmp_path):
    d = tmp_path / "art"
    write_artifact(d, LUT3D.identity(9), NormalizeParams(), GrainParams())
    return Pipeline(Artifacts.load(d))


def _session(tmp_path, pipeline, camera=None, now=None):
    camera = camera or FakeCamera([synthetic_frame(90, 160)])
    return CaptureSession(camera, pipeline, tmp_path / "shots", now=now,
                          seed_rng=np.random.default_rng(0))


def test_raw_mode_writes_the_camera_bytes_verbatim(tmp_path, pipeline):
    rgb = synthetic_frame(48, 64)
    data = _jpeg_bytes(rgb)
    cam = FakeCamera(jpeg_bytes=[data], source="raw-mjpeg")
    fixed = datetime(2026, 9, 3, 21, 5, 7)
    result = _session(tmp_path, pipeline, cam, now=lambda: fixed).capture()
    assert result.original.name == "210507_original.jpg"
    assert result.original.read_bytes() == data, "the camera's own bytes must be saved unchanged"
    assert result.record["frame_source"] == "raw-mjpeg"


def test_decoded_mode_names_the_file_ungraded(tmp_path, pipeline):
    fixed = datetime(2026, 9, 3, 21, 5, 7)
    result = _session(tmp_path, pipeline, now=lambda: fixed).capture()
    assert result.original.name == "210507_ungraded.jpg"
    assert result.record["frame_source"] == "decoded"


def test_both_outputs_come_from_one_acquisition(tmp_path, pipeline):
    """A camera whose frames differ every read would produce mismatched files."""

    class ChangingCamera:
        stream_info = StreamInfo(64, 48, 30.0, "MJPG", False)

        def __init__(self):
            self.n = 0

        def read(self):
            self.n += 1
            return Frame(np.full((48, 64, 3), self.n * 20, np.uint8), None, "decoded")

        def close(self):
            pass

    cam = ChangingCamera()
    result = _session(tmp_path, pipeline, cam).capture()
    saved = np.asarray(Image.open(result.original).convert("RGB"))
    assert cam.n == 1, "capture must read exactly one frame"
    assert abs(int(saved.mean()) - 20) <= 2


def test_log_line_carries_full_provenance(tmp_path, pipeline):
    fixed = datetime(2026, 9, 3, 21, 5, 7)
    session = _session(tmp_path, pipeline, now=lambda: fixed)
    session.capture()
    line = json.loads((tmp_path / "shots" / "2026-09-03" / "captures.jsonl").read_text().strip())
    assert set(line) >= {
        "timestamp", "original", "kodachrome", "frame_source", "wb_gains", "exposure_gain",
        "clamped", "grain_seed", "lut_sha1", "params_version", "package_version",
        "width", "height", "fourcc", "fps", "pipeline_ms", "shutter_to_saved_ms",
    }
    assert line["shutter_to_saved_ms"] >= line["pipeline_ms"]
    assert isinstance(line["grain_seed"], int)


def test_recorded_seed_reproduces_the_graded_file(tmp_path, pipeline):
    """The seed is the only randomness, so it must pin the grade exactly.

    Two claims, checked separately, because conflating them hides which one
    broke. In memory the reproduction is bit-exact. Through the saved files
    it cannot be: both are quality-95 JPEGs, and grain is precisely the
    high-frequency content JPEG discards, so the bounds below are what the
    format allows (measured: mean 1.1, 99th percentile 4, worst pixel 8-12).
    """
    session = _session(tmp_path, pipeline)
    result = session.capture()
    seed = result.record["grain_seed"]
    assert isinstance(seed, int)

    frame = session.camera.read().rgb  # FakeCamera repeats the same frame
    first, _ = pipeline.process(frame, rng=np.random.default_rng(seed))
    second, _ = pipeline.process(frame, rng=np.random.default_rng(seed))
    assert np.array_equal(first, second), "the same seed must give the same pixels"

    original = np.asarray(Image.open(result.original).convert("RGB"))
    saved = np.asarray(Image.open(result.kodachrome).convert("RGB"))
    again, _ = pipeline.process(original, rng=np.random.default_rng(seed))
    diff = np.abs(again.astype(int) - saved.astype(int))
    assert diff.mean() < 3.0
    assert np.percentile(diff, 99) <= 10


def test_same_second_captures_do_not_collide(tmp_path, pipeline):
    fixed = datetime(2026, 9, 3, 21, 5, 7)
    session = _session(tmp_path, pipeline, now=lambda: fixed)
    a, b = session.capture(), session.capture()
    assert a.original.name == "210507_ungraded.jpg"
    assert b.original.name == "210507-2_ungraded.jpg"


def test_preview_frame_is_small_rgb(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    assert session.preview_frame(graded=True).shape == (360, 640, 3)
    assert session.preview_frame(graded=False).shape == (360, 640, 3)


def test_headless_loop_captures_on_space_and_quits_on_q(tmp_path, pipeline):
    session = _session(tmp_path, pipeline)
    keys = iter([None, " ", "x", " ", "q"])
    messages = []
    assert run_headless_loop(session, read_key=lambda: next(keys), out=messages.append) == 2
    assert len(list((tmp_path / "shots").rglob("*_kodachrome.jpg"))) == 2
    assert any("Saved" in m for m in messages)


def test_headless_loop_survives_a_camera_error(tmp_path, pipeline):
    class FlakyCamera(FakeCamera):
        def read(self):
            raise CameraError("boom")

    session = _session(tmp_path, pipeline, FlakyCamera())
    keys = iter([" ", "q"])
    messages = []
    assert run_headless_loop(session, read_key=lambda: next(keys), out=messages.append) == 0
    assert any("boom" in m for m in messages)


def test_preview_loop_falls_back_when_gui_is_unavailable(tmp_path, pipeline, monkeypatch):
    import cv2

    def boom(*a, **k):
        raise cv2.error("no GUI support")

    monkeypatch.setattr(cv2, "namedWindow", boom)
    assert run_preview_loop(_session(tmp_path, pipeline)) is False


def test_preview_loop_falls_back_when_imshow_fails(tmp_path, pipeline, monkeypatch, capture_gui):
    import cv2

    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: (_ for _ in ()).throw(cv2.error("no GUI")))
    assert run_preview_loop(_session(tmp_path, pipeline)) is False


def test_preview_loop_survives_a_frame_error(tmp_path, pipeline, monkeypatch, capture_gui):
    import cv2

    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    keys = iter([ord("q")])
    monkeypatch.setattr(cv2, "waitKey", lambda delay: -1 if delay == 50 else next(keys))

    session = _session(tmp_path, pipeline)
    calls = {"n": 0}
    real = session.preview_frame

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise CameraError("dropped frame")
        return real(*a, **k)

    monkeypatch.setattr(session, "preview_frame", flaky)
    assert run_preview_loop(session) is True  # a dropped frame must not end the session


@pytest.fixture
def capture_gui(monkeypatch):
    import cv2

    shown = []
    monkeypatch.setattr(cv2, "namedWindow", Mock())
    monkeypatch.setattr(cv2, "resizeWindow", Mock())
    monkeypatch.setattr(cv2, "setWindowProperty", Mock())
    monkeypatch.setattr(cv2, "getWindowProperty", lambda name, prop: cv2.WINDOW_FREERATIO)
    monkeypatch.setattr(cv2, "getWindowImageRect", lambda name: (0, 0, 640, 360))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "imshow", lambda name, frame: shown.append(frame.copy()))
    return shown


@pytest.mark.parametrize("terminal_controls", [False, True])
def test_capture_display_changes_only_after_snapshot(
    tmp_path, pipeline, monkeypatch, capture_gui, terminal_controls,
):
    import cv2

    camera = FakeCamera([synthetic_frame(90, 160), 255 - synthetic_frame(90, 160)])
    camera.read = Mock(wraps=camera.read)
    pipeline.process = Mock(wraps=pipeline.process)
    session = _session(tmp_path, pipeline, camera)
    capture = session.capture
    repaints = []

    def capture_after_loading_screen():
        assert len(repaints) == camera.read.call_count + 1
        assert np.array_equal(repaints[-1], capture_gui[-1])
        # Seven distinct TV colour bars, already painted before camera acquisition/grading.
        assert len(np.unique(repaints[-1][0], axis=0)) == 7
        return capture()

    monkeypatch.setattr(session, "capture", capture_after_loading_screen)
    events = iter([(-1, 0), (32, 0), (-1, 1), (ord("p"), 1), (32, 1), (-1, 2), (ord("Q"), 2)])

    def next_key():
        key, captures = next(events)
        assert len(capture_gui) == captures * 2 + 1  # prompt, then loading screen and photo
        assert camera.read.call_count == captures
        assert pipeline.process.call_count == captures
        return key

    def wait_key(delay):
        if delay == 50:
            return -1
        if delay == 1:
            repaints.append(capture_gui[-1].copy())
            return -1
        return -1 if terminal_controls else next_key()

    monkeypatch.setattr(cv2, "waitKey", wait_key)
    if terminal_controls:

        def read_key():
            key = next_key()
            return chr(key) if key >= 0 else None
    else:
        read_key = None
    assert run_preview_loop(session, captures_only=True, read_key=read_key)

    files = sorted(
        (tmp_path / "shots").rglob("*_kodachrome.jpg"), key=lambda p: p.stat().st_mtime_ns,
    )
    assert len(files) == 2
    for displayed, saved in zip(capture_gui[2::2], files, strict=True):
        rgb, _ = load_rgb(saved)
        expected = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)
        assert np.array_equal(displayed, expected[:, :, ::-1])
    assert not np.array_equal(capture_gui[2], capture_gui[4])


@pytest.mark.parametrize("failed_attempt", [1, 2])
def test_capture_display_keeps_last_photo_after_failed_capture(
    tmp_path, pipeline, monkeypatch, capture_gui, capsys, failed_attempt,
):
    import cv2

    session = _session(tmp_path, pipeline)
    capture = session.capture
    attempts = 0

    def flaky_capture():
        nonlocal attempts
        attempts += 1
        if attempts == failed_attempt:
            raise CameraError("dropped snapshot")
        return capture()

    monkeypatch.setattr(session, "capture", flaky_capture)
    events = iter([(32, 1), (32, 3), (-1, 5), (32, 5), (ord("q"), 7)])

    def next_key(delay):
        if delay in (1, 50):
            return -1
        key, updates = next(events)
        assert len(capture_gui) == updates
        return key

    monkeypatch.setattr(cv2, "waitKey", next_key)
    assert run_preview_loop(session, captures_only=True)
    assert "dropped snapshot" in capsys.readouterr().out
    before_failure = capture_gui[(failed_attempt - 1) * 2]
    after_failure = capture_gui[failed_attempt * 2]
    assert np.array_equal(after_failure, before_failure)


def test_capture_display_preserves_portrait_aspect_ratio(
    tmp_path, pipeline, monkeypatch, capture_gui,
):
    import cv2

    session = _session(tmp_path, pipeline, FakeCamera([synthetic_frame(160, 90)]))
    keys = iter([32, ord("q")])
    monkeypatch.setattr(cv2, "waitKey", lambda delay: -1 if delay in (1, 50) else next(keys))
    assert run_preview_loop(session, captures_only=True)
    assert capture_gui[-1].shape == (360, 640, 3)
    assert not capture_gui[-1][:, :219].any()
    assert not capture_gui[-1][:, 421:].any()
    assert capture_gui[-1][:, 219:421].any()


@pytest.mark.parametrize(
    "size, image_rect",
    [
        ((1920, 1080), (240, 0, 1440, 1080)),
        ((1280, 1024), (0, 32, 1280, 960)),
        ((800, 480), (80, 0, 640, 480)),
        ((1080, 1920), (0, 555, 1080, 810)),
        ((3840, 2160), (480, 0, 2880, 2160)),
    ],
)
def test_fullscreen_capture_fits_display_resolution(
    tmp_path, pipeline, monkeypatch, capture_gui, size, image_rect,
):
    import cv2

    monkeypatch.setattr(cv2, "getWindowImageRect", lambda name: (0, 0, *size))
    session = _session(tmp_path, pipeline, FakeCamera([synthetic_frame(120, 160)]))
    keys = iter([32, ord("q")])
    monkeypatch.setattr(cv2, "waitKey", lambda delay: -1 if delay in (1, 50) else next(keys))
    assert run_preview_loop(session, captures_only=True)
    cv2.namedWindow.assert_called_once_with(
        cv2.namedWindow.call_args.args[0], cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO,
    )
    cv2.setWindowProperty.assert_called_once_with(
        cv2.namedWindow.call_args.args[0], cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN,
    )
    assert len(capture_gui) == 4  # initial drawable, fullscreen prompt, loading bars, photo
    for frame in capture_gui[1:]:
        assert frame.shape == (size[1], size[0], 3)
    assert len(np.unique(capture_gui[2][0], axis=0)) == 7

    left, top, width, height = image_rect
    saved, = (tmp_path / "shots").rglob("*_kodachrome.jpg")
    rgb, _ = load_rgb(saved)
    expected = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)[:, :, ::-1]
    actual = capture_gui[-1]
    assert np.array_equal(actual[top:top + height, left:left + width], expected)
    borders = actual.copy()
    borders[top:top + height, left:left + width] = 0
    assert not borders.any()


def test_resolution_change_redraws_full_resolution_photo_without_recapture(
    tmp_path, pipeline, monkeypatch, capture_gui,
):
    import cv2

    camera = FakeCamera([synthetic_frame(720, 1280)])
    camera.read = Mock(wraps=camera.read)
    pipeline.process = Mock(wraps=pipeline.process)
    session = _session(tmp_path, pipeline, camera)
    keys = iter([32, -1, ord("q")])

    def wait_key(delay):
        if delay in (1, 50):
            return -1
        key = next(keys)
        if key == -1:
            monkeypatch.setattr(cv2, "getWindowImageRect", lambda name: (0, 0, 1280, 720))
        return key

    monkeypatch.setattr(cv2, "waitKey", wait_key)
    assert run_preview_loop(session, captures_only=True)
    assert camera.read.call_count == pipeline.process.call_count == 1
    assert len(capture_gui) == 4  # prompt, loading, photo, resized photo
    saved, = (tmp_path / "shots").rglob("*_kodachrome.jpg")
    rgb, _ = load_rgb(saved)
    assert np.array_equal(capture_gui[-1], rgb[:, :, ::-1])


def test_fullscreen_live_preview_preserves_camera_aspect_ratio(
    tmp_path, pipeline, monkeypatch, capture_gui,
):
    import cv2

    session = _session(tmp_path, pipeline, FakeCamera([synthetic_frame(120, 160)]))
    monkeypatch.setattr(cv2, "getWindowImageRect", lambda name: (0, 0, 1920, 1080))
    monkeypatch.setattr(cv2, "waitKey", lambda delay: ord("q"))
    assert run_preview_loop(session)
    assert capture_gui[-1].shape == (1080, 1920, 3)
    assert not capture_gui[-1][:, :240].any()
    assert not capture_gui[-1][:, 1680:].any()
    assert capture_gui[-1][:, 240:1680].any()


@pytest.mark.parametrize("geometry", [(0, 0, 0, 0), (0, 0, -1, -1), (0, 0, 1, 1), None])
def test_fullscreen_tolerates_unavailable_geometry(
    tmp_path, pipeline, monkeypatch, capture_gui, geometry,
):
    import cv2

    def get_rect(name):
        if geometry is None:
            raise cv2.error("geometry unavailable")
        return geometry

    monkeypatch.setattr(cv2, "getWindowImageRect", get_rect)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: ord("q"))
    assert run_preview_loop(_session(tmp_path, pipeline), captures_only=True)
    assert capture_gui[-1].shape == (360, 640, 3)


@pytest.mark.parametrize(
    "image_size, display_size",
    [((1280, 720), (1280, 1024)), ((1080, 608), (1080, 1920)), ((1440, 1080), (1920, 1080))],
)
def test_gtk_geometry_includes_borders_outside_the_image(
    monkeypatch, capture_gui, image_size, display_size,
):
    import cv2

    from kodachrome.capture.app import _window_size

    monkeypatch.setattr(cv2, "getWindowImageRect", lambda name: (0, 0, *image_size))
    monkeypatch.setattr(
        cv2, "getWindowProperty", lambda name, prop: display_size[0] / display_size[1],
    )
    assert _window_size("capture", (640, 360)) == display_size


def test_gtk_first_image_allocation_does_not_shrink_fullscreen_drawable(
    tmp_path, pipeline, monkeypatch, capture_gui,
):
    """GTK's first-image allocation overrides fullscreen if it hasn't been painted yet."""
    import cv2

    state = {"painted": False, "fullscreen": False, "stuck": False, "size": (1, 1)}

    def fullscreen(name, prop, value):
        if prop == cv2.WND_PROP_FULLSCREEN:
            state["fullscreen"] = True
            state["stuck"] = not state["painted"]

    keys = iter([32, ord("q")])

    def wait_key(delay):
        if capture_gui and not state["painted"]:
            state["painted"] = True
            state["size"] = capture_gui[0].shape[1::-1]
        if state["fullscreen"] and not state["stuck"]:
            state["size"] = (1920, 1080)
        return -1 if delay in (1, 50) else next(keys)

    monkeypatch.setattr(cv2, "setWindowProperty", fullscreen)
    monkeypatch.setattr(cv2, "waitKey", wait_key)
    monkeypatch.setattr(cv2, "getWindowImageRect", lambda name: (0, 0, *state["size"]))
    assert run_preview_loop(_session(tmp_path, pipeline), captures_only=True)
    assert not state["stuck"]
    assert capture_gui[-2].shape == (1080, 1920, 3)  # loading bars
    assert capture_gui[-1].shape == (1080, 1920, 3)  # saved photo
    cv2.resizeWindow.assert_any_call(cv2.namedWindow.call_args.args[0], 1920, 1080)


@pytest.mark.parametrize(
    "failure_at",
    ["namedWindow", "resizeWindow", "setWindowProperty", "imshow", "waitKey",
     "loading_image", "saved_image"],
)
def test_capture_display_gui_failure_cleans_up_and_allows_fallback(
    tmp_path, pipeline, monkeypatch, capture_gui, failure_at,
):
    import cv2

    cleanup = Mock()
    monkeypatch.setattr(cv2, "destroyAllWindows", cleanup)
    monkeypatch.setattr(cv2, "waitKey", lambda delay: 32)

    def boom(*args, **kwargs):
        raise cv2.error("no GUI")

    if failure_at in ("loading_image", "saved_image"):
        def show(name, frame):
            if len(capture_gui) == (1 if failure_at == "loading_image" else 2):
                boom()
            capture_gui.append(frame.copy())
        monkeypatch.setattr(cv2, "imshow", show)
    else:
        monkeypatch.setattr(cv2, failure_at, boom)
    assert run_preview_loop(_session(tmp_path, pipeline), captures_only=True) is False
    assert cleanup.call_count == (0 if failure_at == "namedWindow" else 1)
    if failure_at == "saved_image":
        assert len(list((tmp_path / "shots").rglob("*_kodachrome.jpg"))) == 1


@pytest.mark.parametrize("no_preview", [False, True])
@pytest.mark.parametrize("has_terminal", [False, True])
def test_main_show_captures_selects_snapshot_mode(
    tmp_path, monkeypatch, no_preview, has_terminal,
):
    from kodachrome.capture import app

    monkeypatch.setattr(app, "has_display", lambda: True)
    monkeypatch.setattr("sys.stdin.isatty", lambda: has_terminal)
    terminal = Mock()
    terminal.__enter__ = Mock(return_value=terminal)
    terminal.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(app, "TerminalKeys", lambda: terminal)
    window = Mock(return_value=True)
    monkeypatch.setattr(app, "run_preview_loop", window)
    args = ["--fake", "--show-captures", "--out", str(tmp_path)]
    if no_preview:
        args.append("--no-preview")
    assert main(args) == 0
    assert window.call_args.kwargs["captures_only"] is True
    if has_terminal:
        window.call_args.kwargs["read_key"]()
        terminal.read.assert_called_once_with(timeout=0)
        terminal.__exit__.assert_called_once()


@pytest.mark.parametrize("display_available", [False, True])
def test_main_show_captures_falls_back_to_terminal(tmp_path, monkeypatch, display_available):
    from kodachrome.capture import app

    monkeypatch.setattr(app, "has_display", lambda: display_available)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    terminal = Mock()
    terminal.__enter__ = Mock(return_value=terminal)
    terminal.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(app, "TerminalKeys", lambda: terminal)
    monkeypatch.setattr(app, "run_preview_loop", Mock(return_value=False))
    headless = Mock()
    monkeypatch.setattr(app, "run_headless_loop", headless)
    assert main(["--fake", "--no-preview", "--show-captures", "--out", str(tmp_path)]) == 0
    headless.assert_called_once()


def test_main_headless_without_tty_exits_2(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    code = main(["--fake", "--no-preview", "--out", str(tmp_path)])
    assert code == 2
    assert "terminal" in capsys.readouterr().err.lower()


def test_main_reports_bad_artifacts(tmp_path, capsys):
    code = main(
        ["--fake", "--no-preview", "--out", str(tmp_path), "--artifacts", str(tmp_path / "nope")]
    )
    assert code == 2
    assert "params.json" in capsys.readouterr().err
