import io
import json
from datetime import datetime

import numpy as np
import pytest
from PIL import Image

from kodachrome.artifacts import Artifacts, write_artifact
from kodachrome.capture.app import CaptureSession, main, run_headless_loop, run_preview_loop
from kodachrome.capture.camera import CameraError, FakeCamera, Frame, StreamInfo, synthetic_frame
from kodachrome.grain import GrainParams
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


def test_preview_loop_falls_back_when_imshow_fails(tmp_path, pipeline, monkeypatch):
    import cv2

    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: (_ for _ in ()).throw(cv2.error("no GUI")))
    assert run_preview_loop(_session(tmp_path, pipeline)) is False


def test_preview_loop_survives_a_frame_error(tmp_path, pipeline, monkeypatch):
    import cv2

    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    keys = iter([ord("q")])
    monkeypatch.setattr(cv2, "waitKey", lambda _n: next(keys))

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
