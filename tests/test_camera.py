import io

import cv2
import numpy as np
import pytest
from PIL import Image

from kodachrome.capture import camera as camera_module
from kodachrome.capture.camera import (
    CameraError,
    FakeCamera,
    Frame,
    StreamInfo,
    V4L2Camera,
    is_valid_jpeg,
    parse_device,
    synthetic_frame,
)


def _jpeg_bytes(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=95)
    return buf.getvalue()


def test_synthetic_frame_shape_and_colour():
    f = synthetic_frame(90, 160)
    assert f.shape == (90, 160, 3) and f.dtype == np.uint8
    assert f[..., 0].mean() != f[..., 2].mean()


def test_is_valid_jpeg_markers():
    good = _jpeg_bytes(synthetic_frame(16, 16))
    assert is_valid_jpeg(good)
    assert not is_valid_jpeg(good[:-2])        # truncated: no EOI
    assert not is_valid_jpeg(b"\x00\x01" + good[2:])  # no SOI
    assert not is_valid_jpeg(b"")
    assert not is_valid_jpeg(b"\xff\xd8\xff\xd9")     # too short to be a frame


def test_fake_camera_defaults_to_decoded_mode():
    cam = FakeCamera()
    frame = cam.read()
    assert isinstance(frame, Frame)
    assert frame.rgb.shape == (1080, 1920, 3) and frame.rgb.dtype == np.uint8
    assert frame.jpeg is None and frame.source == "decoded"
    assert isinstance(cam.stream_info, StreamInfo)
    cam.close()


def test_fake_camera_raw_mode_returns_bytes_that_decode_to_the_frame():
    rgb = synthetic_frame(48, 64)
    data = _jpeg_bytes(rgb)
    cam = FakeCamera(jpeg_bytes=[data], source="raw-mjpeg")
    frame = cam.read()
    assert frame.source == "raw-mjpeg"
    assert frame.jpeg == data
    decoded = np.asarray(Image.open(io.BytesIO(frame.jpeg)).convert("RGB"))
    assert np.array_equal(frame.rgb, decoded), "rgb must be the decode of the same buffer"


def test_fake_camera_cycles_and_copies():
    frames = [np.zeros((4, 4, 3), np.uint8), np.ones((4, 4, 3), np.uint8)]
    cam = FakeCamera(frames)
    assert cam.read().rgb.max() == 0
    assert cam.read().rgb.max() == 1
    assert cam.read().rgb.max() == 0
    cam.read().rgb[0, 0, 0] = 99
    assert frames[0].max() == 0


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        (3, 3),
        ("3", 3),
        ("/dev/video7", 7),
        ("/dev/v4l/by-id/usb-Innomaker-video-index0", "/dev/v4l/by-id/usb-Innomaker-video-index0"),
    ],
)
def test_parse_device(value, expected):
    assert parse_device(value) == expected


def test_parse_device_rejects_garbage():
    with pytest.raises(CameraError):
        parse_device("camera")


class FakeCapture:
    """Stands in for ``cv2.VideoCapture`` so the real V4L2Camera can be driven.

    Without this, none of the raw-mode, fallback, negotiation or retry logic is
    executed by any test: a build where raw mode silently never engaged would
    pass the whole suite, and that logic is the project's headline promise.

    Two things to know when writing tests against it. Constructing a
    ``V4L2Camera`` consumes one buffer, because ``_enable_raw_mode`` reads a
    frame to verify the mode really works — so a queue must include that
    verification frame before the ones a test intends ``read()`` to return.
    And format properties are reported, not stored (see ``set``).
    """

    def __init__(
        self,
        *,
        buffers=None,
        decoded=None,
        set_convert_fails=False,
        set_raises=False,
        mutate_then_raise=False,
        props=None,
    ):
        self.buffers = list(buffers or [])
        self.decoded = decoded if decoded is not None else np.zeros((1080, 1920, 3), np.uint8)
        self.set_convert_fails = set_convert_fails
        self.set_raises = set_raises
        self.mutate_then_raise = mutate_then_raise
        self.props = props or {
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"MJPG")),
        }
        self.convert_rgb = 1
        self.released = False
        self.requested = []

    def isOpened(self):
        return True

    def set(self, prop, val):
        """Accept a request but do not pretend it was honoured.

        A real camera reports the format it actually negotiated, which is the
        entire reason ``_negotiated`` reads the properties back. Storing the
        requested value here would make every negotiation check tautological:
        the code would read back exactly what it just wrote and never warn.
        So format writes are recorded and ignored; only CONVERT_RGB takes
        effect, because that one genuinely changes what ``read`` returns.
        """
        self.requested.append((prop, val))
        if prop == cv2.CAP_PROP_CONVERT_RGB:
            if self.mutate_then_raise:
                self.convert_rgb = val
                raise cv2.error("device rejected the mode after applying it")
            if self.set_raises:
                raise cv2.error("simulated")
            if self.set_convert_fails:
                return False
            self.convert_rgb = val
        return True

    def get(self, prop):
        return self.props.get(prop, 0)

    def read(self):
        if self.convert_rgb == 0:
            if not self.buffers:
                return False, None
            return True, np.frombuffer(self.buffers.pop(0), np.uint8)
        return True, self.decoded

    def grab(self):
        return True

    def release(self):
        self.released = True


@pytest.fixture
def fake_capture(monkeypatch):
    """Install a FakeCapture in place of cv2.VideoCapture and hand it back."""
    holder = {}

    def install(**kwargs):
        cap = FakeCapture(**kwargs)
        holder["cap"] = cap
        monkeypatch.setattr(camera_module.cv2, "VideoCapture", lambda *a, **k: cap)
        return cap

    return install


def test_raw_mode_engages_and_preserves_the_camera_bytes(fake_capture, capsys):
    rgb = synthetic_frame(48, 64)
    data = _jpeg_bytes(rgb)
    # One buffer is consumed by _enable_raw_mode's verification read.
    cap = fake_capture(buffers=[data] * 4)
    cam = V4L2Camera(device=0, warmup_frames=0)
    assert cam.stream_info.raw_mjpeg is True
    assert cap.convert_rgb == 0

    frame = cam.read()
    assert frame.source == "raw-mjpeg"
    assert frame.jpeg == data
    decoded = np.asarray(Image.open(io.BytesIO(frame.jpeg)).convert("RGB"))
    assert np.array_equal(frame.rgb, decoded)


def test_an_invalid_buffer_falls_back_once_and_stays_fallen_back(fake_capture, capsys):
    rgb = synthetic_frame(48, 64)
    data = _jpeg_bytes(rgb)
    # First buffer is eaten by the constructor's verification read, second is
    # the good frame the first read() returns, third is the truncated one.
    cap = fake_capture(
        buffers=[data, data, data[:-2]], decoded=np.zeros((48, 64, 3), np.uint8)
    )
    cam = V4L2Camera(device=0, warmup_frames=0)
    assert cam.read().source == "raw-mjpeg"

    assert cam.read().source == "decoded"
    assert cam.stream_info.raw_mjpeg is False
    assert cap.convert_rgb == 1
    warnings = capsys.readouterr().out.count("falling back to decoded frames")
    assert warnings == 1, "the fallback must announce itself once, not per frame"

    assert cam.read().source == "decoded"
    assert capsys.readouterr().out.count("falling back to decoded frames") == 0


@pytest.mark.parametrize(
    "kwargs",
    [{"set_convert_fails": True}, {"set_raises": True}, {"mutate_then_raise": True}],
    ids=["set-returns-false", "set-raises", "set-mutates-then-raises"],
)
def test_every_raw_mode_failure_leaves_the_device_in_decoded_mode(fake_capture, kwargs):
    """The half-state is silently destructive, so no failure path may leave it."""
    cap = fake_capture(decoded=np.zeros((1080, 1920, 3), np.uint8), **kwargs)
    cam = V4L2Camera(device=0, warmup_frames=0)
    assert cam.stream_info.raw_mjpeg is False
    assert cap.convert_rgb == 1, "CONVERT_RGB left at 0 while the object thinks it is decoding"
    assert cam.read().source == "decoded"


def test_negotiation_mismatch_warns_but_does_not_abort(fake_capture, capsys):
    cap = fake_capture(
        decoded=np.zeros((720, 1280, 3), np.uint8),
        props={
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
            cv2.CAP_PROP_FPS: 15.0,
            cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"YUYV")),
        },
        set_convert_fails=True,
    )
    cam = V4L2Camera(device=0, warmup_frames=0)
    out = capsys.readouterr().out
    assert "negotiated 1280x720" in out
    assert "YUYV" in out
    assert cam.stream_info.width == 1280 and cam.stream_info.fps == 15.0
    assert cam.read().rgb.shape == (720, 1280, 3)
    assert cap is not None


def test_read_raises_after_three_failed_attempts(fake_capture):
    cap = fake_capture(buffers=[])
    cam = V4L2Camera(device=0, warmup_frames=0)
    cap.convert_rgb = 0
    cap.buffers = []
    with pytest.raises(CameraError, match="3 attempts"):
        cam.read()


def test_close_releases_the_device(fake_capture):
    cap = fake_capture(set_convert_fails=True)
    cam = V4L2Camera(device=0, warmup_frames=0)
    cam.close()
    assert cap.released is True


def test_v4l2_camera_reports_missing_device():
    with pytest.raises(CameraError, match="video"):
        V4L2Camera(device=99, warmup_frames=0)


def test_stream_info_to_dict():
    info = StreamInfo(width=1920, height=1080, fps=30.0, fourcc="MJPG", raw_mjpeg=True)
    assert info.to_dict() == {
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "fourcc": "MJPG",
        "raw_mjpeg": True,
    }
