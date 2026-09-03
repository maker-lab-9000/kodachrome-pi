import io

import numpy as np
import pytest
from PIL import Image

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
