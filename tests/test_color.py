import numpy as np
import pytest

from kodachrome import color


def test_srgb_linear_roundtrip():
    x = np.random.default_rng(0).random((100, 3), dtype=np.float32)
    back = color.linear_to_srgb(color.srgb_to_linear(x))
    assert back.dtype == np.float32
    assert np.allclose(back, x, atol=1e-4)


def test_srgb_to_linear_known_points():
    assert color.srgb_to_linear(np.array([0.0, 1.0]))[1] == pytest.approx(1.0)
    # 18% grey card is about sRGB 0.461
    assert color.srgb_to_linear(np.array([0.4613]))[0] == pytest.approx(0.18, abs=1e-3)


@pytest.mark.parametrize(
    "rgb, expected",
    [
        ((1.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
        ((1.0, 0.0, 0.0), (0.627955, 0.224863, 0.125846)),
        ((0.0, 1.0, 0.0), (0.866440, -0.233888, 0.179498)),
        ((0.0, 0.0, 1.0), (0.452014, -0.032457, -0.311528)),
    ],
)
def test_oklab_reference_values(rgb, expected):
    # Reference values from Björn Ottosson's Oklab article (linear sRGB inputs).
    lab = color.linear_to_oklab(np.array(rgb, dtype=np.float32))
    assert np.allclose(lab, expected, atol=1e-3)


def test_oklab_roundtrip():
    rgb = np.random.default_rng(1).random((500, 3), dtype=np.float32)
    back = color.oklab_to_srgb(color.srgb_to_oklab(rgb))
    assert np.allclose(back, rgb, atol=1e-4)


def test_lch_roundtrip_and_hue_range():
    lab = color.srgb_to_oklab(np.random.default_rng(2).random((200, 3), dtype=np.float32))
    lch = color.oklab_to_lch(lab)
    assert lch[..., 1].min() >= 0
    assert np.all(np.abs(lch[..., 2]) <= np.pi + 1e-6)
    assert np.allclose(color.lch_to_oklab(lch), lab, atol=1e-5)


def test_luminance_weights_sum_to_one():
    assert color.LUMA_709.sum() == pytest.approx(1.0, abs=1e-4)
    assert color.luminance(np.ones(3, dtype=np.float32)) == pytest.approx(1.0, abs=1e-4)
