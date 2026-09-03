import numpy as np
import pytest

from kodachrome.grain import GrainParams, add_grain


def test_disabled_is_identity():
    img = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    out = add_grain(img, GrainParams(enabled=False))
    assert np.array_equal(out, img)
    assert out is not img


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"strength": -0.1}, "strength"),
        ({"blur_sigma": -1.0}, "blur_sigma"),
        ({"strength": float("inf")}, "strength"),
    ],
)
def test_invalid_params_name_the_field(kwargs, field):
    with pytest.raises(ValueError, match=field):
        GrainParams(**kwargs)


def test_preserves_mean_luminance_and_adds_no_colour_bias():
    img = np.full((256, 256, 3), 128, dtype=np.uint8)
    out = add_grain(img, GrainParams(strength=0.05), rng=np.random.default_rng(1))
    assert out.dtype == np.uint8 and out.shape == img.shape
    assert np.allclose(out.reshape(-1, 3).mean(axis=0), 128, atol=0.5)
    assert out.std() > 5


def test_strength_scales_noise():
    img = np.full((256, 256, 3), 128, dtype=np.uint8)
    lo = add_grain(img, GrainParams(strength=0.02), rng=np.random.default_rng(2))
    hi = add_grain(img, GrainParams(strength=0.06), rng=np.random.default_rng(2))
    assert lo[..., 1].std() == pytest.approx(0.02 * 255, rel=0.25)
    assert hi[..., 1].std() == pytest.approx(0.06 * 255, rel=0.25)


def test_black_and_white_are_untouched():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[16:] = 255
    out = add_grain(img, GrainParams(strength=0.1), rng=np.random.default_rng(3))
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 1


def test_seeded_is_reproducible():
    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    a = add_grain(img, GrainParams(), rng=np.random.default_rng(7))
    b = add_grain(img, GrainParams(), rng=np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_params_dict_roundtrip():
    p = GrainParams(strength=0.03, blur_sigma=0.5, enabled=False)
    assert GrainParams.from_dict({**p.to_dict(), "extra": 1}) == p
