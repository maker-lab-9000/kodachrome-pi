import numpy as np
import pytest

from kodachrome import color
from kodachrome.normalize import (
    Gains,
    NormalizeParams,
    apply_gains_float,
    compute_gains,
    gains_to_luts,
    normalize_float,
    normalize_u8,
)


def _gradient_image(h=48, w=64, cast=(1.0, 0.9, 0.75)):
    ramp = np.linspace(0.1, 0.9, w, dtype=np.float32)[None, :, None]
    rows = np.linspace(0.8, 1.2, h, dtype=np.float32)[:, None, None]
    return np.clip(ramp * rows * np.array(cast, dtype=np.float32), 0, 1).astype(np.float32)


def test_params_from_dict_ignores_unknown_and_roundtrips():
    p = NormalizeParams.from_dict({"exposure_target_median": 0.2, "bogus": 1})
    assert p.exposure_target_median == 0.2
    assert p.white_balance is True
    assert NormalizeParams.from_dict(p.to_dict()) == p


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"wb_gain_min": 2.0}, "wb_gain_min"),           # min above max
        ({"exposure_gain_min": 0.0}, "exposure_gain_min"),  # not positive
        ({"exposure_target_median": 1.5}, "exposure_target_median"),
        ({"stats_lum_min": 0.95}, "stats_lum_min"),      # min above max
        ({"wb_gain_max": float("nan")}, "wb_gain_max"),
    ],
)
def test_invalid_params_name_the_field(kwargs, field):
    with pytest.raises(ValueError, match=field):
        NormalizeParams(**kwargs)


def test_gains_combined_and_dict():
    g = Gains(
        wb=np.array([1.0, 1.1, 1.2], dtype=np.float32),
        exposure=2.0,
        clamped={"wb": False, "exposure": True},
    )
    assert np.allclose(g.combined, [2.0, 2.2, 2.4])
    assert g.to_dict() == {
        "wb": [1.0, 1.1, 1.2],
        "exposure": 2.0,
        "clamped": {"wb": False, "exposure": True},
    }


def test_grey_world_neutralises_a_mild_cast():
    # gains land near 0.83/1.04/1.33, inside the clamps, so the cast fully clears
    img = np.full((32, 32, 3), (0.5, 0.45, 0.4), dtype=np.float32)
    out, gains = normalize_float(img, NormalizeParams())
    assert np.allclose(out[..., 0], out[..., 1], atol=1 / 255)
    assert np.allclose(out[..., 1], out[..., 2], atol=1 / 255)
    assert np.median(color.luminance(color.srgb_to_linear(out))) == pytest.approx(0.18, abs=0.005)
    assert gains.wb[0] < 1.0 < gains.wb[2]
    assert gains.clamped == {"wb": False, "exposure": False}


def test_white_balance_can_be_disabled():
    img = np.full((8, 8, 3), (0.5, 0.4, 0.3), dtype=np.float32)
    gains = compute_gains(img, NormalizeParams(white_balance=False))
    assert np.array_equal(gains.wb, np.ones(3, dtype=np.float32))
    assert gains.exposure > 1.0


def test_gains_are_clamped_and_report_it():
    p = NormalizeParams()
    dark = np.full((8, 8, 3), 0.02, dtype=np.float32)
    g = compute_gains(dark, p)
    assert g.exposure == pytest.approx(p.exposure_gain_max)
    assert g.clamped["exposure"] is True
    bright = np.full((8, 8, 3), 0.95, dtype=np.float32)
    assert compute_gains(bright, p).exposure == pytest.approx(p.exposure_gain_min)
    red = np.full((8, 8, 3), (0.9, 0.1, 0.1), dtype=np.float32)
    g = compute_gains(red, p)
    assert g.wb[0] == pytest.approx(p.wb_gain_min)
    assert g.wb[1] == pytest.approx(p.wb_gain_max)
    assert g.clamped["wb"] is True


def test_normalising_twice_is_nearly_a_no_op_and_converges():
    """Normalisation is close to idempotent, but not exactly, and that is fine.

    The statistics mask is recomputed on the transformed image, so a second
    pass measures a slightly different set of pixels (2784 of 3072 becomes
    2752 for this fixture) whose median is 0.1835 rather than the 0.18
    target. The residual correction is real, not floating-point noise. What
    matters is that it is small and that repeated passes settle instead of
    drifting; nothing in the project normalises an image twice.
    """
    img = _gradient_image()
    once, _ = normalize_float(img, NormalizeParams())
    twice, gains2 = normalize_float(once, NormalizeParams())
    thrice, _ = normalize_float(twice, NormalizeParams())

    # Measured for this fixture: gains within 0.019 of 1, pixels within 2.05/255.
    assert np.allclose(gains2.wb, 1.0, atol=0.05)
    assert gains2.exposure == pytest.approx(1.0, abs=0.05)
    assert np.abs(twice - once).max() < 6 / 255

    # Measured: 2.05/255 then 0.53/255. Each pass corrects less than the last.
    assert np.abs(thrice - twice).max() < np.abs(twice - once).max()


def test_float_and_u8_paths_agree():
    img_u8 = (_gradient_image() * 255).round().astype(np.uint8)
    p = NormalizeParams()
    out_f, gains_f = normalize_float(img_u8.astype(np.float32) / 255.0, p)
    out_u8, gains_u8 = normalize_u8(img_u8, p)
    assert out_u8.dtype == np.uint8 and out_u8.shape == img_u8.shape
    assert np.allclose(gains_f.combined, gains_u8.combined, atol=1e-6)
    assert np.abs(out_u8.astype(int) - np.round(out_f * 255).astype(int)).max() <= 1


def test_u8_subsampling_branch_matches_the_reference():
    """At 1080p the Pi always subsamples for statistics, so prove that path.

    The 48x64 fixture above gives ``step == 1``, which quietly skips the
    strided branch entirely — the branch every real capture takes. A
    1920x1080 frame gives ``step == 3``, so the gains come from about a ninth
    of the pixels.

    Two claims, kept apart because they can fail independently: the table
    must apply exactly the gains it computed, and gains from a subsample must
    be close enough to full-image statistics that the result is
    indistinguishable at 8-bit precision. Measured: relative gain difference
    1.4e-4, table exact, fast path within one level.
    """
    img_u8 = (_gradient_image(1080, 1920) * 255).round().astype(np.uint8)
    p = NormalizeParams()
    out_u8, gains_u8 = normalize_u8(img_u8, p)
    reference, gains_full = normalize_float(img_u8.astype(np.float32) / 255.0, p)

    # The subsample really was taken: fewer pixels give slightly different gains.
    assert not np.array_equal(gains_u8.combined, gains_full.combined)
    assert np.allclose(gains_u8.combined, gains_full.combined, rtol=1e-3)

    # The table applies its own gains faithfully.
    own = apply_gains_float(img_u8.astype(np.float32) / 255.0, gains_u8)
    assert np.abs(out_u8.astype(int) - np.round(own * 255).astype(int)).max() <= 1

    # And the fast path is indistinguishable from the full-statistics path.
    assert np.abs(out_u8.astype(int) - np.round(reference * 255).astype(int)).max() <= 1


def test_the_float_path_refuses_uint8_input():
    """uint8 would be clipped to 1.0 by srgb_to_linear and silently destroyed."""
    u8 = np.full((16, 16, 3), 128, dtype=np.uint8)
    with pytest.raises(ValueError, match="float image"):
        normalize_float(u8, NormalizeParams())
    with pytest.raises(ValueError, match="float image"):
        compute_gains(u8, NormalizeParams())


def test_luts_are_monotone():
    luts = gains_to_luts(
        Gains(wb=np.array([0.8, 1.0, 1.5], dtype=np.float32), exposure=1.3,
              clamped={"wb": False, "exposure": False})
    )
    assert luts.shape == (3, 256) and luts.dtype == np.uint8
    assert np.all(np.diff(luts.astype(int), axis=1) >= 0)
