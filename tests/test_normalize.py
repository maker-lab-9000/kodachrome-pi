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
    target = NormalizeParams().exposure_target_median
    assert np.median(color.luminance(color.srgb_to_linear(out))) == pytest.approx(target, abs=0.005)
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
    pass measures a slightly different set of pixels (2774 of 3072 becomes
    2746 for this fixture) whose median is 0.228 rather than the 0.25
    target. The residual correction is real, not floating-point noise, and it
    grew when the target moved from 0.18 to 0.25: a brighter target pushes
    more of this ramp past ``stats_lum_max`` (3.1% -> 5.0% of pixels). What
    matters is that it is bounded and that repeated passes settle instead of
    drifting; nothing in the project normalises an image twice.
    """
    img = _gradient_image()
    once, _ = normalize_float(img, NormalizeParams())
    twice, gains2 = normalize_float(once, NormalizeParams())
    thrice, _ = normalize_float(twice, NormalizeParams())

    # Measured for this fixture at target 0.25: wb within 0.001 of 1, exposure
    # 1.097, pixels within 10.2/255.
    assert np.allclose(gains2.wb, 1.0, atol=0.05)
    assert gains2.exposure == pytest.approx(1.0, abs=0.15)
    assert np.abs(twice - once).max() < 12 / 255

    # Measured: 10.2/255 then 5.4/255. Each pass corrects less than the last.
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


# --- levels normalisation: the training-side step for scanned targets -------


def _low_contrast_grey(lo=0.05, hi=0.60, n=64):
    """A neutral ramp that never reaches black or white, like an archival scan."""
    lin = np.linspace(lo, hi, n * n, dtype=np.float32).reshape(n, n)
    rgb = color.linear_to_srgb(np.stack([lin, lin, lin], axis=-1))
    return rgb.astype(np.float32)


def test_levels_pins_black_white_and_median_and_keeps_neutrals_neutral():
    """The scans' whites sit at 0.72 linear luminance (measured, 150 scans).

    Learning from them unstretched taught the LUT to push white to 0.85.
    Levels puts p0.5 at black, p99.5 at white, then a gamma puts the median
    on the exposure target without moving either end.
    """
    p = NormalizeParams(white_balance=False, levels=True)
    out, gains = normalize_float(_low_contrast_grey(), p)
    lum = color.luminance(color.srgb_to_linear(out))
    lo, hi = np.percentile(lum, [p.levels_low_pct, p.levels_high_pct])
    assert lo == pytest.approx(0.0, abs=0.01)
    assert hi == pytest.approx(1.0, abs=0.01)
    # The gamma targets the median on the statistics mask, exactly as the
    # source-side exposure gain does, so source and target agree on what
    # "median" means. Recompute it from the recorded black point and stretch
    # and check the exponent lands it on target. The whole-image median sits
    # a little above (0.29 here) because the mask drops the ramp's ends.
    lv = gains.levels
    stretched = np.clip((color.luminance(color.srgb_to_linear(_low_contrast_grey())) - lv["low"])
                        * lv["stretch"], 0.0, 1.0)
    m = (stretched >= p.stats_lum_min) & (stretched <= p.stats_lum_max)
    landed = np.median(stretched[m]) ** lv["gamma"]
    assert landed == pytest.approx(p.exposure_target_median, abs=0.005)
    assert np.median(lum) == pytest.approx(p.exposure_target_median, abs=0.05)
    assert np.allclose(out[..., 0], out[..., 1], atol=1 / 255)
    assert np.allclose(out[..., 1], out[..., 2], atol=1 / 255)
    assert gains.levels is not None
    assert gains.exposure == 1.0 and np.allclose(gains.wb, 1.0)
    assert gains.clamped == {"wb": False, "exposure": False, "levels": False}


def test_levels_clamps_are_recorded_not_silent():
    p = NormalizeParams(white_balance=False, levels=True)
    # Nearly flat: the raw stretch would be ~20x, far past the 4x ceiling.
    flat = _low_contrast_grey(0.40, 0.45)
    _, g = normalize_float(flat, p)
    assert g.levels["stretch"] == pytest.approx(p.levels_max_stretch)
    assert g.clamped["levels"] is True
    # Mostly black with a small bright patch: median after stretch ~0.02, so
    # the gamma that would reach 0.25 is ~0.35, below the 0.5 floor.
    dark = _low_contrast_grey(0.0, 0.02)
    dark[:4, :4] = 1.0
    _, g = normalize_float(dark, p)
    assert g.levels["gamma"] == pytest.approx(p.levels_gamma_min)
    assert g.clamped["levels"] is True


def test_levels_off_is_exactly_the_old_path():
    img = _gradient_image()
    out, gains = normalize_float(img, NormalizeParams())
    ref = apply_gains_float(img, compute_gains(img, NormalizeParams()))
    assert np.array_equal(out, ref)
    assert gains.levels is None
    assert "levels" not in gains.clamped


def test_levels_with_white_balance_neutralises_a_cast_and_pins_white():
    """The camera path: grey-world first, then the stretch and gamma."""
    lin = np.linspace(0.05, 0.60, 48 * 48, dtype=np.float32).reshape(48, 48)
    cast = np.stack([lin * 1.15, lin, lin * 0.85], axis=-1)              # warm, flat
    srgb = color.linear_to_srgb(cast).astype(np.float32)
    out, gains = normalize_float(srgb, NormalizeParams(levels=True))
    assert gains.wb[0] < 1.0 < gains.wb[2] and gains.levels is not None
    assert np.allclose(out[..., 0], out[..., 1], atol=1.5 / 255)
    assert np.allclose(out[..., 1], out[..., 2], atol=1.5 / 255)
    lum = color.luminance(color.srgb_to_linear(out))
    assert np.percentile(lum, 99.5) == pytest.approx(1.0, abs=0.01)
    assert np.percentile(lum, 0.5) == pytest.approx(0.0, abs=0.01)


def test_levels_params_validate_and_round_trip():
    with pytest.raises(ValueError, match="levels_gamma"):
        NormalizeParams(
            white_balance=False, levels=True, levels_gamma_min=2.0, levels_gamma_max=1.0
        )
    with pytest.raises(ValueError, match="levels_low_pct"):
        NormalizeParams(
            white_balance=False, levels=True, levels_low_pct=99.0, levels_high_pct=1.0
        )
    p = NormalizeParams(white_balance=False, levels=True, levels_max_stretch=3.0)
    assert NormalizeParams.from_dict(p.to_dict()) == p
    assert "levels" in Gains(wb=np.ones(3), exposure=1.0, levels={"gamma": 0.7}).to_dict()


def test_float_and_u8_paths_agree_under_levels():
    """The Pi bakes WB, stretch and gamma into three tables; they must match the
    float reference the trainer used, or the LUT sees inputs it was not fit on."""
    img_u8 = (_gradient_image() * 255).round().astype(np.uint8)
    p = NormalizeParams(levels=True)
    out_f, gains_f = normalize_float(img_u8.astype(np.float32) / 255.0, p)
    out_u8, gains_u8 = normalize_u8(img_u8, p)
    assert gains_u8.levels is not None
    for key in ("low", "high", "stretch", "gamma"):
        assert gains_u8.levels[key] == pytest.approx(gains_f.levels[key], abs=1e-6), key
    assert np.abs(out_u8.astype(int) - np.round(out_f * 255).astype(int)).max() <= 1


def test_u8_subsampling_under_levels_stays_close_to_the_reference():
    """At 1080p statistics come from a strided subsample; percentiles are noisier
    than a median, so allow a little more than the exact-path test does."""
    rng = np.random.default_rng(0)
    base = np.linspace(0.1, 0.7, 1920, dtype=np.float32)[None, :].repeat(1080, 0)
    noise = rng.normal(0, 0.01, (1080, 1920, 3))
    img = np.clip(np.stack([base * 1.05, base, base * 0.95], -1) + noise, 0, 1)
    img_u8 = (img * 255).round().astype(np.uint8)
    p = NormalizeParams(levels=True)
    out_f, gf = normalize_float(img_u8.astype(np.float32) / 255.0, p)
    out_u8, gu = normalize_u8(img_u8, p)
    assert gu.levels["gamma"] == pytest.approx(gf.levels["gamma"], abs=0.02)
    assert np.abs(out_u8.astype(int) - np.round(out_f * 255).astype(int)).max() <= 3
def test_levels_reports_how_coloured_the_scan_black_is_without_correcting_it():
    """A per-channel black point was tried and made the fit worse than identity
    (see normalize.py). The spread is still reported so a coloured base shows
    up in the corpus report; the darkest tones keep their colour."""
    p = NormalizeParams(white_balance=False, levels=True)
    ramp = np.linspace(0.0, 0.6, 64 * 64, dtype=np.float32).reshape(64, 64)
    lin = np.stack([ramp + 0.010, ramp + 0.012, ramp + 0.030], axis=-1)   # blue-black base
    out, gains = normalize_float(color.linear_to_srgb(lin).astype(np.float32), p)
    assert gains.levels["black_rgb_spread"] == pytest.approx(0.020, abs=0.003)
    o = color.srgb_to_linear(out).reshape(-1, 3)
    dark = o[np.argsort(o @ np.array([0.2126, 0.7152, 0.0722]))[200:260]]
    assert dark[:, 2].mean() > dark[:, 0].mean(), "the shared black point must keep the base colour"


def test_levels_target_median_can_differ_from_the_exposure_target():
    """Slides are dense; gamma-lifting them to the camera's median taught the LUT
    to lift shadows. The target median is therefore its own parameter."""
    # This ramp's stretched median is ~0.5, so anything below ~0.25 needs a
    # gamma past the 2.0 clamp and would test the clamp, not the target. 0.35
    # exercises the mechanism: a target that is not the exposure target.
    p = NormalizeParams(white_balance=False, levels=True, levels_target_median=0.35)
    out, gains = normalize_float(_low_contrast_grey(), p)
    lv = gains.levels
    assert gains.clamped["levels"] is False
    stretched = np.clip((color.luminance(color.srgb_to_linear(_low_contrast_grey())) - lv["low"])
                        * lv["stretch"], 0.0, 1.0)
    m = (stretched >= p.stats_lum_min) & (stretched <= p.stats_lum_max)
    assert np.median(stretched[m]) ** lv["gamma"] == pytest.approx(0.35, abs=0.005)
    assert NormalizeParams.from_dict(p.to_dict()) == p
    with pytest.raises(ValueError, match="levels_target_median"):
        NormalizeParams(white_balance=False, levels=True, levels_target_median=1.5)
