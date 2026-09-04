import numpy as np
import pytest

from kodachrome.color import lch_to_oklab, srgb_to_oklab
from kodachrome.lut import LUT3D
from kodachrome.train.dataset import PixelPool
from kodachrome.train.evaluate import (
    Evaluator,
    channels_are_monotone,
    check_gates,
    clipped_volume_fraction,
    evaluate,
    grey_axis_is_monotone,
    hue_bin_shifts,
    hue_hist_residual,
    neutral_axis_max_chroma,
    swd_seed_spread,
)


def _pool(seed, n=4000, scale=1.0, offset=0.0):
    rng = np.random.default_rng(seed)
    srgb = np.clip(rng.random((n, 3), dtype=np.float32) * scale + offset, 0, 1).astype(np.float32)
    return PixelPool(srgb=srgb, n_images=5)


def _darkening_lut(n=9, gamma=1.5):
    return LUT3D(LUT3D.identity(n).table**gamma)


def test_identity_lut_gives_identical_before_and_after():
    """The whole point of a paired evaluator: no LUT change, no metric change."""
    src, tgt = _pool(0), _pool(1, scale=0.8)
    weights = np.ones(len(tgt.srgb))
    ev = Evaluator.build(src, tgt, weights, seed=0)
    before = ev.distance(src.lab)
    after = ev.distance(srgb_to_oklab(LUT3D.identity(33).apply_numpy(src.srgb)))
    assert after == pytest.approx(before, abs=1e-6)


def test_pairing_holds_when_the_pool_is_actually_subsampled():
    """The production case, which no other test here reaches.

    Every other pool in this file is smaller than ``max_points``, so
    ``rng.choice(n, n, replace=False)`` returns a full permutation and
    ``distance`` — which sorts the projections before comparing — gives the
    same answer whether or not the indices are applied at all. Measured:
    identical to ten decimal places. So an implementation that ignored
    ``src_idx`` entirely would pass every other assertion in this file.

    A real run caps pools at 400,000 pixels against a 100,000 sample, so the
    indices genuinely select a subset. This forces that, and then re-checks
    the property the whole evaluator exists for.
    """
    src, tgt = _pool(11), _pool(12, scale=0.8)
    weights = np.ones(len(tgt.srgb))
    ev = Evaluator.build(src, tgt, weights, seed=0, max_points=500)

    assert len(ev.src_idx) == 500, "the sample must be a subset, not the whole pool"
    assert ev.tgt_points.shape[0] == 500
    assert ev.distance(src.lab) != Evaluator.build(src, tgt, weights, seed=0).distance(src.lab)

    same = Evaluator.build(src, tgt, weights, seed=0, max_points=500)
    assert np.array_equal(same.src_idx, ev.src_idx)
    other = Evaluator.build(src, tgt, weights, seed=1, max_points=500)
    assert not np.array_equal(other.src_idx, ev.src_idx)

    # And the property the evaluator exists for still holds on a real subset.
    before = ev.distance(src.lab)
    after = ev.distance(srgb_to_oklab(LUT3D.identity(33).apply_numpy(src.srgb)))
    assert after == pytest.approx(before, abs=1e-6)


def test_evaluator_is_reusable_and_deterministic():
    src, tgt = _pool(2), _pool(3)
    ev = Evaluator.build(src, tgt, np.ones(len(tgt.srgb)), seed=7)
    assert ev.distance(src.lab) == ev.distance(src.lab)
    again = Evaluator.build(src, tgt, np.ones(len(tgt.srgb)), seed=7)
    assert ev.distance(src.lab) == pytest.approx(again.distance(src.lab), abs=1e-12)


def test_a_real_improvement_beats_the_seed_spread():
    """A LUT that genuinely moves the source toward the target must clear the noise floor."""
    rng = np.random.default_rng(4)
    src = PixelPool(rng.random((6000, 3), dtype=np.float32).astype(np.float32), 5)
    noisy = np.clip(src.srgb**1.5 + rng.normal(0, 0.01, src.srgb.shape), 0, 1)
    tgt = PixelPool(noisy.astype(np.float32), 5)
    weights = np.ones(len(tgt.srgb))
    ev = Evaluator.build(src, tgt, weights, seed=0)
    before = ev.distance(src.lab)
    lut = _darkening_lut(17, gamma=1.5)
    after = ev.distance(srgb_to_oklab(lut.apply_numpy(src.srgb)))
    _mean, spread = swd_seed_spread(src, tgt, weights, lut)
    assert after < before
    assert before - after > 3 * spread


def test_seed_spread_is_small_and_positive():
    src, tgt = _pool(5), _pool(6, scale=0.7)
    mean, spread = swd_seed_spread(src, tgt, np.ones(len(tgt.srgb)), LUT3D.identity(9))
    assert mean > 0 and spread >= 0
    assert spread < mean


def test_grey_axis_monotonicity():
    assert grey_axis_is_monotone(LUT3D.identity(9))
    t = LUT3D.identity(9).table.copy()
    t[4, 4, 4] = 0.05
    assert not grey_axis_is_monotone(LUT3D(t))


def test_channel_monotonicity_catches_a_per_channel_inversion():
    assert channels_are_monotone(LUT3D.identity(9))
    t = LUT3D.identity(9).table.copy()
    t[5, :, :, 0] = 0.1  # red output dips as red input rises
    assert not channels_are_monotone(LUT3D(t))


def test_neutral_axis_chroma_catches_a_tinting_lut():
    assert neutral_axis_max_chroma(LUT3D.identity(33)) < 0.01
    t = LUT3D.identity(33).table.copy()
    for i in range(33):
        t[i, i, i, 2] = min(1.0, t[i, i, i, 2] + 0.25)  # push greys blue
    assert neutral_axis_max_chroma(LUT3D(t)) > 0.05


@pytest.mark.parametrize("size", [9, 17, 33])
def test_identity_and_tone_curves_clip_nothing(size):
    """Interior nodes only, so grid size must not change the answer."""
    assert clipped_volume_fraction(LUT3D.identity(size)) == 0.0
    assert clipped_volume_fraction(LUT3D(LUT3D.identity(size).table**1.5)) == 0.0


def test_clipped_volume_fraction_catches_a_crushing_lut():
    crushed = LUT3D(np.clip(LUT3D.identity(9).table * 3.0, 0, 1))
    assert clipped_volume_fraction(crushed) > 0.9


def test_hue_bin_shifts_report_darkening():
    rng = np.random.default_rng(0)
    src = (rng.random((5000, 3), dtype=np.float32) * 0.6 + 0.2).astype(np.float32)
    shifts = hue_bin_shifts(src, _darkening_lut(), n_bins=12)
    assert len(shifts) == 13
    populated = [s for s in shifts if s["count"] > 0]
    assert populated and all(s["delta_L"] < 0 for s in populated)
    assert {"bin", "hue_deg", "count", "delta_L", "chroma_ratio", "delta_hue_deg"} <= set(shifts[0])


def test_hue_hist_residual_is_zero_when_reweighting_is_exact():
    from kodachrome.train.transport import hue_weights

    rng = np.random.default_rng(8)
    src_lab = lch_to_oklab(
        np.stack(
            [rng.uniform(0.3, 0.8, 8000), np.full(8000, 0.12), rng.uniform(0, 2 * np.pi, 8000)], 1
        )
    )
    # Uneven around the whole circle, not confined to half of it. Reweighting
    # can only scale samples that exist, so a hue with no samples at all leaves
    # a residual pinned at 1/n_bins — measured 0.0459 against a 1/24 = 0.0417
    # floor. That would make the test assert something unachievable rather than
    # something false.
    tgt_lab = lch_to_oklab(
        np.concatenate(
            [
                np.stack(
                    [rng.uniform(0.3, 0.8, 5600), np.full(5600, 0.12), rng.uniform(0, np.pi, 5600)],
                    1,
                ),
                np.stack(
                    [
                        rng.uniform(0.3, 0.8, 2400),
                        np.full(2400, 0.12),
                        rng.uniform(np.pi, 2 * np.pi, 2400),
                    ],
                    1,
                ),
            ]
        )
    )
    w = hue_weights(src_lab, tgt_lab, 24)
    assert hue_hist_residual(src_lab, tgt_lab, w, 24, 0.03) < 0.02


def test_evaluate_returns_the_documented_metric_block():
    src, tgt = _pool(9), _pool(10, scale=0.8)
    weights = np.ones(len(tgt.srgb))
    lut = _darkening_lut(9)
    partners = srgb_to_oklab(lut.apply_numpy(src.srgb))
    metrics = evaluate(
        lut=lut,
        val_src=src,
        val_tgt=tgt,
        val_weights=weights,
        train_src=src,
        train_tgt=tgt,
        train_weights=weights,
        transported_lab=partners,
        n_bins=12,
        chroma_floor=0.03,
        seed=0,
    )
    for key in (
        "swd_before", "swd_after", "swd_identity", "swd_seed_spread",
        "transport_gamut_clip_deltaE", "lut_fit_rms_deltaE", "grey_axis_monotone",
        "channel_monotone", "neutral_axis_max_chroma", "clipped_volume_fraction",
        "hue_bins", "train_swd_before", "train_swd_after",
    ):
        assert key in metrics, key
    assert metrics["swd_identity"] == pytest.approx(metrics["swd_before"], abs=1e-6)


def test_gates_pass_and_fail_explicitly():
    good = {
        "swd_before": 0.10, "swd_after": 0.04, "swd_seed_spread": 0.001,
        "grey_axis_monotone": True, "channel_monotone": True,
        "neutral_axis_max_chroma": 0.005, "clipped_volume_fraction": 0.01,
    }
    assert all(g.passed for g in check_gates(good))

    marginal = {**good, "swd_after": 0.0999}          # improvement inside the noise floor
    failed = [g.name for g in check_gates(marginal) if not g.passed]
    assert "improvement_exceeds_noise" in failed

    tinted = {**good, "neutral_axis_max_chroma": 0.09}
    assert "neutral_axis_chroma" in [g.name for g in check_gates(tinted) if not g.passed]
