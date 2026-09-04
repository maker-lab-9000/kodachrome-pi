import numpy as np
import pytest

from kodachrome.color import lch_to_oklab
from kodachrome.train.transport import (
    hue_bin_index,
    hue_histogram,
    hue_weights,
    iterative_distribution_transfer,
    random_rotation,
    sliced_wasserstein,
    weighted_quantile_map,
)


def _cloud(rng, n, hue_lo=0.0, hue_hi=2 * np.pi, chroma=0.12):
    hue = rng.uniform(hue_lo, hue_hi, n)
    lum = rng.uniform(0.3, 0.8, n)
    return lch_to_oklab(np.stack([lum, np.full(n, chroma), hue], axis=1))


def test_hue_bins_cover_circle_and_achromatic():
    rng = np.random.default_rng(0)
    lab = _cloud(rng, 5000)
    idx = hue_bin_index(lab, 24, 0.03)
    assert idx.min() == 0 and idx.max() == 23
    grey = np.array([[0.5, 0.0, 0.0], [0.5, 0.01, -0.01]])
    assert np.all(hue_bin_index(grey, 24, 0.03) == 24)


def test_hue_histogram_sums_to_one():
    lab = _cloud(np.random.default_rng(1), 1000)
    h = hue_histogram(lab, 12, 0.03)
    assert h.shape == (13,) and h.sum() == pytest.approx(1.0)


def test_hue_weights_make_target_histogram_match_source():
    rng = np.random.default_rng(2)
    src = _cloud(rng, 20000)  # uniform hue
    tgt = np.concatenate([_cloud(rng, 14000, 0, np.pi), _cloud(rng, 6000, np.pi, 2 * np.pi)])
    w = hue_weights(src, tgt, n_bins=24)
    assert w.mean() == pytest.approx(1.0)
    h_src = hue_histogram(src, 24, 0.03)
    h_tgt_w = hue_histogram(tgt, 24, 0.03, weights=w)
    assert np.abs(h_src - h_tgt_w).max() < 0.01


def test_weighted_quantile_map_matches_moments_and_keeps_order():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 20000)
    y = rng.normal(5, 2, 20000)
    mapped = weighted_quantile_map(x, y, np.ones_like(y))
    assert mapped.mean() == pytest.approx(5, abs=0.1)
    assert mapped.std() == pytest.approx(2, rel=0.05)
    order = np.argsort(x)
    assert np.all(np.diff(mapped[order]) >= 0)


def test_weighted_quantile_map_respects_weights():
    x = np.linspace(0, 1, 1001)
    y = np.concatenate([np.zeros(1000), np.full(1000, 10.0)])
    w = np.concatenate([np.full(1000, 3.0), np.ones(1000)])
    mapped = weighted_quantile_map(x, y, w)
    # three quarters of the weighted mass sits at 0, so the 60th percentile maps to 0
    assert mapped[600] < 0.5
    assert mapped[-1] > 9.5
    unweighted = weighted_quantile_map(x, y, np.ones_like(y))
    assert unweighted[600] > 9.5  # without weights the 60th percentile is already at 10


def test_random_rotation_is_proper():
    r = random_rotation(np.random.default_rng(4))
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-10)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_idt_moves_source_onto_target_distribution():
    rng = np.random.default_rng(5)
    src = rng.normal([0.5, 0.0, 0.0], [0.1, 0.05, 0.05], (20000, 3))
    tgt = rng.normal([0.6, 0.05, -0.05], [0.15, 0.08, 0.04], (20000, 3))
    before = sliced_wasserstein(src, tgt, rng=np.random.default_rng(0))
    moved = iterative_distribution_transfer(src, tgt, iterations=20, rng=np.random.default_rng(1))
    after = sliced_wasserstein(moved, tgt, rng=np.random.default_rng(0))
    assert moved.shape == src.shape and moved.dtype == np.float32
    assert after < before * 0.1
    assert np.allclose(moved.mean(axis=0), tgt.mean(axis=0), atol=0.01)
    assert np.allclose(moved.std(axis=0), tgt.std(axis=0), rtol=0.1)


def test_transport_preserves_pixel_identity():
    """Row i of the output must be where source pixel i went, not merely a
    point drawn from the right distribution.

    Everything else in this file is permutation-invariant: shuffling the
    output rows leaves the sliced Wasserstein distance, the means and the
    standard deviations bit-identical, because a permutation does not change
    a distribution at all. Measured, to be sure of it. So a regression that
    reordered rows would sail through every other assertion here while
    handing the LUT fitter pairs that no longer correspond — a film look
    fitted to noise, with nothing to say so.

    The property is tested as equivariance: permuting the input must permute
    the output the same way. That holds exactly for a rank-based mapping and
    fails for any reordering.
    """
    rng = np.random.default_rng(0)
    src = rng.normal([0.5, 0.0, 0.0], [0.1, 0.05, 0.05], (3000, 3))
    tgt = rng.normal([0.6, 0.05, -0.05], [0.15, 0.08, 0.04], (3000, 3))
    perm = np.random.default_rng(7).permutation(len(src))

    straight = iterative_distribution_transfer(src, tgt, iterations=8, rng=np.random.default_rng(1))
    permuted = iterative_distribution_transfer(
        src[perm], tgt, iterations=8, rng=np.random.default_rng(1)
    )
    assert np.array_equal(permuted, straight[perm]), "row correspondence was not preserved"


def test_sliced_wasserstein_zero_for_identical_and_positive_for_shift():
    rng = np.random.default_rng(6)
    a = rng.normal(size=(5000, 3))
    identical = sliced_wasserstein(a, a.copy(), rng=np.random.default_rng(0))
    assert identical == pytest.approx(0.0, abs=1e-9)
    assert sliced_wasserstein(a, a + 1.0, rng=np.random.default_rng(0)) > 0.3
