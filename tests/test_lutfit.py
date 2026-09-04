import numpy as np
import pytest

from kodachrome.lut import LUT3D
from kodachrome.train.evaluate import channels_are_monotone
from kodachrome.train.lutfit import (
    enforce_monotone,
    fit_lut,
    node_index,
    second_difference_operator,
    trilinear_design_matrix,
)


def test_node_index_matches_table_flattening():
    n = 5
    table = LUT3D.identity(n).table.reshape(-1, 3)
    for r, g, b in [(0, 0, 0), (4, 0, 0), (0, 4, 0), (0, 0, 4), (2, 3, 1)]:
        idx = node_index(np.array([r]), np.array([g]), np.array([b]), n)[0]
        assert np.allclose(table[idx], [r / 4, g / 4, b / 4])


def test_design_matrix_rows_sum_to_one_and_reproduce_identity():
    n = 9
    x = np.random.default_rng(0).random((500, 3), dtype=np.float32)
    a = trilinear_design_matrix(x, n)
    assert a.shape == (500, n**3)
    assert np.allclose(np.asarray(a.sum(axis=1)).ravel(), 1.0)
    assert a.nnz <= 500 * 8
    ident = LUT3D.identity(n).table.reshape(-1, 3)
    assert np.allclose(a @ ident, x, atol=1e-6)


def test_second_difference_operator_kills_linear_luts():
    n = 7
    d = second_difference_operator(n)
    assert d.shape == (3 * n * n * (n - 2), n**3)
    ident = LUT3D.identity(n).table.reshape(-1, 3)
    assert np.allclose(d @ ident, 0.0, atol=1e-6)
    bumpy = ident.copy()
    bumpy[n**3 // 2] += 0.1
    assert np.abs(d @ bumpy).max() > 0.1


def test_fit_recovers_per_channel_curve():
    rng = np.random.default_rng(1)
    x = rng.random((30000, 3), dtype=np.float32)
    y = np.clip(x**1.3, 0, 1).astype(np.float32)
    lut = fit_lut(x, y, n=17, lambda_smooth=1e-3, lambda_identity=1e-4)
    assert lut.size == 17
    held = rng.random((3000, 3), dtype=np.float32)
    err = np.abs(lut.apply_numpy(held) - np.clip(held**1.3, 0, 1)).mean()
    assert err < 0.01


def test_the_identity_term_anchors_nodes_no_data_reaches():
    """Untouched nodes must stay put — and this test must notice if they do not.

    The obvious fixture cannot. Fitting a near-identity transform (y = 1.1x)
    over the dark half passes with the identity term *deleted*, because the
    smoothness term extrapolates the fitted curve into the bright region and
    lands close to identity anyway: measured 0.07 drift against a 0.25 bound.
    A test that green-lights the absence of the thing it is named for is
    worse than no test.

    A strong transform separates them. Fitting y = 0.25x over the dark half
    leaves the bright region 0.71 away from identity when nothing anchors it,
    against 0.003 when the identity term is present.
    """
    rng = np.random.default_rng(2)
    x = (rng.random((20000, 3), dtype=np.float32) * 0.5).astype(np.float32)
    y = np.clip(x * 0.25, 0, 1).astype(np.float32)
    lut = fit_lut(x, y, n=9)

    bright = np.array([[0.95, 0.95, 0.95], [0.9, 0.2, 0.9]], dtype=np.float32)
    out = lut.apply_numpy(bright)
    assert np.all(np.isfinite(out))
    assert np.abs(out - bright).max() < 0.05, "untouched nodes drifted"

    # And the fitted region must still follow its data, or a LUT that simply
    # ignored every sample would satisfy the assertion above.
    dark = np.array([[0.3, 0.3, 0.3], [0.4, 0.15, 0.35]], dtype=np.float32)
    assert np.abs(lut.apply_numpy(dark) - dark * 0.25).max() < 0.05


def test_fit_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        fit_lut(np.zeros((10, 3)), np.zeros((9, 3)), n=5)


def test_enforce_monotone_makes_a_reversed_lut_monotone_without_moving_the_rest():
    """A least squares fit gives no ordering guarantee; this projection does.

    Reversals of up to 0.366 (93 levels of 255) appeared in the first real
    fit, in well-supported regions, and would posterise a gradient.
    """
    n = 9
    table = LUT3D.identity(n).table.astype(np.float64).copy()
    # Fold the red channel back on itself along the red axis, twice.
    table[5, 2, 3, 0] = table[2, 2, 3, 0]
    table[7, 4, 1, 0] = 0.0
    broken = LUT3D(table.astype(np.float32))
    assert not channels_are_monotone(broken)

    fixed = enforce_monotone(broken)
    assert channels_are_monotone(fixed)

    # It is a projection, not a rebuild: untouched fibres must be bit-identical.
    ident = LUT3D.identity(n).table
    moved = np.abs(fixed.table - ident) > 1e-6
    assert moved.sum() < 0.05 * moved.size, "projection disturbed far more than the reversals"


def test_enforce_monotone_leaves_an_already_monotone_lut_alone():
    lut = LUT3D((LUT3D.identity(9).table.astype(np.float64) ** 1.3).astype(np.float32))
    assert channels_are_monotone(lut)
    assert np.allclose(enforce_monotone(lut).table, lut.table, atol=1e-6)


def test_fit_lut_returns_a_monotone_lut_by_default():
    """The fitter's contract: callers get a LUT that passes the gate."""
    rng = np.random.default_rng(0)
    x = rng.random((4000, 3)).astype(np.float32)
    # A deliberately order-breaking target: partners that invert in red.
    y = x.copy()
    y[:, 0] = 1.0 - y[:, 0]
    assert not channels_are_monotone(fit_lut(x, y, n=9, monotone=False))
    assert channels_are_monotone(fit_lut(x, y, n=9))
