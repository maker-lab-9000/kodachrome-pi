import numpy as np
import pytest

from kodachrome.lut import LUT3D
from kodachrome.train.evaluate import channels_are_monotone, neutral_axis_max_chroma
from kodachrome.train.lutfit import (
    cap_neutral_axis,
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


@pytest.mark.parametrize("channel", [0, 1, 2])
def test_enforce_monotone_repairs_each_channel_along_its_own_axis(channel):
    """Channel c must be repaired along axis c, and no other channel touched.

    Exercising only the red channel leaves `moveaxis(.., c, 0)` free to be
    `moveaxis(.., 0, 0)`: green and blue would then be projected along the
    RED axis, which still satisfies the gate while silently corrupting them.
    Only the tone-curve acceptance test caught that, and only by 0.0017.
    """
    n = 9
    table = LUT3D.identity(n).table.astype(np.float64).copy()
    idx = [3, 4, 5]
    idx[channel] = 6
    table[tuple(idx) + (channel,)] = 0.0          # a hard reversal along axis `channel`
    broken = LUT3D(table.astype(np.float32))
    assert not channels_are_monotone(broken)

    fixed = enforce_monotone(broken)
    assert channels_are_monotone(fixed)

    # Repaired along its OWN axis...
    steps = np.diff(fixed.table[..., channel].astype(np.float64), axis=channel)
    assert steps.min() >= -1e-9
    # ...and the other two channels are untouched.
    moved = [c for c in range(3)
             if np.abs(fixed.table[..., c].astype(np.float64) - table[..., c]).max() > 1e-9]
    assert moved == [channel], f"projection altered channels {moved}, expected only [{channel}]"


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


def _tinted_lut(n=17, a=0.0, b=0.03):
    """Identity with a uniform Oklab shift: every input, grey or not, gains (a, b)."""
    from kodachrome.color import oklab_to_srgb, srgb_to_oklab
    lab = srgb_to_oklab(LUT3D.identity(n).table.reshape(-1, 3))
    lab[:, 1] += a
    lab[:, 2] += b
    return LUT3D(np.clip(oklab_to_srgb(lab), 0, 1).reshape(n, n, n, 3).astype(np.float32))


def test_cap_neutral_axis_limits_grey_tint_and_leaves_colour_alone():
    """Dark greys came out olive at 0.036 on the first levels fit; the gate is 0.02.

    A uniform +0.03 b-shift is well over a 0.01 cap on the grey ramp. After
    capping, greys carry at most the cap and a saturated red node is
    unchanged, because the correction tapers to zero for colourful input.
    """
    from kodachrome.color import srgb_to_oklab
    lut = _tinted_lut()
    assert neutral_axis_max_chroma(lut) > 0.025
    capped = cap_neutral_axis(lut, 0.01)
    assert neutral_axis_max_chroma(capped) <= 0.0105
    red = np.array([[0.9, 0.1, 0.1]], dtype=np.float32)
    assert np.allclose(capped.apply_numpy(red), lut.apply_numpy(red), atol=1e-6)
    # And a mid grey now carries no more than the cap, in the same hue direction.
    grey_lab = srgb_to_oklab(capped.apply_numpy(np.array([[0.5, 0.5, 0.5]], dtype=np.float32)))
    assert np.hypot(grey_lab[0, 1], grey_lab[0, 2]) <= 0.0105


def test_cap_neutral_axis_zero_neutralises_and_none_is_a_no_op():
    """cap=0 drives the grey ramp toward neutral; the floor is the grid.

    The correction lands on nodes and the ramp is read back by trilinear
    interpolation, so a coarse grid keeps a residual (measured 0.0041 at 17
    nodes, 0.0007 at 33, for a uniform 0.03 shift). Production is 33.
    """
    assert neutral_axis_max_chroma(cap_neutral_axis(_tinted_lut(n=17), 0.0)) < 0.006
    assert neutral_axis_max_chroma(cap_neutral_axis(_tinted_lut(n=33), 0.0)) < 0.0025
    lut = _tinted_lut()
    with pytest.raises(ValueError, match="non-negative"):
        cap_neutral_axis(lut, -0.1)
    rng = np.random.default_rng(0)
    x = rng.random((3000, 3)).astype(np.float32)
    y = np.clip(x * np.array([1.05, 1.0, 0.9], dtype=np.float32), 0, 1)   # a warm cast
    uncapped = fit_lut(x, y, n=9, neutral_axis_cap=None)
    default = fit_lut(x, y, n=9)
    assert neutral_axis_max_chroma(uncapped) > 0.012
    assert neutral_axis_max_chroma(default) <= 0.0105 + 1e-3     # monotone runs after the cap
