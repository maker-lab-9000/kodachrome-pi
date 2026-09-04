"""Fit a smooth 3D LUT to (input colour, target colour) pairs.

After transport (``transport.py``) every source pixel ``x_i`` has a partner
``y_i``. A trilinear LUT is *linear in its node values*: the output for
``x_i`` is a fixed weighted sum of the eight surrounding nodes. So fitting
the LUT is ordinary least squares with a sparse design matrix ``A`` (eight
non-zeros per row), solved once per output channel:

    minimise  (1/M) ||A L - y||^2
            + lambda_smooth   / |D|  * ||D L||^2
            + lambda_identity / N^3  * ||L - I||^2

* The **data term** pulls nodes toward the transported partners.
* ``D`` stacks second-difference operators along the three grid axes. The
  **smoothness term** stops individual nodes chasing noisy partners, which
  would show up as banding or speckle in gradients.
* ``I`` is the identity LUT. The **identity term** decides what happens to
  nodes no source pixel ever touches (a saturated magenta the camera never
  saw): they stay where they were instead of drifting. On real corpora this
  matters more than it sounds -- 71% of the cube held no source pixels in the
  first real fit, and at the original 1e-4 weight those nodes drifted far
  enough to fail the grey-axis and neutral-tint gates.

Least squares alone cannot promise the result is monotone, so a final
projection (``enforce_monotone``) enforces it exactly. See that function.

Each term is divided by its own row count so the lambdas are relative
weights that do not change meaning when the sample count or grid size does.

The normal equations ``(A'A/M + ... ) L = A'y/M + ...`` are symmetric positive
definite (the identity term guarantees it), so they are solved with
conjugate gradients and a Jacobi preconditioner. That was chosen over a
direct sparse solve because the 3D grid's fill-in makes factorisation
memory-hungry at N=33 (35,937 unknowns), while CG converges in a few
thousand cheap sparse products.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import isotonic_regression
from scipy.sparse.linalg import cg

from ..color import oklab_to_srgb, srgb_to_oklab
from ..lut import LUT3D


class FitConvergenceError(RuntimeError):
    """Conjugate gradients did not converge."""


def node_index(r: np.ndarray, g: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    return (r * n + g) * n + b


def trilinear_design_matrix(x_srgb: np.ndarray, n: int) -> sp.csr_matrix:
    x = np.clip(np.asarray(x_srgb, dtype=np.float64), 0.0, 1.0) * (n - 1)
    i0 = np.minimum(np.floor(x).astype(np.int64), n - 2)
    f = x - i0
    m = len(x)
    rows = np.arange(m)
    row_parts, col_parts, val_parts = [], [], []
    for dr in (0, 1):
        wr = f[:, 0] if dr else 1.0 - f[:, 0]
        for dg in (0, 1):
            wg = f[:, 1] if dg else 1.0 - f[:, 1]
            for db in (0, 1):
                wb = f[:, 2] if db else 1.0 - f[:, 2]
                row_parts.append(rows)
                col_parts.append(node_index(i0[:, 0] + dr, i0[:, 1] + dg, i0[:, 2] + db, n))
                val_parts.append(wr * wg * wb)
    a = sp.csr_matrix(
        (np.concatenate(val_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
        shape=(m, n**3),
    )
    a.sum_duplicates()
    return a


def second_difference_operator(n: int) -> sp.csr_matrix:
    idx = np.arange(n**3).reshape(n, n, n)
    blocks = []
    for axis in range(3):
        moved = np.moveaxis(idx, axis, 0)
        prev = moved[:-2].ravel()
        centre = moved[1:-1].ravel()
        nxt = moved[2:].ravel()
        k = len(centre)
        rows = np.arange(k)
        blocks.append(
            sp.csr_matrix(
                (
                    np.concatenate([np.ones(k), -2.0 * np.ones(k), np.ones(k)]),
                    (np.concatenate([rows, rows, rows]), np.concatenate([prev, centre, nxt])),
                ),
                shape=(k, n**3),
            )
        )
    return sp.vstack(blocks).tocsr()


# Inputs with Oklab chroma below the first value are treated as neutral and
# fully corrected; above the second they are colour and left alone; between,
# a smoothstep. 0.06 sits just under skin (measured 0.07 on a real face).
NEUTRAL_TAPER = (0.03, 0.06)


def cap_neutral_axis(
    lut: LUT3D, cap: float, taper: tuple[float, float] = NEUTRAL_TAPER, passes: int = 4
) -> LUT3D:
    """Limit the tint the LUT gives a neutral input to ``cap`` Oklab chroma.

    Distribution transport learns the target's cast on greys along with
    everything else, and on the first levels-normalised fit it left dark
    greys olive at chroma 0.036 against a 0.02 gate. Zeroing the cast
    entirely costs about 10% of the held-out match, because that cast is
    part of what the transport was matching. Capping it keeps a residual
    below visibility and removes only the excess.

    The grey ramp's own output tint is measured as a function of input
    lightness, the excess over ``cap`` is subtracted from every node, and the
    subtraction is tapered to zero for colourful input so hue rendering is
    untouched. ``cap=0`` neutralises greys fully. Runs before
    ``enforce_monotone`` so the ordering constraint is re-imposed last.

    The correction is applied at the nodes but the ramp is read back through
    trilinear interpolation, and Oklab-to-sRGB is not linear between nodes,
    so one pass leaves a residual (0.006 of a 0.03 shift at 17 nodes). A few
    passes converge; the loop stops early once the ramp is within the cap.
    """
    if cap < 0:
        raise ValueError(f"cap must be non-negative, got {cap}")
    n = lut.size
    lab_in = srgb_to_oklab(LUT3D.identity(n).table.reshape(-1, 3)).astype(np.float64)
    c_in = np.hypot(lab_in[:, 1], lab_in[:, 2])
    x = np.clip((c_in - taper[0]) / (taper[1] - taper[0]), 0.0, 1.0)
    w = 1.0 - x * x * (3.0 - 2.0 * x)
    ramp = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    grey = np.stack([ramp, ramp, ramp], axis=1)
    l_grey = srgb_to_oklab(grey)[:, 0].astype(np.float64)

    for _ in range(passes):
        tint = srgb_to_oklab(lut.apply_numpy(grey)).astype(np.float64)[:, 1:]
        mag = np.hypot(tint[:, 0], tint[:, 1])
        if mag.max() <= cap + 1e-4:
            break
        keep = np.where(mag > cap, cap / np.maximum(mag, 1e-12), 1.0)
        excess = tint * (1.0 - keep)[:, None]                 # what to remove, per ramp L
        lab_out = srgb_to_oklab(lut.table.reshape(-1, 3)).astype(np.float64)
        lab_out[:, 1] -= w * np.interp(lab_in[:, 0], l_grey, excess[:, 0])
        lab_out[:, 2] -= w * np.interp(lab_in[:, 0], l_grey, excess[:, 1])
        out = np.clip(oklab_to_srgb(lab_out.astype(np.float32)), 0.0, 1.0)
        lut = LUT3D(out.reshape(n, n, n, 3).astype(np.float32))
    return lut


def _grey_axis_luminance(table: np.ndarray) -> np.ndarray:
    """Linear luminance of the LUT's 256-point grey ramp, as the gate measures it."""
    from ..color import luminance, srgb_to_linear

    ramp = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    grey = np.stack([ramp, ramp, ramp], axis=1)
    return luminance(srgb_to_linear(LUT3D(table).apply_numpy(grey)))


def enforce_grey_axis(lut: LUT3D, passes: int = 8, tol: float = 2e-4) -> LUT3D:
    """Make luminance non-decreasing along the 256-point grey ramp.

    Per-channel monotonicity along each channel's own axis does not imply it:
    along the diagonal all three inputs rise together and the cross terms can
    dip. The first symmetric-levels fit dipped by 0.00108 at input 0.937 --
    one step, just past the gate's 0.001 tolerance.

    Each ramp sample is a trilinear blend of the eight corners of one cell.
    Where the ramp departs from its isotonic regression, a small least
    squares problem over the affected cells finds the corner luminance
    changes that put every sample in those cells on target at once; the
    blend is done in sRGB while the change is applied in linear light, so
    the pass repeats until the ramp is clean. A scaling indexed by lightness
    alone cannot do this: a dip caused by corners at the same lightness as
    the diagonal node survives any per-lightness factor.
    """
    from scipy.optimize import isotonic_regression

    from ..color import linear_to_srgb, luminance, srgb_to_linear

    n = lut.size
    table = np.array(lut.table, dtype=np.float32)
    grey = np.linspace(0.0, 1.0, 256)
    pos = grey * (n - 1)
    cell = np.minimum(np.floor(pos).astype(int), n - 2)
    frac = pos - cell
    offsets = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]

    for _ in range(passes):
        lum_ramp = _grey_axis_luminance(table).astype(np.float64)
        if np.diff(lum_ramp).min() >= -tol:
            break
        target = isotonic_regression(lum_ramp).x
        bad_cells = set(cell[np.abs(target - lum_ramp) > tol / 4].tolist())
        samples = [k for k in range(256) if cell[k] in bad_cells]
        corners = sorted(
            {(cell[k] + a, cell[k] + b, cell[k] + c) for k in samples for a, b, c in offsets}
        )
        col = {c: j for j, c in enumerate(corners)}
        # One row per ramp sample in an affected cell: its trilinear weights on
        # the corners' luminance changes must add up to its shortfall. Samples
        # that are already on target constrain their corners to stay put, which
        # is what a per-sample update could not do -- consecutive samples share
        # corners and were undoing each other.
        A = np.zeros((len(samples), len(corners)))
        b = np.zeros(len(samples))
        for r, k in enumerate(samples):
            f = frac[k]
            for a, g, c in offsets:
                w = (f if a else 1 - f) * (f if g else 1 - f) * (f if c else 1 - f)
                A[r, col[(cell[k] + a, cell[k] + g, cell[k] + c)]] = w
            b[r] = target[k] - lum_ramp[k]
        delta = np.linalg.solve(A.T @ A + 1e-4 * np.eye(len(corners)), A.T @ b)
        lin = srgb_to_linear(table).astype(np.float64)
        for c, d in zip(corners, delta, strict=True):
            lum_c = float(luminance(lin[c][None])[0])
            factor = float(np.clip((lum_c + d) / max(lum_c, 1e-6), 0.5, 2.0))
            lin[c] = np.clip(lin[c] * factor, 0.0, 1.0)
        table = linear_to_srgb(lin.astype(np.float32))
    return LUT3D(table)


def enforce_monotone(lut: LUT3D) -> LUT3D:
    """Project each output channel onto the monotone cone along its own axis.

    Least squares fitting gives no ordering guarantee, and neither does the
    transport that produced its targets: where transport asks a dark pixel to
    brighten and its slightly lighter neighbour to darken, the LUT obliges.
    Measured on the first real fit that produced reversals of up to 0.366,
    which is 93 levels out of 255 -- enough to posterise or invert a gradient.

    ``channels_are_monotone`` requires output channel ``c`` not to fall as
    input axis ``c`` rises. Those three constraints touch disjoint variables,
    so each is an exact 1-D isotonic regression along every fibre and a single
    pass per channel is the optimal least squares projection. It is also
    nearly free in match quality: on the first real fit it moved the held-out
    sliced Wasserstein distance by less than 0.01%, because the reversals
    contributed nothing to the match in the first place.
    """
    table = np.array(lut.table, dtype=np.float64)
    n = lut.size
    for c in range(3):
        plane = np.moveaxis(table[..., c], c, 0).copy()
        fibres = plane.reshape(n, -1)
        for j in range(fibres.shape[1]):
            fibres[:, j] = isotonic_regression(fibres[:, j]).x
        table[..., c] = np.moveaxis(fibres.reshape(plane.shape), 0, c)
    return LUT3D(np.clip(table, 0.0, 1.0).astype(np.float32))


def fit_lut(
    x_srgb: np.ndarray,
    y_srgb: np.ndarray,
    n: int = 33,
    lambda_smooth: float = 1e-2,
    lambda_identity: float = 1.0,
    rtol: float = 1e-8,
    maxiter: int = 5000,
    monotone: bool = True,
    neutral_axis_cap: float | None = 0.01,
) -> LUT3D:
    x = np.asarray(x_srgb, dtype=np.float64)
    y = np.clip(np.asarray(y_srgb, dtype=np.float64), 0.0, 1.0)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"x and y must both be (M, 3); got {x.shape} and {y.shape}")
    m = len(x)
    a = trilinear_design_matrix(x, n)
    d = second_difference_operator(n)
    ident = LUT3D.identity(n).table.reshape(-1, 3).astype(np.float64)

    lhs = (a.T @ a) / m
    lhs = lhs + (d.T @ d) * (lambda_smooth / d.shape[0])
    lhs = lhs + sp.identity(n**3, format="csr") * (lambda_identity / n**3)
    lhs = lhs.tocsr()
    precond = sp.diags(1.0 / np.maximum(lhs.diagonal(), 1e-12))

    table = np.empty((n**3, 3), dtype=np.float64)
    for c in range(3):
        rhs = (a.T @ y[:, c]) / m + (lambda_identity / n**3) * ident[:, c]
        sol, info = cg(lhs, rhs, x0=ident[:, c], rtol=rtol, maxiter=maxiter, M=precond)
        if info != 0:
            raise FitConvergenceError(
                f"CG did not converge for channel {c} (info={info}); "
                "raise --lambda-identity or --lambda-smooth slightly"
            )
        table[:, c] = sol
    lut = LUT3D(np.clip(table, 0.0, 1.0).reshape(n, n, n, 3).astype(np.float32))
    if neutral_axis_cap is not None:
        lut = cap_neutral_axis(lut, neutral_axis_cap)
    if monotone:
        lut = enforce_monotone(enforce_grey_axis(enforce_monotone(lut)))
    return lut
