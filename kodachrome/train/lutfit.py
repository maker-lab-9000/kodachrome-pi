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
* ``I`` is the identity LUT. The **identity term** is tiny but decides what
  happens to nodes no source pixel ever touches (a saturated magenta the
  camera never saw): they stay where they were instead of drifting.

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
from scipy.sparse.linalg import cg

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


def fit_lut(
    x_srgb: np.ndarray,
    y_srgb: np.ndarray,
    n: int = 33,
    lambda_smooth: float = 1e-3,
    lambda_identity: float = 1e-4,
    rtol: float = 1e-8,
    maxiter: int = 5000,
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
    return LUT3D(np.clip(table, 0.0, 1.0).reshape(n, n, n, 3).astype(np.float32))
