"""Unpaired colour-distribution transport in Oklab.

The problem
-----------
We have two unpaired sets of pixels: colours the camera produces, and
colours Kodachrome produced. Nothing links a particular camera pixel to a
particular film pixel. What we can ask is: "if the camera's cloud of colours
had to become the film's cloud of colours, moving each point as little as
possible, where would each camera colour go?" That is a transport problem;
its answer gives every source pixel a partner, and the LUT is then fitted to
those pairs (see ``lutfit.py``).

Content bias and hue reweighting: a heuristic, not a constraint
---------------------------------------------------------------
The two corpora do not show the same things. The 1940s FSA photographs are
full of fields, khaki and weathered wood; a modern indoor sample set is not.
Raw distribution matching would happily turn a blue wall green because the
film corpus contains more green. ``hue_weights`` reduces that: target pixels
are reweighted so the target's hue histogram matches the source's, removing
the largest content-driven bias.

Be precise about what this does **not** do. The transport still operates on
the full three-dimensional distribution and can move individual pixels
across hue bins; matching the aggregate histogram does not constrain where
any particular pixel goes. Clipping the weights to ``[0.2, 5.0]`` also means
the reweighted histograms match only approximately when a hue is nearly
absent from one side. So this is a bias-reduction heuristic, not a guarantee
about what is learned, and the report publishes the residual histogram
difference (``hue_hist_residual``) so its size is visible rather than
assumed. Spec section 1 states the resulting limit on the project's claim.

Iterative distribution transfer (Pitié, Kokaram, Dahyot 2005)
-------------------------------------------------------------
Matching a 3D distribution directly is hard; matching a 1D one is a sort.
IDT repeats: pick a random 3D rotation, project both clouds onto its three
axes, match the source marginal to the (weighted) target marginal along
each axis by quantile mapping, rotate back. Each round moves the source
cloud closer to the target in every direction; a few dozen rounds converge.
Because each round is a monotone map along each axis, pixel identities are
preserved: row ``i`` of the output is where source pixel ``i`` went.

Sliced Wasserstein distance is the same idea used as a metric: average
1D Wasserstein-2 distance over random projections.
"""

from __future__ import annotations

import numpy as np

from ..color import oklab_to_lch


def hue_bin_index(lab: np.ndarray, n_bins: int, chroma_floor: float) -> np.ndarray:
    lch = oklab_to_lch(lab)
    hue = np.mod(lch[..., 2], 2 * np.pi)
    idx = np.minimum(np.floor(hue / (2 * np.pi) * n_bins).astype(np.int64), n_bins - 1)
    idx[lch[..., 1] < chroma_floor] = n_bins
    return idx


def hue_histogram(
    lab: np.ndarray, n_bins: int, chroma_floor: float, weights: np.ndarray | None = None
) -> np.ndarray:
    idx = hue_bin_index(lab, n_bins, chroma_floor)
    hist = np.bincount(idx, weights=weights, minlength=n_bins + 1).astype(np.float64)
    return hist / max(hist.sum(), 1e-12)


def hue_weights(
    src_lab: np.ndarray,
    tgt_lab: np.ndarray,
    n_bins: int = 24,
    chroma_floor: float = 0.03,
    w_min: float = 0.2,
    w_max: float = 5.0,
) -> np.ndarray:
    h_src = hue_histogram(src_lab, n_bins, chroma_floor)
    h_tgt = hue_histogram(tgt_lab, n_bins, chroma_floor)
    ratio = np.where(h_tgt > 0, h_src / np.maximum(h_tgt, 1e-12), 1.0)
    ratio = np.clip(ratio, w_min, w_max)
    w = ratio[hue_bin_index(tgt_lab, n_bins, chroma_floor)]
    return (w / w.mean()).astype(np.float64)


def weighted_quantile_map(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Monotone map sending the empirical distribution of ``x`` onto the
    ``w``-weighted empirical distribution of ``y``."""
    order = np.argsort(y, kind="stable")
    y_sorted = y[order]
    w_sorted = w[order]
    cum = np.cumsum(w_sorted)
    q_y = (cum - 0.5 * w_sorted) / cum[-1]
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[np.argsort(x, kind="stable")] = np.arange(len(x))
    q_x = (ranks + 0.5) / len(x)
    return np.interp(q_x, q_y, y_sorted)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def iterative_distribution_transfer(
    src_lab: np.ndarray,
    tgt_lab: np.ndarray,
    tgt_w: np.ndarray | None = None,
    iterations: int = 40,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng if rng is not None else np.random.default_rng(0)
    x = np.asarray(src_lab, dtype=np.float64).copy()
    y = np.asarray(tgt_lab, dtype=np.float64)
    w = np.ones(len(y)) if tgt_w is None else np.asarray(tgt_w, dtype=np.float64)
    for _ in range(iterations):
        rot = random_rotation(rng)
        xr = x @ rot
        yr = y @ rot
        for axis in range(3):
            xr[:, axis] = weighted_quantile_map(xr[:, axis], yr[:, axis], w)
        x = xr @ rot.T
    return x.astype(np.float32)


def sliced_wasserstein(
    a: np.ndarray,
    b: np.ndarray,
    n_proj: int = 64,
    rng: np.random.Generator | None = None,
    b_weights: np.ndarray | None = None,
    max_points: int = 100_000,
) -> float:
    rng = rng if rng is not None else np.random.default_rng(0)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b), max_points)
    a = a[rng.choice(len(a), n, replace=False)]
    if b_weights is not None:
        p = np.asarray(b_weights, dtype=np.float64)
        b = b[rng.choice(len(b), n, replace=True, p=p / p.sum())]
    else:
        b = b[rng.choice(len(b), n, replace=False)]
    dirs = rng.standard_normal((n_proj, a.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pa = np.sort(a @ dirs.T, axis=0)
    pb = np.sort(b @ dirs.T, axis=0)
    return float(np.sqrt(np.mean((pa - pb) ** 2)))
