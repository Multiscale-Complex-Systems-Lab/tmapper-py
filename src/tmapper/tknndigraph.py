"""Port of tmapper_tools/tknndigraph.m from the MATLAB toolbox."""

import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist


def _percentile_with_inf(values, prct):
    """Like ``np.percentile`` (linear interpolation method), but robust to
    ``inf`` entries. ``np.percentile`` can return NaN when the interpolation
    boundary falls between two ``inf`` values, because its formula computes
    ``a + weight * (b - a)`` and ``inf - inf`` is NaN even when ``weight``
    is (mathematically) zero -- 0 * NaN is still NaN, not 0. This matters
    here since D routinely contains inf entries (masked self-loops/temporal
    exclusions) and MATLAB's prctile does not have this problem.
    """
    sorted_vals = np.sort(np.asarray(values).ravel())
    n = sorted_vals.size
    if n == 0:
        return np.inf
    virtual_idx = (prct / 100.0) * (n - 1)
    lo = int(np.floor(virtual_idx))
    hi = int(np.ceil(virtual_idx))
    if lo == hi:
        return sorted_vals[lo]
    a, b = sorted_vals[lo], sorted_vals[hi]
    if np.isinf(a) and np.isinf(b) and a == b:
        return a
    frac = virtual_idx - lo
    return a + frac * (b - a)


def tknndigraph(
    X_or_D,
    k,
    tidx,
    *,
    reciprocal=True,
    time_exclude_space=True,
    time_exclude_range=1,
    max_neighbor_dist=np.inf,
    max_neighbor_dist_prct=100.0,
):
    """Construct a directed graph based on k-nearest neighbors that also
    includes temporal neighbors (t -> t+1 links).

    Port of MATLAB's ``tknndigraph.m``. Node i of the returned graph
    corresponds to row i of ``X_or_D`` / element i of ``tidx`` (0-indexed,
    unlike the 1-indexed MATLAB original).

    Parameters
    ----------
    X_or_D : array_like, shape (N, d) or (N, N)
        Either raw coordinates (N points in d-dim space) or a precomputed
        (N, N) distance matrix. Auto-detected: treated as a distance
        matrix only if square *and* symmetric; otherwise treated as raw
        coordinates and converted to a Euclidean distance matrix.
    k : int
        Max number of spatial neighbors per point. Must be a positive
        integer strictly less than N.
    tidx : array_like of int, shape (N,)
        Time index per point. Two points x, y are temporal neighbors iff
        ``tidx[y] == tidx[x] + 1``.
    reciprocal : bool, default True
        Whether to require spatial k-NN links to be mutual (both
        directions) to be kept.
    time_exclude_space : bool, default True
        Whether temporal neighbors are barred from also counting as
        spatial neighbors.
    time_exclude_range : int, default 1
        How many following time points are excluded from spatial-kNN
        consideration.
    max_neighbor_dist : float, default inf
        Absolute distance cutoff for spatial neighbors.
    max_neighbor_dist_prct : float, default 100
        Percentile distance cutoff for spatial neighbors. The stricter
        (smaller) of this and ``max_neighbor_dist`` is applied.

    Returns
    -------
    g : networkx.DiGraph
        Unweighted directed graph, one node per input point.
    par : dict
        Parameters actually used, including the resolved
        ``max_neighbor_dist``.
    """
    X_or_D = np.asarray(X_or_D, dtype=float)
    if X_or_D.ndim != 2:
        raise ValueError("X_or_D must be a 2D array.")
    if not (np.isscalar(k) and float(k) == round(float(k)) and k >= 1):
        raise ValueError("k must be a positive integer scalar.")
    k = int(k)

    tidx = np.asarray(tidx, dtype=float).ravel()
    if tidx.shape[0] != X_or_D.shape[0]:
        raise ValueError(
            "tidx must have the same number of elements as rows of X_or_D."
        )

    par = {
        "reciprocal": reciprocal,
        "time_exclude_space": time_exclude_space,
        "time_exclude_range": time_exclude_range,
        "max_neighbor_dist": max_neighbor_dist,
        "max_neighbor_dist_prct": max_neighbor_dist_prct,
        "k": k,
    }

    # -- check input and obtain distance matrix D
    nr, nc = X_or_D.shape
    if nr != nc or not np.allclose(X_or_D, X_or_D.T):
        D = cdist(X_or_D, X_or_D)
    else:
        D = X_or_D.copy()
    Nn = D.shape[0]

    if k >= Nn:
        raise ValueError(f"k must be smaller than the number of points ({Nn}).")

    np.fill_diagonal(D, np.inf)  # exclude self-loops

    # -- find indices for temporal links D[i(t), i(t+1)]
    # t_wafter[i] is True iff point i has a point immediately after it in time
    # (mirrors MATLAB's circshift(tidx,-1,1) - 1 == tidx, including its
    # wraparound behavior at the last index, which is virtually always False
    # for non-cyclic tidx and is harmless here for the same reason it is in
    # the original).
    tidx_next = np.roll(tidx, -1)
    t_wafter = tidx_next - 1 == tidx

    # t_after_idx1: unconditional immediate temporal edges i -> i+1
    # Built by index arithmetic rather than a Python loop over rows: the
    # bands are just the pairs (i, i+n) for the i that have a temporal
    # successor.
    i_after = np.flatnonzero(t_wafter)
    t_after_idx1 = np.zeros((Nn, Nn), dtype=bool)
    t_after_idx1[i_after, (i_after + 1) % Nn] = True

    # t_after_idx: temporal-exclusion mask, extended up to time_exclude_range
    # steps, restricted to strictly-forward (i < j) positions
    t_after_idx = np.zeros((Nn, Nn), dtype=bool)
    for n in range(1, time_exclude_range + 1):
        cols = (i_after + n) % Nn
        # keep only strictly-forward pairs, which is what the triu below
        # used to do -- doing it here avoids allocating a whole Nn-by-Nn
        # mask of ones purely to AND against.
        fwd = cols > i_after
        t_after_idx[i_after[fwd], cols[fwd]] = True

    if time_exclude_space:
        D[t_after_idx] = np.inf

    # -- compute adjacency matrix (k nearest neighbors per row)
    # argpartition, not a full argsort: only the k nearest per row are ever
    # used, and k is typically single digits while a full sort of every row
    # costs ~10x as much. Ties are still picked up by the D <= dmax pass
    # below, so which of several equal-distance neighbours lands in the
    # first k does not matter.
    part = np.argpartition(D, k - 1, axis=1)[:, :k]
    D_k = np.take_along_axis(D, part, axis=1)
    A = np.zeros((Nn, Nn), dtype=bool)
    rows = np.repeat(np.arange(Nn), k)
    A[rows, part.ravel()] = True

    # -- check for duplicate points: any other point tied with the k-th
    # nearest neighbor's distance is also included
    dmax = D_k.max(axis=1)[:, None]  # the k-th smallest distance per row
    A |= D <= dmax

    # -- get distance threshold (the stricter of the absolute and percentile caps).
    # NOTE: matches MATLAB's prctile(D(:), Prct) exactly by computing the
    # percentile over the *full* D, Inf entries included (the masked
    # diagonal/temporal-exclusion entries) -- not just the finite values.
    if max_neighbor_dist_prct >= 100.0:
        # At the default there is nothing to compute: D's masked entries are
        # inf, so the 100th percentile is always inf and the absolute cutoff
        # wins. Sorting all Nn^2 distances to learn that dominated this
        # function.
        prct_dist = np.inf
    else:
        prct_dist = _percentile_with_inf(D, max_neighbor_dist_prct)
    resolved_max_dist = min(prct_dist, max_neighbor_dist)
    par["max_neighbor_dist"] = resolved_max_dist

    # -- remove neighbors that exceed max distance
    A[D > resolved_max_dist] = False

    # -- exclude or retain temporal links as spatial links
    if time_exclude_space:
        A_space = A & ~t_after_idx
    else:
        A_space = A.copy()

    # -- enforce symmetry of spatial links
    if reciprocal:
        A_space = A_space & A_space.T
    else:
        A_space = A_space | A_space.T

    # -- (re-)incorporate temporal links
    A_final = t_after_idx1 | A_space

    g = nx.from_numpy_array(A_final, create_using=nx.DiGraph)
    return g, par
