"""Port of tmapper_tools/tknndigraph.m from the MATLAB toolbox."""

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.spatial.distance import cdist

# Values per chunk when accumulating the low-memory percentile histogram
# (see _blocked_build). Caps the integer bin-index temporary at ~64 MB, so
# the histogram does not undo the point of blocking. Module-level so tests
# can shrink it and actually exercise the multi-chunk path on a small
# fixture, which no realistic test-sized input would otherwise reach.
_HIST_CHUNK = 8_000_000

# Bin count for that histogram. Sets the precision of the low-memory
# percentile cutoff: exact to one bin width, i.e. one part in a million of
# the bounding-box diagonal. Module-level so tests can reason about which
# bin a given distance lands in.
_HIST_BINS = 1_000_000


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
    low_memory=False,
    block_size=None,
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
    low_memory : bool, default False
        Build the graph from ``X`` a block of rows at a time, never
        allocating an (N, N) array. Peak memory becomes O(block_size * N)
        instead of O(N^2), which is what makes large N feasible at all.
        Requires coordinates, not a precomputed distance matrix -- with D
        in hand the full matrix already exists and there is nothing to
        save.
    block_size : int, optional
        Rows per block in ``low_memory`` mode. Larger is slightly faster,
        smaller uses less memory (peak tracks block_size * N * 8 bytes).
        Defaults to roughly 400 MB per block.

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

    # -- low-memory path: never form the (N, N) distance matrix at all.
    # Must branch before D is built, since building it is the thing being
    # avoided.
    if low_memory:
        nr_lm, nc_lm = X_or_D.shape
        if nr_lm == nc_lm and nr_lm > 1 and np.allclose(X_or_D, X_or_D.T):
            raise ValueError(
                "low_memory builds distances block by block, so it needs the "
                "coordinates X (N, d), not a precomputed (N, N) distance "
                "matrix -- with D in hand the full matrix already exists and "
                "there is nothing left to save."
            )
        return _blocked_build(X_or_D, k, tidx, par, block_size)

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
    # The percentile is taken over FINITE distances only. By this point D
    # holds inf wherever a pair is already barred from being a spatial
    # neighbor -- the diagonal, and every temporal pair within
    # time_exclude_range -- and those infs sit at the top of the
    # distribution, so counting them dragged the cutoff upward and made it
    # more permissive than asked for. With texclude=30 on 1500 points the
    # 99th percentile came out as inf outright: a request to drop the most
    # distant 1% of neighbors silently applied no cutoff at all.
    # Matches MATLAB tmapper2 v2.2, which made the same correction.
    if max_neighbor_dist_prct >= 100.0:
        # At the default there is nothing to compute: D's masked entries are
        # inf, so the 100th percentile is always inf and the absolute cutoff
        # wins. Sorting all Nn^2 distances to learn that dominated this
        # function.
        prct_dist = np.inf
    else:
        finite_D = D[np.isfinite(D)]
        if finite_D.size == 0:
            prct_dist = np.inf  # nothing finite to take a percentile of
        else:
            # method="hazen": numpy's DEFAULT ("linear") uses plotting
            # positions i/(n-1), while MATLAB's prctile uses (i-0.5)/n.
            # Those genuinely differ -- prctile([1 2 3 4 5], 95) is 5 but
            # np.percentile(..., 95) is 4.8 -- so the port had been
            # resolving a slightly different cutoff than the MATLAB
            # original it is checked against. Hazen IS (i-0.5)/n, and
            # reproduces prctile exactly.
            prct_dist = float(
                np.percentile(finite_D, max_neighbor_dist_prct, method="hazen")
            )
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


def _blocked_build(X, k, tidx, par, block_size):
    """tknndigraph's low-memory path: same graph, built a block of rows at
    a time so no (N, N) array is ever allocated.

    Every step except the percentile is row-local, so it blocks cleanly.
    The percentile needs the global distance distribution, which is exactly
    what blocking refuses to hold, so it is accumulated as a histogram
    during the same pass.
    """
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    if k >= N:
        raise ValueError(f"k must be smaller than the number of points ({N}).")
    if np.isnan(X).any():
        raise ValueError(
            "X_or_D contains NaN values (or produces them once converted to a "
            "distance matrix). Remove or impute missing data before calling "
            "tknndigraph, e.g. via numpy or pandas dropna."
        )

    B = block_size
    if B is None:
        B = max(1, min(N, int(50e6 // max(N, 1))))  # ~400 MB of float64 per block
    par["block_size"] = int(B)

    tex = par["time_exclude_range"]
    time_exclude_space = par["time_exclude_space"]
    max_nd = par["max_neighbor_dist"]
    prct = par["max_neighbor_dist_prct"]

    # the temporal band, same construction as the dense path: pairs
    # (i, i+n) for n = 1..tex wherever i has a temporal successor
    t_wafter = np.roll(tidx, -1) - 1 == tidx
    i_after = np.flatnonzero(t_wafter)
    succ = i_after[i_after + 1 < N]
    rows1 = succ
    cols1 = succ + 1

    is_after = np.zeros(N, dtype=bool)
    is_after[i_after] = True

    # Histogram range without an extra pass: the bounding-box diagonal is a
    # hard upper bound on any pairwise Euclidean distance, and costs O(N*d).
    need_prct = prct < 100.0
    n_bins = _HIST_BINS
    if need_prct:
        hi = float(np.linalg.norm(X.max(axis=0) - X.min(axis=0)))
        if not np.isfinite(hi) or hi <= 0:
            hi = 1.0
        # edges only serves the final edges[ib + 1] lookup below -- the
        # per-block accumulation derives its bin index arithmetically.
        edges = np.linspace(0.0, hi, n_bins + 1)
        counts = np.zeros(n_bins, dtype=np.int64)

    rows_all, cols_all, dist_all = [], [], []
    for i0 in range(0, N, B):
        i1 = min(i0 + B, N)
        idx = np.arange(i0, i1)
        Db = cdist(X[idx], X)  # (B, N) -- the only large array

        Db[np.arange(idx.size), idx] = np.inf  # self-loops
        if time_exclude_space:
            for n in range(1, tex + 1):
                r = idx[is_after[idx] & (idx + n < N)]
                if r.size:
                    Db[r - i0, r + n] = np.inf

        if need_prct:
            # Bin by arithmetic rather than np.histogram. Handing np.histogram
            # an ARRAY of bin edges forces its general path -- a searchsorted
            # over a million edges for every one of ~50M values per block --
            # and that single call was ~98% of this function's runtime (37.7s
            # per block, against 143ms for the cdist that produced the data).
            # The edges are uniform by construction, so floor(v / hi * n_bins)
            # gives the same bin directly: ~71x faster, counts bit-identical.
            #   Chunked over rows so the intp index array stays bounded; the
            # obvious `Db[np.isfinite(Db)]` for the whole block would also add
            # a ~400 MB copy of it.
            rows_per_chunk = max(1, _HIST_CHUNK // max(Db.shape[1], 1))
            scale = n_bins / hi
            for c0 in range(0, Db.shape[0], rows_per_chunk):
                vals = Db[c0:c0 + rows_per_chunk]
                vals = vals[np.isfinite(vals)]
                if not vals.size:
                    continue
                ib_chunk = (vals * scale).astype(np.intp)
                # Not defensive. hi is the bounding-box diagonal, so a real
                # pair can sit exactly at hi (two opposite corners), and
                # whether hi * scale then lands on n_bins -- one past the
                # last bin -- comes down to how the two round: it did for
                # ~22% of random point sets in a quick scan. Without the
                # clip np.bincount returns an array one element longer than
                # counts and the += raises.
                np.clip(ib_chunk, 0, n_bins - 1, out=ib_chunk)
                counts += np.bincount(ib_chunk, minlength=n_bins)

        part = np.argpartition(Db, k - 1, axis=1)[:, :k]
        dmax = np.take_along_axis(Db, part, axis=1).max(axis=1)[:, None]
        keep = Db <= dmax  # the k nearest, plus any ties at the k-th distance
        rr, cc = np.nonzero(keep)
        rows_all.append(rr + i0)
        cols_all.append(cc)
        dist_all.append(Db[rr, cc])

    # -- resolve the distance threshold. The percentile comes from the
    # histogram above: exact to one bin width (range / 1e6), far below any
    # scale that moves an edge.
    if need_prct:
        total = counts.sum()
        target = prct / 100.0 * total
        cum = np.cumsum(counts)
        ib = int(np.searchsorted(cum, target))
        ib = min(ib, n_bins - 1)
        prct_dist = float(edges[ib + 1])  # upper edge: inclusive, matching D <= thr
    else:
        prct_dist = np.inf
    resolved = min(prct_dist, max_nd)
    par["max_neighbor_dist"] = resolved

    rows = np.concatenate(rows_all) if rows_all else np.array([], dtype=int)
    cols = np.concatenate(cols_all) if cols_all else np.array([], dtype=int)
    dist = np.concatenate(dist_all) if dist_all else np.array([], dtype=float)
    ok = dist <= resolved
    A = sp.csr_matrix(
        (np.ones(int(ok.sum()), dtype=bool), (rows[ok], cols[ok])),
        shape=(N, N), dtype=bool,
    )

    if time_exclude_space:
        band_r, band_c = [], []
        for n in range(1, tex + 1):
            r = i_after[i_after + n < N]
            band_r.append(r)
            band_c.append(r + n)
        if band_r:
            br = np.concatenate(band_r)
            bc = np.concatenate(band_c)
            band = sp.csr_matrix(
                (np.ones(br.size, dtype=bool), (br, bc)), shape=(N, N), dtype=bool
            )
            # Subtract the band rather than A & ~band: the complement of
            # a sparse matrix is DENSE, which would undo the point of
            # blocking.
            #   This is defensive rather than redundant. The blocked pass
            # already set band distances to inf, so those pairs normally
            # cannot be selected -- but if an entire row is inf (every
            # candidate excluded), dmax is inf too and `Db <= dmax` picks
            # them up again. Removing this line passes the test sweep,
            # because that degenerate row does not arise there.
            A = A > band
    A_space = A.multiply(A.T) if par["reciprocal"] else (A + A.T)
    A_space = A_space.astype(bool)

    temporal = sp.csr_matrix(
        (np.ones(rows1.size, dtype=bool), (rows1, cols1)), shape=(N, N), dtype=bool
    )
    A_final = (A_space + temporal).astype(bool)

    g = nx.from_scipy_sparse_array(A_final, create_using=nx.DiGraph)
    return g, par
