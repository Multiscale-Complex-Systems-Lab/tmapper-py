"""Port of tmapper_tools/cknngraph.m from the MATLAB toolbox."""

import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist


def cknngraph(X_or_D, k, delta, *, average=False):
    """Construct a graph based on continuous k-nearest neighbors (Berry &
    Sauer, 2019).

    Port of MATLAB's ``cknngraph.m``. Connects points x, y iff::

        D(x, y) < delta * sqrt(D(x, x_k) * D(y, y_k))

    where ``D(~, ~)`` is the distance and ``~_k`` is the k-th nearest
    neighbor of ``~``.

    Parameters
    ----------
    X_or_D : array_like, shape (N, d) or (N, N)
        Either raw coordinates (N points in d-dim space) or a precomputed
        (N, N) distance matrix. Auto-detected: treated as a distance
        matrix only if square *and* symmetric.
    k : int
        Used to normalize distance by local point density. Must be a
        positive integer strictly less than N.
    delta : float
        Threshold for linking two nodes (must be a positive real
        number).
    average : bool, default False
        Whether ``x_k`` is the average distance over the first k nearest
        neighbors (True) or just the k-th nearest neighbor (False).

    Returns
    -------
    networkx.Graph
        Unweighted undirected graph, one node per input point.
    """
    X_or_D = np.asarray(X_or_D, dtype=float)
    if X_or_D.ndim != 2:
        raise ValueError("X_or_D must be a 2D array.")
    if not (np.isscalar(k) and float(k) == round(float(k)) and k >= 1):
        raise ValueError("k must be a positive integer scalar.")
    k = int(k)
    if not (np.isscalar(delta) and np.isreal(delta) and delta > 0):
        raise ValueError("delta must be a positive real scalar.")

    nr, nc = X_or_D.shape
    if nr != nc or not np.allclose(X_or_D, X_or_D.T):
        D = cdist(X_or_D, X_or_D)
    else:
        D = X_or_D.copy()
    Nn = D.shape[0]

    if k >= Nn:
        raise ValueError(f"k must be smaller than the number of points ({Nn}).")

    # -- find nearest neighbors and compute distance
    D_sorted = np.sort(D, axis=1)
    if average:
        Dk = D_sorted[:, 1:1 + k].mean(axis=1)  # col 0 is distance to self
    else:
        Dk = D_sorted[:, k]

    # -- normalize D by local density
    D_norm = D / np.sqrt(np.outer(Dk, Dk))

    # -- construct graph from adjacency matrix
    A = D_norm < delta
    np.fill_diagonal(A, False)

    return nx.from_numpy_array(A)
