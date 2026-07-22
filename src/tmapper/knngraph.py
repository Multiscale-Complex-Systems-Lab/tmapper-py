"""Port of tmapper_tools/knngraph.m from the MATLAB toolbox."""

import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist


def knngraph(X_or_D, k, *, reciprocal=True):
    """Construct an undirected graph based on k-nearest neighbors: each
    point is a node, linked to its k nearest neighbors.

    Port of MATLAB's ``knngraph.m``. A standalone graph builder with no
    temporal component (unlike :func:`tknndigraph`).

    Parameters
    ----------
    X_or_D : array_like, shape (N, d) or (N, N)
        Either raw coordinates (N points in d-dim space) or a precomputed
        (N, N) distance matrix. Auto-detected: treated as a distance
        matrix only if square *and* symmetric.
    k : int
        Number of nearest neighbors. Must be a positive integer strictly
        less than N.
    reciprocal : bool, default True
        Whether to require a k-NN link to be mutual (both directions) to
        be kept.

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

    nr, nc = X_or_D.shape
    if nr != nc or not np.allclose(X_or_D, X_or_D.T):
        D = cdist(X_or_D, X_or_D)
    else:
        D = X_or_D.copy()
    Nn = D.shape[0]

    if k >= Nn:
        raise ValueError(f"k must be smaller than the number of points ({Nn}).")

    order = np.argsort(D, axis=1, kind="stable")
    A = np.zeros((Nn, Nn), dtype=bool)
    rows = np.repeat(np.arange(Nn), k)
    cols = order[:, 1:1 + k].ravel()  # skip column 0: self (distance 0)
    A[rows, cols] = True

    if reciprocal:
        A = A & A.T
    else:
        A = A | A.T

    return nx.from_numpy_array(A)
