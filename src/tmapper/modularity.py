"""Port of tmapper_tools/Qasym.m and calMod.m from the MATLAB toolbox."""

import numpy as np


def qasym(A, C):
    """Modularity measure of an asymmetric, weighted network.

    Port of MATLAB's ``Qasym.m``. Adapted from the canonical modularity
    definition (Fortunato 2010, Physics Reports) for asymmetric networks
    with weighted edges.

    Parameters
    ----------
    A : array_like, shape (N, N)
        Adjacency matrix, which may or may not be symmetric.
    C : array_like, shape (N,)
        Community assignment of each node.

    Returns
    -------
    float
        Modularity Q. Returns 0 for a zero-edge network (neither modular
        nor non-modular).
    """
    A = np.asarray(A, dtype=float)
    N_edges = A.sum()  # "2m" in other notations

    if N_edges == 0:
        return 0.0

    k_source = A.sum(axis=1, keepdims=True)  # source degree
    k_sink = A.sum(axis=0, keepdims=True)  # sink degree
    P_ij = k_source @ k_sink / N_edges  # null model

    C = np.asarray(C).ravel()
    same_community = C[:, None] == C[None, :]

    return float(np.sum((A - P_ij) * same_community) / N_edges)


def cal_mod(W, m0):
    """Modularity score of a (typically symmetric) network given a
    community assignment.

    Port of MATLAB's ``calMod.m``.

    Parameters
    ----------
    W : array_like, shape (N, N)
        Graph adjacency matrix.
    m0 : array_like, shape (N,)
        Categorical node labels / community assignment.

    Returns
    -------
    float
        Modularity score. Returns 0 for a zero-edge network (neither
        modular nor non-modular).
    """
    W = np.asarray(W, dtype=float)
    s = W.sum()  # sum of degrees of nodes (assuming symmetric)

    if s == 0:
        return 0.0

    gamma = 1  # scaling parameter
    B = (W - gamma * (W.sum(axis=1, keepdims=True) @ W.sum(axis=0, keepdims=True)) / s) / s
    B = (B + B.T) / 2  # symmetrize

    m0 = np.asarray(m0).ravel()
    same_community = m0[:, None] == m0[None, :]

    return float(B[same_community].sum())
