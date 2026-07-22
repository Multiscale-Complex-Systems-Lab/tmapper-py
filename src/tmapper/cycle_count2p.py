"""Port of tmapper_tools/CycleCount2p.m and reorgCycles.m from the
MATLAB toolbox."""

import numpy as np
import networkx as nx


def cycle_count2p(A, *, simple=True):
    """Estimate the number of cycles of different lengths by finding the
    smallest cycle passing through every pair of vertices.

    Port of MATLAB's ``CycleCount2p.m``. Only counts unique cycles, and
    by default only unique simple cycles (no repeated nodes other than
    the start/end).

    Parameters
    ----------
    A : array_like, shape (N, N)
        Adjacency matrix for a simple directed graph.
    simple : bool, default True
        Whether to only count simple cycles.

    Returns
    -------
    cyc_count : numpy.ndarray
        Number of cycles per unique length (shortest to longest).
    cyc_len : numpy.ndarray
        The unique cycle lengths themselves (shortest to longest).
    cyc_path : list of numpy.ndarray
        ``cyc_path[m]`` is an (Nm, Lm) array of the Nm cycles of length
        ``cyc_len[m]``.
    all_cycles : list of list
        Every cycle's path, flattened into a single list.
    """
    g = nx.from_numpy_array(np.asarray(A), create_using=nx.DiGraph)
    n = g.number_of_nodes()

    all_closed = []
    for s in range(n):
        for t in range(s + 1, n):
            if not (nx.has_path(g, s, t) and nx.has_path(g, t, s)):
                continue
            spf = nx.shortest_path(g, s, t)  # forward path
            spb = nx.shortest_path(g, t, s)  # backward path
            cycle = spf + spb[1:-1]
            if not simple or len(set(cycle)) == len(cycle):
                root_idx = cycle.index(min(cycle))
                all_closed.append(cycle[root_idx:] + cycle[:root_idx])  # start at smallest node

    cyc_length = np.array([len(c) for c in all_closed], dtype=int)
    cyc_len = np.unique(cyc_length) if cyc_length.size else np.array([], dtype=int)

    cyc_path = []
    cyc_count = []
    for L in cyc_len:
        rows = sorted({tuple(c) for c, l in zip(all_closed, cyc_length) if l == L})
        cyc_path.append(np.array(rows, dtype=int))
        cyc_count.append(len(rows))
    cyc_count = np.array(cyc_count, dtype=int)

    all_cycles = [list(row) for block in cyc_path for row in block]

    return cyc_count, cyc_len, cyc_path, all_cycles


def reorg_cycles(cyc_path):
    """Unpack the ``cyc_path`` output of :func:`cycle_count2p` into a flat
    list where each element is a single cycle's path.

    Port of MATLAB's ``reorgCycles.m``.
    """
    return [list(row) for block in cyc_path for row in np.asarray(block)]
