"""Port of tmapper_tools/TCMdistance.m from the MATLAB toolbox."""

import numpy as np
import networkx as nx

from ._shortest_path import all_pairs_distance


def tcm_distance(g, nodet, weighted=False):
    """Compute a temporal connectivity matrix: for every pair of original
    time points, the shortest path length between the nodes (of a
    simplified graph ``g``) that contain them.

    Port of MATLAB's ``TCMdistance.m``. Sizes its output by the full range
    covered by ``nodet`` (``max - min + 1``), not just the count of
    distinct points referenced -- so time points genuinely not covered by
    any node in ``nodet`` correctly stay ``NaN`` rather than being silently
    zero-filled.

    Parameters
    ----------
    g : networkx.Graph or networkx.DiGraph
        The simplified graph.
    nodet : sequence of sequence of int
        ``nodet[i]`` gives the original time-point indices belonging to
        node i of ``g`` (in the same node order as ``g.nodes()``).
    weighted : bool, default False
        Whether to use ``g``'s edge weights. If False, every edge is
        treated as weight 1 (hop count).

    Returns
    -------
    numpy.ndarray, shape (Nt, Nt)
        The temporal connectivity matrix, where Nt = max(nodet) - min(nodet) + 1.
        Entries for time points never covered by any node stay NaN.
    """
    nodelist = list(g.nodes())
    all_t = np.concatenate([np.asarray(list(nt), dtype=int) for nt in nodet]) if nodet else np.array([], dtype=int)
    if all_t.size == 0:
        return np.zeros((0, 0))
    t_0 = all_t.min()
    Nt = all_t.max() - t_0 + 1

    distmat = all_pairs_distance(g, nodelist, weight="weight" if weighted else None)

    tcm = np.full((Nt, Nt), np.nan)

    n_nodes = len(nodelist)
    idx_lists = [np.asarray(list(nt), dtype=int) - t_0 for nt in nodet]

    # When each time point belongs to exactly one node -- which is what
    # filtergraph produces, since its members are connected components and
    # so partition the time points -- every (i, j) block below is written
    # exactly once into an all-NaN matrix, so the fmin is a no-op. The
    # double loop then collapses into a lookup: find each time point's
    # node, and read off the node-to-node distance. The loop pays Python
    # overhead on n_nodes^2/2 fancy-index blocks, so this is a very large
    # win (150s -> under a second on a 2672-node network).
    #   The loop is kept for the general case, where a caller may pass
    # overlapping membership and the fmin genuinely arbitrates.
    if all_t.size == np.unique(all_t).size:
        node_of_t = np.full(Nt, -1, dtype=int)
        for i in range(n_nodes):
            node_of_t[idx_lists[i]] = i
        covered = np.flatnonzero(node_of_t >= 0)  # uncovered stay NaN
        node_idx = node_of_t[covered]
        # Filled in row blocks rather than as one indexed assignment: the
        # latter builds a second full Nt-by-Nt array for the right-hand
        # side before writing it, doubling peak memory on the largest
        # thing here.
        rows_per_block = max(1, int(2e7 // max(covered.size, 1)))
        for b0 in range(0, covered.size, rows_per_block):
            sl = covered[b0:b0 + rows_per_block]
            tcm[np.ix_(sl, covered)] = distmat[
                np.ix_(node_idx[b0:b0 + rows_per_block], node_idx)
            ]
        return tcm

    for i in range(n_nodes):
        ii = idx_lists[i]
        tcm[np.ix_(ii, ii)] = 0

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            ii, jj = idx_lists[i], idx_lists[j]
            block_ij = tcm[np.ix_(ii, jj)]
            tcm[np.ix_(ii, jj)] = np.fmin(block_ij, distmat[i, j])
            block_ji = tcm[np.ix_(jj, ii)]
            tcm[np.ix_(jj, ii)] = np.fmin(block_ji, distmat[j, i])

    return tcm
