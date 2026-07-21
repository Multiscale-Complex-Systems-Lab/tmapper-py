"""Port of tmapper_tools/filtergraph.m from the MATLAB toolbox."""

import numpy as np
import networkx as nx

from ._shortest_path import all_pairs_distance


def filtergraph(g, d, *, reciprocal=True):
    """Contract a graph ``g`` into a simplified graph (a Mapper-style shape
    graph) by merging nodes that are within geodesic distance ``d`` of one
    another.

    Port of MATLAB's ``filtergraph.m``. This is the Mapper-style contraction
    step: nodes of ``g`` within geodesic distance ``d`` of each other are
    connected in an intermediate graph, and its connected components become
    the nodes of the simplified graph.

    Parameters
    ----------
    g : networkx.Graph or networkx.DiGraph
        Graph to simplify.
    d : float
        Threshold under which original nodes are collapsed together. Must
        be a positive real number.
    reciprocal : bool, default True
        Whether to require both the path length from x to y *and* from y
        to x to be under ``d`` (True), or either direction (False).

    Returns
    -------
    g_simp : networkx.DiGraph
        The simplified graph -- the attractor transition network.
    members : list of list
        ``members[n]`` is the list of original node labels (from ``g``)
        belonging to new node n. If ``g``'s nodes are the default 0..N-1
        integers straight from :func:`tknndigraph`, these are *positions*
        into your original data, not real time labels.
    nodesize : numpy.ndarray, shape (n_new,)
        Number of original-graph members in each new node.
    D_simp : numpy.ndarray, shape (n_new, n_new)
        Shortest "distance" (from the original graph) between each pair
        of new nodes' member sets.
    """
    if not isinstance(g, (nx.Graph, nx.DiGraph)):
        raise ValueError("g must be a networkx Graph or DiGraph.")
    if not (np.isscalar(d) and np.isreal(d) and d > 0):
        raise ValueError("d must be a positive real scalar.")

    nodelist = list(g.nodes())
    Nn = len(nodelist)
    idx_of = {node: i for i, node in enumerate(nodelist)}

    A = nx.to_numpy_array(g, nodelist=nodelist)  # weighted adjacency (weight 1 if unweighted)
    D = all_pairs_distance(g, nodelist)  # geodesic distance, inf if unreachable

    # -- connectivity within a distance threshold
    if reciprocal:
        A_ = (D < d) & (D.T < d)
    else:
        A_ = (D < d) | (D.T < d)
    np.fill_diagonal(A_, False)  # remove self-loops

    # -- create graph out of nodes within said threshold, find connected components
    g_ = nx.from_numpy_array(A_)
    components = list(nx.connected_components(g_))
    # sort components by their smallest member index, for deterministic/stable ordering
    components = sorted(components, key=min)

    idx_newnodes = np.empty(Nn, dtype=int)
    for new_idx, comp in enumerate(components):
        for i in comp:
            idx_newnodes[i] = new_idx
    n_new = len(components)

    # -- group original nodes by new-node label via a stable sort, so each
    # group is a contiguous run. This lets the block min/mean below use
    # ufunc.reduceat instead of an O(n_new^2) Python loop over Nn-sized
    # boolean masks (the previous approach was O(n_new^2 * Nn) and became
    # the dominant cost at realistic graph sizes).
    order = np.argsort(idx_newnodes, kind="stable")
    group_start = np.searchsorted(idx_newnodes[order], np.arange(n_new))
    group_bounds = np.append(group_start, Nn)

    members = [
        [nodelist[i] for i in order[group_bounds[n]:group_bounds[n + 1]]]
        for n in range(n_new)
    ]
    nodesize = np.array([len(m) for m in members])

    D_sorted = D[np.ix_(order, order)]
    A_sorted = A[np.ix_(order, order)]

    # -- define distance between new nodes: shortest path between member sets
    D_row = np.minimum.reduceat(D_sorted, group_start, axis=0)
    D_simp = np.minimum.reduceat(D_row, group_start, axis=1)

    # -- construct simplified graph: A_simp(n,m) = average connectivity between blocks
    A_row_sum = np.add.reduceat(A_sorted, group_start, axis=0)
    A_col_sum = np.add.reduceat(A_row_sum, group_start, axis=1)
    group_sizes = np.diff(group_bounds)
    A_simp = A_col_sum / np.outer(group_sizes, group_sizes)

    g_simp = nx.from_numpy_array(A_simp, create_using=nx.DiGraph)
    g_simp.remove_edges_from(nx.selfloop_edges(g_simp))  # OmitSelfLoops

    return g_simp, members, nodesize, D_simp
