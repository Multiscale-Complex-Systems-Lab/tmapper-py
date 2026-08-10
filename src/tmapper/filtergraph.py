"""Port of tmapper_tools/filtergraph.m from the MATLAB toolbox."""

import numpy as np
import networkx as nx
import scipy.sparse as sp

from ._shortest_path import all_pairs_distance


def filtergraph(g, d, *, reciprocal=True, compute_dsimp=True):
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
    compute_dsimp : bool, default True
        Whether to compute the ``D_simp`` output. It is the most expensive
        remaining step and nothing in the toolbox itself consumes it, so
        pass ``False`` to skip it and get ``None`` back.
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
    D_simp : numpy.ndarray, shape (n_new, n_new), or None
        Shortest "distance" (from the original graph) between each pair
        of new nodes' member sets. ``None`` when ``compute_dsimp=False``.
    """
    if not isinstance(g, (nx.Graph, nx.DiGraph)):
        raise ValueError("g must be a networkx Graph or DiGraph.")
    if not (np.isscalar(d) and np.isreal(d) and d > 0):
        raise ValueError("d must be a positive real scalar.")

    nodelist = list(g.nodes())
    Nn = len(nodelist)
    idx_of = {node: i for i, node in enumerate(nodelist)}

    A = nx.to_numpy_array(g, nodelist=nodelist)  # weighted adjacency (weight 1 if unweighted)

    # -- connectivity within a distance threshold.
    # The geodesics are only ever compared against d, and for an UNWEIGHTED
    # graph a geodesic is a hop count -- so "within distance d" is just
    # "reachable within a bounded number of hops", which sparse boolean
    # products give without the all-pairs shortest path (the single most
    # expensive step here: ~5.9s of 7.8s at 8000 nodes).
    #   Weighted graphs need real shortest paths, and so does D_simp, so
    # those keep the dense route.
    A_bool = A != 0
    is_unweighted = bool(np.all(A[A_bool] == 1)) if A_bool.any() else True
    # hop counts are integers, so D < d means D <= hmax:
    #   d integer    -> hmax = d-1       (D<3 admits 1 and 2 hops)
    #   d fractional -> hmax = floor(d)  (D<3.5 admits 1, 2 and 3)
    hmax = int(d) - 1 if float(d) == int(d) else int(np.floor(d))
    use_reach = is_unweighted and np.isfinite(d) and not compute_dsimp

    D = None
    if use_reach:
        Ab = sp.csr_matrix(A_bool)
        R = Ab.copy() if hmax >= 1 else sp.csr_matrix(Ab.shape, dtype=bool)
        P = Ab
        for _ in range(2, hmax + 1):
            P = (P @ Ab).astype(bool)
            newR = (R + P).astype(bool)
            if newR.nnz == R.nnz:
                break  # reachability saturated
            R = newR
            if R.nnz > 0.25 * Nn * Nn:
                use_reach = False  # densifying: sparsity has stopped paying
                break

    if use_reach:
        Rd = R.toarray()
        A_ = (Rd & Rd.T) if reciprocal else (Rd | Rd.T)
    else:
        D = all_pairs_distance(g, nodelist)  # geodesic distance, inf if unreachable
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

    group_sizes = np.diff(group_bounds)

    # -- construct simplified graph: A_simp(n,m) = average connectivity
    # between blocks. Summing over blocks IS a matrix product: with S the
    # (n_new, Nn) group-indicator matrix, S @ A @ S.T is the block-sum in
    # one sparse call, instead of two reduceat passes over a reordered
    # copy of A. That also drops the A[np.ix_(order, order)] fancy-index
    # copy, a full Nn-by-Nn temporary.
    S = sp.csr_matrix(
        (np.ones(Nn), (idx_newnodes, np.arange(Nn))), shape=(n_new, Nn)
    )
    A_simp = np.asarray((S @ sp.csr_matrix(A) @ S.T).todense())
    A_simp /= np.outer(group_sizes, group_sizes)

    # -- define distance between new nodes: shortest path between member
    # sets. Block-min has no matrix-product equivalent, so it keeps the
    # reduceat pass -- but only when the caller wants it.
    if compute_dsimp:
        if D is None:
            D = all_pairs_distance(g, nodelist)
        D_sorted = D[np.ix_(order, order)]
        D_row = np.minimum.reduceat(D_sorted, group_start, axis=0)
        D_simp = np.minimum.reduceat(D_row, group_start, axis=1)
    else:
        D_simp = None

    g_simp = nx.from_numpy_array(A_simp, create_using=nx.DiGraph)
    g_simp.remove_edges_from(nx.selfloop_edges(g_simp))  # OmitSelfLoops

    return g_simp, members, nodesize, D_simp
