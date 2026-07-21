"""Port of tmapper_tools/filtergraph.m from the MATLAB toolbox."""

import numpy as np
import networkx as nx


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
    D = nx.floyd_warshall_numpy(g, nodelist=nodelist)  # geodesic distance, inf if unreachable

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

    members = [
        [nodelist[i] for i in range(Nn) if idx_newnodes[i] == n] for n in range(n_new)
    ]
    nodesize = np.array([len(m) for m in members])

    # -- define distance between new nodes: shortest path between member sets
    D_simp = np.full((n_new, n_new), np.inf)
    for n in range(n_new):
        rows = idx_newnodes == n
        for m in range(n_new):
            cols = idx_newnodes == m
            block = D[np.ix_(rows, cols)]
            D_simp[n, m] = block.min()

    # -- construct simplified graph: A_simp(n,m) = average connectivity between blocks
    A_simp = np.zeros((n_new, n_new))
    for n in range(n_new):
        rows = idx_newnodes == n
        for m in range(n_new):
            cols = idx_newnodes == m
            block = A[np.ix_(rows, cols)]
            A_simp[n, m] = block.mean()

    g_simp = nx.from_numpy_array(A_simp, create_using=nx.DiGraph)
    g_simp.remove_edges_from(nx.selfloop_edges(g_simp))  # OmitSelfLoops

    return g_simp, members, nodesize, D_simp
