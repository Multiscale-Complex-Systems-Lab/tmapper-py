"""Shared all-pairs shortest-path helper.

Uses scipy's compiled Dijkstra (via ``scipy.sparse.csgraph.shortest_path``)
instead of networkx's ``floyd_warshall_numpy``. The latter is O(n^3) with
substantial per-iteration numpy overhead and becomes the dominant cost at
realistic graph sizes (a few thousand nodes); our graphs are sparse (a
handful of edges per node from the k-NN construction), where Dijkstra from
every source is asymptotically and practically much faster.
"""

import networkx as nx
from scipy.sparse.csgraph import shortest_path


def all_pairs_distance(g, nodelist, weight="weight"):
    """All-pairs shortest-path distance matrix for ``g``, ordered by
    ``nodelist``. ``weight=None`` treats every edge as weight 1 (hop
    count); ``weight="weight"`` (default) uses ``g``'s edge weights.
    Unreachable pairs are ``inf``, matching ``nx.floyd_warshall_numpy``.
    """
    adj = nx.to_scipy_sparse_array(g, nodelist=nodelist, weight=weight, dtype=float)
    return shortest_path(adj, method="auto", directed=True)
