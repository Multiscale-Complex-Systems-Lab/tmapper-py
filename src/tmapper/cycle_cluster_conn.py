"""Port of tmapper_tools/CycleClusterConn.m from the MATLAB toolbox."""

import numpy as np


def cycle_cluster_conn(dg, allcycles, cluster_idx):
    """Connectivity between M clusters of N cycles.

    Port of MATLAB's ``CycleClusterConn.m``.

    Parameters
    ----------
    dg : networkx.DiGraph
        The directed graph the cycles/paths live on.
    allcycles : sequence of sequence of int
        ``allcycles[n]`` is the path of one cycle.
    cluster_idx : array_like of int
        Cluster label of each cycle (e.g. from :func:`cycle_cluster`).

    Returns
    -------
    cluster_conn : list of list of set
        ``cluster_conn[i][j]`` (== ``cluster_conn[j][i]``) contains
        nodes on the boundary between loop-clusters i and j.
    cluster_conn_dir : list of list of set
        ``cluster_conn_dir[i][j]`` contains nodes in the boundary of
        cluster j that receive a link from at least one node of cluster
        i (excluding shared nodes).
    clusters_nodes : list of list
        All nodes in each cluster (first-seen order).
    clusters_boundary : list of set
        Boundary nodes of each cluster.
    clusters_interior : list of list
        Nodes of each cluster that are not on its boundary.
    clusters_crtpts : list of list
        Critical points (nodes with more than one source or target) in
        each cluster.
    clusters_intcrtpts : list of list
        Critical points in each cluster that are not on its boundary.
    """
    cluster_idx = np.asarray(cluster_idx)
    clusters = np.unique(cluster_idx)
    n_clusters = len(clusters)

    out_degree = dict(dg.out_degree())
    in_degree = dict(dg.in_degree())
    crtpts = {n for n in dg.nodes() if out_degree[n] > 1 or in_degree[n] > 1}

    clusters_nodes = [None] * n_clusters
    clusters_crtpts = [None] * n_clusters
    clusters_intcrtpts = [None] * n_clusters
    clusters_inneighbors = [None] * n_clusters
    clusters_outneighbors = [None] * n_clusters
    clusters_boundary = [None] * n_clusters
    clusters_interior = [None] * n_clusters

    for ii, c in enumerate(clusters):
        cycles_ii = [allcycles[k] for k in range(len(allcycles)) if cluster_idx[k] == c]

        seen, seen_set = [], set()
        for cyc in cycles_ii:
            for node in cyc:
                if node not in seen_set:
                    seen_set.add(node)
                    seen.append(node)
        clusters_nodes[ii] = seen

        crtpts_ii = [n for n in seen if n in crtpts]
        clusters_crtpts[ii] = crtpts_ii

        innbg, outnbg = set(), set()
        for cp in crtpts_ii:
            innbg |= set(dg.predecessors(cp))
            outnbg |= set(dg.successors(cp))
        innbg -= seen_set
        outnbg -= seen_set

        bd = set()
        for n in innbg:
            bd |= set(dg.successors(n))
        for n in outnbg:
            bd |= set(dg.predecessors(n))
        bd &= seen_set

        clusters_inneighbors[ii] = innbg
        clusters_outneighbors[ii] = outnbg
        clusters_boundary[ii] = bd
        clusters_interior[ii] = [n for n in seen if n not in bd]
        clusters_intcrtpts[ii] = [n for n in crtpts_ii if n not in bd]

    cluster_conn = [[None] * n_clusters for _ in range(n_clusters)]
    cluster_conn_dir = [[None] * n_clusters for _ in range(n_clusters)]
    cluster_conn_in = [[None] * n_clusters for _ in range(n_clusters)]
    cluster_conn_out = [[None] * n_clusters for _ in range(n_clusters)]

    for ii in range(n_clusters):
        for jj in range(n_clusters):
            cluster_conn[ii][jj] = clusters_boundary[ii] & set(clusters_nodes[jj])
            cluster_conn_in[ii][jj] = clusters_inneighbors[jj] & set(clusters_nodes[ii])
            cluster_conn_out[ii][jj] = clusters_outneighbors[ii] & set(clusters_nodes[jj])

    for ii in range(n_clusters):
        for jj in range(n_clusters):
            bd = set()
            for nn in cluster_conn[ii][jj]:
                preds = set(dg.predecessors(nn))
                if preds & cluster_conn_in[ii][jj]:
                    bd.add(nn)
            cluster_conn_dir[ii][jj] = bd & cluster_conn[ii][jj]

    return (cluster_conn, cluster_conn_dir, clusters_nodes, clusters_boundary,
            clusters_interior, clusters_crtpts, clusters_intcrtpts)
