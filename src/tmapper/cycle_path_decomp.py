"""Port of tmapper_tools/CyclePathDecomp.m from the MATLAB toolbox.

The MATLAB original's diagonal cluster-block overlay (``addDiagBlock.m``)
is not ported literally -- it's tightly coupled to MATLAB axes internals
(``ax.Children(end).CData``) with no Python equivalent. The same visual
annotation is reproduced directly here with matplotlib patches, reusing
:func:`tmapper.graph_utils.find_blocks`.
"""

import numpy as np
import networkx as nx

from .cycle_count2p import cycle_count2p
from .cycle_cluster import cycle_cluster
from .cycle_cluster_conn import cycle_cluster_conn
from .cycles_to_paths import cycles_to_paths
from .graph_utils import find_blocks


def _add_diag_block(ax, cluster_labels, *, reordermat=True):
    from matplotlib.patches import Rectangle

    cluster_labels = np.asarray(cluster_labels)
    order = np.argsort(cluster_labels, kind="stable") if reordermat else np.arange(len(cluster_labels))
    sorted_labels = cluster_labels[order]

    for label in np.unique(sorted_labels):
        starts, _, sizes = find_blocks((sorted_labels == label).astype(int))
        for s, size in zip(starts, sizes):
            ax.add_patch(Rectangle((s - 0.5, s - 0.5), size, size,
                                    fill=False, edgecolor="k", linewidth=2))


def cycle_path_decomp(dg, *, clusterthres=0.5, plotmat=True, plotmds=False,
                       plothist=False, reordermat=True):
    """Cycle-path decomposition of a directed graph.

    Port of MATLAB's ``CyclePathDecomp.m``. Cycles on the graph are
    first computed, then clustered such that cycles with ``clusterthres``
    amount of overlap belong to the same cluster. Boundaries between
    clusters of cycles (nodes where one cycle enters into another) are
    used to cut cycles into paths, which are then clustered themselves.

    Parameters
    ----------
    dg : networkx.DiGraph
        The graph to decompose.
    clusterthres : float, default 0.5
        Overlap threshold for cycle/path clustering.
    plotmat : bool, default True
        Whether to plot the overlap matrices (with cluster-block
        outlines) for diagnostics.
    plotmds : bool, default False
        Whether to plot a 2D classical-MDS projection of cycles/paths.
    plothist : bool, default False
        Whether to plot linkage-distance histograms.
    reordermat : bool, default True
        Whether to reorder the overlap matrices by cluster assignment.

    Returns
    -------
    allbd : list
        All boundary points between cycle clusters (used to cut cycles
        into paths).
    pclusters_nodes : list of list
        Nodes in each path cluster.
    pclusters_interior : list of list
        Interior (non-boundary) nodes of each path cluster.
    pclusters_boundary : list of set
        Boundary nodes of each path cluster.
    pcluster_conn_dir : list of list of set
        Directed boundary connectivity between path clusters.
    pclusters_intcrtpts : list of list
        Interior critical points of each path cluster.
    allupath : list of list
        All unique decomposed paths.
    Tp : numpy.ndarray
        Cluster assignment of each path in ``allupath``.
    """
    A = nx.to_numpy_array(dg, nodelist=list(dg.nodes()), weight=None)
    _, _, _, allcycles = cycle_count2p(A)

    T, ax1 = cycle_cluster(allcycles, clusterthres, plotmat=plotmat, plotmds=plotmds,
                           plothist=plothist, reordermat=reordermat, return_ax=True)
    _, _, _, clusters_boundary, _, _, _ = cycle_cluster_conn(dg, allcycles, T)

    if plotmat and ax1 is not None:
        _add_diag_block(ax1, T, reordermat=reordermat)
    print(f"# cycle clusters = {int(T.max()) if len(T) else 0}")

    allbd = sorted(set().union(*clusters_boundary)) if clusters_boundary else []
    allupath = cycles_to_paths(allcycles, allbd)

    Tp, ax2 = cycle_cluster(allupath, clusterthres, plotmat=plotmat, plotmds=plotmds,
                            plothist=plothist, reordermat=reordermat, return_ax=True)
    (_, pcluster_conn_dir, pclusters_nodes, pclusters_boundary,
     pclusters_interior, _, pclusters_intcrtpts) = cycle_cluster_conn(dg, allupath, Tp)

    if plotmat and ax2 is not None:
        _add_diag_block(ax2, Tp, reordermat=reordermat)
    print(f"# path clusters = {int(Tp.max()) if len(Tp) else 0}")

    return (allbd, pclusters_nodes, pclusters_interior, pclusters_boundary,
            pcluster_conn_dir, pclusters_intcrtpts, allupath, Tp)
