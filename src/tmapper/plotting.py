"""Port of tmapper_tools/plottmgraph.m and plotgraphtcm.m from the MATLAB
toolbox."""

import numpy as np
import networkx as nx
from scipy import stats

from .labeling import find_node_label
from .tcm_distance import tcm_distance
from ._shortest_path import all_pairs_distance


def _rescale(x, lo, hi):
    """Linearly map x's [min, max] to [lo, hi], matching MATLAB's rescale."""
    x = np.asarray(x, dtype=float)
    xmin, xmax = x.min(), x.max()
    return lo + (x - xmin) / (xmax - xmin) * (hi - lo)


def _normalize_positions(pos):
    """Rescale a layout so the farthest node sits at radius 1."""
    pts = np.array(list(pos.values()))
    max_r = np.max(np.linalg.norm(pts, axis=1))
    scale = 1.0 / max_r if max_r > 0 else 1.0
    return {n: xy * scale for n, xy in pos.items()}


def _expand_center(pos, gamma):
    """Log-radial de-clumping: remap each node's radius r -> log1p(gamma*r).

    Force-directed layouts (spring/ForceAtlas2) tend to pack the bulk of
    nodes into a dense central clump with only a few outliers stretching
    the frame. Since log1p(gamma*r)/r is largest at small r and flattens
    out at large r, this pushes the dense core outward proportionally
    more than the already-spread-out periphery, without disturbing
    angular position or relative radial order.
    """
    out = {}
    for n, xy in pos.items():
        r = np.linalg.norm(xy)
        if r > 0:
            rn = np.log1p(gamma * r)
            out[n] = xy * (rn / r)
        else:
            out[n] = xy
    return out


def plot_tmgraph(
    g,
    x_label=None,
    nodemembers=None,
    *,
    ax=None,
    nodesizerange=(1, 10),
    nodesizemode="log",
    colorlabel="x_label",
    cmap="jet",
    labelmethod="mode",
    nodeclim=None,
    nodescatter=False,
    center_expand=4.0,
):
    """Plot a temporal mapper graph (without a recurrence plot).

    Port of MATLAB's ``plottmgraph.m``.

    Parameters
    ----------
    g : networkx.Graph or networkx.DiGraph
        The graph to plot.
    x_label : array_like, optional
        A label for each member of each node, indexed the same way as the
        indices inside ``nodemembers``. Defaults to a constant array of
        ones sized to the number of unique members across all nodes.
    nodemembers : sequence of sequence of int, optional
        ``nodemembers[n]`` gives the indices belonging to node n. Defaults
        to one singleton member per node (0..N-1).
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. A new figure/axes is created if not given.
    nodesizerange : (float, float), default (1, 10)
        (min, max) marker size.
    nodesizemode : {'log', 'rank', 'original'}, default 'log'
        How to transform node sizes before rescaling to ``nodesizerange``.
    colorlabel : str, default 'x_label'
        Colorbar label.
    cmap : str or matplotlib.colors.Colormap, default 'jet'
        Colormap for node coloring.
    labelmethod : {'mode', 'mean', 'median', 'none'} or callable, default 'mode'
        See :func:`tmapper.labeling.find_node_label`.
    nodeclim : (float, float), optional
        Color axis limits. Defaults to (min(x_label), max(x_label)).
    nodescatter : bool, default False
        Whether to overlay a scatter plot on top of the graph nodes.
    center_expand : float, default 4.0
        Log-radial de-clumping strength applied to the layout after
        normalizing it to the unit circle (0 disables it). Force-directed
        layouts tend to pack most nodes into a dense central clump with a
        few outliers stretching the frame; this spreads the dense core
        outward without disturbing the outer structure. See
        :func:`_expand_center`.

    Returns
    -------
    ax : matplotlib.axes.Axes
    cbar : matplotlib.colorbar.Colorbar
    node_collection : matplotlib.collections.PathCollection
        The drawn graph nodes.
    scatter_collection : matplotlib.collections.PathCollection or None
        The scatter overlay, if ``nodescatter`` is True.
    """
    import matplotlib.pyplot as plt

    nodelist = list(g.nodes())
    n_nodes = len(nodelist)

    if nodemembers is None:
        nodemembers = [[i] for i in range(n_nodes)]

    if x_label is None:
        all_members = sorted({m for members in nodemembers for m in members})
        x_label = np.ones(len(all_members))
    x_label = np.asarray(x_label, dtype=float)

    if nodeclim is None:
        nodeclim = (float(np.min(x_label)), float(np.max(x_label)))

    # -- define node size
    nodesize = np.array([len(m) for m in nodemembers], dtype=float)
    buniform = len(set(nodesize.tolist())) == 1
    if not buniform:
        if nodesizemode == "rank":
            nodesize = stats.rankdata(nodesize, method="average")
        elif nodesizemode == "log":
            nodesize = np.log10(nodesize)
        # 'original' (or anything else): leave nodesize as-is, matching
        # MATLAB's switch with no matching/default case
        nodesize = _rescale(nodesize, nodesizerange[0], nodesizerange[1])
    else:
        span = nodesizerange[1] - nodesizerange[0]
        nodesize = np.full(n_nodes, nodesizerange[0] + span / n_nodes)

    # -- define node labels/colors
    nodelabel = find_node_label(nodemembers, x_label, labelmethod=labelmethod)

    # -- plotting
    if ax is None:
        _, ax = plt.subplots()

    # igraph's DrL layout (designed for exactly this scale of large,
    # dense graph) separates clustered regions far more cleanly than
    # networkx's spring_layout/ForceAtlas2 in practice; falls back to
    # spring_layout for the trivial 1-node case.
    if n_nodes > 1:
        import igraph as ig

        idx_of = {node: i for i, node in enumerate(nodelist)}
        edges = [(idx_of[u], idx_of[v]) for u, v in g.edges()]
        g_ig = ig.Graph(n=n_nodes, edges=edges, directed=g.is_directed())
        coords = np.array(g_ig.layout_drl().coords)
        pos = {nodelist[i]: coords[i] for i in range(n_nodes)}
    else:
        pos = nx.spring_layout(g, weight=None, seed=0)

    pos = _normalize_positions(pos)
    if center_expand:
        pos = _expand_center(pos, center_expand)

    xs = np.array([pos[n][0] for n in nodelist])
    ys = np.array([pos[n][1] for n in nodelist])

    # Edges visible but behind (zorder=1); nodes outlined and drawn in
    # front (zorder=3), smallest last, so small nodes aren't buried
    # under large ones and stay crisp regardless of edge clutter.
    edge_collection = nx.draw_networkx_edges(
        g, pos, ax=ax, alpha=0.6, edge_color="#888888", width=0.8,
        arrowsize=6, nodelist=nodelist,
    )
    for obj in (edge_collection if isinstance(edge_collection, list) else [edge_collection]):
        if obj is not None:
            obj.set_zorder(1)

    order = np.argsort(nodesize, kind="stable")[::-1]  # largest first, smallest drawn last (on top)
    node_collection = ax.scatter(
        xs[order], ys[order], s=(nodesize[order] ** 2), c=nodelabel[order], cmap=cmap,
        vmin=nodeclim[0] if nodeclim[1] != nodeclim[0] else None,
        vmax=nodeclim[1] if nodeclim[1] != nodeclim[0] else None,
        edgecolors="black", linewidths=0.5, zorder=3,
    )
    ax.set_aspect("equal")
    ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(nodelabel)
    if nodeclim[1] != nodeclim[0]:
        sm.set_clim(nodeclim[0], nodeclim[1])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(colorlabel)

    scatter_collection = None
    if nodescatter:
        order = np.argsort(nodesize, kind="stable")
        scatter_collection = ax.scatter(
            xs[order], ys[order], s=(nodesize[order] ** 2),
            c=nodelabel[order], cmap=cmap,
            vmin=nodeclim[0] if nodeclim[1] != nodeclim[0] else None,
            vmax=nodeclim[1] if nodeclim[1] != nodeclim[0] else None,
            edgecolors="k", linewidths=0.2,
        )

    return ax, cbar, node_collection, scatter_collection


def plot_tmgraph_tcm(g, x_label, t, nodemembers, **kwargs):
    """Plot a temporal mapper graph alongside its geodesic recurrence plot.

    Port of MATLAB's ``plotgraphtcm.m``. Calls :func:`plot_tmgraph`
    internally for the left subplot (all ``**kwargs`` are passed through),
    so it accepts the same styling parameters.

    Parameters
    ----------
    g : networkx.Graph or networkx.DiGraph
        The (simplified) graph to plot.
    x_label : array_like
        A label for each time point in the time series.
    t : array_like
        Actual time (or a time index) associated with each time point,
        used as the recurrence plot's axis labels.
    nodemembers : sequence of sequence of int
        ``nodemembers[n]`` gives the original time-point indices
        belonging to node n.
    **kwargs
        Passed through to :func:`plot_tmgraph`.

    Returns
    -------
    ax1, ax2 : matplotlib.axes.Axes
        The network plot and recurrence plot axes.
    cbar, cbar2 : matplotlib.colorbar.Colorbar
    node_collection, scatter_collection
        As returned by :func:`plot_tmgraph`.
    D_geo : numpy.ndarray
        The recurrence plot matrix.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    nodesize = np.array([len(m) for m in nodemembers])
    bsinglemember = np.all(nodesize == 1)

    # constrained_layout resolves spacing between subplots/colorbars
    # automatically, avoiding overlap between cbar's label and ax2's ylabel.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    ax1, cbar, node_collection, scatter_collection = plot_tmgraph(
        g, x_label, nodemembers, ax=ax1, **kwargs
    )

    if bsinglemember:
        D_geo = all_pairs_distance(g, list(g.nodes()), weight=None)
    else:
        D_geo = tcm_distance(g, nodemembers)

    t = np.asarray(t)
    is_datetime = np.issubdtype(t.dtype, np.datetime64)
    t_num = mdates.date2num(t) if is_datetime else t.astype(float)

    im = ax2.imshow(
        D_geo, cmap="hot",
        extent=[t_num.min(), t_num.max(), t_num.max(), t_num.min()],
        aspect="equal", interpolation="nearest",  # matches MATLAB's imagesc (flat, unsmoothed cells)
    )
    cbar2 = plt.colorbar(im, ax=ax2)
    cbar2.set_label("path length")

    if is_datetime:
        for axis in (ax2.xaxis, ax2.yaxis):
            locator = mdates.AutoDateLocator()
            axis.set_major_locator(locator)
            axis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax2.set_xlabel("time")
        ax2.set_ylabel("time")
    else:
        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("time (s)")
    ax2.set_title("geodesic recurrence plot")

    return ax1, ax2, cbar, cbar2, node_collection, scatter_collection, D_geo
