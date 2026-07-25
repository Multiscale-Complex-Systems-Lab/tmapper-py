"""Regression tests for plotting/labeling/tcm_distance, ported from the
MATLAB toolbox's tests/test_visualization.m. Expected values are the same
hand-derived, MATLAB-verified oracle values used there.
"""

import matplotlib
matplotlib.use("Agg")  # headless backend for CI/no-display environments

import numpy as np
import pytest
import networkx as nx

from tmapper import (
    tknndigraph, filtergraph, find_node_label, tcm_distance,
    plot_tmgraph, plot_tmgraph_tcm, plot_tmgraph_interactive,
)


def test_find_node_label_all_methods():
    # 3 nodes with distinct label distributions to discriminate each method.
    # node0 -> {10,10,20}: mode=10, mean=13.333, median=10
    # node1 -> {5,7}: mode=5 (smallest of an all-tied set), mean=6, median=6
    # node2 -> {100}: 100 regardless of method
    members = [[0, 1, 2], [3, 4], [5]]
    x_label = np.array([10, 10, 20, 5, 7, 100], dtype=float)

    assert np.allclose(find_node_label(members, x_label, labelmethod="mode"), [10, 5, 100])
    assert np.allclose(find_node_label(members, x_label, labelmethod="mean"), [13 + 1 / 3, 6, 100])
    assert np.allclose(find_node_label(members, x_label, labelmethod="median"), [10, 6, 100])
    assert np.allclose(find_node_label(members, x_label, labelmethod="none"), [0, 0, 0])
    # custom callable: range = max - min
    result = find_node_label(members, x_label, labelmethod=lambda x: x.max() - x.min())
    assert np.allclose(result, [10, 2, 0])


def test_plot_tmgraph_nodesizemode():
    # sizes [1,2,10] (not evenly spaced, not geometric) so rank/log/original
    # genuinely disagree. plot_tmgraph draws largest-to-smallest (so small
    # nodes render on top, unburied), so get_sizes() comes back in
    # descending order rather than node order.
    g3 = nx.DiGraph()
    g3.add_nodes_from(range(3))
    members = [[0], [1, 2], list(range(3, 13))]  # sizes 1, 2, 10
    x_label = np.ones(13)

    _, _, nc_rank, _ = plot_tmgraph(g3, x_label, members, nodesizemode="rank")
    assert np.allclose(np.sqrt(nc_rank.get_sizes()), [10, 5.5, 1], atol=1e-9)

    _, _, nc_log, _ = plot_tmgraph(g3, x_label, members, nodesizemode="log")
    assert np.allclose(np.sqrt(nc_log.get_sizes()), [10, 3.709, 1], atol=1e-3)

    _, _, nc_orig, _ = plot_tmgraph(g3, x_label, members, nodesizemode="original")
    assert np.allclose(np.sqrt(nc_orig.get_sizes()), [10, 2, 1], atol=1e-9)
    matplotlib.pyplot.close("all")


def test_plot_tmgraph_default_uniform_nodesize():
    # default nodemembers/x_label (both omitted): singleton members ->
    # buniform=True -> uniform marker size 1 + range/numnodes = 1 + 9/3 = 4
    g3 = nx.DiGraph()
    g3.add_nodes_from(range(3))
    _, _, nc, _ = plot_tmgraph(g3)
    assert np.allclose(np.sqrt(nc.get_sizes()), 4.0)
    matplotlib.pyplot.close("all")


def test_tcm_distance_weighted_vs_unweighted():
    # 3 nodes: 0->1 (w=5), 1->2 (w=1), 0->2 (w=100). Unweighted: dist(0,2)=1
    # (direct edge, hop count). Weighted: dist(0,2)=6 (5+1 via node 1,
    # cheaper than the expensive direct edge).
    g = nx.DiGraph()
    g.add_weighted_edges_from([(0, 1, 5), (1, 2, 1), (0, 2, 100)])
    nodet = [[0], [1], [2]]
    tcm_unweighted = tcm_distance(g, nodet, weighted=False)
    tcm_weighted = tcm_distance(g, nodet, weighted=True)
    assert tcm_unweighted[0, 2] == 1
    assert tcm_weighted[0, 2] == 6


def test_tcm_distance_gap_coverage():
    # chain 0->1->2, nodet = [[0],[1],[4]] (points 2,3 never covered by any
    # node). Output should be sized 5x5 (range 0..4), with points 2,3
    # correctly staying NaN throughout instead of being 0-filled.
    g = nx.DiGraph()
    g.add_edges_from([(0, 1), (1, 2)])
    nodet = [[0], [1], [4]]
    tcm = tcm_distance(g, nodet)
    assert tcm.shape == (5, 5)
    assert np.all(np.isnan(tcm[2, :])) and np.all(np.isnan(tcm[:, 2]))
    assert np.all(np.isnan(tcm[3, :])) and np.all(np.isnan(tcm[:, 3]))
    assert tcm[0, 0] == 0 and tcm[1, 1] == 0 and tcm[4, 4] == 0
    assert tcm[0, 1] == 1 and tcm[1, 0] == np.inf
    assert tcm[0, 4] == 2 and tcm[4, 0] == np.inf
    assert tcm[1, 4] == 1 and tcm[4, 1] == np.inf


def test_bsinglemember_matches_tcm_distance():
    # with d such that every node is singleton, plot_tmgraph_tcm's shortcut
    # branch should agree exactly with tcm_distance computed directly.
    Xd = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0]).reshape(-1, 1)
    tidxd = np.arange(6)
    Dd = np.abs(Xd - Xd.T)
    g, _ = tknndigraph(Dd, 2, tidxd)
    g_simp, members, _, _ = filtergraph(g, 1, reciprocal=True)
    assert g_simp.number_of_nodes() == 6  # confirms every node is singleton

    colorvar = Xd[:, 0]
    t = tidxd
    _, _, _, _, _, _, D_geo = plot_tmgraph_tcm(g_simp, colorvar, t, members)
    tcm = tcm_distance(g_simp, members)
    assert np.allclose(D_geo, tcm, equal_nan=True)
    matplotlib.pyplot.close("all")


def test_plot_tmgraph_nodescatter():
    g3 = nx.DiGraph()
    g3.add_nodes_from(range(3))
    members = [[0], [1, 2], list(range(3, 13))]
    x_label = np.ones(13)
    ax, cbar, nc, sc = plot_tmgraph(g3, x_label, members, nodescatter=True)
    assert sc is not None
    assert len(sc.get_offsets()) == 3
    matplotlib.pyplot.close("all")

    _, _, _, sc_off = plot_tmgraph(g3, x_label, members, nodescatter=False)
    assert sc_off is None
    matplotlib.pyplot.close("all")


def test_plot_tmgraph_clim_edge_case():
    # a constant x_label (no variation) should run without error
    g3 = nx.DiGraph()
    g3.add_nodes_from(range(3))
    members = [[0], [1, 2], list(range(3, 13))]
    x_label_const = 5 * np.ones(13)
    ax, cbar, nc, _ = plot_tmgraph(g3, x_label_const, members)
    assert ax is not None and cbar is not None
    matplotlib.pyplot.close("all")


def test_plot_tmgraph_interactive_returns_html_by_default(tmp_path):
    # -- default output_path=None should return the HTML string with no
    # disk side effect, so callers (e.g. a Streamlit app) can embed it
    # directly without a write-then-read-back round trip.
    g3 = nx.DiGraph()
    g3.add_nodes_from(range(3))
    g3.add_edges_from([(0, 1), (1, 2), (2, 0)])
    members = [[0], [1, 2], list(range(3, 13))]
    x_label = np.ones(13)

    files_before = set(tmp_path.iterdir())
    net, html = plot_tmgraph_interactive(g3, x_label, members, title="unit test network")
    assert set(tmp_path.iterdir()) == files_before, \
        "output_path=None (the default) should not write any file."

    # heading appears exactly once (pyvis's own template duplicates it
    # unconditionally; we strip the second copy)
    assert html.count("<h1>unit test network</h1>") == 1
    # a legend image was embedded
    assert "data:image/png;base64," in html
    # physics disabled, so the network stays exactly where the layout put it
    assert '"physics": {"enabled": false}' in html or '"enabled": false' in html
    assert len(net.nodes) == 3
    assert len(net.edges) == 3


def test_plot_tmgraph_interactive_writes_html_when_path_given(tmp_path):
    g3 = nx.DiGraph()
    g3.add_nodes_from(range(3))
    g3.add_edges_from([(0, 1), (1, 2), (2, 0)])
    members = [[0], [1, 2], list(range(3, 13))]
    x_label = np.ones(13)
    out = tmp_path / "network.html"

    net, html = plot_tmgraph_interactive(
        g3, x_label, members, title="unit test network", output_path=str(out)
    )

    assert out.exists()
    # the written file and the returned string must be identical -- the
    # whole point of returning html is that it's the same content you'd
    # otherwise have to write-then-read-back from output_path.
    assert out.read_text(encoding="utf-8") == html
    assert len(net.nodes) == 3
    assert len(net.edges) == 3
    assert len(net.nodes) == 3
    assert len(net.edges) == 3
