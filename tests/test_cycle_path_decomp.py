"""Tests for cycle_cutter.py / cycles_to_paths.py / cycle_cluster_conn.py /
cycle_path_decomp.py, ported from the MATLAB toolbox's test_cyclepath.m.
Expected values are the same hand-derived, MATLAB-verified oracle values
used there, with node indices translated from 1-indexed to 0-indexed.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from tmapper.cycle_cutter import cycle_cutter
from tmapper.cycle_path_decomp import cycle_path_decomp


def _two_triangle_digraph():
    # two simple cycles sharing node 2 (0-indexed): 0->1->2->0 and
    # 2->3->4->2 (1-indexed in the MATLAB original: 1->2->3->1, 3->4->5->3)
    dg = nx.DiGraph()
    dg.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)])
    return dg


def test_cycle_path_decomp_two_triangles():
    dg = _two_triangle_digraph()

    allbd1, pclusters_nodes1, _, _, _, _, allupath1, Tp1 = cycle_path_decomp(
        dg, plotmat=False, plotmds=False, plothist=False
    )
    plt.close("all")

    assert len(allbd1) > 0, "expected at least one boundary node."
    assert 2 in allbd1, "node 2 (shared between the two cycles) should be a boundary node."
    assert len(pclusters_nodes1) == max(Tp1), \
        "path-cluster count should match the number of unique cluster labels."
    assert len(allupath1) > 0, "expected at least one decomposed path."

    # -- enabling plotmat should not change the boundary-node/clustering result
    allbd2, _, _, _, _, _, _, Tp2 = cycle_path_decomp(dg, plotmat=True, plotmds=False, plothist=False)
    plt.close("all")
    assert allbd1 == allbd2
    assert np.array_equal(Tp1, Tp2)


def test_cycle_path_decomp_acyclic_graph():
    # chain, no cycles at all
    dg_acyclic = nx.DiGraph()
    dg_acyclic.add_edges_from([(0, 1), (1, 2), (2, 3)])
    allbd, pcn, _, _, _, _, allupath, Tp = cycle_path_decomp(
        dg_acyclic, plotmat=False, plotmds=False, plothist=False
    )
    plt.close("all")
    assert allbd == []
    assert pcn == []
    assert allupath == []
    assert len(Tp) == 0


def test_cycle_path_decomp_single_isolated_cycle():
    dg_single = nx.DiGraph()
    dg_single.add_edges_from([(0, 1), (1, 2), (2, 0)])
    allbd, pcn, _, _, _, _, _, Tp = cycle_path_decomp(
        dg_single, plotmat=False, plotmds=False, plothist=False
    )
    plt.close("all")
    assert allbd == [], "a single isolated cycle has no boundary (nothing to share with)."
    assert len(pcn) == 1, "a single isolated cycle should form exactly one path cluster."
    assert max(Tp) == 1


def test_cycle_path_decomp_merge_threshold():
    dg = _two_triangle_digraph()
    # the two cycles overlap by exactly 1/5=0.2 (shared node 2 only); a
    # lower threshold (0.1 < 0.2) should merge them into one cluster
    _, pcn_merged, _, _, _, _, _, Tp_merged = cycle_path_decomp(
        dg, plotmat=False, plotmds=False, plothist=False, clusterthres=0.1
    )
    plt.close("all")
    assert max(Tp_merged) == 1
    assert len(pcn_merged) == 1


def test_cycle_cutter_no_cutting_points():
    result = cycle_cutter([0, 1, 2], [])
    assert result == [[0, 1, 2]]
