"""Regression tests for tknndigraph/filtergraph, ported from the MATLAB
toolbox's tests/test_core_pipeline.m. All expected values here are the same
hand-derived, MATLAB-verified oracle values used there, translated from
1-indexed MATLAB node labels to 0-indexed Python labels (subtract 1 from
every node index). See that file's comments for the full derivations.
"""

import numpy as np
import pytest
import networkx as nx

from tmapper import tknndigraph, filtergraph


# -- deterministic dataset: 6 points on a line, two well-separated triples
Xd = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0]).reshape(-1, 1)
tidxd = np.arange(6)
Dd = np.abs(Xd - Xd.T)
kd = 2


def test_basic_construction():
    g, par = tknndigraph(Dd, kd, tidxd)
    assert isinstance(g, nx.DiGraph)
    assert g.number_of_nodes() == 6
    assert g.number_of_edges() == 9
    assert par["k"] == kd

    # reciprocal spatial edges (MATLAB 1<->3, 4<->6 -> Python 0<->2, 3<->5)
    for u, v in [(0, 2), (2, 0), (3, 5), (5, 3)]:
        assert g.has_edge(u, v), f"expected edge {u}->{v}"
    # temporal edges (always present)
    for i in range(5):
        assert g.has_edge(i, i + 1), f"expected temporal edge {i}->{i + 1}"
    # non-reciprocal spatial candidates should NOT survive
    assert not g.has_edge(0, 3)
    assert not g.has_edge(3, 0)


def test_raw_coordinates_equal_precomputed_distance():
    g_from_x, _ = tknndigraph(Xd, kd, tidxd)
    g_from_d, _ = tknndigraph(Dd, kd, tidxd)
    assert nx.to_numpy_array(g_from_x) is not None
    assert (nx.to_numpy_array(g_from_x, nodelist=range(6))
            == nx.to_numpy_array(g_from_d, nodelist=range(6))).all()


def test_reciprocal_false():
    # 10 pairs become fully bidirectional (20 directed edges); 5 "far" pairs
    # (0-4,0-5,1-4,1-5,2-5) get none.
    g, _ = tknndigraph(Dd, kd, tidxd, reciprocal=False)
    assert g.number_of_edges() == 20
    bidirectional_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]
    for u, v in bidirectional_pairs:
        assert g.has_edge(u, v) and g.has_edge(v, u), f"expected {u}-{v} bidirectional"
    no_edge_pairs = [(0, 4), (0, 5), (1, 4), (1, 5), (2, 5)]
    for u, v in no_edge_pairs:
        assert not g.has_edge(u, v) and not g.has_edge(v, u), f"expected no edge {u}-{v}"


def test_time_exclude_space_false():
    # each triple becomes a fully bidirectional clique, plus the one-way
    # bridge 2->3 (MATLAB 3->4)
    g, _ = tknndigraph(Dd, kd, tidxd, time_exclude_space=False)
    assert g.number_of_edges() == 13
    for u, v in [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1),
                 (3, 4), (4, 3), (3, 5), (5, 3), (4, 5), (5, 4)]:
        assert g.has_edge(u, v), f"expected edge {u}->{v} within a triple"
    assert g.has_edge(2, 3)
    assert not g.has_edge(3, 2)


def test_time_exclude_range_2():
    # excluding both 1st and 2nd temporal successors leaves no mutual
    # nearest-neighbor pairs on this dataset -> only the 5 temporal edges
    g, _ = tknndigraph(Dd, kd, tidxd, time_exclude_range=2)
    assert g.number_of_edges() == 5
    for i in range(5):
        assert g.has_edge(i, i + 1)
    assert not g.has_edge(0, 2)
    assert not g.has_edge(3, 5)


def test_max_neighbor_dist():
    # cuts the two spatial shortcuts (both at distance 2), leaving only
    # the 5 temporal edges
    g, _ = tknndigraph(Dd, kd, tidxd, max_neighbor_dist=1.5)
    assert g.number_of_edges() == 5
    assert not g.has_edge(0, 2)
    assert not g.has_edge(3, 5)


def test_max_neighbor_dist_prct_matches_equivalent_absolute():
    # cross-check: maxNeighborDistPrct should give the same graph as the
    # equivalent maxNeighborDist value, computed on the same *masked* D
    # (self-loops + default time_exclude_range=1 entries set to inf) that
    # tknndigraph itself uses internally.
    prct = 30
    D_masked = Dd.copy()
    np.fill_diagonal(D_masked, np.inf)
    for i in range(5):
        D_masked[i, i + 1] = np.inf
    equivalent_threshold = np.percentile(D_masked, prct)

    g_prct, _ = tknndigraph(Dd, kd, tidxd, max_neighbor_dist_prct=prct)
    g_equiv, _ = tknndigraph(Dd, kd, tidxd, max_neighbor_dist=equivalent_threshold)
    assert (nx.to_numpy_array(g_prct, nodelist=range(6))
            == nx.to_numpy_array(g_equiv, nodelist=range(6))).all()


def test_tidx_shift_invariance():
    # tidx only matters via relative adjacency between array positions,
    # never absolute values -- shifting every entry by a constant should
    # not change the result
    g_base, _ = tknndigraph(Dd, kd, tidxd)
    g_shifted, _ = tknndigraph(Dd, kd, tidxd + 1)
    assert (nx.to_numpy_array(g_base, nodelist=range(6))
            == nx.to_numpy_array(g_shifted, nodelist=range(6))).all()


def test_tidx_gap():
    # a genuine time gap between positions 2 and 3 (0-indexed; MATLAB 3,4)
    # suppresses exactly the unconditional temporal edge 2->3
    tidx_gap = np.array([0, 1, 2, 6, 7, 8])
    g, _ = tknndigraph(Dd, kd, tidx_gap)
    assert g.number_of_edges() == 8
    for u, v in [(0, 2), (2, 0), (3, 5), (5, 3)]:
        assert g.has_edge(u, v)
    for i in [0, 1, 3, 4]:
        assert g.has_edge(i, i + 1)
    assert not g.has_edge(2, 3)


def test_duplicate_point_tie_handling():
    # 4 points, two exact-duplicate pairs: positions [0,0,5,5]. Point 0's
    # only two candidates (points 2,3) are tied at distance 5; with
    # reciprocal=False (so the tie's effect isn't masked by reciprocal
    # filtering) this should connect point 0 to BOTH 2 and 3, despite k=1.
    Xdup = np.array([0.0, 0.0, 5.0, 5.0]).reshape(-1, 1)
    tidxdup = np.arange(4)
    g, _ = tknndigraph(Xdup, 1, tidxdup, reciprocal=False)
    assert g.number_of_edges() == 10
    assert g.has_edge(0, 3) or g.has_edge(3, 0)
    assert not g.has_edge(1, 3) and not g.has_edge(3, 1)


def test_filtergraph_partial_merge():
    g, _ = tknndigraph(Dd, kd, tidxd)
    g_simp, members, nodesize, D_simp = filtergraph(g, 2, reciprocal=True)
    assert g_simp.number_of_nodes() == 4
    assert sorted(nodesize) == [1, 1, 2, 2]
    assert g_simp.number_of_edges() == 5

    def node_containing(point):
        return next(i for i, m in enumerate(members) if point in m)

    idx02 = node_containing(0)
    idx1 = node_containing(1)
    idx35 = node_containing(3)
    idx4 = node_containing(4)
    assert sorted(members[idx02]) == [0, 2]
    assert members[idx1] == [1]
    assert sorted(members[idx35]) == [3, 5]
    assert members[idx4] == [4]

    # D_simp: hand-derived shortest cross-block distances on the original
    # directed geodesics (asymmetric: {3,5} cannot reach back to {0,2}/{1})
    assert D_simp[idx02, idx02] == 0 and D_simp[idx35, idx35] == 0
    assert D_simp[idx02, idx1] == 1 and D_simp[idx1, idx02] == 1
    assert D_simp[idx02, idx35] == 1
    assert D_simp[idx35, idx02] == np.inf
    assert D_simp[idx02, idx4] == 2
    assert D_simp[idx4, idx02] == np.inf
    assert D_simp[idx1, idx35] == 2
    assert D_simp[idx35, idx1] == np.inf
    assert D_simp[idx1, idx4] == 3
    assert D_simp[idx4, idx1] == np.inf
    assert D_simp[idx35, idx4] == 1 and D_simp[idx4, idx35] == 1


def test_filtergraph_fuller_merge():
    g, _ = tknndigraph(Dd, kd, tidxd)
    g_simp, members, nodesize, D_simp = filtergraph(g, 3, reciprocal=True)
    assert g_simp.number_of_nodes() == 2
    assert sorted(nodesize) == [3, 3]
    assert g_simp.number_of_edges() == 1

    def node_containing(point):
        return next(i for i, m in enumerate(members) if point in m)

    idx012 = node_containing(0)
    idx345 = node_containing(3)
    assert sorted(members[idx012]) == [0, 1, 2]
    assert sorted(members[idx345]) == [3, 4, 5]
    assert g_simp.has_edge(idx012, idx345)
    assert not g_simp.has_edge(idx345, idx012)


def test_filtergraph_reciprocal_false_merges_everything():
    # the OR condition is much more permissive: 0-1, 0-2, 1-2, and
    # critically 2-3 bridge the two triples, then 3-4, 3-5, 4-5 pull in
    # the rest -- all 6 points merge into one node
    g, _ = tknndigraph(Dd, kd, tidxd)
    g_simp, members, nodesize, D_simp = filtergraph(g, 2, reciprocal=False)
    assert g_simp.number_of_nodes() == 1
    assert nodesize[0] == 6
    assert g_simp.number_of_edges() == 0


def test_tknndigraph_input_validation():
    with pytest.raises(ValueError):
        tknndigraph(Dd, kd, tidxd[:-1])  # mismatched tidx length
    with pytest.raises(ValueError):
        tknndigraph(Dd, 6, tidxd)  # k >= number of points
    with pytest.raises(ValueError):
        tknndigraph(Dd, 1.5, tidxd)  # non-integer k


def test_filtergraph_input_validation():
    with pytest.raises(ValueError):
        filtergraph(Dd, 1)  # not a graph/digraph
    g, _ = tknndigraph(Dd, kd, tidxd)
    with pytest.raises(ValueError):
        filtergraph(g, -1)  # non-positive threshold
