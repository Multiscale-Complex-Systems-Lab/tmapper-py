"""Regression tests for tknndigraph/filtergraph, ported from the MATLAB
toolbox's tests/test_core_pipeline.m. All expected values here are the same
hand-derived, MATLAB-verified oracle values used there, translated from
1-indexed MATLAB node labels to 0-indexed Python labels (subtract 1 from
every node index). See that file's comments for the full derivations.
"""

import importlib

import numpy as np
import pytest
import networkx as nx

from tmapper import tknndigraph, filtergraph, tcm_distance


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
    # finite entries only, and MATLAB's (i-0.5)/n convention -- see
    # tknndigraph's note on method="hazen".
    equivalent_threshold = np.percentile(
        D_masked[np.isfinite(D_masked)], prct, method="hazen"
    )

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


def test_compute_dsimp_false_skips_only_that_output():
    """D_simp is the most expensive step in filtergraph and nothing in the
    toolbox consumes it, so it can be skipped -- but skipping it must not
    disturb anything else."""
    rng = np.random.RandomState(0)
    N = 150
    X = np.column_stack([
        np.sin(np.arange(N) / 12),
        np.cos(np.arange(N) / 12),
        np.cumsum(rng.randn(N)) / 20,
    ])
    g, _ = tknndigraph(X, 3, np.arange(N), time_exclude_range=5)

    gs_a, mem_a, ns_a, D_a = filtergraph(g, 3, reciprocal=True)
    gs_b, mem_b, ns_b, D_b = filtergraph(g, 3, reciprocal=True, compute_dsimp=False)

    assert D_a is not None, "D_simp should be computed by default."
    assert D_b is None, "compute_dsimp=False should return None for D_simp."
    assert np.array_equal(
        nx.to_numpy_array(gs_a, nodelist=sorted(gs_a.nodes())),
        nx.to_numpy_array(gs_b, nodelist=sorted(gs_b.nodes())),
    ), "skipping D_simp must not change the simplified graph."
    assert mem_a == mem_b, "skipping D_simp must not change the members."
    assert np.array_equal(ns_a, ns_b), "skipping D_simp must not change nodesize."


def _hazen_by_hand(values, prct):
    """The (i-0.5)/n percentile, written out from the definition.

    Deliberately NOT np.percentile(..., method="hazen") -- that is what the
    implementation calls, so using it here would make the test agree with
    the code by construction rather than by being right. MATLAB's prctile
    places the i-th of n sorted values at percentile 100*(i-0.5)/n and
    interpolates linearly between them, clamping outside that range.
    """
    x = np.sort(np.asarray(values, dtype=float).ravel())
    n = x.size
    pos = np.array([100.0 * (i + 0.5) / n for i in range(n)])
    if prct <= pos[0]:
        return float(x[0])
    if prct >= pos[-1]:
        return float(x[-1])
    j = np.searchsorted(pos, prct) - 1
    w = (prct - pos[j]) / (pos[j + 1] - pos[j])
    return float(x[j] + w * (x[j + 1] - x[j]))


def test_percentile_uses_matlabs_convention_not_numpys_default():
    """MATLAB's prctile and np.percentile's default disagree, and this port
    is checked against MATLAB.

    prctile places the i-th of n sorted values at 100*(i-0.5)/n; numpy's
    default "linear" uses i/(n-1). On [1,2,3,4,5] at the 95th percentile
    that is 5 versus 4.8 -- so picking the wrong one silently resolves a
    different neighbour cutoff.
    """
    x = np.array([1.0, 2, 3, 4, 5])
    assert _hazen_by_hand(x, 95) == 5.0, "MATLAB's convention gives 5 here."
    assert abs(np.percentile(x, 95) - 4.8) < 1e-12, "numpy's default gives 4.8."

    # the helper must agree with the method the implementation relies on
    for vals in (x, np.array([1.0, 3, 7, 10]), np.linspace(0, 1, 9) ** 2):
        for p in (25, 50, 75, 95, 99):
            assert abs(
                _hazen_by_hand(vals, p) - np.percentile(vals, p, method="hazen")
            ) < 1e-12, f"hand-derived Hazen should match numpy's, p={p}"

    # ...and, the point of all this, tknndigraph must resolve its cutoff the
    # same way. A small fixture, because the two conventions converge as n
    # grows and a large one would not tell them apart.
    N, tex, prct = 8, 3, 95.0
    rng = np.random.RandomState(0)
    Xs = np.column_stack([
        np.sin(np.arange(N) / 3),
        np.cos(np.arange(N) / 3),
        np.cumsum(rng.randn(N)) / 5,
    ])
    D = np.linalg.norm(Xs[:, None, :] - Xs[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    for n in range(1, tex + 1):
        i = np.arange(N - n)
        D[i, i + n] = np.inf
    finite = D[np.isfinite(D)]

    matlab_way = _hazen_by_hand(finite, prct)
    numpy_way = float(np.percentile(finite, prct))
    assert abs(matlab_way - numpy_way) > 1e-3, (
        "this fixture must be one where the two conventions visibly disagree"
    )

    _, par = tknndigraph(
        Xs, 3, np.arange(N), time_exclude_range=tex, max_neighbor_dist_prct=prct
    )
    assert abs(par["max_neighbor_dist"] - matlab_way) < 1e-12, (
        f"tknndigraph should resolve MATLAB's percentile ({matlab_way}), "
        f"not numpy's default ({numpy_way}); got {par['max_neighbor_dist']}"
    )


def test_prct_cutoff_ignores_the_masked_inf_entries():
    """The percentile must be taken over pairs that could actually be
    spatial neighbours.

    By the time the cutoff is resolved, D holds inf on the diagonal and on
    every temporal pair inside time_exclude_range. Those sit at the top of
    the distribution, so counting them pushes the cutoff up and makes it
    more permissive than asked for -- at high texclude the percentile can
    come back as inf outright, applying no cutoff at all.
    """
    N = 40
    rng = np.random.RandomState(0)
    X = np.column_stack([
        np.sin(np.arange(N) / 6),
        np.cos(np.arange(N) / 6),
        np.cumsum(rng.randn(N)) / 10,
    ])
    tidx = np.arange(N)
    tex, prct = 5, 90.0

    _, par = tknndigraph(
        X, 3, tidx, time_exclude_range=tex, max_neighbor_dist_prct=prct
    )

    # rebuild the same masking the function does, then take the percentile
    # over the finite entries only
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    for n in range(1, tex + 1):
        i = np.arange(N - n)
        D[i, i + n] = np.inf
    finite = D[np.isfinite(D)]

    expected = _hazen_by_hand(finite, prct)
    assert abs(par["max_neighbor_dist"] - expected) < 1e-9, (
        f"cutoff should come from the finite distances only "
        f"(expected {expected}, got {par['max_neighbor_dist']})"
    )

    # and it must genuinely differ from counting the infs, or this proves
    # nothing about which set was used
    with_inf = _hazen_by_hand(D.ravel(), prct)
    assert not np.isclose(with_inf, expected), (
        "this fixture must be one where including the infs changes the answer"
    )


def test_prct_100_short_circuits_to_no_cutoff():
    """At the default there is nothing to compute: every masked entry is
    inf, so the percentile is inf and the absolute cutoff wins."""
    N = 30
    X = np.column_stack([np.sin(np.arange(N) / 5), np.cos(np.arange(N) / 5)])
    _, par = tknndigraph(X, 3, np.arange(N), time_exclude_range=2)
    assert par["max_neighbor_dist"] == np.inf, (
        "with no absolute cutoff and prct=100, nothing should be cut."
    )
    _, par2 = tknndigraph(
        X, 3, np.arange(N), time_exclude_range=2, max_neighbor_dist=0.5
    )
    assert par2["max_neighbor_dist"] == 0.5, (
        "the absolute cutoff should win at prct=100."
    )


def test_reachability_route_matches_the_dense_route():
    """With compute_dsimp=False on an unweighted graph, filtergraph
    thresholds geodesics by sparse reachability instead of computing the
    all-pairs shortest path. Same answer, or the shortcut is not valid.

    Fractional d is included deliberately: hop counts are integers, so
    d=3 admits 1-2 hops while d=3.5 admits 1-3, and getting that boundary
    wrong would silently merge or split nodes.
    """
    for N in (80, 200):
        for tex in (1, 5):
            for dd in (1, 2, 3, 5, 2.5, 3.5):
                for recip in (True, False):
                    rng = np.random.RandomState(0)
                    X = np.column_stack([
                        np.sin(np.arange(N) / 12),
                        np.cos(np.arange(N) / 12),
                        np.cumsum(rng.randn(N)) / 20,
                    ])
                    g, _ = tknndigraph(X, 3, np.arange(N), time_exclude_range=tex)
                    ga, ma, na, _ = filtergraph(g, dd, reciprocal=recip)
                    gb, mb, nb, Db = filtergraph(
                        g, dd, reciprocal=recip, compute_dsimp=False
                    )
                    assert np.array_equal(
                        nx.to_numpy_array(ga, nodelist=sorted(ga.nodes())),
                        nx.to_numpy_array(gb, nodelist=sorted(gb.nodes())),
                    ), f"routes disagree at N={N}, tex={tex}, d={dd}, recip={recip}"
                    assert ma == mb and np.array_equal(na, nb)
                    assert Db is None


def test_tcm_distance_fast_path_matches_the_pairwise_loop():
    """tcm_distance short-circuits when each time point belongs to exactly
    one node, which is what filtergraph guarantees. Overlapping membership
    must still fall back to the pairwise loop, where the fmin arbitrates.
    """
    N = 200
    rng = np.random.RandomState(3)
    X = np.column_stack([
        np.sin(np.arange(N) / 12),
        np.cos(np.arange(N) / 12),
        np.cumsum(rng.randn(N)) / 20,
    ])
    g, _ = tknndigraph(X, 3, np.arange(N), time_exclude_range=5)
    gs, mem, _, _ = filtergraph(g, 3, reciprocal=True, compute_dsimp=False)

    D = tcm_distance(gs, mem)
    flat = np.concatenate([np.asarray(list(m), dtype=int) for m in mem])
    assert flat.size == np.unique(flat).size, (
        "filtergraph members should partition the time points."
    )
    assert D.shape == (N, N)
    assert np.all(np.diag(D) == 0), "a time point is 0 from itself."

    # time points sharing a node are 0 apart
    big = next((m for m in mem if len(m) > 1), None)
    if big is not None:
        ii = np.asarray(list(big), dtype=int)
        assert np.all(D[np.ix_(ii, ii)] == 0)

    # The fast path must agree with what the definition says, computed here
    # by an independent loop rather than by calling the fallback -- the
    # fallback is the same function, so it would not be independent.
    from tmapper._shortest_path import all_pairs_distance
    nodelist = list(gs.nodes())
    dm = all_pairs_distance(gs, nodelist, weight=None)
    node_of_t = {}
    for i, m in enumerate(mem):
        for t in m:
            node_of_t[int(t)] = i
    rng2 = np.random.RandomState(11)
    for _ in range(200):  # spot-check pairs rather than all N^2
        a_t = int(rng2.randint(N))
        b_t = int(rng2.randint(N))
        expected = dm[node_of_t[a_t], node_of_t[b_t]]
        assert D[a_t, b_t] == expected or (
            np.isnan(D[a_t, b_t]) and np.isnan(expected)
        ), f"tcm[{a_t},{b_t}] should be the geodesic between their nodes"

    # overlapping membership takes the loop and must still work
    g_ov = nx.from_numpy_array(np.array([[0, 1], [1, 0]]), create_using=nx.DiGraph)
    D_ov = tcm_distance(g_ov, [[0, 1, 2], [2, 3, 4]])
    assert D_ov.shape == (5, 5)
    assert D_ov[2, 2] == 0, "a shared time point is still 0 from itself."

    # time points covered by no node stay NaN rather than defaulting to 0
    D_gap = tcm_distance(g_ov, [[0, 1], [6, 7]])
    assert D_gap.shape == (8, 8)
    assert np.all(np.isnan(D_gap[2:6, 2:6])), (
        "uncovered time points should remain NaN, not 0."
    )


def test_low_memory_reproduces_the_dense_graph():
    """The blocked path never forms an (N, N) array, so it must be checked
    against the dense one rather than trusted -- including at block sizes
    that leave an awkward remainder, and with a gapped tidx, which is what
    drives the temporal-band construction."""
    for N in (60, 200):
        for k in (2, 3):
            for tex in (1, 5, 30):
                for recip in (True, False):
                    for tes in (True, False):
                        if tex >= N:
                            continue
                        rng = np.random.RandomState(0)
                        X = np.column_stack([
                            np.sin(np.arange(N) / 12),
                            np.cos(np.arange(N) / 12),
                            np.cumsum(rng.randn(N)) / 20,
                        ])
                        kw = dict(
                            time_exclude_range=tex, reciprocal=recip,
                            time_exclude_space=tes,
                        )
                        g1, _ = tknndigraph(X, k, np.arange(N), **kw)
                        g2, _ = tknndigraph(
                            X, k, np.arange(N), low_memory=True, block_size=37, **kw
                        )
                        assert np.array_equal(
                            nx.to_numpy_array(g1, nodelist=range(N)).astype(bool),
                            nx.to_numpy_array(g2, nodelist=range(N)).astype(bool),
                        ), f"low_memory differs at N={N}, k={k}, tex={tex}"

    # block size is a memory dial only: it must not change the result, and a
    # block larger than N (a single block) must work too
    rng = np.random.RandomState(1)
    N = 200
    X = np.column_stack([
        np.sin(np.arange(N) / 12),
        np.cos(np.arange(N) / 12),
    ])
    tgap = np.concatenate([np.arange(100), np.arange(150, 250)])
    ref, _ = tknndigraph(X, 3, tgap, time_exclude_range=5)
    for bs in (1, 17, 199, 200, 1000):
        gb, _ = tknndigraph(
            X, 3, tgap, time_exclude_range=5, low_memory=True, block_size=bs
        )
        assert np.array_equal(
            nx.to_numpy_array(ref, nodelist=range(N)).astype(bool),
            nx.to_numpy_array(gb, nodelist=range(N)).astype(bool),
        ), f"block_size={bs} changed the result; it should only trade memory for speed"


def test_low_memory_percentile_matches_closely():
    """The percentile needs the global distance distribution, which blocking
    never holds, so it is accumulated as a histogram in the same pass. That
    is exact to one bin width -- far below the scale that moves an edge, so
    the graphs must still match."""
    for prct in (95, 75, 50):
        rng = np.random.RandomState(0)
        N = 200
        X = np.column_stack([
            np.sin(np.arange(N) / 12),
            np.cos(np.arange(N) / 12),
            np.cumsum(rng.randn(N)) / 20,
        ])
        g1, p1 = tknndigraph(
            X, 3, np.arange(N), time_exclude_range=5, max_neighbor_dist_prct=prct
        )
        g2, p2 = tknndigraph(
            X, 3, np.arange(N), time_exclude_range=5, max_neighbor_dist_prct=prct,
            low_memory=True, block_size=37,
        )
        rel = abs(p2["max_neighbor_dist"] - p1["max_neighbor_dist"]) / p1["max_neighbor_dist"]
        assert rel < 1e-3, f"histogram percentile off by {rel:.2e} at prct={prct}"
        assert np.array_equal(
            nx.to_numpy_array(g1, nodelist=range(N)).astype(bool),
            nx.to_numpy_array(g2, nodelist=range(N)).astype(bool),
        ), f"low_memory should give the same graph under a percentile cutoff (prct={prct})"


def test_simplified_edge_weights_are_block_average_connectivity():
    """g_simp's edge weight from node n to node m is the MEAN connectivity
    between their member sets: the number of original edges running n -> m,
    divided by |n| * |m|.

    Nothing asserted this before. Deleting the normalisation outright, or
    dividing by only the row group size, left all 122 tests passing while
    changing 600 of 1,296 swept configurations -- so the weights were
    entirely untested. The oracle below is computed straight from the
    definition, independently of how filtergraph gets there."""
    rng = np.random.RandomState(0)
    N = 200
    X = np.column_stack([
        np.sin(np.arange(N) / 12),
        np.cos(np.arange(N) / 12),
        np.cumsum(rng.randn(N)) / 20,
    ])
    g, _ = tknndigraph(X, 3, np.arange(N), time_exclude_range=5)
    g_simp, members, nodesize, _ = filtergraph(g, 3, compute_dsimp=False)

    A = nx.to_numpy_array(g, nodelist=sorted(g.nodes()))
    n_new = len(members)

    # the test is only meaningful if some block is bigger than a single
    # node -- otherwise |n| * |m| == 1 and the normalisation is invisible
    assert max(len(m) for m in members) > 1, "fixture merges nothing"

    checked = 0
    for n in range(n_new):
        for m in range(n_new):
            if n == m:
                continue  # self-loops are removed from g_simp
            block = A[np.ix_(members[n], members[m])]
            expected = block.sum() / (len(members[n]) * len(members[m]))
            actual = g_simp[n][m]["weight"] if g_simp.has_edge(n, m) else 0.0
            assert np.isclose(actual, expected), (
                f"edge {n}->{m}: weight {actual!r}, expected block average "
                f"{expected!r} (block sum {block.sum()!r} over "
                f"{len(members[n])}x{len(members[m])} members)"
            )
            if expected:
                checked += 1
    assert checked > 0, "no non-zero edges were actually compared"


def test_low_memory_histogram_survives_chunking():
    """The percentile histogram is accumulated a chunk of rows at a time, to
    keep the bin-index temporary bounded. Chunking is pure bookkeeping, so the
    number of chunks must not change the answer -- but no test-sized input
    comes near the 8M-value default, so force it small and check."""
    # importlib, not `import tmapper.tknndigraph as tkm`: the package
    # re-exports the FUNCTION under that name, which shadows the module.
    tkm = importlib.import_module("tmapper.tknndigraph")

    rng = np.random.RandomState(0)
    N = 200
    X = np.column_stack([
        np.sin(np.arange(N) / 12),
        np.cos(np.arange(N) / 12),
        np.cumsum(rng.randn(N)) / 20,
    ])
    kw = dict(time_exclude_range=5, max_neighbor_dist_prct=90.0)

    original = tkm._HIST_CHUNK
    try:
        # one chunk per block (the default's effective behaviour here)...
        tkm._HIST_CHUNK = 10_000_000
        g_one, p_one = tknndigraph(X, 3, np.arange(N), low_memory=True,
                                   block_size=64, **kw)
        # ...versus many chunks per block, several of them partial
        tkm._HIST_CHUNK = 50
        g_many, p_many = tknndigraph(X, 3, np.arange(N), low_memory=True,
                                     block_size=64, **kw)
    finally:
        tkm._HIST_CHUNK = original

    assert p_many["max_neighbor_dist"] == p_one["max_neighbor_dist"], (
        "chunking changed the resolved cutoff: "
        f"{p_many['max_neighbor_dist']!r} vs {p_one['max_neighbor_dist']!r}"
    )
    assert np.array_equal(
        nx.to_numpy_array(g_one, nodelist=range(N)).astype(bool),
        nx.to_numpy_array(g_many, nodelist=range(N)).astype(bool),
    ), "chunking changed the graph"

    # and both must still agree with the exact, unblocked percentile
    g_dense, p_dense = tknndigraph(X, 3, np.arange(N), **kw)
    rel = abs(p_many["max_neighbor_dist"] - p_dense["max_neighbor_dist"])
    rel /= p_dense["max_neighbor_dist"]
    assert rel < 1e-3, f"chunked histogram drifted from the exact percentile: {rel:.2e}"
    assert np.array_equal(
        nx.to_numpy_array(g_dense, nodelist=range(N)).astype(bool),
        nx.to_numpy_array(g_many, nodelist=range(N)).astype(bool),
    ), "chunked low_memory graph differs from the dense one"


def test_low_memory_handles_a_pair_at_the_bounding_box_diagonal():
    """The histogram's upper bound is the bounding-box diagonal, so a pair of
    opposite corners sits exactly AT it -- and depending on how hi and the
    scale factor round, dist * scale lands on bin index n_bins, one past the
    last bin. np.bincount then returns an array one element longer than the
    accumulator and the += raises. The clip is what prevents that.

    Whether the rounding goes that way is fixture-specific (~22% of random
    point sets in a quick scan), so the fixture below is picked to trip it
    and asserts that it still does."""
    tkm = importlib.import_module("tmapper.tknndigraph")

    # Two opposite corners of the box, so the max distance equals the box
    # diagonal exactly, plus interior points so there is a distribution to
    # take a percentile of.
    #   The corners go FIRST and LAST, not adjacent: neighbouring rows are
    # temporal neighbours, and time_exclude_range would mask their distance
    # to inf before the histogram ever saw it.
    #   The 1.5 factor is not cosmetic. On the unit box the diagonal is
    # sqrt(3), whose scale factor rounds down and keeps the top distance
    # inside the last bin -- the clip never fires and the test passes even
    # with it deleted.
    rng = np.random.RandomState(0)
    interior = rng.rand(48, 3) * 0.6 + 0.2
    X = np.vstack([np.zeros(3), interior, np.ones(3)]) * 1.5
    N = X.shape[0]

    span = X.max(axis=0) - X.min(axis=0)
    hi = float(np.linalg.norm(span))
    dmax_actual = float(np.max(np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)))
    assert dmax_actual == hi, (
        "fixture is not exercising the top edge: the largest distance "
        f"({dmax_actual!r}) should equal the bounding-box diagonal ({hi!r})"
    )
    assert int(dmax_actual * (tkm._HIST_BINS / hi)) >= tkm._HIST_BINS, (
        "fixture no longer overflows the last bin, so it stopped testing the "
        "clip -- rescale it until it does"
    )

    kw = dict(time_exclude_range=2, max_neighbor_dist_prct=80.0)
    g_lm, p_lm = tknndigraph(X, 3, np.arange(N), low_memory=True,
                             block_size=7, **kw)
    g_d, p_d = tknndigraph(X, 3, np.arange(N), **kw)

    rel = abs(p_lm["max_neighbor_dist"] - p_d["max_neighbor_dist"])
    rel /= p_d["max_neighbor_dist"]
    assert rel < 1e-3, f"cutoff off by {rel:.2e} with a pair at the top edge"
    assert np.array_equal(
        nx.to_numpy_array(g_d, nodelist=range(N)).astype(bool),
        nx.to_numpy_array(g_lm, nodelist=range(N)).astype(bool),
    ), "a pair at the bounding-box diagonal changed the low_memory graph"


def test_low_memory_rejects_a_precomputed_distance_matrix():
    """Handed D there is nothing left to save, so say so rather than
    silently doing the dense thing."""
    rng = np.random.RandomState(0)
    N = 40
    X = np.column_stack([np.sin(np.arange(N) / 6), np.cos(np.arange(N) / 6)])
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    with pytest.raises(ValueError, match="coordinates"):
        tknndigraph(D, 3, np.arange(N), low_memory=True)

    # and missing data must still be rejected on this path
    Xnan = X.copy()
    Xnan[5, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        tknndigraph(Xnan, 3, np.arange(N), low_memory=True)
