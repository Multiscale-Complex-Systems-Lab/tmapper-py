"""Tests for graph_utils.py, ported from the MATLAB toolbox's
test_graph_utils.m / test_symdyn.m / test_subgraph.m. Expected values are
the same hand-derived, MATLAB-verified oracle values used there, with
node/time indices translated from 1-indexed to 0-indexed.
"""

import numpy as np
import pytest
import networkx as nx

from tmapper import tknndigraph, filtergraph
from tmapper.graph_utils import (
    node_size,
    node_measure,
    normalize_geodesic,
    normalize_tcm,
    members_to_tidx,
    subgraph_from_members,
    sym_dyn_to_digraph,
    digraph_to_graph,
    find_blocks,
)


def test_node_size_and_measure():
    members = [[0], [1, 2], list(range(3, 13))]  # sizes 1, 2, 10
    assert np.array_equal(node_size(members), [1, 2, 10])
    assert np.allclose(node_measure(members), np.array([1, 2, 10]) / 13)


def test_normalize_geodesic_default_nsize():
    geod = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    geod_n, nm = normalize_geodesic(geod)
    assert np.allclose(nm, [1 / 3, 1 / 3, 1 / 3])
    # uniform nsize: normfactor = sqrt(sum((1/9)*geod^2)) = sqrt(sum(geod^2)/9)
    normfactor = np.sqrt(np.sum(geod ** 2) / 9)
    assert np.allclose(geod_n, geod / normfactor)


def test_normalize_geodesic_handles_inf_and_weighted_nsize():
    geod = np.array([[0, 1, np.inf], [1, 0, 2], [np.inf, 2, 0]])
    geod_n, nm = normalize_geodesic(geod, nsize=[1, 2, 1], exclude_diag=True)
    assert np.allclose(nm, [0.25, 0.5, 0.25])
    # inf replaced with max finite value (2)
    geod_clean = np.array([[0, 1, 2], [1, 0, 2], [2, 2, 0]])
    weighted = np.outer(nm, nm) * geod_clean ** 2
    mask = ~np.eye(3, dtype=bool)
    normfactor = np.sqrt(weighted[mask].sum())
    assert np.allclose(geod_n, geod_clean / normfactor)


def test_normalize_tcm_max_and_norm():
    tcm = np.array([[0, 2, np.inf], [2, 0, 4], [np.inf, 4, 0]])
    tcm_max = normalize_tcm(tcm, normtype="max", infreplace="max")
    tcm_clean = np.array([[0, 2, 4], [2, 0, 4], [4, 4, 0]])
    assert np.allclose(tcm_max, tcm_clean / 4)

    tcm_nan = normalize_tcm(tcm, normtype="max", infreplace="nan")
    expected = tcm_clean.astype(float)
    expected[np.isinf(tcm)] = np.nan
    assert np.allclose(tcm_nan, expected / np.nanmax(expected), equal_nan=True)


def test_members_to_tidx():
    members = [[0, 1], [2, 3, 4]]
    tidx = np.array([100, 101, 102, 103, 104])
    result = members_to_tidx(members, tidx)
    assert np.array_equal(result[0], [100, 101])
    assert np.array_equal(result[1], [102, 103, 104])


def test_sym_dyn_to_digraph_core():
    sym_dyn = np.array([1, 1, 1, 2, 2, 3, 3, 3, 1, 1, 4, 4, 2, 2])
    N = len(sym_dyn)
    dg, dwelltime, nodemembers = sym_dyn_to_digraph(sym_dyn)

    unique_states = np.unique(sym_dyn)
    assert dg.number_of_nodes() == len(unique_states)
    assert dwelltime.sum() == N
    assert len(nodemembers) == dg.number_of_nodes()

    # reconstruct symDyn from nodemembers
    reconstructed = np.zeros(N, dtype=int)
    for n, state in enumerate(dg.nodes()):
        reconstructed[nodemembers[n]] = state
    assert np.array_equal(reconstructed, sym_dyn)

    changepoints = np.flatnonzero(np.diff(sym_dyn) != 0)
    transitions = np.unique(
        np.stack([sym_dyn[changepoints], sym_dyn[changepoints + 1]], axis=1), axis=0
    )
    assert dg.number_of_edges() == transitions.shape[0]


def test_sym_dyn_to_digraph_degenerate_inputs():
    # constant symDyn: single node, no edges
    dg_const, dwelltime_const, members_const = sym_dyn_to_digraph([5, 5, 5, 5])
    assert dg_const.number_of_nodes() == 1
    assert dg_const.number_of_edges() == 0
    assert dwelltime_const[0] == 4
    assert np.array_equal(members_const[0], [0, 1, 2, 3])

    # length-1 symDyn
    dg_one, dwelltime_one, members_one = sym_dyn_to_digraph([7])
    assert dg_one.number_of_nodes() == 1
    assert dg_one.number_of_edges() == 0
    assert dwelltime_one[0] == 1
    assert np.array_equal(members_one[0], [0])


def test_digraph_to_graph_averages_weights():
    dg = nx.DiGraph()
    dg.add_weighted_edges_from([("A", "B", 2), ("B", "A", 4)])
    g = digraph_to_graph(dg)
    assert isinstance(g, nx.Graph)
    assert g["A"]["B"]["weight"] == 3
    assert set(g.nodes()) == {"A", "B"}


def test_digraph_to_graph_defaults_missing_weight_to_one():
    dg = nx.DiGraph()
    dg.add_edge(0, 1)  # no explicit weight, and no reverse edge 1->0
    g = digraph_to_graph(dg)
    # missing weight defaults to 1, but only one direction exists, so the
    # symmetrized average is (1 + 0) / 2 = 0.5
    assert g[0][1]["weight"] == 0.5


def test_find_blocks():
    ind = [0, 1, 1, 0, 0, 1, 0, 1, 1, 1]
    starts, ends, sizes = find_blocks(ind)
    assert np.array_equal(starts, [1, 5, 7])
    assert np.array_equal(ends, [2, 5, 9])
    assert np.array_equal(sizes, [2, 1, 3])


def test_subgraph_from_members():
    rng = np.random.default_rng(0)
    N1, N2 = 10, 10
    X = np.vstack([
        np.array([0, 0]) + 0.01 * rng.standard_normal((N1, 2)),
        np.array([10, 10]) + 0.01 * rng.standard_normal((N2, 2)),
    ])
    N = X.shape[0]
    tidx = np.arange(N)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    g, _ = tknndigraph(D, 3, tidx)
    g_simp, members, _, _ = filtergraph(g, 5, reciprocal=True)

    include_members = list(range(N1))
    g_sub, members_sub, sub_orig_nodeidx = subgraph_from_members(g_simp, members, include_members)

    assert g_sub.number_of_nodes() == len(members_sub)
    assert g_sub.number_of_nodes() == len(sub_orig_nodeidx)
    assert g_sub.number_of_nodes() <= g_simp.number_of_nodes()

    include_set = set(include_members)
    for m in members_sub:
        assert set(m) <= include_set

    expected_orig_idx = [i for i, m in enumerate(members) if set(m) & include_set]
    assert sorted(sub_orig_nodeidx) == sorted(expected_orig_idx)

    # empty includemembers -> empty subgraph
    g_sub_empty, members_sub_empty, idx_empty = subgraph_from_members(g_simp, members, [])
    assert g_sub_empty.number_of_nodes() == 0
    assert members_sub_empty == []
    assert idx_empty == []

    # full includemembers -> reproduces g_simp exactly
    g_sub_full, members_sub_full, idx_full = subgraph_from_members(g_simp, members, list(range(N)))
    assert g_sub_full.number_of_nodes() == g_simp.number_of_nodes()
    assert [sorted(m) for m in members_sub_full] == [sorted(m) for m in members]
    assert idx_full == list(range(g_simp.number_of_nodes()))

    # out-of-range includemembers values are silently ignored
    g_sub_a, members_sub_a, idx_a = subgraph_from_members(g_simp, members, [0, 1, 2])
    g_sub_b, members_sub_b, idx_b = subgraph_from_members(g_simp, members, [0, 1, 2, 9999])
    assert g_sub_a.number_of_nodes() == g_sub_b.number_of_nodes()
    assert members_sub_a == members_sub_b
    assert idx_a == idx_b
