"""Tests for knngraph.py / cknngraph.py, ported from the MATLAB
toolbox's test_knngraph.m. Expected values are the same hand-derived,
MATLAB-verified oracle values used there, with node indices translated
from 1-indexed to 0-indexed.
"""

import numpy as np
import pytest

from tmapper import knngraph, cknngraph


def test_knngraph_two_disjoint_triangles():
    # k=2, no temporal component, so ties fill both k=2 slots exactly and
    # every pair ends up mutual: two disjoint triangles, 6 edges total.
    Xd = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0]).reshape(-1, 1)
    g = knngraph(Xd, 2)
    assert g.number_of_nodes() == 6
    assert g.number_of_edges() == 6
    for u, v in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]:
        assert g.has_edge(u, v), f"expected edge {u}-{v} within a triple."
    assert not g.has_edge(2, 3), "expected no edge across the two triples."


def test_knngraph_reciprocal_effect():
    # Xd2=[0,1,3], k=1. Point0's nearest is point1; point1's nearest is
    # point0; point2's nearest is point1, but point1's nearest is point0,
    # not point2 -- so 1-2 is one-directional.
    Xd2 = np.array([0.0, 1.0, 3.0]).reshape(-1, 1)
    g_rec = knngraph(Xd2, 1, reciprocal=True)
    g_norec = knngraph(Xd2, 1, reciprocal=False)

    assert g_rec.number_of_edges() == 1 and g_rec.has_edge(0, 1)
    assert g_norec.number_of_edges() == 2 and g_norec.has_edge(0, 1) and g_norec.has_edge(1, 2)


def test_cknngraph_paths_not_triangles():
    # Every point's nearest-neighbor distance (Dk) is exactly 1, so
    # D_norm equals the raw distance matrix unchanged. Within each
    # triple, adjacent-point pairs (distance 1) connect but the
    # end-to-end pair (distance 2) does not (2 is not < 1.5) -- two
    # disjoint 3-point paths, not triangles.
    Xd = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0]).reshape(-1, 1)
    g = cknngraph(Xd, 1, 1.5)
    assert g.number_of_nodes() == 6
    assert g.number_of_edges() == 4
    assert g.has_edge(0, 1) and g.has_edge(1, 2)
    assert not g.has_edge(0, 2)
    assert g.has_edge(3, 4) and g.has_edge(4, 5)
    assert not g.has_edge(3, 5)


def test_knngraph_input_validation():
    Xd = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0]).reshape(-1, 1)
    with pytest.raises(ValueError):
        knngraph(Xd, 1.5)  # non-integer k
    with pytest.raises(ValueError):
        knngraph(Xd, 6)  # k >= number of points


def test_cknngraph_input_validation():
    Xd = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0]).reshape(-1, 1)
    with pytest.raises(ValueError):
        cknngraph(Xd, 1.5, 1)  # non-integer k
    with pytest.raises(ValueError):
        cknngraph(Xd, 6, 1)  # k >= number of points
    with pytest.raises(ValueError):
        cknngraph(Xd, 1, -1)  # non-positive delta
