"""Tests for cycle_overlap.py / cycle_cluster.py, ported from the
MATLAB toolbox's test_cyclepath.m. Expected values are the same
hand-derived, MATLAB-verified oracle values used there, with node
indices translated from 1-indexed to 0-indexed.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np

from tmapper.cycle_overlap import cycle_path_overlap
from tmapper.cycle_cluster import cycle_cluster


def test_cycle_path_overlap_node_type():
    # two paths sharing 2 of 4 total unique nodes
    c_overlap = [[0, 1, 2], [1, 2, 3]]
    CO_node, _ = cycle_path_overlap(c_overlap, overlap_type="node")
    assert abs(CO_node[0, 1] - 0.5) < 1e-10


def test_cycle_path_overlap_edge_type():
    # cycle1 edges {0-1,1-2,2-0}, cycle2 edges {1-2,2-3,3-1}; only edge
    # 1-2 is shared, union has 5 unique edges -> overlap 1/5=0.2
    c_overlap = [[0, 1, 2], [1, 2, 3]]
    CO_edge, _ = cycle_path_overlap(c_overlap, overlap_type="edge")
    assert abs(CO_edge[0, 1] - 0.2) < 1e-10


def test_cycle_cluster_degenerate_cases():
    assert cycle_cluster([], 0.5, plotmat=False, plothist=False).tolist() == []
    assert cycle_cluster([[0, 1, 2]], 0.5, plotmat=False, plothist=False).tolist() == [1]


def test_cycle_cluster_threshold_behavior():
    # two cycles sharing one node (overlap 1/5=0.2)
    allcycles = [[0, 1, 2], [2, 3, 4]]
    # default-like threshold 0.5 > 0.2: cycles stay in separate clusters
    idx_separate = cycle_cluster(allcycles, 0.5, plotmat=False, plothist=False)
    assert len(set(idx_separate.tolist())) == 2

    # lower threshold 0.1 < 0.2: cycles merge into one cluster
    idx_merged = cycle_cluster(allcycles, 0.1, plotmat=False, plothist=False)
    assert len(set(idx_merged.tolist())) == 1
