"""Tests for modularity.py (Qasym/calMod), ported from the MATLAB
toolbox's test_qasym.m. Expected values are the same hand-derived,
MATLAB-verified oracle values used there.
"""

import numpy as np

from tmapper.modularity import qasym, cal_mod


def test_qasym_symmetric_blocks():
    # two disjoint fully-connected 3-node blocks (block-diagonal
    # adjacency, no self-loops, no edges across blocks)
    block = np.ones((3, 3)) - np.eye(3)
    A = np.block([
        [block, np.zeros((3, 3))],
        [np.zeros((3, 3)), block],
    ])

    C_correct = np.array([1, 1, 1, 2, 2, 2])
    Q_correct = qasym(A, C_correct)

    C_merged = np.ones(6)
    Q_merged = qasym(A, C_merged)

    tol = 1e-10
    assert abs(Q_correct - 0.5) < tol
    assert abs(Q_merged - 0) < tol
    assert Q_correct > Q_merged


def test_qasym_asymmetric_weighted():
    # 2 communities {0,1},{2,3}, directed unevenly-weighted edges:
    # 0->1 (w=2), 1->0 (w=1), 2->3 (w=3), 3->2 (w=1), and one
    # cross-community edge 1->2 (w=1, one direction only).
    A_dir = np.zeros((4, 4))
    A_dir[0, 1] = 2
    A_dir[1, 0] = 1
    A_dir[2, 3] = 3
    A_dir[3, 2] = 1
    A_dir[1, 2] = 1
    C_dir = np.array([1, 1, 2, 2])

    Q_dir = qasym(A_dir, C_dir)
    assert abs(Q_dir - 0.375) < 1e-10

    # binarized version should give a genuinely different Q (weights matter)
    A_bin = (A_dir != 0).astype(float)
    Q_bin = qasym(A_bin, C_dir)
    assert abs(Q_bin - 0.32) < 1e-10
    assert abs(Q_dir - Q_bin) > 1e-10


def test_qasym_zero_edge_guard():
    C_dir = np.array([1, 1, 2, 2])
    assert qasym(np.zeros((4, 4)), C_dir) == 0


def test_cal_mod_zero_edge_guard():
    m0 = np.array([1, 1, 2, 2])
    assert cal_mod(np.zeros((4, 4)), m0) == 0
