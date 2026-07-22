"""Tests for cycle_count.py / cycle_count2p.py, ported from the MATLAB
toolbox's test_cyclepath.m. Expected values are the same hand-derived,
MATLAB-verified oracle values used there, with node indices translated
from 1-indexed to 0-indexed.
"""

import numpy as np

from tmapper.cycle_count import cycle_count
from tmapper.cycle_count2p import cycle_count2p, reorg_cycles


def _two_triangle_graph():
    # two simple cycles sharing node 2 (0-indexed): 0->1->2->0 and
    # 2->3->4->2 (1-indexed in the MATLAB original: 1->2->3->1, 3->4->5->3)
    A = np.zeros((5, 5))
    A[0, 1] = 1
    A[1, 2] = 1
    A[2, 0] = 1
    A[2, 3] = 1
    A[3, 4] = 1
    A[4, 2] = 1
    return A


def test_cycle_count_directed_triangle():
    A_tri = np.zeros((3, 3))
    A_tri[0, 1] = 1
    A_tri[1, 2] = 1
    A_tri[2, 0] = 1
    primes_tri = cycle_count(A_tri, 3)
    assert np.max(np.abs(primes_tri - [0, 0, 1])) < 1e-8


def test_cycle_count_mutual_pair():
    A_pair = np.zeros((2, 2))
    A_pair[0, 1] = 1
    A_pair[1, 0] = 1
    primes_pair = cycle_count(A_pair, 2)
    assert np.max(np.abs(primes_pair - [0, 1])) < 1e-8


def test_cycle_count_vs_cycle_count2p_agree():
    A = _two_triangle_graph()
    primes5 = cycle_count(A, 3)
    assert np.max(np.abs(primes5 - [0, 0, 2])) < 1e-8

    cyc_count5, cyc_len5, cyc_path5, all_cycles5 = cycle_count2p(A)
    assert cyc_count5.tolist() == [2]
    assert cyc_len5.tolist() == [3]

    # reorgCycles should exactly reproduce cycle_count2p's own flattening
    all_cycles_reorg = reorg_cycles(cyc_path5)
    assert all_cycles_reorg == all_cycles5
