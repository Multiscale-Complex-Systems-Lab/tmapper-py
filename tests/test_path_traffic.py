"""Tests for path_traffic.py, ported from the pathtraffic portion of the
MATLAB toolbox's test_cyclepath.m. Expected values are the same
hand-derived, MATLAB-verified oracle values used there, with node
indices translated from 1-indexed to 0-indexed.
"""

import numpy as np

from tmapper.path_traffic import path_traffic


def test_path_traffic_hand_derived():
    allpath = [[0, 1, 2], [3, 4]]
    nodesize = [10, 20, 30, 5, 15]
    traf_mean, traf_med, traf_min, traf_max, traf_std = path_traffic(allpath, nodesize)

    assert np.array_equal(traf_mean, [20, 10])
    assert np.array_equal(traf_med, [20, 10])
    assert np.array_equal(traf_min, [10, 5])
    assert np.array_equal(traf_max, [30, 15])
    assert np.max(np.abs(traf_std - [10, np.sqrt(50)])) < 1e-9
