"""Port of tmapper_tools/pathtraffic.m from the MATLAB toolbox."""

import numpy as np


def path_traffic(allpath, nodesize):
    """Compute traffic statistics (based on node size) along each path.

    Port of MATLAB's ``pathtraffic.m``. Note: matches MATLAB's ``std()``
    convention of dividing by ``N-1`` (sample standard deviation), not
    numpy's default ``N``.

    Parameters
    ----------
    allpath : sequence of sequence of int
        Each element is a path of node indices.
    nodesize : array_like
        Size of each node, indexed the same way as the indices in
        ``allpath``.

    Returns
    -------
    traf_mean, traf_med, traf_min, traf_max, traf_std : numpy.ndarray
        One value per path in ``allpath``.
    """
    nodesize = np.asarray(nodesize, dtype=float)
    path_nodesize = [nodesize[np.asarray(p, dtype=int)] for p in allpath]

    traf_mean = np.array([p.mean() for p in path_nodesize])
    traf_med = np.array([np.median(p) for p in path_nodesize])
    traf_min = np.array([p.min() for p in path_nodesize])
    traf_max = np.array([p.max() for p in path_nodesize])
    traf_std = np.array([p.std(ddof=1) if len(p) > 1 else 0.0 for p in path_nodesize])

    return traf_mean, traf_med, traf_min, traf_max, traf_std
