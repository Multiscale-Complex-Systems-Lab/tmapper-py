"""Port of tmapper_tools/CycleCutter.m from the MATLAB toolbox."""

import numpy as np


def cycle_cutter(cyc, node_name):
    """Cut a cycle into multiple paths at given nodes.

    Port of MATLAB's ``CycleCutter.m``.

    Parameters
    ----------
    cyc : sequence
        A single cycle's path (node names, usually integers).
    node_name : sequence or scalar
        Names of nodes that are the cutting points of the cycle. Names
        not contained in the cycle are ignored.

    Returns
    -------
    list of list
        Each element is a path from one cutting point to the next.
        Wraps around: if there are ``M`` cutting points, this list has
        ``M`` elements, and the last one wraps from the last cutting
        point, through the end of ``cyc``, back through the start, up to
        (and including) the first cutting point.
    """
    cyc = list(cyc)

    if np.isscalar(node_name):
        node_set = {node_name}
    else:
        node_set = set(node_name)

    node_idx = [i for i, v in enumerate(cyc) if v in node_set]
    if not node_idx:
        return [cyc]

    n_path = len(node_idx)
    nodepath = [None] * n_path

    for n in range(n_path - 1):
        nodepath[n] = cyc[node_idx[n]: node_idx[n + 1] + 1]

    nodepath[n_path - 1] = cyc[node_idx[-1]:] + cyc[:node_idx[0] + 1]

    return nodepath
