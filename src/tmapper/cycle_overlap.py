"""Port of tmapper_tools/CyclePathOverlap.m from the MATLAB toolbox."""

import numpy as np


def cycle_path_overlap(c, *, cycle=True, overlap_type="edge", grpvar=None):
    """Calculate the overlap between cycles or paths.

    Port of MATLAB's ``CyclePathOverlap.m``.

    Parameters
    ----------
    c : sequence of sequence of int
        ``c[n]`` is a path of node indices.
    cycle : bool, default True
        Whether the paths in ``c`` are cycles (adds a wraparound edge
        from the last node back to the first when ``overlap_type``
        is 'edge').
    overlap_type : {'edge', 'node'}, default 'edge'
        Whether to calculate overlap between edges or between nodes.
    grpvar : array_like of int, optional
        A grouping variable for the paths. By default, each path is its
        own group.

    Returns
    -------
    CO : numpy.ndarray, shape (P, P)
        Percent overlap between every pair of groups (1 on the
        diagonal).
    grpnames : numpy.ndarray
        The unique group labels, in the same order as ``CO``'s rows.
    """
    if overlap_type == "edge":
        def as_items(x):
            x = list(x)
            edges = list(zip(x, x[1:] + [x[0]]))
            if not cycle:
                edges = edges[:-1]  # drop the wraparound edge
            return edges
    elif overlap_type == "node":
        def as_items(x):
            return list(x)
    else:
        raise ValueError(f"Unknown type: {overlap_type!r}")

    c_repr = [as_items(x) for x in c]

    Nc = len(c_repr)
    if grpvar is None:
        grpvar = np.arange(Nc)
    grpvar = np.asarray(grpvar)
    grpnames = np.unique(grpvar)
    Ngrp = len(grpnames)

    groups = []
    for name in grpnames:
        items = set()
        for idx in np.flatnonzero(grpvar == name):
            items |= set(c_repr[idx])
        groups.append(items)

    CO = np.full((Ngrp, Ngrp), np.nan)
    for ii in range(Ngrp):
        for jj in range(ii):
            inter = len(groups[ii] & groups[jj])
            union = len(groups[ii] | groups[jj])
            CO[ii, jj] = inter / union
            CO[jj, ii] = CO[ii, jj]
    np.fill_diagonal(CO, 1)

    return CO, grpnames
