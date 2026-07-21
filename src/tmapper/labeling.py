"""Port of tmapper_tools/findnodelabel.m from the MATLAB toolbox."""

import numpy as np
from scipy import stats


def find_node_label(members, x_label, *, labelmethod="mode"):
    """Aggregate a per-time-point label to a per-node label.

    Port of MATLAB's ``findnodelabel.m``.

    Parameters
    ----------
    members : sequence of sequence of int
        ``members[n]`` gives the positional indices belonging to node n.
    x_label : array_like
        Label/value for each time point (indexed the same way as the
        indices inside ``members``).
    labelmethod : {'mode', 'mean', 'median', 'none'} or callable, default 'mode'
        How to aggregate each node's members' x_label values. A callable
        is applied as ``labelmethod(x_label[members[n]])`` per node and
        must return a scalar. 'none' gives every node the same label (0).

    Returns
    -------
    numpy.ndarray, shape (len(members),)
        The aggregated label for each node.
    """
    x_label = np.asarray(x_label)

    if callable(labelmethod):
        return np.array([labelmethod(x_label[list(m)]) for m in members])

    if labelmethod == "mode":
        # scipy.stats.mode returns the smallest value among ties, matching
        # MATLAB's mode().
        return np.array([stats.mode(x_label[list(m)], keepdims=False).mode for m in members])
    elif labelmethod == "mean":
        return np.array([np.mean(x_label[list(m)]) for m in members])
    elif labelmethod == "median":
        return np.array([np.median(x_label[list(m)]) for m in members])
    elif labelmethod == "none":
        return np.zeros(len(members))
    else:
        raise ValueError(f"Unknown labelmethod: {labelmethod!r}")
