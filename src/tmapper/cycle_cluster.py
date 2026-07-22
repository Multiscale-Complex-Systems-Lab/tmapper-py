"""Port of tmapper_tools/CycleCluster.m from the MATLAB toolbox."""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def _classical_mds(dist, n_components=2):
    """Classical (Torgerson) multidimensional scaling, matching MATLAB's
    ``cmdscale``. ``dist`` is a square distance matrix."""
    dist = np.asarray(dist, dtype=float)
    n = dist.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (dist ** 2) @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    eigvals = np.clip(eigvals[:n_components], 0, None)
    return eigvecs[:, :n_components] * np.sqrt(eigvals)


def cycle_cluster(allcycles, thres, *, plotmat=True, plotmds=False, plothist=True,
                   reordermat=True, ax=None, return_ax=False):
    """Cluster cycles based on the fraction of overlap between them
    (shared nodes / union of nodes).

    Port of MATLAB's ``CycleCluster.m``.

    Parameters
    ----------
    allcycles : sequence of sequence of int
        ``allcycles[n]`` is the path of one cycle.
    thres : float
        Cut-off threshold (0-1) for single-linkage clustering: if
        overlap > thres, two cycles belong to the same cluster.
    plotmat : bool, default True
        Whether to plot the overlap matrix.
    plotmds : bool, default False
        Whether to plot a 2D classical-MDS projection of the cycles.
    plothist : bool, default True
        Whether to plot the histogram of linkage distances.
    reordermat : bool, default True
        Whether to reorder the overlap matrix by cluster assignment.
    ax : matplotlib.axes.Axes, optional
        Axes for the overlap-matrix plot (only used if ``plotmat``).
        A new figure is created if not given.
    return_ax : bool, default False
        If True, also return the axes used for the overlap-matrix plot
        (None if ``plotmat`` is False), so a caller can add further
        annotations to it (e.g. :func:`cycle_path_decomp`'s cluster-block
        outlines).

    Returns
    -------
    numpy.ndarray
        1-indexed cluster label for each cycle (matching MATLAB's
        ``cluster()`` convention).
    ax : matplotlib.axes.Axes or None
        Only returned if ``return_ax`` is True.
    """
    Nc = len(allcycles)

    if Nc <= 1:
        cluster_idx = np.ones(Nc, dtype=int)
        return (cluster_idx, None) if return_ax else cluster_idx

    prct_overlap = np.zeros((Nc, Nc))
    sets = [set(c) for c in allcycles]
    for ii in range(Nc):
        for jj in range(Nc):
            prct_overlap[ii, jj] = len(sets[ii] & sets[jj]) / len(sets[ii] | sets[jj])

    if plotmds:
        import matplotlib.pyplot as plt
        Y = _classical_mds(1 - prct_overlap, 2)
        plt.figure()
        plt.scatter(Y[:, 0], Y[:, 1])

    Z = linkage(squareform(1 - prct_overlap, checks=False), method="complete")

    if plothist:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(1 - Z[:, 2], bins=np.linspace(0, 1, 21))
        ylim = plt.ylim()
        plt.plot([thres, thres], ylim)
        plt.xlabel("overlap")
        plt.ylabel("count")

    cluster_idx = fcluster(Z, t=1 - thres, criterion="distance")

    if plotmat:
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        if reordermat:
            order = np.argsort(cluster_idx, kind="stable")
            im = ax.imshow(prct_overlap[np.ix_(order, order)], origin="lower", vmin=0, vmax=1)
            ax.set_xlabel("cycle/path index (reordered)")
            ax.set_ylabel("cycle/path index (reordered)")
        else:
            im = ax.imshow(prct_overlap, origin="lower", vmin=0, vmax=1)
            ax.set_xlabel("cycle/path index")
            ax.set_ylabel("cycle/path index")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("fraction of overlap")
        ax.set_aspect("equal")
    else:
        ax = None

    return (cluster_idx, ax) if return_ax else cluster_idx
