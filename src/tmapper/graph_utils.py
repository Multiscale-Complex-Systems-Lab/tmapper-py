"""Small graph/data utilities. Ports of tmapper_tools/nodesize.m,
nodemeasure.m, normgeo.m, normtcm.m, members2tidx.m,
subgraphFromMembers.m, symDyn2digraph.m, digraph2graph.m, and
findtaskn.m from the MATLAB toolbox.

weightedAdj.m, zerodiag.m, and toVec.m are not ported: they are pure
MATLAB-object-compatibility shims with no standalone meaning in Python
(nx.to_numpy_array, np.fill_diagonal, and flat numpy arrays already
cover that ground natively).
"""

import numpy as np
import networkx as nx


def node_size(nodemembers):
    """Number of members in each node. Port of ``nodesize.m``."""
    return np.array([len(m) for m in nodemembers])


def node_measure(nodemembers):
    """Number of members in each node, normalized to sum to 1. Port of
    ``nodemeasure.m``."""
    n = node_size(nodemembers).astype(float)
    return n / n.sum()


def normalize_geodesic(geod, nsize=None, *, exclude_diag=False):
    """Normalize a geodesic distance matrix by node measure.

    Port of ``normgeo.m``. Note: unlike the MATLAB original, an omitted
    ``nsize`` defaults to a uniform *vector* of ones (equal-size nodes),
    which is clearly the intended behavior from the docstring -- the
    MATLAB version's ``ones(N_nodes)`` (a matrix, not ``ones(N_nodes,1)``)
    appears to be an unexercised bug in that rarely-hit default-value path.

    Parameters
    ----------
    geod : array_like, shape (N, N)
        Geodesic distance matrix.
    nsize : array_like, shape (N,), optional
        Size of each node. Defaults to a uniform vector of ones.
    exclude_diag : bool, default False
        Whether to exclude the diagonal when computing the normalizing
        factor.

    Returns
    -------
    geod_n : numpy.ndarray, shape (N, N)
        Normalized geodesic distance matrix.
    nm : numpy.ndarray, shape (N,)
        Node measure (probability), sums to 1.
    """
    geod = np.array(geod, dtype=float, copy=True)
    n_nodes = geod.shape[0]

    if nsize is None:
        nsize = np.ones(n_nodes)
    nsize = np.asarray(nsize, dtype=float)

    # -- handle inf
    if np.any(np.isinf(geod)):
        finite = geod[np.isfinite(geod)]
        geod[np.isinf(geod)] = finite.max() if finite.size else 0.0

    # -- weight nodes and geodesics
    nm = nsize / nsize.sum()
    geod_n = np.outer(nm, nm) * geod ** 2

    # -- normalizing factor: the sum of weighted geodesics
    if exclude_diag:
        mask = ~np.eye(n_nodes, dtype=bool)
        normfactor = np.sqrt(geod_n[mask].sum())
    else:
        normfactor = np.sqrt(geod_n.sum())

    # -- normalize
    if normfactor != 0:
        geod_n = geod / normfactor

    return geod_n, nm


def normalize_tcm(tcm, *, normtype="max", infreplace="max"):
    """Normalize a temporal connectivity matrix by its maximal finite
    value (or its norm). Port of ``normtcm.m``.

    Parameters
    ----------
    tcm : array_like
        Temporal connectivity matrix.
    normtype : {'max', 'norm'}, default 'max'
        How to compute the normalizing factor: the matrix's maximum
        finite value, or its (Frobenius/vector) norm.
    infreplace : {'max', 'nan'}, default 'max'
        How to handle inf entries before normalizing: replace with the
        matrix's maximum finite value, or with NaN.

    Returns
    -------
    numpy.ndarray
    """
    tcm = np.array(tcm, dtype=float, copy=True)

    if infreplace == "max":
        finite = tcm[np.isfinite(tcm)]
        if finite.size:
            tcm[np.isinf(tcm)] = finite.max()
    elif infreplace == "nan":
        tcm[np.isinf(tcm)] = np.nan
    else:
        raise ValueError(f"Unknown infreplace: {infreplace!r}")

    if normtype == "max":
        normfactor = np.nanmax(tcm)
    elif normtype == "norm":
        normfactor = np.linalg.norm(tcm.ravel())
    else:
        raise ValueError(f"Unknown normtype: {normtype!r}")

    if normfactor != 0:
        return tcm / normfactor
    return tcm


def members_to_tidx(members, tidx):
    """Translate positional-index members to real ``tidx`` values.

    Port of ``members2tidx.m``. Use this when ``tidx`` is non-contiguous
    or offset (e.g. real timestamps with gaps): :func:`filtergraph`'s
    ``members`` output gives positional indices into your original data,
    not ``tidx`` values.

    Parameters
    ----------
    members : sequence of sequence of int
        Positional indices into the original data, e.g. from
        :func:`filtergraph`.
    tidx : array_like
        The same ``tidx`` array originally passed to :func:`tknndigraph`.

    Returns
    -------
    list of numpy.ndarray
    """
    tidx = np.asarray(tidx)
    return [tidx[np.asarray(list(m), dtype=int)] for m in members]


def subgraph_from_members(g_simp, members, include_members):
    """Extract the subgraph of a transition network restricted to a
    subset of original time points. Port of ``subgraphFromMembers.m``.

    Parameters
    ----------
    g_simp : networkx.Graph or networkx.DiGraph
        The simplified transition network.
    members : sequence of sequence of int
        ``members[n]`` gives the original time-point indices belonging
        to node n (in the same node order as ``g_simp.nodes()``).
    include_members : iterable of int
        The subset of original time points to keep.

    Returns
    -------
    g_sub : networkx.Graph or networkx.DiGraph
        The induced subgraph containing only nodes with at least one
        member in ``include_members``.
    members_sub : list of list
        Members of each node of ``g_sub``, restricted to
        ``include_members``.
    sub_orig_nodeidx : list of int
        Positional indices (into the original ``g_simp.nodes()``) of the
        nodes kept in ``g_sub``.
    """
    include_set = set(include_members)
    members_sub_all = [sorted(set(m) & include_set) for m in members]
    sub_orig_nodeidx = [i for i, m in enumerate(members_sub_all) if len(m) > 0]

    nodelist = list(g_simp.nodes())
    sub_nodes = [nodelist[i] for i in sub_orig_nodeidx]
    g_sub = g_simp.subgraph(sub_nodes).copy()
    members_sub = [members_sub_all[i] for i in sub_orig_nodeidx]

    return g_sub, members_sub, sub_orig_nodeidx


def sym_dyn_to_digraph(sym_dyn):
    """Construct a directed graph from a single time series of symbolic
    dynamics. Port of ``symDyn2digraph.m``.

    Parameters
    ----------
    sym_dyn : array_like of int
        A vector of N labels (purely nominal -- numeric differences
        between labels are irrelevant), one per time point.

    Returns
    -------
    dg : networkx.DiGraph
        One node per unique state in ``sym_dyn``; an edge wherever a
        transition between two states is observed.
    dwelltime : numpy.ndarray
        Number of time points belonging to each state (in the same
        order as ``dg.nodes()``).
    nodemembers : list of numpy.ndarray
        ``nodemembers[n]`` gives the time-point indices where the
        system was in ``dg``'s n-th state.
    """
    sym_dyn = np.asarray(sym_dyn)
    tidx = np.arange(len(sym_dyn))

    state_names, state_idx = np.unique(sym_dyn, return_inverse=True)
    dwelltime = np.bincount(state_idx, minlength=len(state_names))

    dg = nx.DiGraph()
    dg.add_nodes_from(state_names.tolist())

    if len(sym_dyn) > 1:
        changed = np.diff(sym_dyn) != 0
        tidx_trans = tidx[:-1][changed]
        state_s = sym_dyn[tidx_trans]
        state_t = sym_dyn[tidx_trans + 1]
        trans = np.unique(np.stack([state_s, state_t], axis=1), axis=0) if tidx_trans.size else np.empty((0, 2))
        dg.add_edges_from((s, t) for s, t in trans.tolist())

    nodemembers = [tidx[state_idx == n] for n in range(len(state_names))]

    return dg, dwelltime, nodemembers


def digraph_to_graph(dg):
    """Convert a directed graph to an undirected graph, averaging the
    weights of the two edges between each pair of nodes. Port of
    ``digraph2graph.m``.

    Parameters
    ----------
    dg : networkx.DiGraph
        The directed graph to symmetrize.

    Returns
    -------
    networkx.Graph
    """
    nodelist = list(dg.nodes())
    A = nx.to_numpy_array(dg, nodelist=nodelist, weight="weight")
    A_sym = (A + A.T) / 2
    g = nx.from_numpy_array(A_sym)
    return nx.relabel_nodes(g, {i: nodelist[i] for i in range(len(nodelist))})


def find_blocks(indicator):
    """Find the start, end, and size of contiguous True/1 runs
    ("blocks") in a 0/1 indicator array. Port of ``findtaskn.m``.

    Parameters
    ----------
    indicator : array_like
        A 1D array of 0s/1s (or booleans).

    Returns
    -------
    block_start : numpy.ndarray
        0-indexed position of the first sample of each block.
    block_end : numpy.ndarray
        0-indexed position of the last sample of each block.
    block_size : numpy.ndarray
        Number of samples in each block.
    """
    ind = np.asarray(indicator).astype(int).ravel()

    padded_start = np.concatenate([[0], ind])
    block_start = np.flatnonzero(np.diff(padded_start) == 1)

    padded_end = np.concatenate([ind, [0]])
    block_end = np.flatnonzero(np.diff(padded_end) == -1)

    if len(block_start) != len(block_end):
        raise ValueError("Block structure incomplete.")

    block_size = block_end - block_start + 1
    return block_start, block_end, block_size
