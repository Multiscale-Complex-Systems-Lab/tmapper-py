"""Port of tmapper_tools/Cycles2Paths.m from the MATLAB toolbox."""

from .cycle_cutter import cycle_cutter


def cycles_to_paths(allcycles, cutpts):
    """Cut a set of cycles to obtain a set of unique paths connecting a
    set of cutting points (nodes).

    Port of MATLAB's ``Cycles2Paths.m``.

    Parameters
    ----------
    allcycles : sequence of sequence of int
        A list of cycle paths.
    cutpts : sequence of int
        Node indices at which each cycle should be cut (cutting points).

    Returns
    -------
    list of list
        Unique paths linking the cutting points, sorted by (length, then
        lexicographically within each length).
    """
    if not allcycles:
        return []

    allpath = []
    for x in allcycles:
        allpath.extend(cycle_cutter(x, cutpts))

    pathlen = [len(p) for p in allpath]
    upathlen = sorted(set(pathlen))

    allupath = []
    for L in upathlen:
        rows = sorted({tuple(p) for p, l in zip(allpath, pathlen) if l == L})
        allupath.extend(list(r) for r in rows)

    return allupath
