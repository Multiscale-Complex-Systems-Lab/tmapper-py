# Changelog

## 0.2.0 — 2026-08-25

### Changed results — read this before upgrading

**`max_neighbor_dist_prct` now resolves the same cutoff MATLAB does.** Two
fixes land together, and both change which edges you get if you set a
percentile cutoff:

- The percentile is taken over **finite distances only**. Previously the
  masked entries — the diagonal and every temporally-excluded pair — counted
  as `inf` and sat at the top of the distribution, dragging the cutoff upward
  and making it more permissive than requested. With `time_exclude_range=30`
  on 1,500 points, the 99th percentile came out as `inf` outright: a request
  to drop the most distant 1% of neighbours silently applied no cutoff at all.
- The percentile now uses MATLAB's plotting-position convention.
  `numpy.percentile`'s default (`linear`, `i/(n-1)`) and MATLAB's `prctile`
  (`(i-0.5)/n`) genuinely differ — `prctile([1 2 3 4 5], 95)` is `5` where
  `np.percentile(..., 95)` is `4.8`. The port now uses `method="hazen"`, which
  is `(i-0.5)/n` and reproduces `prctile` exactly.

If you used `max_neighbor_dist_prct < 100` in 0.1.0, **your graphs will
change**, and the new ones are the correct ones. The default
(`max_neighbor_dist_prct=100`) is unaffected.

### Added

- `tknndigraph(..., low_memory=True)` builds the graph a block of rows at a
  time and never allocates the N×N distance matrix, so peak memory scales with
  `block_size × N` instead of `N²`. This is what makes large builds possible:
  the full 56,835-point sample dataset needs a 25.8 GB dense `D` otherwise.
  Optional `block_size` trades memory against speed. Requires coordinates, not
  a precomputed distance matrix.
- `filtergraph(..., compute_dsimp=False)` skips the `D_simp` output, which is
  the most expensive remaining step and which nothing in the toolbox consumes.
- New [Performance & scaling](https://multiscale-complex-systems-lab.github.io/tmapper-py/performance/)
  documentation: what the pipeline costs at scale, when to reach for
  `low_memory`, and how to compare results against the MATLAB toolbox.

### Performance

Measured on the full sample dataset (56,835 points, `k=3`, `texclude=30`,
`prct=95`, `maxdist=0.5`), same input both sides. The network built is
identical — 6,049 nodes / 44,561 edges:

| | 0.1.0 | 0.2.0 |
|---|---:|---:|
| `tknndigraph` | 1667.1 s | **97.2 s** |
| `filtergraph` | 23.7 s | **2.6 s** |
| total | 1690.8 s | **99.8 s** |
| peak memory | 7.68 GB | **1.33 GB** |

MATLAB Temporal Mapper 2 v2.2 takes ~103.9 s and 1.96 GB on the same input,
so the two toolboxes are now at parity.

The two largest wins: the low-memory percentile histogram was passing
`np.histogram` an explicit array of a million bin edges, which forces its
general `searchsorted` path and accounted for ~98% of `tknndigraph`'s runtime;
and `filtergraph` was densifying a matrix that is 0.0859% dense into two
3.23 GB arrays to run a connected-component search that takes 0.19 s.

Verified equal to the previous implementation across 540 (`tknndigraph`) and
1,296 (`filtergraph`) parameter configurations, compared on adjacency, member
sets, node sizes, `D_simp`, and the exact resolved distance cutoff.

### Fixed

- `filtergraph`'s simplified-graph edge weights — the block-average
  connectivity between merged member sets — were never asserted by any test.
  No evidence they were ever wrong; nothing would have reported it if they
  became wrong. Now covered by a test deriving the expected value from the
  definition rather than the implementation.
- Date parsing in the Streamlit app no longer depends on a pandas dtype that
  changed in pandas 3.x.

### Note for anyone comparing against MATLAB

Match preprocessing **exactly**, not merely equivalently. MATLAB's `zscore`
and pandas' `(x - mean) / std` are algebraically identical and differ in the
last decimal (~3e-13), which is enough to change ~6% of edges on quantized
data — because `tknndigraph` deliberately admits *every* neighbour tied at the
k-th distance, and a perturbed input has no exact ties left to admit. Handed
bit-identical input, the two toolboxes produce bit-identical graphs at every
size tested. See the
[performance docs](https://multiscale-complex-systems-lab.github.io/tmapper-py/performance/#comparing-results-against-the-matlab-toolbox).

## 0.1.0 — 2026-07-27

First release. Python port of
[Temporal Mapper 2](https://github.com/Multiscale-Complex-Systems-Lab/tmapper2):
the core `tknndigraph` → `filtergraph` pipeline, the cycle/path analysis
toolkit, plotting, and an interactive Streamlit app (`pip install
"tmapper-py[app]"`).
