# Python `filtergraph`: it densified a 99.91%-empty matrix to run a 0.19 s search

**Date:** 2026-08-24
**Repo/branch:** `tmapper-py`, `perf-optimization`
**Produced by:** `scratchpad/fg_profile.py` (step timing at full scale),
`scratchpad/cmp_fg.py` + `cmp_fg_diff.py` (1,296-config old-vs-new sweep),
`scratchpad/mut_equiv.py` (mutation / equivalent-mutant separation),
`scratchpad/full_onmatx.py` (full-scale run on MATLAB's exact X).
**Status:** `[PROPOSED]` — measurements recorded as taken; conclusion not yet
reviewed by Mengsen.

**Reliability note.** Wall times are single runs on a shared Windows machine,
so ±20%. Equality claims (1,296 configurations, identical node/edge counts)
are exact comparisons and carry no such caveat.

## Finding

Step timing of `filtergraph`'s reachability route at N=56,835:

| step | time | share |
|---|---:|---:|
| `to_scipy_sparse_array` | 2.82 s | 12.5% |
| sparse reachability | 0.20 s | 0.9% |
| `R.toarray()` | 1.21 s | 5.3% |
| `Rd & Rd.T` | 5.36 s | 23.7% |
| **`nx.from_numpy_array`** | **12.85 s** | **56.8%** |
| `nx.connected_components` | 0.19 s | 0.9% |
| total | 22.64 s | |

`R` holds 2,775,824 nonzeros out of 3.23 billion cells — **0.0859% dense**.
The code converted it to a dense bool array (3.23 GB), AND-ed it with its
transpose (another 3.23 GB), then had `nx.from_numpy_array` walk all 3.2
billion cells building Python objects — all to feed a connected-component
search that takes 0.19 s.

## The change

`src/tmapper/filtergraph.py`. Stay sparse end to end:

```python
# before
Rd = R.toarray()                                  # 3.23 GB
A_ = (Rd & Rd.T) if reciprocal else (Rd | Rd.T)   # 3.23 GB
g_ = nx.from_numpy_array(A_)                      # 12.85 s
components = sorted(nx.connected_components(g_), key=min)

# after
A_sp = R.multiply(R.T) if reciprocal else (R + R.T)
n_new, idx_newnodes = connected_components(A_sp, directed=False)
```

`A_simp` is likewise kept sparse through the block-average division and fed to
`nx.from_scipy_sparse_array`, since it carries one entry per output edge
(~44 k) against a dense `n_new²`.

**Ordering is load-bearing.** scipy labels components in order of smallest
member index, which is exactly what `sorted(components, key=min)` guaranteed;
component order fixes the numbering of every node in the output. Verified
against the old route on 400 random graphs — 0 disagreed.

## Verification

- **0 of 1,296 configurations differ** from the previous implementation:
  N ∈ {120, 400, 1200} × k ∈ {2,3,5} × texclude ∈ {1,5,30} × prct ∈
  {100, 95, 70} × d ∈ {2, 3, 3.5, 7} × reciprocal ∈ {T,F} × compute_dsimp ∈
  {T,F}. Compared on weighted adjacency, `members`, `nodesize`, `D_simp`, and
  node/edge counts. Covers both the sparse and the dense route.
- Full suite **123 passed** (+1 new test).
- Full-scale run reproduces the network exactly: 6,049 nodes / 44,561 edges,
  same as before the change and same as MATLAB on identical input.

### A real test gap this exposed

Of five mutations, four survived the full 122-test suite. Two were
**equivalent mutants** (0/1,296 configurations changed): leaving self-loops in
(components don't care) and `directed=True` (`A_sp` is symmetric, and scipy
defaults to weak connectivity). Those need no test.

The other two were genuine gaps — **600 of 1,296 configurations changed while
all 122 tests passed**:

- deleting `A_simp.data / denom` entirely, and
- normalising by only the row group size.

Nothing asserted `g_simp`'s edge weights, which are the block-average
connectivity between merged member sets. Added
`test_simplified_edge_weights_are_block_average_connectivity`, computing the
expected weight from the definition — `block.sum() / (|n| * |m|)` — rather
than from the implementation. It catches all three weight mutations tried.

*Process note: a 2-minute tool timeout killed an earlier mutation run mid-way
and left a mutated `filtergraph.py` on disk — `finally` does not run when the
process is killed. Caught by diffing against a saved copy rather than trusting
the harness's "restored" message. Worth keeping mutation runs in the
background, and verifying source integrity afterwards regardless.*

## Result: Python is now at or ahead of MATLAB on this pipeline

N=56,835, identical input (MATLAB's X), identical parameters:

| | MATLAB v2.2 | Python (start of session) | Python (now) |
|---|---:|---:|---:|
| `tknndigraph` | 102.2 s | 1667.1 s | **97.2 s** |
| `filtergraph` | 1.7 s | 23.7 s | **2.6 s** |
| total | 103.9 s | 1690.8 s | **99.8 s** |
| peak memory | 1.96 GB | 7.68 GB | **1.33 GB** |
| network | 6,049 nodes / 44,561 edges | identical | identical |

Both remaining gaps against MATLAB are closed: `filtergraph` went 26.1 s →
2.6 s (10×) and peak memory 7.60 GB → 1.33 GB (5.7×). Python's total is now
slightly ahead of MATLAB's, on ~68% of the memory.

`[UNRESOLVED]` — whether the ~4 s in which Python leads is meaningful, or
inside run-to-run noise on a shared machine. Treat the two as at parity rather
than claiming Python is faster.
