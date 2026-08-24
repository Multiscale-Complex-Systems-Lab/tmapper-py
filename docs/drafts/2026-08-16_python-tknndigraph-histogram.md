# Python `tknndigraph`: the low-memory percentile histogram was ~98% of the runtime

**Date:** 2026-08-16
**Repo/branch:** `tmapper-py`, `perf-optimization`
**Produced by:** step-by-step timing of `_blocked_build` (inline harness, see
"How the numbers were produced"); full-scale A/B via
`scratchpad/full_ab.py`, config grid via `scratchpad/cmp_lowmem.py`,
mutation check via `scratchpad/mutate_hist.py`.
**Status:** `[PROPOSED]` — measurements below are recorded as taken; the
conclusion has not yet been reviewed by Mengsen.

**Reliability note.** Timings are single runs on one Windows machine with
other processes present, so treat them as ±20% on wall time. The
*correctness* claims (bit-identical cutoffs, identical adjacency) are exact
comparisons, not estimates, and carry no such caveat.

## Background

`low_memory=True` (added in `17737ec`) builds the k-NN digraph a block of rows
at a time, so no (N, N) distance matrix is ever allocated. That made the full
56.8k-point sample dataset buildable at all — a dense `D` would be 25.8 GB.
But it was slow: an earlier full run measured **854 s total, of which
`tknndigraph` was 842 s and `filtergraph` only 11 s**, against ~84 s for the
equivalent MATLAB v2.2 build.

*(That 854 s run used a delay embedding whose parameters I could not
reconstruct — 56,746 points against 56,835 here — so it is quoted only as the
motivation. Every before/after number below comes from running both
implementations on the same input, which is why the "old" full-scale figure is
1667 s rather than 842 s. I have not explained that 2× gap; the likely
candidates are the different embedding dimension and machine load. It does not
affect the A/B, which holds the input fixed.)*

## Scope: this only ever bit when a percentile cutoff was requested

`need_prct = prct < 100.0`, so at the default `max_neighbor_dist_prct=100`
the histogram is skipped entirely and none of this applied. **The Streamlit
app also defaults to 100.** So the affected users are those who set a
percentile cutoff explicitly — which the demo parameters (95) and the MATLAB
comparison runs do. Worth stating plainly so the speedup is not read as
applying to every build.

## Finding: one `np.histogram` call was almost the entire cost

Timing one representative block (N=20,000, k=3, texclude=30,
block_size=2,500 → 8 blocks; each block is 2,500 × 20,000 = 50 M distances):

| step | per block | × 8 blocks |
|---|---:|---:|
| `cdist` | 143 ms | 1.15 s |
| `isfinite` + extract | 169 ms | 1.35 s |
| **`np.histogram(vals, bins=edges)`** | **37,709 ms** | **301.67 s** |
| `argpartition` + `take_along_axis` | 361 ms | 2.89 s |
| compare + `nonzero` | 205 ms | 1.64 s |

The histogram was **~98% of `tknndigraph`'s runtime** — 264× the cost of the
`cdist` that produced the data it consumes.

**Cause.** The percentile cutoff needs the global distance distribution, which
blocking refuses to hold, so it is accumulated as a histogram during the same
pass. That accumulation passed `np.histogram` an explicit 1,000,001-element
array of bin edges. Handing `np.histogram` an *array* of edges forces its
general code path — a `searchsorted` over a million edges for every one of the
50 M values per block — even though these edges are `np.linspace(0, hi, ...)`,
uniform by construction.

Measured alternatives on one real block (50 M finite values, 10⁶ bins), counts
verified bit-identical to the current call in every case:

| approach | time | speedup |
|---|---:|---:|
| current: `np.histogram(vals, bins=edges)` | 37.7 s | 1× |
| `np.histogram(vals, bins=n_bins, range=(0,hi))` | 5.49 s | 6.9× |
| **manual `(v*scale).astype(intp)` → `np.bincount`** | **0.53 s** | **71×** |
| manual, chunked to bound the index temporary | 0.59 s | 64× |

Manual bin arithmetic beats numpy's own uniform-bin fast path by ~10× because
it skips numpy's internal re-chunking, which re-allocates and re-zeros a
10⁶-element accumulator every 65,536 values.

## The change

`src/tmapper/tknndigraph.py`, `_blocked_build` only. The histogram is
accumulated by computing the bin index arithmetically —
`floor(v / hi * n_bins)`, clipped to the last bin — and summing with
`np.bincount`. Chunked over rows (`_HIST_CHUNK`, 8 M values) so the `intp`
index array stays bounded, which also removes the previous whole-block
`Db[np.isfinite(Db)]` copy (~400 MB at the default block size).

**That memory saving did not show up in the measurement** — peak was 7.68 GB
before and 7.71 GB after, i.e. unchanged within noise. Peak is evidently set
elsewhere in the pipeline (the candidate-edge accumulation lists, or
`filtergraph`), not by this temporary. Recording it as a null result: the
change is a speed fix, and should not be described as a memory fix.

Bin count and range are unchanged, so the documented accuracy — exact to one
bin width, one part in 10⁶ of the bounding-box diagonal — is unchanged.

### The clip is load-bearing, and only sometimes

`hi` is the bounding-box diagonal, so a real pair *can* sit exactly at `hi`
(two opposite corners). Whether `hi * scale` then lands on index `n_bins` —
one past the last bin, which makes `np.bincount` return an array one element
longer than the accumulator and the `+=` raise — depends on how `hi` and the
reciprocal round. **In a scan of 4,000 random point sets it went that way 887
times (22%).** The unit-cube fixture I first wrote happened to be in the safe
78%, and passed with the clip deleted; the committed fixture is scaled by 1.5
specifically to trip it, and asserts that it still does.

## Verification

- Full suite: **122 passed** (was 120; +2 new tests).
- **Mutation check** — all four mutations caught by the two new tests: scale
  factor halved, clip removed, only the first chunk accumulated, chunk stride
  ignoring the row offset. *(A fifth, `(n_bins-1)/hi`, is deliberately not
  tested: it shifts the bin index by at most one bin, i.e. exactly the
  documented tolerance, so it is an equivalent mutant rather than a defect.)*
- **Cutoff equality, old vs new**, N ∈ {1500, 4000} × prct ∈ {95, 80, 99}:
  resolved `max_neighbor_dist` **bit-identical in all six**, and each within
  one bin width of the exact dense-path percentile.
- **Config grid, old vs new on the blocked path: 0 of 540 configurations
  differ.** N ∈ {120, 400, 1200} × 2 seeds × k ∈ {2,3,5} × texclude ∈
  {1,5,30} × prct ∈ {100, 99, 95, 80, 50} × reciprocal ∈ {True, False},
  alternating `block_size` between the default and 37. Compared on both the
  full adjacency matrix and the exact resolved `max_neighbor_dist`.
- **Full 56,835-point A/B, old vs new, identical input and parameters**
  (`k=3, texclude=30, prct=95, maxdist=0.5, low_memory=True`), each in its own
  process:

  | | old | new |
  |---|---:|---:|
  | `tknndigraph` | 1667.1 s | **98.2 s** (17.0×) |
  | `filtergraph` | 23.7 s | 24.0 s |
  | total | 1690.8 s | **122.2 s** (13.8×) |
  | peak memory | 7.68 GB | 7.71 GB |
  | network | 6,517 nodes / 44,537 edges | identical |

  The per-node `nodesize` vectors are element-wise identical (6,517 entries),
  so the two builds agree on the graph, not merely on its size.

## Where Python now stands against MATLAB v2.2

Same machine, same input, same parameters (`k=3, texclude=30, prct=95,
maxdist=0.5, lowMemory`), N=56,835. The Python column is a run on **MATLAB's
exact X**, loaded from `parity_X.mat`, so the two do identical work:

| | MATLAB v2.2 | Python (before) | Python (now) |
|---|---:|---:|---:|
| `tknndigraph` | 102.2 s | 1667.1 s | **105.3 s** |
| `filtergraph` | 1.7 s | 23.7 s | 26.1 s |
| total | 103.9 s | 1690.8 s | **131.4 s** |
| peak memory | 1.96 GB | 7.68 GB | 7.60 GB |
| network | 6,049 nodes / 44,561 edges | — | **identical** |

`tknndigraph` is now at parity (105.3 s vs 102.2 s, ~3%). **Two gaps remain,
and they are now the dominant ones:** `filtergraph` is ~15× slower in Python
(26.1 s vs 1.7 s), and peak memory is ~3.9× higher (7.60 GB vs 1.96 GB). The
memory ceiling clearly does not live in `tknndigraph`, which is why removing
the histogram's 400 MB temporary changed nothing.

**MATLAB does not have the same bug.** Its `blockedBuild` makes the
structurally identical call — `histcounts(finiteVals, edges)` with an explicit
edges vector — but pays ~102 s where the same pattern cost Python ~1,600 s, so
`histcounts` evidently special-cases uniformly-spaced edges where
`np.histogram` does not. Checked by measurement rather than assumed, since
assuming is what produced the Python version.

## Parity: the ports agree bit-for-bit; the *preprocessing* does not

The first MATLAB/Python full-scale pair produced different networks (6,049 vs
6,517 nodes), which looked like a port defect. It is not:

| input | result |
|---|---|
| MATLAB's X, n=1500 | identical (0 edges either way) |
| MATLAB's X, n=5000 | identical (0 edges either way) |
| MATLAB's X, N=56,835 (full) | identical: 916,608 edges → 6,049 nodes / 44,561 edges |
| each side deriving its own X, n=1500 | 296 Python-only, 198 MATLAB-only |
| each side deriving its own X, n=5000 | 1,286 Python-only, 1,066 MATLAB-only |

MATLAB's `zscore` and pandas' `(x - mean) / std` differ by **~3e-13** —
normally invisible. It matters here because the data carries large numbers of
*exactly equal* pairwise distances, and the `Db <= dmax` rule admits every tie
at the k-th distance.

Tie density, measured on this dataset:

| | |
|---|---:|
| distinct `(tmax, tmin)` | 4,122 |
| distinct `(tmax, tmin, prcp)` | 20,173 |
| exact duplicate rows | 36,662 of 56,835 (65%) |
| rows whose k-th distance is exactly tied (k=3, n=5,000) | 3,401 (68%) |
| mean neighbours admitted per row | 2.94 (max 18) |

`prcp` genuinely adds resolution — it more than quadruples the distinct states
— but it does not break the ties, being itself quantized (268 distinct values)
and zero on 37,168 of 56,835 days.

**~~Struck 2026-08-24: an earlier version of this section called the pipeline
"ill-conditioned on quantized data" and marked the behaviour `[UNRESOLVED]`.
That was wrong, and inverted the point.~~** Admitting every tie is deliberate:
when distances are exactly equal, selecting exactly k of them would be
arbitrary, so `Db <= dmax` takes all of them and the result is *deterministic*.
Ties are pervasive here, so that rule is load-bearing rather than incidental.

The correct reading of the 6% divergence: a perturbed input has **no exact ties
at all**, so "admit every tied neighbour" silently degrades into "admit an
arbitrary k". The float noise does not expose fragility in the tie rule — it
destroys the exact arithmetic the rule depends on. That argues *for* the
design, and against letting preprocessing introduce last-decimal noise.

Practical consequence, and the part still worth care: matching results across
the two toolboxes requires preprocessing matched *exactly*, not merely
equivalently. `zscore` and `(x - mean) / std` are algebraically identical and
differ in the last decimal, which is enough. Whether the divergence changes the
final network's *structure* — as opposed to its node and edge counts — has not
been investigated.

## What was measured and deliberately left alone

- `argpartition` → `np.partition` (only the k-th smallest is used): measured
  **374 ms vs 361 ms — slower**, same temporary size. Not changed.
- `cdist`, the `nonzero` gather, `block_size` default: all small relative to
  the total and all irreducible O(N²) scans.
- `filtergraph`: 11 s of the 854 s. **An earlier claim in conversation that
  `filtergraph` was the bottleneck was wrong and is withdrawn** — it was based
  on a run I mistakenly believed had failed. Python buffers stdout when
  redirected, so the empty output file meant "still running", not "hung".

## Open follow-up (not in this change)

`tknndigraph` remains O(N²): every pair is visited to find k neighbours. A
spatial index (`scipy.spatial.cKDTree`, already available via scipy) could make
it O(N log N), but interacts awkwardly with the temporal-exclusion mask and the
global-percentile cutoff, both of which currently assume the full distance
distribution is seen. `[UNRESOLVED]` — worth a separate look, not folded in
here.
