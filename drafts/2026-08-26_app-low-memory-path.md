# The Streamlit app on the low-memory path

**Status: APPROVED — reviewed and agreed by Mengsen, 2026-08-26**

Produced by: `scratchpad/measure_one.py` (one measurement per process),
run on 2026-08-26 against `tmapper-py` at branch `app-low-memory`
(commits `83c609f`, `c452f31`).

**Reliability note.** Timings and peak-memory figures are single runs on one
Windows laptop, one dataset, one parameter set. They support the *ordering*
claims below (blocked beats dense, by margins well outside plausible noise
from N=2000 up) and the *scaling* claim (quadratic). They do not support
precise per-machine time predictions; the estimate shipped in the app is
labelled "roughly" for that reason.

---

## What prompted this

`tmapper-py` 0.2.0 added `tknndigraph(..., low_memory=True)`, a builder that
scans coordinates a block of rows at a time instead of allocating the
(N, N) distance matrix. The Streamlit app never adopted it: `build_network`
still did `cdist(X, X)` and handed over a dense matrix.

Consequently the app carried a hard refusal at 8000 rows — `cdist` would
have allocated N² float64s — while the library it wraps could build the
whole 57 709-row file. **The app was the one place in the toolbox that could
not do what the toolbox could do.**

## Correction to an earlier claim

An earlier commit on this branch (`83c609f`) introduced a size threshold and
justified it in both a code comment and the commit message:

> ~~"the dense path stays for small windows because it is faster"~~

**This is wrong.** It was reasoning about blocking overhead, never measured.
Measurement (below) shows the blocked path is faster at every size tried.
The threshold selected between a worse option and a better one; `c452f31`
removes it.

## Measurement

**Method.** One fresh Python process per (N, path) — required because
`psutil`'s `peak_wset` is a process high-water mark that never decreases, so
measuring both paths in one process makes every run after the first report
the first one's peak. (An earlier run of mine did exactly that and reported
`0.00 GB` for every blocked build; those numbers were discarded.)

Each process runs `app.build_network` (= `tknndigraph` + `filtergraph`) on
the **last N rows** of the bundled `EL_temp.csv`, variables
`(tmax, tmin, prcp)`, z-score on, `k=3, d=3.0, texclude=30, prct=100,
maxdist=0.5, reciprocal=True, downsample=1`.

- `wall` — seconds, `time.perf_counter` around the call. Lower is better.
- `peak` — `Process().memory_info().peak_wset` delta from process start, ÷1e9.
  Peak *physical* memory the process held. Lower is better.
- `graph` — node and edge counts, as a guard against a silently degenerate build.

| N | dense wall | blocked wall | dense peak | blocked peak | nodes/edges (both) |
|---:|---:|---:|---:|---:|---|
| 500 | 0.019 s | 0.017 s | 0.00 GB | 0.00 GB | 150 / 357 |
| 1 000 | 0.045 s | 0.027 s | 0.02 GB | 0.01 GB | 258 / 829 |
| 2 000 | 0.145 s | 0.062 s | 0.12 GB | 0.06 GB | 441 / 1 658 |
| 4 000 | 0.703 s | 0.190 s | 0.53 GB | 0.27 GB | 898 / 3 688 |
| 8 000 | 2.804 s | 0.645 s | 2.16 GB | 0.95 GB | 1 814 / 7 656 |
| 16 000 | 10.593 s | 2.564 s | 8.69 GB | 1.24 GB | 3 284 / 15 393 |

Larger blocked-only runs, same setup: **32 000 → 10.29 s**, **57 000 → 31.82 s**.

**Conclusion** (agreed 2026-08-26). The blocked builder dominates the dense one on both
axes at every size measured — 1.1× faster at N=500 (within noise), rising to
~4× from N=8000 on, with 2–7× less peak memory. Graphs are identical. There
is no size at which the dense path is the better choice, so the app should
use the blocked one unconditionally.

**Why blocked is also faster, not just smaller** — proposed, not established:
the dense path allocates and repeatedly scans an N² array, while the blocked
path works in chunks that stay in cache. This was not measured (no cache
counters), and it is not load-bearing for the decision.

## Scaling of the remaining cost

`SECONDS_PER_PAIR = wall / N²`, from the blocked runs:

| N | wall | wall / N² |
|---:|---:|---:|
| 8 000 | 0.645 s | 1.008 × 10⁻⁸ |
| 16 000 | 2.564 s | 1.002 × 10⁻⁸ |
| 32 000 | 10.29 s | 1.005 × 10⁻⁸ |
| 57 000 | 31.82 s | 0.979 × 10⁻⁸ |

Flat to ~3%, consistent with the quadratic scaling the algorithm implies
(every pair of points is compared). The app uses `1.0e-8` for its estimate.
**This constant is machine-specific** — it is a laptop figure and the message
says "roughly".

## What changed in the app

1. `build_network` always calls `tknndigraph(X, ..., low_memory=True)`. No
   threshold, no branch.
2. The 8000-row **refusal** became a **warning** above 16 000 points, with a
   time estimate, that does not gate the build. *(Decision: Mengsen, 2026-08-26,
   asked to warn rather than refuse.)* Rationale: memory was the reason for a
   hard limit and memory is no longer the constraint.
3. The code panel emits the `low_memory=True` form and no longer imports
   `cdist`.

## Test notes

Six mutations applied to the app; five killed immediately, one survived and
exposed a real weakness worth recording:

**Dropping `low_memory=True` from the *generated* code was not caught.** Two
causes, both instructive:

- `tknndigraph` accepts coordinates **or** a distance matrix
  (`tknndigraph.py:151`). Given coordinates without the flag it calls `cdist`
  itself — so the emitted script produces the *identical graph* while making
  exactly the N² allocation the flag exists to avoid. **A graph comparison
  cannot detect a missing memory flag.**
- `assert "low_memory=True" in code` was satisfied by the generated
  *comment*, not the call.

Fixed by asserting on the parsed call line. All six mutations now killed.

Full suite: **130 passed** (123 before this work).

## Open

- `[UNRESOLVED]` The warning threshold (16 000 points ≈ 2.6 s) and the
  90-second seconds→minutes switchover are round numbers chosen for
  legibility, not derived from anything. No evidence they are the right
  round numbers.
- `[UNRESOLVED]` `SECONDS_PER_PAIR` is calibrated on one machine. A slower
  machine will see the estimate under-predict, with no mechanism to notice.
  Alternative not pursued: time the first build and calibrate from it.
- The app's O(N²) is inherited from `tknndigraph` itself; the `cKDTree`
  follow-up noted in the 0.2.0 work would lift it for both.
