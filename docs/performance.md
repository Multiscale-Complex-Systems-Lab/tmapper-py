# Performance & scaling

How large a dataset the pipeline can handle, what it costs, and which options
matter when it gets big.

All figures below were measured on the full `EL_temp` sample dataset —
**56,835 points** in 3 z-scored dimensions, with `k=3`, `texclude=30`,
`prct=95`, `maxdist=0.5` — on one Windows desktop. Treat wall times as ±20%:
they are single runs on a machine doing other things. They are here to convey
*orders of magnitude and which knob matters*, not as a benchmark.

## The short version

| | time | peak memory |
|---|---:|---:|
| `tknndigraph` | 97 s | |
| `filtergraph` | 3 s | |
| **whole pipeline, 56,835 points** | **~100 s** | **1.3 GB** |

The equivalent MATLAB v2.2 build takes ~104 s and 1.96 GB on the same input,
so the two toolboxes are at parity.

## Use `low_memory=True` above ~10,000 points

The default path forms a full N×N distance matrix. That is fine while it fits
and fastest when it does, but it grows quadratically and becomes the binding
constraint quickly:

| points | dense `D` alone |
|---:|---:|
| 5,000 | 0.2 GB |
| 10,000 | 0.8 GB |
| 20,000 | 3.2 GB |
| 56,835 | **25.8 GB** |

`low_memory=True` builds the graph a block of rows at a time and never
allocates that matrix, so peak memory becomes proportional to
`block_size × N` rather than `N²`:

```python
g, par = tknndigraph(X, k=3, tidx=tidx, time_exclude_range=30,
                     max_neighbor_dist_prct=95, low_memory=True)
```

It needs coordinates `X`, not a precomputed distance matrix — with `D` already
in hand there is nothing left to save, and passing one raises. Results are
identical to the default path: verified across 540 parameter configurations on
both the adjacency matrix and the exact resolved distance cutoff.

`block_size` trades memory against speed and defaults to roughly 400 MB per
block. You rarely need to set it.

## Skip `D_simp` if you do not use it

`filtergraph`'s fourth return value is the shortest-path distance between every
pair of merged nodes. Computing it requires all-pairs shortest paths, which is
the most expensive step in that function, and nothing else in the toolbox
consumes it:

```python
g_simp, members, nodesize, _ = filtergraph(g, d=3, compute_dsimp=False)
```

At full scale this is the difference between the reachability route and a
dense all-pairs computation.

## What is still quadratic

`tknndigraph` visits every pair to find each point's k nearest neighbours, so
its *time* remains O(N²) even under `low_memory` — that option bounds memory,
not work. Doubling the number of points roughly quadruples the runtime. A
spatial index would change that, but interacts awkwardly with the
temporal-exclusion mask and the global percentile cutoff, both of which assume
the full distance distribution is seen. Not currently implemented.

## Comparing results against the MATLAB toolbox

If you are checking Python output against MATLAB output, **match the
preprocessing exactly, not merely equivalently.**

MATLAB's `zscore` and pandas' `(x - mean) / std` are algebraically identical
and differ in the last decimal place (~3e-13 on this dataset). That is normally
invisible, but it is enough to change the graph, because `tknndigraph` admits
*every* neighbour tied at the k-th distance — deliberately, since choosing k of
several exactly-equal candidates would be arbitrary.

Ties are common in real data. On the `EL_temp` sample, where `tmax` and `tmin`
are whole degrees:

| | |
|---|---:|
| exact duplicate rows | 36,662 of 56,835 (65%) |
| rows whose k-th distance is exactly tied | 68% |
| mean neighbours admitted per row (k=3) | 2.94 |

A perturbed input has **no exact ties at all**, so "admit every tied
neighbour" quietly becomes "admit an arbitrary k". Deriving `X` separately in
each toolbox changed ~6% of edges; feeding both the *same* `X` produced
bit-identical graphs at every size tested, up to and including all 56,835
points.

So: export `X` from one toolbox and load it in the other — do not re-derive it
on both sides and expect agreement.
