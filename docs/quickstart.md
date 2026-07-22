# Quickstart

This walkthrough builds an attractor transition network end to end and
reproduces the figure on the [home page](index.md). It mirrors the MATLAB
toolbox's own `tmapper_demo.m`, translated step by step into Python.

The sample data is a slice of historical **East Lansing daily weather**
(temperature and precipitation), included in the repo at
`sampledata/EL_temp.csv`.

## Step 0 — Imports

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.spatial.distance import cdist

from tmapper import tknndigraph, filtergraph, plot_tmgraph_tcm
```

## Step 1 — Load and select the data

Read the sample CSV, drop rows with missing values, and keep a recent slice
for a manageable demo:

```python
dat = pd.read_csv("sampledata/EL_temp.csv")
dat = dat.dropna()
dat = dat.iloc[53883:].reset_index(drop=True)   # keep a recent slice for the demo
```

Next, choose which columns define the **state** of the system. Here we take
three weather variables and z-score them so dimensions with different units
are comparable:

```python
X = zscore(dat[["tmax", "tmin", "prcp"]].to_numpy(), axis=0, ddof=1)  # (1)!
t = pd.to_datetime(dat["Date"]).to_numpy()
```

1.  `X` must be organized as **rows = time points, columns = state variables**
    — that convention holds throughout the toolbox.

!!! warning "ddof=1 matters"
    MATLAB's `zscore` divides by the *sample* standard deviation (`N-1`), while
    `scipy.stats.zscore` defaults to the *population* standard deviation (`N`).
    Pass `ddof=1` to match MATLAB's convention. This is the one line where a
    default Python call silently diverges from the MATLAB pipeline — see
    [Concepts & coming from MATLAB](concepts.md) for the full story.

!!! note "Why z-score?"
    Z-scoring puts each variable on a common scale so that one high-variance
    dimension does not dominate the distance computation. It is a sensible
    default, not a hard requirement — skip it if your variables are already
    comparable.

## Step 2 — (Optional) Delay embedding

When data is low-dimensional, distinct dynamical states can overlap and
become hard to separate. A quick fix is **delay embedding**: augment each
time point with a copy of the state from some time earlier (here, 90 days),
lifting the data into a higher-dimensional space where states separate more
cleanly.

```python
X = np.hstack([X[:-89, :], X[89:, :]])   # concatenate current state with the 90-day-lagged state
t = t[89:]                               # align the time vector to match
```

!!! tip "Is this always needed?"
    No. Delay embedding helps when your state space is too compressed to
    distinguish attractors. For richer, higher-dimensional data you can skip
    this step entirely.

## Step 3 — Build the distance matrix

tmapper works from a **pairwise distance matrix** `D`, where `D[i, j]` is the
distance between the state at time `i` and the state at time `j`. This is
exactly a classical recurrence plot. Here we use Euclidean distance
(Minkowski with `p=2`):

```python
D = cdist(X, X, metric="minkowski", p=2)   # N-by-N pairwise distance matrix
```

## Step 4 — Build the spatiotemporal graph

Now the first step of the pipeline. [`tknndigraph`](api.md#tmapper.tknndigraph)
turns the distance matrix into a directed *k*-nearest-neighbor graph — one
node per time point — while preserving temporal order.

```python
# --- construction parameters
k = 3              # (1)!  max number of spatial neighbors per time point
texclude = 30      # (2)!  temporal window (in time points) excluded from being "spatial" neighbors
maxdistprct = 95   # (3)!  neighbor cutoff by percentile of all distances
maxdist = 0.5      # (4)!  neighbor cutoff by absolute distance (the tighter of the two wins)

tidx = np.arange(len(t))   # integer index per time point, used to define temporal neighborhoods

g, par = tknndigraph(
    D, k, tidx,
    time_exclude_range=texclude,
    max_neighbor_dist_prct=maxdistprct,
    max_neighbor_dist=maxdist,
)
```

1.  **`k`** — the maximum number of spatial (nearest-neighbor) links each time
    point may form. Larger `k` produces a denser graph.
2.  **`texclude`** — points within this many steps in time are treated as
    temporal neighbors and are *not* allowed to also count as spatial
    neighbors. This prevents "recurrence" being trivially detected between
    adjacent moments. `1` is a common default; the demo uses `30`.
3.  **`max_neighbor_dist_prct`** — a distance cutoff expressed as a percentile
    of all pairwise distances (95 = ignore the most distant 5% as neighbors).
4.  **`max_neighbor_dist`** — the same cutoff as an absolute distance. When
    both are given, the **smaller** (stricter) threshold is applied.

!!! tip "Which parameters matter most"
    **`k`** is the essential one — it largely determines the topology of the
    graph, so it is the parameter to explore first. **`texclude`** should be
    set from your *sampling density* rather than the dynamics: when a signal
    is sampled quickly, a few consecutive samples reflect the same underlying
    state rather than genuine recurrence, so they should be excluded from
    counting as spatial neighbors. The two distance cutoffs are optional:
    leave them at their defaults unless you suspect outliers you want to keep
    from being linked as neighbors.

The output `g` is a `networkx.DiGraph` with one node per time point (node
labels are **0-indexed**, unlike MATLAB); `par` records the parameters
actually used (handy for reading back the resolved distance cutoff).

## Step 5 — Simplify into the transition network

The second step contracts the fine-grained graph into the attractor
transition network. [`filtergraph`](api.md#tmapper.filtergraph) collapses
time points that lie within distance `d` of one another into a single node.

```python
d = 3   # compression threshold: loops shorter than this are absorbed into a node

g_simp, members, nodesize, D_simp = filtergraph(g, d, reciprocal=True)
```

This returns everything you need to describe and plot the network:

| Output | Meaning |
| --- | --- |
| `g_simp` | the simplified graph — **this is the attractor transition network** |
| `members` | which original time points map into each new node |
| `nodesize` | number of member time points per node (a proxy for stability) |
| `D_simp` | shortest "distances" between the groups of members |

!!! note "What `reciprocal` does"
    With `reciprocal=True`, two time points are only merged when the short
    path holds in **both** directions (x→y *and* y→x). This is the stricter,
    recommended setting for directed graphs.

## Step 6 — Visualize

Finally, plot the network alongside its recurrence plot. First pick a
variable to color the nodes — any per-time-point quantity works; here we use
the daily maximum temperature, `tmax`:

```python
colorvarname = "tmax"
colorvar = dat["tmax"].to_numpy()[89:]

ax1, ax2, cbar, cbar2, node_collection, scatter_collection, D_geo = plot_tmgraph_tcm(
    g_simp, colorvar, t, members,
    nodesizerange=(1, 10),
    colorlabel=colorvarname,
    labelmethod="median",
    nodesizemode="log",
)
ax1.set_title(f"sample data\nk={k}, d={d}\ntx={texclude}, maxdist={par['max_neighbor_dist']:.3g}")
plt.show()
```

You should now see the attractor transition network on the left and the
geodesic recurrence plot on the right — the same figure shown on the
[home page](index.md).

## How to read the network

At a glance, here is what the figure is showing you:

- **Nodes** are attractors — recurring states the system settles into. A
  node's **size** reflects how long the system dwells there (its *local
  stability*).
- **Edges** are transitions between states, and they are **directed**: the
  arrow is the direction of time.
- **Loops** (cycles) are excursions that leave a node and eventually return
  to it. A loop's **length** — the number of transitions — reflects how
  easily that state recurs (a measure of *global stability*).

See [Concepts & coming from MATLAB](concepts.md) for the deeper theory behind
this, and the original [MATLAB toolbox's docs](https://multiscale-complex-systems-lab.github.io/tmapper2/concepts/)
for the full conceptual write-up.

## What's next

- Try changing **`k`** and **`d`** and watch how the network coarsens or
  fragments — these two parameters control the resolution of the result.
- Swap in your **own data**: organize it as rows = time points, columns =
  state variables, build `D` with `scipy.spatial.distance.cdist`, and run the
  same two-step pipeline.
- The toolbox also includes a cycle/path analysis toolkit
  (`tmapper.cycle_count2p`, `tmapper.cycle_path_decomp`, etc.) for probing the
  topology of the resulting network — see the [API Reference](api.md).
