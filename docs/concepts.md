# Concepts & coming from MATLAB

The underlying method — what a node, edge, and loop *mean*, why the two-step
construction works, what `k` and `d` really control — is identical between
the MATLAB and Python versions. Rather than duplicate that write-up, read it
on the MATLAB toolbox's site:

- [**Concepts**](https://multiscale-complex-systems-lab.github.io/tmapper2/concepts/) — the full conceptual explanation
- [**Applications**](https://multiscale-complex-systems-lab.github.io/tmapper2/applications/) — published examples in brain and social dynamics

This page instead covers what's specific to using tmapper **from Python**: the
function-name mapping, and the handful of places where a direct MATLAB→Python
translation isn't quite 1:1.

## Function name mapping

| MATLAB | Python |
| --- | --- |
| `tknndigraph` | `tknndigraph` |
| `filtergraph` | `filtergraph` |
| `findnodelabel` | `find_node_label` |
| `TCMdistance` | `tcm_distance` |
| `plottmgraph` | `plot_tmgraph` |
| `plotgraphtcm` | `plot_tmgraph_tcm` |
| `knngraph` | `knngraph` |
| `cknngraph` | `cknngraph` |
| `nodesize` | `node_size` |
| `nodemeasure` | `node_measure` |
| `normgeo` | `normalize_geodesic` |
| `normtcm` | `normalize_tcm` |
| `members2tidx` | `members_to_tidx` |
| `subgraphFromMembers` | `subgraph_from_members` |
| `symDyn2digraph` | `sym_dyn_to_digraph` |
| `digraph2graph` | `digraph_to_graph` |
| `findtaskn` | `find_blocks` |
| `CycleCount` | `cycle_count` |
| `CycleCount2p` | `cycle_count2p` |
| `reorgCycles` | `reorg_cycles` |
| `CyclePathOverlap` | `cycle_path_overlap` |
| `CycleCluster` | `cycle_cluster` |
| `CycleClusterConn` | `cycle_cluster_conn` |
| `CycleCutter` | `cycle_cutter` |
| `Cycles2Paths` | `cycles_to_paths` |
| `CyclePathDecomp` | `cycle_path_decomp` |
| `pathtraffic` | `path_traffic` |
| `Qasym` | `qasym` |
| `calMod` | `cal_mod` |
| `tknngraph`, `weightedAdj`, `zerodiag`, `toVec`, `addDiagBlock` | *not ported* — legacy/unused, or pure MATLAB-object shims with a direct numpy/networkx/matplotlib equivalent (see below) |

## Behavioral differences from the MATLAB toolbox

!!! warning "0-indexed nodes"
    Every graph produced by this package uses **0-indexed** node labels
    (`0..N-1`), matching numpy's row-indexing convention — unlike the MATLAB
    originals, which are 1-indexed. If you're translating a MATLAB snippet by
    hand, subtract 1 from any hand-computed node index.

!!! warning "z-score convention"
    MATLAB's `zscore` divides by the *sample* standard deviation (`N-1`, i.e.
    `ddof=1`). `scipy.stats.zscore` defaults to the *population* standard
    deviation (`ddof=0`). If you z-score your data before calling
    `tknndigraph` — as the [Quickstart](quickstart.md) does — pass `ddof=1`
    to match the MATLAB pipeline exactly. This was verified: with `ddof=1`,
    the Python and MATLAB pipelines produce identical output down to
    floating-point precision on the sample dataset.

!!! note "Network layout"
    `plot_tmgraph` lays out the network with networkx's `kamada_kawai_layout`
    (falling back to `spring_layout` for disconnected graphs), rather than
    MATLAB's gravity-assisted force layout. Both are force/stress-based and
    produce visually similar (and topologically equivalent) layouts, but node
    *positions* will not match pixel-for-pixel between the two toolboxes.

!!! note "Skipped MATLAB-object shims"
    `weightedAdj.m`, `zerodiag.m`, and `toVec.m` have no standalone Python
    port: they exist in MATLAB purely to smooth over `graph`/`digraph`/`table`
    object quirks across MATLAB versions. Use `networkx.to_numpy_array(g)`,
    `numpy.fill_diagonal(A, 0)`, and plain numpy arrays (already flat)
    respectively wherever you'd reach for these in MATLAB.

## Verification

Every ported function was cross-checked against real MATLAB output on
deterministic test graphs — not just self-consistent Python tests. For the
core pipeline (`tknndigraph` + `filtergraph`), this included a node-for-node,
edge-for-edge comparison of the full output (`D_simp`, `A_simp`, node
membership) on the real sample dataset, matching MATLAB exactly once the
z-score convention above was accounted for.
