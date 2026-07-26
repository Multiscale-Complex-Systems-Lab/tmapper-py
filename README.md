# tmapper (Python)

A Python port of [Temporal Mapper 2](https://github.com/Multiscale-Complex-Systems-Lab/tmapper2), a toolbox for building **attractor transition networks** from time-series data.

**Status: feature-complete port.** Both the core two-step pipeline (`tknndigraph` + `filtergraph` + plotting) and the secondary cycle/path analysis toolkit (cycle counting/clustering, path decomposition, traffic, modularity) are ported and tested. `tknngraph.m` (a legacy, unused MATLAB function) was intentionally not ported.

For the full background, citation, and the original MATLAB implementation, see [tmapper2](https://github.com/Multiscale-Complex-Systems-Lab/tmapper2).

**Documentation, quickstart, and full API reference:** https://multiscale-complex-systems-lab.github.io/tmapper-py/

## What's included

- **Core pipeline**: `tknndigraph`, `filtergraph`, `find_node_label`, `tcm_distance`, `plot_tmgraph`, `plot_tmgraph_tcm`, `plot_tmgraph_interactive` (draggable/zoomable/hoverable HTML via pyvis, no MATLAB equivalent)
- **Standalone graph builders**: `knngraph`, `cknngraph`
- **Graph/data utilities**: `node_size`, `node_measure`, `normalize_geodesic`, `normalize_tcm`, `members_to_tidx`, `subgraph_from_members`, `sym_dyn_to_digraph`, `digraph_to_graph`, `find_blocks`
- **Cycle/path analysis toolkit**: `cycle_count` (Giscard/Kriege/Wilson combinatorial-sieve counter, third-party algorithm carrying its own BSD license -- see `cycle_count.py`), `cycle_count2p`, `reorg_cycles`, `cycle_path_overlap`, `cycle_cluster`, `cycle_cluster_conn`, `cycle_cutter`, `cycles_to_paths`, `cycle_path_decomp`, `path_traffic`, `qasym`, `cal_mod`
- **Interactive app**: `app/streamlit_app.py`, a browser front end for the whole pipeline -- see below

All ported functions were cross-checked node-for-node and edge-for-edge against real MATLAB output on deterministic test graphs (see the test suite for the same hand-derived oracle values used in the MATLAB toolbox's own `tests/`).

## Installation (development)

```bash
git clone https://github.com/Multiscale-Complex-Systems-Lab/tmapper-py.git
cd tmapper-py
pip install -e ".[plot,test]"
```

## Interactive app

Prefer point-and-click? `app/streamlit_app.py` runs the same pipeline in your browser -- load data, pick variables, turn the parameters, and explore the network without writing any code:

```bash
pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

It handles missing data and downsampling for you (dropping incomplete rows before an anti-aliasing lowpass), exports both figures and analysis-ready data (interactive HTML, PNGs, a per-timepoint `timeline.csv`, GraphML, and a provenance JSON), and shows a runnable script reproducing the current build.

See the [Interactive App guide](https://multiscale-complex-systems-lab.github.io/tmapper-py/app/) for a full walkthrough.

## Running tests

```bash
pytest
```

## Notes for users porting workflows from MATLAB

- Node labels are 0-indexed (Pythonic), unlike the MATLAB original's 1-indexed convention.
- If you z-score your data before calling `tknndigraph` (as `tmapper_demo.m` does), note that MATLAB's `zscore` divides by the sample standard deviation (`N-1`), while `scipy.stats.zscore` defaults to the population standard deviation (`N`). Pass `ddof=1` to `scipy.stats.zscore` to match MATLAB's convention -- otherwise results can subtly diverge near k-NN/threshold boundaries. (Verified: with `ddof=1`, the Python and MATLAB pipelines produce identical output down to floating-point precision on the sample dataset.)

## License

BSD 3-Clause License -- see [LICENSE](LICENSE).
