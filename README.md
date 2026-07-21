# tmapper (Python)

A Python port of [Temporal Mapper 2](https://github.com/Multiscale-Complex-Systems-Lab/tmapper2), a toolbox for building **attractor transition networks** from time-series data.

**Status: work in progress.** The core two-step pipeline (`tknndigraph` + `filtergraph`) is being ported and tested first; the secondary cycle/path analysis toolkit will follow.

For the full background, citation, and the original MATLAB implementation, see [tmapper2](https://github.com/Multiscale-Complex-Systems-Lab/tmapper2).

## Installation (development)

```bash
git clone https://github.com/Multiscale-Complex-Systems-Lab/tmapper-py.git
cd tmapper-py
pip install -e ".[plot,test]"
```

## Running tests

```bash
pytest
```

## Notes for users porting workflows from MATLAB

- Node labels are 0-indexed (Pythonic), unlike the MATLAB original's 1-indexed convention.
- If you z-score your data before calling `tknndigraph` (as `tmapper_demo.m` does), note that MATLAB's `zscore` divides by the sample standard deviation (`N-1`), while `scipy.stats.zscore` defaults to the population standard deviation (`N`). Pass `ddof=1` to `scipy.stats.zscore` to match MATLAB's convention -- otherwise results can subtly diverge near k-NN/threshold boundaries. (Verified: with `ddof=1`, the Python and MATLAB pipelines produce identical output down to floating-point precision on the sample dataset.)

## License

BSD 3-Clause License -- see [LICENSE](LICENSE).
