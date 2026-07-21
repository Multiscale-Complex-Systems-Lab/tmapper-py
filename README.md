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

## License

BSD 3-Clause License -- see [LICENSE](LICENSE).
