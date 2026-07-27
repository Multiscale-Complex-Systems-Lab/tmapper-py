# Installation

```bash
pip install tmapper                 # library
pip install "tmapper[app]"          # library + the interactive app
```

That is all most people need. The sections below cover installing from a
clone instead, which you want if you intend to modify the code.

## Requirements

| Requirement | Notes |
| --- | --- |
| **Python** | 3.10 or newer |
| **numpy, scipy, networkx** | installed automatically as dependencies |
| **matplotlib, python-igraph** | optional, needed for `plot_tmgraph` / `plot_tmgraph_tcm` (the `[plot]` extra) — igraph provides the DrL layout algorithm |
| **pyvis** | optional, needed only for `plot_tmgraph_interactive` (also in the `[plot]` extra) |
| **streamlit** | optional, only for the [interactive app](app.md) (the `[app]` extra, which also pulls in the plotting deps) |

!!! note "Trying a pre-release from TestPyPI"
    Release candidates are rehearsed on TestPyPI first. Those builds need
    PyPI as a fallback index, because tmapper's dependencies only live
    there. The project name there is **`tmapper-py`**, not `tmapper` — the
    latter is squatted on TestPyPI by an unrelated placeholder package, so
    rehearsals publish under this repo's name instead. This only affects
    the rehearsal index; the installed module is still `tmapper`.

    ```bash
    pip install --index-url https://test.pypi.org/simple/ \
                --extra-index-url https://pypi.org/simple/ \
                --pre "tmapper-py[app]"
    ```

## Installing from a clone (for development)

```bash
git clone https://github.com/Multiscale-Complex-Systems-Lab/tmapper-py.git
cd tmapper-py
```

## Install it

Install in editable mode, with the plotting extra (recommended so you can
follow the [Quickstart](quickstart.md) as-is):

```bash
pip install -e ".[plot]"
```

If you don't need plotting, the bare install is lighter:

```bash
pip install -e .
```

To run the point-and-click [interactive app](app.md) as well:

```bash
pip install -e ".[app]"
tmapper-app
```

!!! tip "The sample data ships with the package"
    `sample_data_path()` returns the bundled CSV wherever tmapper is
    installed, so example code works from a pip install and not only from
    the repository root:

    ```python
    import pandas as pd
    from tmapper import sample_data_path
    dat = pd.read_csv(sample_data_path())
    ```

## Verify the install

```python
import tmapper
print(tmapper.__all__)
```

If that prints a list of function names with no error, you're ready to go.
Head to the **[Quickstart](quickstart.md)** for a step-by-step walkthrough
that reproduces the network shown on the home page.

!!! note "Optional: run the test suite"
    ```bash
    pip install -e ".[plot,test]"
    pytest
    ```

    It should report every test passing.
