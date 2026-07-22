# Installation

tmapper is not yet published to PyPI, so install it directly from GitHub.

## Requirements

| Requirement | Notes |
| --- | --- |
| **Python** | 3.10 or newer |
| **numpy, scipy, networkx** | installed automatically as dependencies |
| **matplotlib** | optional, needed for `plot_tmgraph` / `plot_tmgraph_tcm` (the `[plot]` extra) |

## Get the code

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
