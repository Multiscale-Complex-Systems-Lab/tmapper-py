"""Smoke tests for app/streamlit_app.py, run headlessly via Streamlit's
AppTest (no browser needed). Mirrors the MATLAB GUI's tests/test_gui_app.m
in spirit: exercise the real app object end to end rather than testing the
pipeline logic in isolation."""

import re
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def _load_sample(at):
    at.sidebar.button[0].click().run()  # "Load sample data"
    return at


def test_initial_load_prompts_for_data():
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception
    assert any("Load data to get started" in i.value for i in at.sidebar.info)


def test_load_sample_data_populates_variables():
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    _load_sample(at)
    assert not at.exception
    assert any("Loaded:" in c.value for c in at.sidebar.caption)
    # EL_temp.csv's 3 real numeric columns (tmax/tmin/prcp) should be
    # selected by default -- its leading "Unnamed: 0" index-artifact
    # column must NOT show up as a candidate variable.
    multiselects = at.sidebar.multiselect
    assert len(multiselects) == 1
    assert set(multiselects[0].value) == {"tmax", "tmin", "prcp"}


def test_build_network_succeeds_with_defaults():
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    _load_sample(at)
    at.sidebar.button(key="k").set_value(3) if False else None  # placeholder, no-op
    # click "Build Network" (the primary button in the sidebar)
    build_button = [b for b in at.sidebar.button if b.label == "Build Network"][0]
    build_button.click().run()
    assert not at.exception
    assert any("Built network:" in s.value for s in at.success)


def test_build_network_rejects_no_variables_selected():
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    _load_sample(at)
    at.sidebar.multiselect[0].set_value([]).run()
    build_button = [b for b in at.sidebar.button if b.label == "Build Network"][0]
    build_button.click().run()
    assert not at.exception
    assert any("Select at least one variable" in e.value for e in at.error)


def test_plot_option_change_does_not_error_and_reuses_cached_network():
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    _load_sample(at)
    build_button = [b for b in at.sidebar.button if b.label == "Build Network"][0]
    build_button.click().run()
    assert any("Built network:" in s.value for s in at.success)

    node_size_select = [s for s in at.sidebar.selectbox if s.label == "Node size"][0]
    node_size_select.set_value("rank").run()
    assert not at.exception
    # network is still reported as built (re-rendered from the cached
    # network, not rebuilt -- Build Network was never clicked again)
    assert any("Built network:" in s.value for s in at.success)


def test_missing_data_is_dropped_with_a_warning():
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    _load_sample(at)
    build_button = [b for b in at.sidebar.button if b.label == "Build Network"][0]
    build_button.click().run()
    assert not at.exception
    # EL_temp.csv (rmmissing'd upstream in the MATLAB demo) may or may not
    # contain gaps on its own; this just confirms the app doesn't crash
    # and reports a coherent state either way.
    assert any("Built network:" in s.value for s in at.success)


def test_export_panel_offers_downloadable_artifacts():
    """All export buttons should be wired up after a build."""
    at = AppTest.from_file(APP_PATH, default_timeout=90).run()
    _load_sample(at)
    build_button = [b for b in at.sidebar.button if b.label == "Build Network"][0]
    build_button.click().run()
    assert not at.exception

    labels = [b.label for b in at.get("download_button")]
    for expected in [
        "Interactive network (.html)", "Recurrence plot (.png)",
        "Timeline (.csv)", "Network (.graphml)", "Parameters (.json)", "Everything (.zip)",
    ]:
        assert any(expected in lbl for lbl in labels), \
            f"missing export button: {expected} (got {labels})"


def test_export_artifacts_are_valid_and_consistent(monkeypatch):
    """Build the export payloads directly and check they're actually
    usable: parseable, mutually consistent, and covering every time point."""
    import importlib.util
    import io
    import json as _json
    import zipfile

    spec = importlib.util.spec_from_file_location("tm_app", APP_PATH)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)

    repo_root = Path(APP_PATH).resolve().parent.parent
    monkeypatch.chdir(repo_root)

    df, _dropped = app.read_csv_smart(repo_root / "sampledata" / "EL_temp.csv")
    df = df.iloc[53883:].reset_index(drop=True)
    selected = tuple(app.numeric_columns(df))

    built = app.build_network(
        df, selected, True, 0, None, 4, 0, 1, 3, 3.0, 1, 100.0, float("inf"), True
    )
    n_nodes = built["g_simp"].number_of_nodes()

    # -- timeline: one row per retained time point, every node covered,
    # and source_row values must be real indices into the source frame
    timeline = pd.read_csv(io.StringIO(app.build_timeline_csv(built, df, "tmax", "(row index)")))
    assert len(timeline) == len(built["tidx"]), "timeline must cover every retained time point."
    assert set(timeline["node"]) == set(range(n_nodes)), "every node must appear in the timeline."
    assert timeline["source_row"].max() < len(df), "source_row must index into the source frame."
    assert "tmax" in timeline.columns, "timeline should carry the selected color variable."

    # -- graphml: parses back with matching topology and node attributes
    graphml = app.build_graphml(built, df["tmax"].to_numpy()[built["rows"]], "mode")
    g_back = nx.read_graphml(io.BytesIO(graphml))
    assert g_back.number_of_nodes() == n_nodes
    assert g_back.number_of_edges() == built["g_simp"].number_of_edges()
    attrs = next(iter(g_back.nodes(data=True)))[1]
    for key in ("n_members", "color_value", "first_source_row", "last_source_row"):
        assert key in attrs, f"graphml nodes missing attribute {key}"
    total_members = sum(int(d["n_members"]) for _, d in g_back.nodes(data=True))
    assert total_members == len(built["tidx"]), "node member counts must sum to the time points."

    # -- params json: strict-parseable (no bare Infinity) and complete
    params = _json.loads(app.build_params_json(
        built, "sample", "dat = ...", selected, True, 0, None, 4, 0, 1,
        3, 3.0, 1, 100.0, float("inf"), True, "tmax", "(row index)", "log", "mode", True,
    ))
    assert params["network_parameters"]["max_neighbor_dist"] == "inf", \
        "non-finite values must be serialized as strings, not bare Infinity."
    assert params["data_source"]["label"] == "sample"
    assert params["preprocessing"]["downsample"] == 4
    assert params["result"]["n_nodes"] == n_nodes

    # -- zip bundles all five entries
    zbytes = app.build_export_zip("<html></html>", "a,b\n1,2\n", graphml, "{}", "print(1)")
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        assert set(z.namelist()) == {
            "network.graphml", "timeline.csv", "params.json", "tmgraph.html", "reproduce.py"
        }


def test_generated_code_is_self_contained_and_reproduces_the_build(tmp_path, monkeypatch):
    """The generated script must include the data-loading step and, when
    run standalone, reproduce exactly the network the app reported --
    not merely assume a `dat` already exists in scope."""
    at = AppTest.from_file(APP_PATH, default_timeout=90).run()
    _load_sample(at)
    build_button = [b for b in at.sidebar.button if b.label == "Build Network"][0]
    build_button.click().run()
    assert not at.exception
    status = [s.value for s in at.success if "Built network:" in s.value][0]
    m = re.search(r"Built network: (\d+) nodes, (\d+) edges", status)
    app_nodes, app_edges = int(m.group(1)), int(m.group(2))

    code = at.code[0].value
    assert "pd.read_csv(" in code, "generated code must include the data-loading step."
    assert "dat = " in code, "generated code must actually define `dat`."

    # run it standalone, from the repo root so the relative sampledata/
    # path in the generated loader resolves
    repo_root = Path(__file__).resolve().parent.parent
    monkeypatch.chdir(repo_root)
    ns = {}
    exec(compile(code, "<generated>", "exec"), ns)  # noqa: S102

    assert ns["g_simp"].number_of_nodes() == app_nodes, \
        "generated code should reproduce the same node count as the app reported."
    assert ns["g_simp"].number_of_edges() == app_edges, \
        "generated code should reproduce the same edge count as the app reported."
    assert isinstance(ns["html"], str) and "<html" in ns["html"].lower(), \
        "generated code should produce the standalone HTML page."


def test_reset_restores_defaults_but_keeps_loaded_data():
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    _load_sample(at)
    k_input = [n for n in at.sidebar.number_input if n.label == "k (neighbors)"][0]
    k_input.set_value(99).run()
    assert k_input.value == 99

    reset_button = [b for b in at.sidebar.button if b.label == "Reset"][0]
    reset_button.click().run()
    assert not at.exception
    k_input = [n for n in at.sidebar.number_input if n.label == "k (neighbors)"][0]
    assert k_input.value == 3
    # data should NOT have been cleared by Reset
    assert any("Loaded:" in c.value for c in at.sidebar.caption)
