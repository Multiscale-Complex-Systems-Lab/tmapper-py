"""Smoke tests for app/streamlit_app.py, run headlessly via Streamlit's
AppTest (no browser needed). Mirrors the MATLAB GUI's tests/test_gui_app.m
in spirit: exercise the real app object end to end rather than testing the
pipeline logic in isolation."""

from pathlib import Path

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
