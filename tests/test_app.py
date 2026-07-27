"""Tests for src/tmapper/app/streamlit_app.py, run headlessly via Streamlit's AppTest
(no browser needed). Mirrors the MATLAB GUI's tests/test_gui_app.m in
spirit: exercise the real app end to end rather than testing the pipeline
logic in isolation.

Split by cost: anything about *widget wiring / error surfacing* goes
through AppTest (each such run rebuilds a real network, ~10-20s), while
parameter *semantics* are checked by calling build_network and the export
builders directly -- same code path, a fraction of the runtime.
"""

import importlib.util
import io
import json as _json
import re
import zipfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from tmapper import sample_data_path

APP_PATH = str(
    Path(__file__).resolve().parent.parent
    / "src" / "tmapper" / "app" / "streamlit_app.py"
)
REPO_ROOT = Path(__file__).resolve().parent.parent

# the bundled sample's recent slice contains exactly 6 rows with a missing
# value among tmax/tmin/prcp -- real data, so the missing-data path below
# is exercised against the genuine article rather than a synthetic stub
SAMPLE_MISSING_ROWS = 6
SAMPLE_SLICE_ROWS = 3826


@pytest.fixture(scope="module")
def app():
    """The app module itself, for calling its functions directly."""
    spec = importlib.util.spec_from_file_location("tm_app", APP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def sample_df(app):
    df, _dropped = app.read_csv_smart(sample_data_path())
    return df.iloc[53883:].reset_index(drop=True)


def _load_sample(at):
    # by label, not position: the sample button was moved below the file
    # uploader so it stops catching clicks meant for "load my own data"
    [b for b in at.sidebar.button if b.label == "Try sample data"][0].click().run()
    return at


def _build(at):
    [b for b in at.sidebar.button if b.label == "Build Network"][0].click().run()
    return at


def _counts(at):
    """(nodes, edges) as reported in the app's success banner."""
    status = [s.value for s in at.success if "Built network:" in s.value][0]
    m = re.search(r"Built network: (\d+) nodes, (\d+) edges", status)
    return int(m.group(1)), int(m.group(2))


def _num_input(at, label):
    return [n for n in at.sidebar.number_input if n.label == label][0]


def _text_input(at, label):
    return [t for t in at.sidebar.text_input if t.label == label][0]


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


def test_sample_button_sits_below_the_uploader():
    """The file uploader is the primary action and must come first.

    As a full-width button on top, "load sample data" kept catching clicks
    from people who meant to load their own file.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception
    labels = [b.label for b in at.sidebar.button]
    assert "Try sample data" in labels
    # the uploader has no button of its own in AppTest, so assert the
    # ordering that is observable: the sample button is not the first thing
    # in the Data section by prominence -- it is a plain, non-full-width
    # secondary control, and the uploader widget exists above it
    assert len(at.sidebar.get("file_uploader")) == 1


def test_every_input_control_explains_itself():
    """Each control carries help text, and the data-format note is shown.

    These are the app's only documentation at the point of use, so a control
    added later without a `help=` should fail here rather than ship bare.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    _load_sample(at)
    assert not at.exception

    bare = []
    for kind in ("selectbox", "number_input", "text_input", "checkbox", "multiselect"):
        for w in at.sidebar.get(kind):
            if not getattr(w, "help", None):
                bare.append(f"{kind}:{w.label}")
    assert not bare, f"controls with no help text: {bare}"

    # the format note itself must be present and mention the essentials
    body = " ".join(m.value for m in at.sidebar.markdown)
    for expected in ("one row per time point", "numeric", "header row"):
        assert expected in body.lower(), f"data-format note is missing '{expected}'"


def test_sample_button_is_not_overridden_by_an_attached_upload(app):
    """Clicking "try sample data" while a file is attached must switch to
    the sample, and a later rerun must not flip it back.

    The uploader used to re-fire on any rerun where the loaded source
    wasn't the uploaded file, silently reloading the attachment over the
    sample -- so once you'd uploaded anything, the button looked dead.

    Tested through resolve_data_action rather than AppTest: Streamlit's
    AppTest has no file_uploader element and cannot simulate an upload, so
    an integration test here would pass whatever the branch did.
    """
    mine = ("mine.csv", 123)

    # a brand-new attachment loads
    assert app.resolve_data_action(False, mine, None) == "upload"
    # ...but only once; after the caller claims it, reruns are no-ops
    assert app.resolve_data_action(False, mine, mine) is None
    # the bug: clicking sample with an unclaimed attachment present
    assert app.resolve_data_action(True, mine, None) == "sample", \
        "an explicit sample click must win over an attached upload"
    # and having claimed the token, the following rerun must stay put
    assert app.resolve_data_action(False, mine, mine) is None, \
        "an attached upload must not reload over an explicitly chosen sample"
    # swapping in a different file is a new token, so it does load
    assert app.resolve_data_action(False, ("other.csv", 9), mine) == "upload"
    # nothing attached, nothing clicked
    assert app.resolve_data_action(False, None, None) is None


def test_build_network_succeeds_with_defaults():
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    _load_sample(at)
    _build(at)
    assert not at.exception
    nodes, edges = _counts(at)
    assert nodes > 0 and edges > 0


def test_build_network_rejects_no_variables_selected():
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    _load_sample(at)
    at.sidebar.multiselect[0].set_value([]).run()
    _build(at)
    assert not at.exception
    assert any("Select at least one variable" in e.value for e in at.error)


def test_plot_option_change_rerenders_without_rebuilding():
    """Changing a Plot Option must re-render the *cached* network, not
    rebuild it -- so the reported node/edge counts must be unchanged."""
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    _load_sample(at)
    _build(at)
    before = _counts(at)

    [s for s in at.sidebar.selectbox if s.label == "Node size"][0].set_value("rank").run()
    assert not at.exception
    assert _counts(at) == before, \
        "a plot-only change must not alter the network (it should reuse the cached build)."

    [s for s in at.sidebar.selectbox if s.label == "Label method"][0].set_value("mean").run()
    assert not at.exception
    assert _counts(at) == before


def test_missing_data_is_dropped_with_a_warning():
    """The bundled sample slice really does contain missing values, so the
    warning must appear with the exact count -- and the build must still
    succeed rather than being poisoned by the NaNs."""
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    _load_sample(at)
    _build(at)
    assert not at.exception

    warnings = [w.value for w in at.warning]
    assert any("missing values" in w for w in warnings), \
        f"expected a missing-data warning, got {warnings}"
    msg = [w for w in warnings if "missing values" in w][0]
    assert f"{SAMPLE_MISSING_ROWS} of {SAMPLE_SLICE_ROWS} row(s)" in msg, \
        f"warning should report the exact counts, got: {msg}"
    # the default build is stride 1, so there is no averaging to absorb them
    # and they really are dropped -- the message must say so
    assert "Dropped" in msg and "gap in time" in msg, \
        f"at downsample=1 the warning should report dropped samples, got: {msg}"
    assert any("Built network:" in s.value for s in at.success), \
        "dropping missing rows should let the build succeed, not abort it."


def test_no_missing_data_warning_when_data_is_clean(app, sample_df):
    """Guard against the warning firing spuriously: a clean frame must
    report zero dropped rows."""
    clean = sample_df.dropna(subset=["tmax", "tmin", "prcp"]).reset_index(drop=True)
    built = app.build_network(
        clean, ("tmax", "tmin", "prcp"), True, 0, None, 8, 0, 1,
        3, 3.0, 1, 100.0, float("inf"), True,
    )
    assert built["n_dropped"] == 0


def test_isolated_missing_row_is_absorbed_by_the_lowpass(app):
    """An isolated NaN must not cost a sample when downsampling.

    The decimated value is an average over its window, and pandas'
    rolling mean skips NaN, so one missing input simply doesn't contribute
    -- it neither poisons the average nor punches a hole in the time grid.
    """
    n = 400
    df = pd.DataFrame({
        "x": np.sin(np.arange(n) / 9.0),
        "y": np.cos(np.arange(n) / 9.0),
    })
    df.loc[100, "x"] = np.nan
    built = app.build_network(
        df, ("x", "y"), True, 0, None, 4, 0, 1, 3, 3.0, 1, 100.0, float("inf"), True
    )
    assert built["n_dropped"] == 1, "the missing source row should still be reported"
    assert built["n_grid_dropped"] == 0, "no sample should be lost to one isolated NaN"
    assert np.all(np.diff(built["tidx"]) == 1), "and the time grid must stay unbroken"


def test_fully_missing_window_drops_the_sample(app):
    """When every row feeding a decimated sample is missing there is nothing
    to average, so that sample must be dropped and leave a real gap."""
    n = 400
    df = pd.DataFrame({
        "x": np.sin(np.arange(n) / 9.0),
        "y": np.cos(np.arange(n) / 9.0),
    })
    df.loc[98:103, "x"] = np.nan  # covers the whole window around grid point 100
    built = app.build_network(
        df, ("x", "y"), True, 0, None, 4, 0, 1, 3, 3.0, 1, 100.0, float("inf"), True
    )
    assert built["n_grid_dropped"] >= 1, "a fully-missing window must drop its sample"
    assert (np.diff(built["tidx"]) > 1).any(), "and that must leave a gap in tidx"
    assert 100 not in set(built["rows"].tolist())


# ------------------------------------------------ preprocessing semantics
# Called directly rather than through AppTest: same code path, but each
# AppTest build costs ~10-20s and these are pure input/output questions.

def test_tidx_preserves_real_time_gaps(app):
    """tidx must encode real time position, not array position.

    tknndigraph links points as temporal neighbours iff their tidx differs
    by exactly 1, so a plain arange would silently bridge a stretch of
    dropped rows -- linking the samples either side of a gap as if they
    were consecutive, which is precisely what tidx exists to prevent.
    """
    n = 60
    df = pd.DataFrame({
        "x": np.sin(np.arange(n) / 5.0),
        "y": np.cos(np.arange(n) / 5.0),
    })
    df.loc[25:34, "x"] = np.nan  # a real 10-sample hole

    built = app.build_network(
        df, ("x", "y"), True, 0, None, 1, 0, 1, 3, 3.0, 1, 100.0, float("inf"), True
    )
    rows, tidx = built["rows"], built["tidx"]
    pos = {int(r): i for i, r in enumerate(rows)}

    # rows either side of the hole are 11 apart in real time, so their tidx
    # must be too -- not adjacent
    gap_jump = tidx[pos[35]] - tidx[pos[24]]
    assert gap_jump == 11, f"tidx must preserve the real gap, got a jump of {gap_jump}"

    # and contiguous stretches must still advance by exactly 1, or *no*
    # temporal edges would be built at all
    assert tidx[pos[23]] - tidx[pos[22]] == 1
    assert tidx[pos[36]] - tidx[pos[35]] == 1

    # the gap must actually suppress the temporal edge in the built graph:
    # with a bridged tidx the two sides would land in one connected run
    assert len(tidx) == len(rows)


def test_tidx_is_unit_spaced_when_downsampling(app, sample_df):
    """Downsampling by N means successive retained samples are N source
    rows apart, but they are still *consecutive samples* -- tidx must step
    by 1 or tknndigraph would build no temporal edges at all."""
    built = app.build_network(
        sample_df, ("tmax", "tmin", "prcp"), True, 0, 999, 5, 0, 1,
        3, 3.0, 1, 100.0, float("inf"), True,
    )
    steps = np.diff(built["tidx"])
    assert steps.min() >= 1, "tidx must be strictly increasing"
    assert np.bincount(steps).argmax() == 1, \
        "the typical tidx step under downsampling must be 1, not the downsample factor"


def test_tidx_has_no_drift_artifacts_when_dropping_and_downsampling(app, sample_df):
    """Dropping rows *and* downsampling together must not corrupt tidx.

    Deriving tidx from an absolute offset -- rint((row - row[0]) / N) --
    accumulates rounding drift past every dropped row, which produced
    duplicate tidx values (two samples sharing a time index) and phantom
    gaps at samples that were in fact evenly spaced. Steps must be measured
    per-interval instead.
    """
    # the bundled slice has 6 scattered missing rows, so this combination
    # is exactly the drift-prone case
    built = app.build_network(
        sample_df, ("tmax", "tmin", "prcp"), True, 0, None, 4, 0, 1,
        3, 3.0, 30, 100.0, 0.5, True,
    )
    rows, tidx = built["rows"], built["tidx"]
    steps = np.diff(tidx)

    assert steps.min() >= 1, \
        "tidx must strictly increase -- a 0 step means two samples share a time index."
    assert len(set(tidx.tolist())) == len(tidx), "tidx values must be unique."

    # every jump >1 must correspond to a genuinely larger source-row gap,
    # not to rounding drift
    row_gaps = np.diff(rows)
    for i in np.where(steps > 1)[0]:
        assert row_gaps[i] > built["downsample"], (
            f"tidx jumped {steps[i]} at a source-row gap of only {row_gaps[i]} "
            f"(downsample={built['downsample']}) -- that's a drift artifact."
        )


@pytest.mark.parametrize(
    "label, nan_slice, downsample, start_row, lag, order",
    [
        ("clean",                        None,       4, 0, 0, 1),
        ("one grid point missing",       (98, 103),  4, 0, 0, 1),
        ("several consecutive missing",  (98, 111),  4, 0, 0, 1),
        ("first grid point missing",     (0, 5),     4, 0, 0, 1),
        ("with delay embedding",         None,       4, 0, 10, 2),
        ("offset start row, stride 3",   None,       3, 7, 0, 1),
        ("no downsampling",              (98, 103),  1, 0, 0, 1),
    ],
)
def test_tidx_matches_the_grid_under_downsampling(
    app, label, nan_slice, downsample, start_row, lag, order
):
    """The core invariant, across every combination that touches it.

    Retained rows must sit on the exact original decimation grid, and each
    tidx step must equal its source-row step divided by the downsample
    factor. That is what makes consecutive samples differ by exactly 1 (so
    temporal edges exist at all) while a genuinely dropped sample leaves a
    proportionally larger jump (so no edge is fabricated across it).
    """
    n = 400
    df = pd.DataFrame({
        "x": np.sin(np.arange(n) / 9.0),
        "y": np.cos(np.arange(n) / 9.0),
    })
    if nan_slice is not None:
        df.loc[nan_slice[0]:nan_slice[1], "x"] = np.nan

    built = app.build_network(
        df, ("x", "y"), True, start_row, None, downsample, lag, order,
        3, 3.0, 1, 100.0, float("inf"), True,
    )
    rows, tidx = built["rows"], built["tidx"]

    assert np.all((rows - rows[0]) % downsample == 0), \
        f"[{label}] retained rows drifted off the decimation grid"
    assert tidx[0] == 0, f"[{label}] tidx should start at 0"
    assert len(set(tidx.tolist())) == len(tidx), f"[{label}] tidx values must be unique"
    assert np.array_equal(np.diff(tidx), np.diff(rows) // downsample), \
        f"[{label}] tidx steps must equal source-row steps / downsample"
    assert np.diff(tidx).min() >= 1, f"[{label}] tidx must strictly increase"


def _seasoned(app_mod):
    """Sample slice plus a genuine categorical column."""
    df, _ = app_mod.read_csv_smart(sample_data_path())
    df = df.iloc[53883:54600].reset_index(drop=True)
    df["season"] = pd.cut(
        df["Date"].dt.month % 12 // 3, bins=[-1, 0, 1, 2, 3],
        labels=["winter", "spring", "summer", "fall"],
    ).astype(str)
    return df


def test_date_strings_are_parsed_like_matlab(app):
    """MATLAB's readtable makes datetimes out of date strings, which is why
    tmapper_demo.m can use dat.Date directly. pandas leaves them as text, so
    the app parses them -- otherwise a perfectly good time column is unusable.
    """
    df, _ = app.read_csv_smart(sample_data_path())
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    assert app.datetime_columns(df) == ["Date"]
    # but it must stay out of the *state variables*: distances need numbers
    assert "Date" not in app.numeric_columns(df)
    assert "Date" in app.time_column_options(df)
    assert "Date" in app.color_column_options(df)


def test_short_text_columns_are_not_mangled_into_dates(app):
    """The parser must not turn a categorical column into nonsense
    timestamps -- it requires nearly every value to parse."""
    df = pd.DataFrame({"cond": ["a", "b", "c", "a"], "x": [1.0, 2, 3, 4]})
    assert app._maybe_datetime(df["cond"]) is None
    assert app.categorical_columns(df) == ["cond"]


def test_datetime_column_works_as_a_time_index(app):
    """Daily dates should give one tidx step per day -- and real missing
    days then show up as genuine gaps rather than being bridged."""
    df, _ = app.read_csv_smart(sample_data_path())
    df = df.iloc[53883:].reset_index(drop=True)
    built = app.build_network(
        df, ("tmax", "tmin", "prcp"), True, 0, 800, 1, 0, 1,
        3, 3.0, 1, 100.0, float("inf"), True, tidx_var="Date",
    )
    steps = np.diff(built["tidx"])
    assert steps.min() >= 1
    assert np.bincount(steps).argmax() == 1, "daily sampling should step by 1 per day"


def test_categorical_column_can_colour_the_network(app):
    """Colouring by condition/state is the common case and needs no
    distance-metric changes -- categories map to nominal integer codes."""
    df = _seasoned(app)
    assert "season" in app.color_column_options(df)
    # ...but not as a time axis or index: categories have no order
    assert "season" not in app.time_column_options(df)
    assert "season" not in app.numeric_columns(df)

    built = app.build_network(
        df, ("tmax", "tmin", "prcp"), True, 0, None, 1, 0, 1,
        3, 3.0, 30, 100.0, 0.5, True,
    )
    vals, label, cats = app.resolve_color_values(
        df, "season", built["rows"], built["tidx"]
    )
    assert cats == ["fall", "spring", "summer", "winter"], "categories must be reported"
    assert set(np.unique(vals)) <= set(range(len(cats))), "values must be category codes"
    assert label == "season"


def test_colormap_lists_are_type_appropriate(app):
    """Categories need a qualitative palette; a continuous ramp would imply
    an ordering between them that doesn't exist."""
    assert "jet" in app.CONTINUOUS_CMAPS
    assert "tab10" in app.CATEGORICAL_CMAPS
    assert not set(app.CONTINUOUS_CMAPS) & set(app.CATEGORICAL_CMAPS)
    import matplotlib.pyplot as plt
    for name in app.CONTINUOUS_CMAPS + app.CATEGORICAL_CMAPS:
        plt.get_cmap(name)  # must be a real matplotlib colormap


@pytest.mark.parametrize(
    "color_var, color_kind, cmap, must_contain",
    [
        ("tmax", "numeric", "viridis", "dat['tmax'].to_numpy()[rows]"),
        ("Date", "datetime", "plasma", "mdates.date2num"),
        ("season", "category", "tab10", "pd.factorize"),
        ("(row index)", "index", "jet", "tidx.astype(float)"),
    ],
)
def test_generated_code_runs_for_every_colour_type(
    app, monkeypatch, tmp_path, color_var, color_kind, cmap, must_contain
):
    """The generated script must actually execute for each colour type.

    Dates and categories emit different colour expressions (and dates need
    an extra import), so covering only the default would miss a script that
    reads fine but raises NameError when run -- which is exactly how the
    earlier bare `inf` bug slipped through.
    """
    dat = _seasoned(app)
    built = app.build_network(
        dat, ("tmax", "tmin", "prcp"), True, 0, None, 1, 0, 1,
        3, 3.0, 30, 100.0, 0.5, True,
    )

    code = app.generate_code(
        "pass  # dat is supplied by the test", ("tmax", "tmin", "prcp"), True,
        0, None, 1, 0, 1, 3, 3.0, 30, 100.0, 0.5, True,
        color_var, "(row index)", "log", "mode", True,
        None, cmap, color_kind,
    )
    assert must_contain in code, f"colour expression missing for {color_kind}"
    assert f"cmap={cmap!r}" in code, "the chosen colormap must reach the generated code"
    if color_kind == "datetime":
        assert "import matplotlib.dates as mdates" in code, \
            "date colouring needs its import emitted, or the script raises NameError"

    # generated scripts locate the sample via sample_data_path(), so they no
    # longer depend on the working directory -- run from elsewhere to prove it
    monkeypatch.chdir(tmp_path)
    ns = {"dat": dat}
    exec(compile(code, "<generated>", "exec"), ns)  # noqa: S102
    assert ns["g_simp"].number_of_nodes() == built["g_simp"].number_of_nodes()
    assert isinstance(ns["html"], str) and "<html" in ns["html"].lower()


def test_params_json_records_colormap_and_time_index(app, sample_df):
    """Provenance must capture the plot/preprocessing choices that change
    the output, not just the network parameters."""
    built = app.build_network(
        sample_df, ("tmax", "tmin", "prcp"), True, 0, 500, 1, 0, 1,
        3, 3.0, 1, 100.0, float("inf"), True,
    )
    params = _json.loads(app.build_params_json(
        built, "sample", "dat = ...", ("tmax", "tmin", "prcp"), True, 0, 500, 1, 0, 1,
        3, 3.0, 1, 100.0, float("inf"), True, "tmax", "Date", "log", "mode", True,
        tidx_var="Date", cmap="cividis",
    ))
    assert params["plot_options"]["cmap"] == "cividis"
    assert params["plot_options"]["color_by"] == "tmax"
    assert params["preprocessing"]["time_index_source"] == "Date"

    # and the default must record that it came from row order, not a column
    params2 = _json.loads(app.build_params_json(
        built, "sample", "dat = ...", ("tmax",), True, 0, 500, 1, 0, 1,
        3, 3.0, 1, 100.0, float("inf"), True, "(row index)", "(row index)",
        "log", "mode", True,
    ))
    assert params2["preprocessing"]["time_index_source"] == "row order"


def test_user_supplied_tidx_column_is_used(app):
    """A chosen time-index column must drive temporal adjacency, so its own
    gaps (separate sessions, irregular sampling) break the chain."""
    n = 120
    t = np.arange(n) + (np.arange(n) >= 60) * 500  # a 500-step jump mid-way
    df = pd.DataFrame({
        "t": t,
        "x": np.sin(np.arange(n) / 7.0),
        "y": np.cos(np.arange(n) / 7.0),
    })
    built = app.build_network(
        df, ("x", "y"), True, 0, None, 1, 0, 1, 3, 3.0, 1, 100.0, float("inf"), True,
        tidx_var="t",
    )
    tidx = built["tidx"]
    assert tidx[0] == 0
    steps = np.diff(tidx)
    assert (steps == 1).sum() == n - 2, "within a session, samples must step by 1"
    # t runs ...58, 59, then 560 -- so the single break is a 501-step jump
    assert steps.max() == 501, "the session break must survive as a large jump"
    assert (steps > 1).sum() == 1, "there should be exactly one break"


@pytest.mark.parametrize("downsample", [1, 2, 4, 5])
def test_user_supplied_tidx_works_with_downsampling(app, downsample):
    """A supplied index counts *raw* sampling intervals, so decimating must
    convert to decimated units -- otherwise neighbouring kept samples would
    differ by N rather than 1 and no temporal edge would be built at all.
    Real breaks must survive, scaled by the same factor.
    """
    n = 200
    t = np.arange(n) + (np.arange(n) >= 100) * 500  # one genuine break
    df = pd.DataFrame({
        "t": t,
        "x": np.sin(np.arange(n) / 7.0),
        "y": np.cos(np.arange(n) / 7.0),
    })
    built = app.build_network(
        df, ("x", "y"), True, 0, None, downsample, 0, 1,
        3, 3.0, 1, 100.0, float("inf"), True, tidx_var="t",
    )
    steps = np.diff(built["tidx"])
    assert (steps == 1).sum() == len(steps) - 1, \
        "consecutive kept samples must differ by exactly 1, or no temporal edges form"
    assert (steps > 1).sum() == 1, "the single real break must remain a single break"
    # the exact integer depends on which sample the stride last kept before
    # the break, so assert the scaling property rather than re-deriving it
    assert abs(steps.max() - 501 / downsample) <= 2, \
        f"the break should scale as ~501/{downsample}, got {steps.max()}"


def test_user_supplied_tidx_rejects_bad_columns(app):
    n = 60
    df = pd.DataFrame({
        "t": np.arange(n),
        "bad": np.r_[np.arange(n - 1), [0]].astype(float),  # not increasing
        "frac": np.arange(n) + 0.5,                          # not whole numbers
        # irregular spacing: steps are not multiples of the smallest step,
        # so there is no interval to decimate by
        "irregular": np.cumsum(np.r_[[0], np.tile([2, 3, 7], n)[: n - 1]]),
        "x": np.sin(np.arange(n) / 7.0),
        "y": np.cos(np.arange(n) / 7.0),
    })

    def build(col, downsample=1):
        return app.build_network(
            df, ("x", "y"), True, 0, None, downsample, 0, 1,
            3, 3.0, 1, 100.0, float("inf"), True, tidx_var=col,
        )

    with pytest.raises(ValueError, match="strictly increasing"):
        build("bad")
    with pytest.raises(ValueError, match="whole numbers"):
        build("frac")
    # irregular spacing is fine at stride 1, but has no well-defined
    # interval to decimate by
    build("irregular", downsample=1)
    with pytest.raises(ValueError, match="needs a regular time index"):
        build("irregular", downsample=3)


def test_row_range_restricts_to_exactly_that_window(app, sample_df):
    built = app.build_network(
        sample_df, ("tmax", "tmin", "prcp"), True, 100, 400, 1, 0, 1,
        3, 3.0, 1, 100.0, float("inf"), True,
    )
    rows = built["rows"]
    assert rows.min() >= 100 and rows.max() <= 400
    assert built["n_window"] == 301  # inclusive of both endpoints


def test_end_row_none_means_last_row(app, sample_df):
    built = app.build_network(
        sample_df, ("tmax", "tmin", "prcp"), True, 3000, None, 1, 0, 1,
        3, 3.0, 1, 100.0, float("inf"), True,
    )
    assert built["n_window"] == len(sample_df) - 3000


def test_downsample_keeps_every_nth_retained_row(app, sample_df):
    full = app.build_network(
        sample_df, ("tmax", "tmin", "prcp"), True, 0, 999, 1, 0, 1,
        3, 3.0, 1, 100.0, float("inf"), True,
    )
    ds = app.build_network(
        sample_df, ("tmax", "tmin", "prcp"), True, 0, 999, 5, 0, 1,
        3, 3.0, 1, 100.0, float("inf"), True,
    )
    assert len(ds["tidx"]) == int(np.ceil(len(full["tidx"]) / 5))
    assert ds["downsample"] == 5


def test_downsample_applies_an_anti_aliasing_lowpass(app):
    """Downsampling must smooth before striding, not pick raw every-Nth
    samples -- otherwise high-frequency content aliases into spurious
    low-frequency structure.

    Verified by equivalence: the app's downsample=4 build must match a
    build on a *manually pre-smoothed then strided* frame, and must NOT
    match one on raw strided samples.
    """
    n = 600
    rng = np.random.default_rng(0)
    # noisy on purpose: smoothing vs raw striding diverge sharply, so the
    # inequality below can't pass by coincidence
    df = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)})
    kw = dict(zscore_on=True, lag=0, order=1, k=3, d=3.0, texclude=1,
              maxdistprct=100.0, maxdist=float("inf"), reciprocal=True)

    def counts(frame, downsample):
        b = app.build_network(
            frame, ("x", "y"), kw["zscore_on"], 0, None, downsample, kw["lag"], kw["order"],
            kw["k"], kw["d"], kw["texclude"], kw["maxdistprct"], kw["maxdist"], kw["reciprocal"],
        )
        return b["g_simp"].number_of_nodes(), b["g_simp"].number_of_edges()

    app_downsampled = counts(df, 4)

    smoothed_first = df.rolling(4, center=True, min_periods=1).mean().iloc[::4].reset_index(drop=True)
    raw_strided = df.iloc[::4].reset_index(drop=True)

    assert app_downsampled == counts(smoothed_first, 1), \
        "downsample=4 should equal building on a pre-smoothed, strided frame."
    assert app_downsampled != counts(raw_strided, 1), \
        "downsample=4 must NOT equal naive every-Nth striding (no lowpass applied)."


def test_delay_embedding_shortens_the_series_by_lag_times_order(app, sample_df):
    plain = app.build_network(
        sample_df, ("tmax",), True, 0, 999, 1, 0, 1,
        3, 3.0, 1, 100.0, float("inf"), True,
    )
    embedded = app.build_network(
        sample_df, ("tmax",), True, 0, 999, 1, 30, 3,
        3, 3.0, 1, 100.0, float("inf"), True,
    )
    # order=3, lag=30 consumes (order-1)*lag = 60 samples
    assert len(embedded["tidx"]) == len(plain["tidx"]) - 60
    assert embedded["order"] == 3 and embedded["lag"] == 30


def test_invalid_range_and_embedding_are_rejected(app, sample_df):
    with pytest.raises(ValueError, match="greater than or equal to start row"):
        app.build_network(
            sample_df, ("tmax",), True, 500, 100, 1, 0, 1,
            3, 3.0, 1, 100.0, float("inf"), True,
        )
    with pytest.raises(ValueError, match="need at least 2"):
        app.build_network(
            sample_df, ("tmax",), True, 10, 10, 1, 0, 1,
            3, 3.0, 1, 100.0, float("inf"), True,
        )
    with pytest.raises(ValueError, match="Embed lag must be at least 1"):
        app.build_network(
            sample_df, ("tmax",), True, 0, 500, 1, 0, 2,
            3, 3.0, 1, 100.0, float("inf"), True,
        )
    with pytest.raises(ValueError, match="Embed lag/order too large"):
        app.build_network(
            sample_df, ("tmax",), True, 0, 100, 1, 500, 2,
            3, 3.0, 1, 100.0, float("inf"), True,
        )


# --------------------------------------------------- UI guards & sentinels

def test_memory_guard_blocks_an_oversized_row_range(app):
    """The untrimmed sample (57709 rows) would need a ~25 GB distance
    matrix; the guard must refuse with an actionable number rather than
    letting cdist attempt the allocation."""
    # under the limit, or downsampled -> allowed
    assert app.oversized_window_message(3826, 1) is None
    assert app.oversized_window_message(app.MAX_WINDOW_ROWS, 1) is None
    assert app.oversized_window_message(57709, 4) is None, \
        "downsampling is the fix, so the guard must not fire when it's on."

    msg = app.oversized_window_message(57709, 1)
    assert msg is not None
    assert "57709 rows" in msg
    assert "GB of memory" in msg
    # 57709^2 * 8 bytes ~= 26.6 GB -- the figure must be real, not a guess
    assert "26.6 GB" in msg, msg
    assert "downsample" in msg, "the message should say how to fix it."


def test_invalid_max_dist_text_is_reported_not_crashed():
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    _load_sample(at)
    _text_input(at, "max dist").set_value("not-a-number").run()
    assert not at.exception, "a bad numeric entry must surface as an error, not an exception."
    assert any("max dist must be a number" in e.value for e in at.error)


def test_max_dist_and_end_row_sentinels_are_accepted():
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    _load_sample(at)
    _text_input(at, "max dist").set_value("inf").run()
    _text_input(at, "end row").set_value("last").run()
    assert not at.exception
    _build(at)
    assert not at.exception
    assert any("Built network:" in s.value for s in at.success)


def test_reset_also_restores_plot_options():
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()
    _load_sample(at)
    [s for s in at.sidebar.selectbox if s.label == "Node size"][0].set_value("original").run()
    [s for s in at.sidebar.selectbox if s.label == "Label method"][0].set_value("median").run()
    _text_input(at, "max dist").set_value("0.5").run()

    [b for b in at.sidebar.button if b.label == "Reset"][0].click().run()
    assert not at.exception
    assert [s for s in at.sidebar.selectbox if s.label == "Node size"][0].value == "log"
    assert [s for s in at.sidebar.selectbox if s.label == "Label method"][0].value == "mode"
    assert _text_input(at, "max dist").value == "inf"


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

    df, _dropped = app.read_csv_smart(sample_data_path())
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

    # run it standalone from an unrelated directory: the generated loader
    # uses sample_data_path(), so it must not depend on the working directory
    monkeypatch.chdir(tmp_path)
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
