"""Interactive Streamlit app for the Temporal Mapper pipeline.

Point-and-click equivalent of the scripted tknndigraph -> filtergraph ->
plot_tmgraph_interactive pipeline, mirroring the MATLAB toolbox's
gui/TemporalMapperApp.m. Launch with:

    streamlit run app/streamlit_app.py

Architecture note (mirrors the MATLAB app): building the network
(tknndigraph/filtergraph) is the expensive step and only runs when
"Build Network" is clicked, cached by st.cache_data so re-clicking Build
with unchanged parameters is instant. Changing a Plot Options widget
(color/time axis, node size, label method, show recurrence) re-renders
the *cached* network on every Streamlit rerun -- cheap, no rebuild.
"""

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from scipy.spatial.distance import cdist
from scipy.stats import zscore as scipy_zscore

from tmapper import tknndigraph, filtergraph, plot_tmgraph_interactive, tcm_distance

SAMPLE_DATA_PATH = Path(__file__).resolve().parent.parent / "sampledata" / "EL_temp.csv"

DEFAULTS = {
    "zscore": True,
    "start_row": 0,
    "end_row_str": "last",
    "downsample": 1,
    "embed_lag": 0,
    "embed_order": 1,
    "k": 3,
    "d": 3.0,
    "texclude": 1,
    "maxdistprct": 100.0,
    "maxdist_str": "inf",
    "reciprocal": True,
    "color_var": "(row index)",
    "time_var": "(row index)",
    "nodesizemode": "log",
    "labelmethod": "mode",
    "show_recurrence": True,
}


# ============================================================= data helpers

def read_csv_smart(path_or_buffer):
    """pandas.read_csv, but drop a leading unnamed column (pandas' own
    "Unnamed: 0" marker for a header-less first column) -- it's virtually
    always a stray row-index column left over from a previous
    ``to_csv()`` without ``index=False``, and including it as a
    candidate build variable would silently dominate the distance
    computation (it's just a monotonic ramp).

    Always returns a plain 0..N-1 RangeIndex: build_network/render_network
    use positional indexing (df[...].to_numpy()[rows]) throughout, which
    requires the DataFrame's index to match row position exactly.

    Returns
    -------
    df : pandas.DataFrame
    dropped_index_col : bool
        Whether a leading unnamed column was dropped -- the caller needs
        this to emit matching code in the "Show equivalent code" panel.
    """
    df = pd.read_csv(path_or_buffer)
    dropped_index_col = df.columns[0].startswith("Unnamed:")
    if dropped_index_col:
        df = df.drop(columns=df.columns[0])
    return df.reset_index(drop=True), dropped_index_col


def numeric_columns(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def set_data(df, source_label, source_code):
    """Register a newly-loaded table, invalidating any cached network and
    resetting variable selection -- mirrors TemporalMapperApp.loadData.

    ``source_code`` is the runnable snippet that reproduces ``df`` as
    ``dat``; it gets emitted verbatim at the top of the generated code
    (mirroring the MATLAB app's DataSourceCode property) so the output
    is a complete script rather than one that assumes ``dat`` exists.
    """
    numvars = numeric_columns(df)
    if not numvars:
        st.sidebar.error("That data has no numeric columns to build a network from.")
        return
    st.session_state.pop("built", None)
    st.session_state["data"] = df
    st.session_state["data_source"] = source_label
    st.session_state["data_source_code"] = source_code
    st.session_state["numeric_vars"] = numvars
    st.session_state["selected_vars"] = numvars


# ========================================================= expensive: build

@st.cache_data(show_spinner=False)
def build_network(
    df, selected_vars, zscore_on, start_row, end_row, downsample,
    lag, order, k, d, texclude, maxdistprct, maxdist, reciprocal,
):
    """tknndigraph -> filtergraph on the resolved parameters. Cached by
    st.cache_data, keyed on every argument here -- re-calling with the
    same parameters (even across reruns triggered by unrelated widgets)
    returns instantly instead of recomputing.

    Row indices throughout are 0-indexed positions into `df`, matching
    this port's Pythonic convention (unlike the MATLAB app's 1-indexed
    UI).
    """
    n_full = len(df)
    end_row = n_full - 1 if end_row is None else min(end_row, n_full - 1)
    if end_row < start_row:
        raise ValueError("End row must be greater than or equal to start row.")

    window = df.iloc[start_row:end_row + 1]

    # -- drop rows with missing (NaN) values in the selected variables
    # BEFORE lowpass filtering/downsampling: a rolling-mean lowpass (like
    # z-scoring) would otherwise smear a gap into its neighbors too. This
    # automates the "clean your data first" convention documented for the
    # scripted pipeline instead of assuming the caller already did it --
    # an unremoved NaN would otherwise poison the whole distance matrix
    # silently.
    missing_mask = window[list(selected_vars)].isna().any(axis=1)
    clean = window[~missing_mask]
    n_dropped = int(missing_mask.sum())
    if len(clean) < 2:
        raise ValueError(
            f"Row range/downsampling/missing-data removal leaves only {len(clean)} "
            "row(s) -- need at least 2."
        )

    # -- anti-aliasing lowpass filter before downsampling: a plain strided
    # pick (every Nth row) can alias high-frequency content into spurious
    # low-frequency structure. A centered rolling mean over a window the
    # size of the downsample factor attenuates that first.
    if downsample > 1:
        smoothed = clean[list(selected_vars)].rolling(
            window=downsample, center=True, min_periods=1
        ).mean()
        values = smoothed.iloc[::downsample].to_numpy()
        base_rows = clean.index.to_numpy()[::downsample]
    else:
        values = clean[list(selected_vars)].to_numpy()
        base_rows = clean.index.to_numpy()

    if zscore_on:
        # ddof=1 matches MATLAB's zscore convention (sample std, N-1) --
        # see this package's own porting notes.
        X_raw = scipy_zscore(values, ddof=1, axis=0)
    else:
        X_raw = values
    n_raw = X_raw.shape[0]

    # -- delay embedding: concatenate `order` copies of the state, each
    # `lag` time points apart. order=1 (default) skips this.
    if order > 1:
        if lag < 1:
            raise ValueError("Embed lag must be at least 1 when embed order > 1.")
        n = n_raw - (order - 1) * lag
        if n < 2:
            raise ValueError(
                f"Embed lag/order too large: only {n_raw} rows of data available."
            )
        nvars = X_raw.shape[1]
        X = np.zeros((n, nvars * order))
        for j in range(order):
            X[:, j * nvars:(j + 1) * nvars] = X_raw[j * lag: j * lag + n, :]
    else:
        n = n_raw
        X = X_raw

    # original df row indices aligned with each embedded state (the most
    # recent slice, since embedding stacks past->present), mapped back
    # through base_rows since X_raw may already be a range/downsample subset.
    rows = base_rows[(n_raw - n):]
    tidx = np.arange(n)

    D = cdist(X, X, metric="euclidean")
    g, par = tknndigraph(
        D, k, tidx,
        time_exclude_range=texclude,
        max_neighbor_dist_prct=maxdistprct,
        max_neighbor_dist=maxdist,
        reciprocal=reciprocal,
    )
    g_simp, members, _nodesize, _D_simp = filtergraph(g, d, reciprocal=reciprocal)

    return {
        "g_simp": g_simp, "members": members, "rows": rows, "tidx": tidx,
        "par": par, "n_dropped": n_dropped, "n_window": len(window),
        "k": k, "d": d, "texclude": texclude, "lag": lag, "order": order,
        "downsample": downsample,
    }


# ============================================================ cheap: render

def render_network(built, df, color_var, time_var, nodesizemode, labelmethod, show_recurrence):
    """Render the (cached) network and optional recurrence plot.

    Returns the rendered artifacts so the caller can offer them as
    downloads without recomputing anything: the standalone interactive
    HTML page, the recurrence-plot figure (or None), and the resolved
    colorvar/colorlabel/title needed to render a static PNG on demand.
    """
    g_simp, members, rows, tidx, par = (
        built["g_simp"], built["members"], built["rows"], built["tidx"], built["par"]
    )

    if color_var == "(row index)":
        colorvar = tidx.astype(float)
        colorlabel = "row index"
    else:
        colorvar = df[color_var].to_numpy()[rows]
        colorlabel = color_var

    title = f"k={built['k']}, d={built['d']}, texclude={built['texclude']}, maxdist={par['max_neighbor_dist']:.4g}"
    if built["order"] > 1:
        title += f", lag={built['lag']}, order={built['order']}"
    if built["downsample"] > 1:
        title += f", downsample={built['downsample']}"

    net, html = plot_tmgraph_interactive(
        g_simp, colorvar, members,
        colorlabel=colorlabel, nodesizemode=nodesizemode, labelmethod=labelmethod,
        title=title,
    )
    components.html(html, height=800, scrolling=True)

    fig = None
    if show_recurrence:
        if time_var == "(row index)":
            t = tidx
        else:
            t = df[time_var].to_numpy()[rows]

        D_geo = tcm_distance(g_simp, members)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5.5))
        im = ax.imshow(D_geo, cmap="hot", extent=[t.min(), t.max(), t.max(), t.min()])
        ax.set_xlabel("time")
        ax.set_ylabel("time")
        ax.set_title("geodesic recurrence plot")
        fig.colorbar(im, ax=ax, label="path length")
        st.pyplot(fig)

    return html, fig, colorvar, colorlabel, title


# ========================================================= code generation

def _coderepr(x):
    """repr() for embedding in generated source, but emitting infinities
    as ``np.inf`` -- plain repr(float('inf')) is the bare token ``inf``,
    which is a NameError when the generated script is actually run."""
    if isinstance(x, float) and np.isinf(x):
        return "np.inf" if x > 0 else "-np.inf"
    return repr(x)


def generate_code(source_code, selected_vars, zscore_on, start_row, end_row, downsample,
                   lag, order, k, d, texclude, maxdistprct, maxdist, reciprocal,
                   color_var, time_var, nodesizemode, labelmethod, show_recurrence):
    end_row_repr = "None" if end_row is None else repr(end_row)
    lines = [
        "# Temporal Mapper -- generated by the Streamlit app's code view",
        "import numpy as np",
        "import pandas as pd",
        "from scipy.spatial.distance import cdist",
        "from scipy.stats import zscore",
        "from tmapper import tknndigraph, filtergraph, plot_tmgraph_interactive, tcm_distance",
        "",
        source_code,
        "",
        f"selected_vars = {list(selected_vars)!r}",
        f"start_row, end_row, downsample = {start_row!r}, {end_row_repr}, {downsample!r}",
        "end_row = (len(dat) - 1) if end_row is None else min(end_row, len(dat) - 1)",
        "window = dat.iloc[start_row:end_row + 1]",
        "missing_mask = window[selected_vars].isna().any(axis=1)",
        "clean = window[~missing_mask]",
        "if downsample > 1:",
        "    smoothed = clean[selected_vars].rolling(window=downsample, center=True, min_periods=1).mean()",
        "    values = smoothed.iloc[::downsample].to_numpy()",
        "    base_rows = clean.index.to_numpy()[::downsample]",
        "else:",
        "    values = clean[selected_vars].to_numpy()",
        "    base_rows = clean.index.to_numpy()",
        "",
    ]
    if zscore_on:
        lines.append("X_raw = zscore(values, ddof=1, axis=0)  # ddof=1 matches MATLAB's convention")
    else:
        lines.append("X_raw = values")
    lines += [
        "n_raw = X_raw.shape[0]",
        f"lag, order = {lag!r}, {order!r}",
    ]
    if order > 1:
        lines += [
            "n = n_raw - (order - 1) * lag",
            "nvars = X_raw.shape[1]",
            "X = np.zeros((n, nvars * order))",
            "for j in range(order):",
            "    X[:, j*nvars:(j+1)*nvars] = X_raw[j*lag : j*lag + n, :]",
        ]
    else:
        lines += ["n = n_raw", "X = X_raw"]
    lines += [
        "rows = base_rows[(n_raw - n):]",
        "tidx = np.arange(n)",
        "",
        "D = cdist(X, X, metric='euclidean')",
        f"g, par = tknndigraph(D, {k!r}, tidx, time_exclude_range={texclude!r}, "
        f"max_neighbor_dist_prct={_coderepr(maxdistprct)}, max_neighbor_dist={_coderepr(maxdist)}, "
        f"reciprocal={reciprocal!r})",
        f"g_simp, members, nodesize, D_simp = filtergraph(g, {d!r}, reciprocal={reciprocal!r})",
        "",
    ]
    color_expr = "tidx.astype(float)" if color_var == "(row index)" else f"dat['{color_var}'].to_numpy()[rows]"
    lines += [
        f"colorvar = {color_expr}",
        f"net, html = plot_tmgraph_interactive(g_simp, colorvar, members, "
        f"colorlabel={color_var!r}, nodesizemode={nodesizemode!r}, labelmethod={labelmethod!r}, "
        f"output_path='tmgraph.html')",
        "# open tmgraph.html in a browser, or embed `html` directly (e.g. in Streamlit)",
    ]
    if show_recurrence:
        time_expr = "tidx" if time_var == "(row index)" else f"dat['{time_var}'].to_numpy()[rows]"
        lines += [
            "",
            f"t = {time_expr}",
            "D_geo = tcm_distance(g_simp, members)",
        ]
    return "\n".join(lines)


# ===================================================================== UI

def main():
    st.set_page_config(page_title="Temporal Mapper", layout="wide")
    st.title("Temporal Mapper")

    # Reset must apply BEFORE any widget below is instantiated this run --
    # Streamlit forbids writing session_state[key] once a run has already
    # created the widget bound to that key. The Reset button (below) just
    # sets this flag and calls st.rerun(); on the resulting fresh run, this
    # check fires first, so the defaults land before any widget exists.
    if st.session_state.pop("_do_reset", False):
        for key, val in DEFAULTS.items():
            st.session_state[key] = val

    with st.sidebar:
        st.header("Data")
        if st.button("Load sample data", use_container_width=True):
            # the bundled CSV is the *full* historical daily record (57709
            # rows) -- same recent-slice trim as this project's Quickstart/
            # tmapper_demo.m (dat.iloc[53883:]), since the untrimmed file
            # is unusable as-is: cdist's pairwise distance matrix is O(N^2)
            # in memory (57709 rows would need ~25 GiB).
            sample_df, dropped_col = read_csv_smart(SAMPLE_DATA_PATH)
            sample_df = sample_df.iloc[53883:].reset_index(drop=True)
            src = ['dat = pd.read_csv("sampledata/EL_temp.csv")']
            if dropped_col:
                src.append("dat = dat.drop(columns=dat.columns[0])  # stray unnamed index column")
            src.append("dat = dat.iloc[53883:].reset_index(drop=True)  # recent slice, as in the Quickstart")
            set_data(sample_df, f"sample data ({SAMPLE_DATA_PATH.name}, recent slice)", "\n".join(src))
        uploaded = st.file_uploader("...or load a CSV file", type=["csv", "txt"])
        if uploaded is not None and st.session_state.get("data_source") != f"uploaded: {uploaded.name}":
            up_df, dropped_col = read_csv_smart(uploaded)
            # only the filename is available (Streamlit uploads are
            # in-memory), so the generated line assumes the file sits in
            # the working directory -- adjust the path when re-running.
            src = [f'dat = pd.read_csv("{uploaded.name}")']
            if dropped_col:
                src.append("dat = dat.drop(columns=dat.columns[0])  # stray unnamed index column")
            src.append("dat = dat.reset_index(drop=True)")
            set_data(up_df, f"uploaded: {uploaded.name}", "\n".join(src))

        if "data" not in st.session_state:
            st.info("Load data to get started.")
            return
        df = st.session_state["data"]
        st.caption(f"Loaded: {len(df)} rows from {st.session_state['data_source']}")

        st.header("Variables & Preprocessing")
        selected_vars = st.multiselect(
            "Variables", st.session_state["numeric_vars"],
            default=st.session_state["selected_vars"], key="selected_vars",
        )
        zscore_on = st.checkbox("z-score variables", value=DEFAULTS["zscore"], key="zscore")
        col1, col2 = st.columns(2)
        start_row = col1.number_input(
            "start row (0-indexed)", min_value=0, max_value=max(len(df) - 1, 0),
            value=DEFAULTS["start_row"], key="start_row",
        )
        end_row_str = col2.text_input(
            "end row", value="last", key="end_row_str",
            help="A number, or 'last' for the final row.",
        )
        end_row = None if end_row_str.strip().lower() == "last" else int(end_row_str)
        downsample = st.number_input(
            "downsample (N)", min_value=1, value=DEFAULTS["downsample"], key="downsample",
            help="Keep every Nth row; a centered rolling-mean lowpass is applied first to avoid aliasing.",
        )
        col3, col4 = st.columns(2)
        lag = col3.number_input("embed lag", min_value=0, value=DEFAULTS["embed_lag"], key="embed_lag")
        order = col4.number_input("embed order", min_value=1, value=DEFAULTS["embed_order"], key="embed_order")

        st.header("Network Parameters")
        col5, col6 = st.columns(2)
        k = col5.number_input("k (neighbors)", min_value=1, value=DEFAULTS["k"], key="k")
        d = col6.number_input("d (compression)", min_value=0.0, value=DEFAULTS["d"], key="d")
        texclude = st.number_input("texclude", min_value=1, value=DEFAULTS["texclude"], key="texclude")
        col7, col8 = st.columns(2)
        maxdistprct = col7.number_input(
            "max dist %ile", min_value=0.0, max_value=100.0, value=DEFAULTS["maxdistprct"], key="maxdistprct"
        )
        # st.number_input rejects inf outright (enforces a finite JS-float
        # bound), so this is a text field with an "inf" sentinel instead --
        # same convention as "end row" above.
        maxdist_str = col8.text_input("max dist", value="inf", key="maxdist_str")
        try:
            maxdist = np.inf if maxdist_str.strip().lower() == "inf" else float(maxdist_str)
        except ValueError:
            st.error(f"max dist must be a number or 'inf', got {maxdist_str!r}.")
            return
        reciprocal = st.checkbox("reciprocal", value=DEFAULTS["reciprocal"], key="reciprocal")

        st.header("Plot Options")
        color_options = ["(row index)"] + st.session_state["numeric_vars"]
        color_var = st.selectbox("Color by", color_options, key="color_var")
        time_var = st.selectbox("Time axis", color_options, key="time_var")
        nodesizemode = st.selectbox("Node size", ["log", "rank", "original"], key="nodesizemode")
        labelmethod = st.selectbox("Label method", ["mode", "mean", "median", "none"], key="labelmethod")
        show_recurrence = st.checkbox(
            "Show recurrence plot", value=DEFAULTS["show_recurrence"], key="show_recurrence"
        )

        col9, col10 = st.columns(2)
        build_clicked = col9.button("Build Network", type="primary", use_container_width=True)
        if col10.button("Reset", use_container_width=True):
            st.session_state["_do_reset"] = True
            st.rerun()

    MAX_WINDOW_ROWS = 8000  # cdist's pairwise distance matrix is O(N^2) in memory

    if build_clicked:
        resolved_end = (len(df) - 1) if end_row is None else min(end_row, len(df) - 1)
        window_rows = resolved_end - start_row + 1
        if not selected_vars:
            st.error("Select at least one variable to build the network from.")
        elif window_rows > MAX_WINDOW_ROWS and downsample == 1:
            st.error(
                f"The selected row range has {window_rows} rows -- computing a full "
                f"pairwise distance matrix at this size needs "
                f"~{8 * window_rows**2 / 1e9:.1f} GB of memory. Restrict the row range "
                f"(start row/end row) or set downsample (N) > 1 first."
            )
        else:
            with st.spinner("Building network..."):
                try:
                    built = build_network(
                        df, tuple(selected_vars), zscore_on, start_row, end_row, downsample,
                        lag, order, k, d, texclude, maxdistprct, maxdist, reciprocal,
                    )
                    st.session_state["built"] = built
                except ValueError as e:
                    st.session_state.pop("built", None)
                    st.error(str(e))

    if "built" in st.session_state:
        built = st.session_state["built"]
        if built["n_dropped"] > 0:
            st.warning(
                f"Dropped {built['n_dropped']} of {built['n_window']} row(s) in the "
                "selected range due to missing values in the selected variables."
            )
        st.success(f"Built network: {built['g_simp'].number_of_nodes()} nodes, "
                   f"{built['g_simp'].number_of_edges()} edges.")
        html, rec_fig, colorvar, colorlabel, title = render_network(
            built, df, color_var, time_var, nodesizemode, labelmethod, show_recurrence
        )

        with st.expander("Export / share"):
            # the pyvis page is built with cdn_resources="in_line", so this
            # file is fully self-contained -- it stays interactive offline
            # and can just be emailed or dropped in a shared folder.
            st.download_button(
                "Interactive network (.html)", data=html, file_name="tmgraph.html",
                mime="text/html", use_container_width=True,
                help="Self-contained page: still draggable/zoomable with no internet or install.",
            )

            if rec_fig is not None:
                buf = BytesIO()
                rec_fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
                st.download_button(
                    "Recurrence plot (.png)", data=buf.getvalue(),
                    file_name="recurrence_plot.png", mime="image/png",
                    use_container_width=True,
                )

            # Static network PNG is gated behind a checkbox: unlike the two
            # above (whose artifacts already exist), it re-runs the layout,
            # which would otherwise cost that on every single rerun.
            if st.checkbox("Render a static network PNG (for figures)"):
                import matplotlib.pyplot as plt
                from tmapper import plot_tmgraph

                with st.spinner("Rendering static figure..."):
                    fig_static, ax_static = plt.subplots(figsize=(7, 6))
                    plot_tmgraph(
                        built["g_simp"], colorvar, built["members"], ax=ax_static,
                        nodesizemode=nodesizemode, labelmethod=labelmethod,
                        colorlabel=colorlabel,
                    )
                    ax_static.set_title(title, fontsize=9)
                    sbuf = BytesIO()
                    fig_static.savefig(sbuf, format="png", dpi=200, bbox_inches="tight")
                st.pyplot(fig_static)
                plt.close(fig_static)
                st.download_button(
                    "Network figure (.png)", data=sbuf.getvalue(),
                    file_name="tmgraph.png", mime="image/png", use_container_width=True,
                )

            # node-level table, for downstream analysis outside the app
            node_csv = pd.DataFrame({
                "node": list(built["g_simp"].nodes()),
                "n_members": [len(m) for m in built["members"]],
                "first_row": [int(built["rows"][m[0]]) for m in built["members"]],
            }).to_csv(index=False)
            st.download_button(
                "Node table (.csv)", data=node_csv, file_name="tmgraph_nodes.csv",
                mime="text/csv", use_container_width=True,
                help="One row per network node: member count and the first original data row it covers.",
            )

        with st.expander("Show equivalent code"):
            st.code(
                generate_code(
                    st.session_state["data_source_code"], selected_vars, zscore_on, start_row, end_row,
                    downsample, lag, order, k, d, texclude, maxdistprct, maxdist, reciprocal,
                    color_var, time_var, nodesizemode, labelmethod, show_recurrence,
                ),
                language="python",
            )
    else:
        st.info("Set parameters in the sidebar and click **Build Network** to get started.")


if __name__ == "__main__":
    main()
