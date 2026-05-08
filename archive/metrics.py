"""Metrics page -- interactive charts for modality, collection size, and disk usage."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from pages.data import apply_page_filters, load_all_indexes, load_summary


def _modality_pie(mod_counts: pd.DataFrame) -> None:
    fig = px.pie(
        mod_counts,
        names="Modality",
        values="count",
        title="Modality distribution",
        hole=0.4,
    )
    fig.update_traces(textinfo="percent+label", textposition="outside")
    fig.update_layout(height=500, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width='stretch')


def _modality_bar(mod_counts: pd.DataFrame) -> None:
    fig = px.bar(
        mod_counts.sort_values("count", ascending=True),
        x="count",
        y="Modality",
        orientation="h",
        title="Images per modality",
        text="count",
        color_discrete_sequence=["#4A90D9"],
    )
    fig.update_layout(
        height=500,
        yaxis=dict(dtick=1),
        xaxis_title="Images",
        yaxis_title="",
        showlegend=False,
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width='stretch')


def _modality_breakdown_per_collection(df: pd.DataFrame) -> None:
    if "Modality" not in df.columns or df["Modality"].dropna().empty:
        st.info("No modality data in the current selection.")
        return

    ds_order = df.groupby("dataset").size().sort_values(ascending=False).index.tolist()
    cross = (
        df.groupby(["dataset", "Modality"])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        cross,
        x="count",
        y="dataset",
        color="Modality",
        orientation="h",
        title="Modality breakdown per collection",
        height=max(400, len(ds_order) * 22),
        category_orders={"dataset": ds_order},
    )
    fig.update_layout(
        yaxis=dict(dtick=1),
        margin=dict(l=220),
        xaxis_title="Images",
        yaxis_title="",
    )
    st.plotly_chart(fig, width='stretch')


def _collection_disk_sizes() -> None:
    """Bar chart of on-disk sizes (GB) from the collections summary."""
    summary_data = load_summary()
    if not summary_data:
        st.info("No collections summary available.")
        return

    rows = [
        {"collection": name, "size_gb": info.get("Size", 0)}
        for name, info in summary_data.items()
    ]
    size_df = pd.DataFrame(rows).sort_values("size_gb", ascending=True)

    total = size_df["size_gb"].sum()
    st.metric("Total size across all collections", f"{total:,.1f} GB")

    fig = px.bar(
        size_df,
        x="size_gb",
        y="collection",
        orientation="h",
        title="Collection size on disk (GB)",
        text=size_df["size_gb"].map(lambda v: f"{v:.2f}"),
        height=max(400, len(size_df) * 22),
        color_discrete_sequence=["#4A90D9"],
    )
    fig.update_layout(
        yaxis=dict(dtick=1),
        margin=dict(l=220),
        xaxis_title="GB",
        yaxis_title="",
        showlegend=False,
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width='stretch')


_PLOT_TYPES = {
    "Bar": "bar",
    "Histogram": "histogram",
    "Scatter": "scatter",
    "Box": "box",
    "Violin": "violin",
    "Pie": "pie",
}


def _custom_explorer(df: pd.DataFrame) -> None:
    """Let the user pick a plot type and assign columns to axes."""
    if df.empty:
        st.info("No data available.")
        return

    skip = {"dataset", "format"}
    usable_cols = [c for c in df.columns if c not in skip and df[c].notna().any()]

    numeric_cols = df[usable_cols].select_dtypes("number").columns.tolist()
    categorical_cols = [c for c in usable_cols if c not in numeric_cols]

    c_plot, c_x, c_y = st.columns(3)
    with c_plot:
        plot_type = st.selectbox("Plot type", list(_PLOT_TYPES.keys()))
    with c_x:
        x_col = st.selectbox("X axis", usable_cols, index=usable_cols.index("Modality") if "Modality" in usable_cols else 0)
    with c_y:
        if plot_type in ("Histogram", "Pie"):
            y_col = None
            st.selectbox("Y axis", ["(not used)"], disabled=True)
        else:
            default_y = numeric_cols[0] if numeric_cols else usable_cols[0]
            y_col = st.selectbox("Y axis", usable_cols, index=usable_cols.index(default_y))

    color_options = ["None"] + categorical_cols
    color_col = st.selectbox("Color / group by", color_options)
    if color_col == "None":
        color_col = None

    plot_df = df

    if plot_df.empty:
        st.warning("No rows match the selected filters.")
        return

    kind = _PLOT_TYPES[plot_type]
    kwargs: dict = dict(data_frame=plot_df, x=x_col, color=color_col)

    if kind == "bar":
        agg = plot_df.groupby([x_col] + ([color_col] if color_col else [])).size().reset_index(name="count")
        fig = px.bar(agg, x=x_col, y="count" if y_col is None else None, color=color_col)
        if y_col is not None:
            agg2 = plot_df.groupby([x_col] + ([color_col] if color_col else []))[y_col].mean().reset_index()
            fig = px.bar(agg2, x=x_col, y=y_col, color=color_col)
    elif kind == "histogram":
        fig = px.histogram(plot_df, x=x_col, color=color_col)
    elif kind == "scatter":
        fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_col)
    elif kind == "box":
        fig = px.box(plot_df, x=x_col, y=y_col, color=color_col)
    elif kind == "violin":
        fig = px.violin(plot_df, x=x_col, y=y_col, color=color_col)
    elif kind == "pie":
        counts = plot_df[x_col].value_counts().reset_index()
        counts.columns = [x_col, "count"]
        fig = px.pie(counts, names=x_col, values="count", hole=0.4)
    else:
        st.error(f"Unknown plot type: {plot_type}")
        return

    fig.update_layout(height=550)
    st.plotly_chart(fig, width='stretch')

    with st.expander("Preview data"):
        display_cols = [c for c in [x_col, y_col, color_col] if c is not None]
        st.dataframe(plot_df[list(dict.fromkeys(display_cols))].head(200), hide_index=True)


def run() -> None:
    st.header("Metrics")
    st.caption("Visual breakdowns of the indexed data. Use the filters below to narrow by collection, format, or modality.")

    raw_df = load_all_indexes()
    df = apply_page_filters(raw_df).df

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Modality distribution",
        "Images per modality",
        "Modality by collection",
        "Disk sizes (GB)",
        "Custom explorer",
    ])

    if "Modality" not in df.columns or df["Modality"].dropna().empty:
        with tab1:
            st.info("No modality data in the current selection.")
        with tab2:
            st.info("No modality data in the current selection.")
        with tab3:
            st.info("No modality data in the current selection.")
    else:
        mod_counts = df["Modality"].value_counts().reset_index()
        mod_counts.columns = ["Modality", "count"]

        with tab1:
            _modality_pie(mod_counts)
        with tab2:
            _modality_bar(mod_counts)
        with tab3:
            _modality_breakdown_per_collection(df)

    with tab4:
        _collection_disk_sizes()

    with tab5:
        _custom_explorer(df)


run()
