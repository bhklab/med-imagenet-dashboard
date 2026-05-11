"""Datasets page -- pick a collection and inspect a tabbed summary card."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import plotly.express as px
import streamlit as st

from pages.data import dataset_summary, get_store, load_index_dataset

_INTERLACER_DIR = Path(__file__).resolve().parent / "interlacer_visualizations"

ACCENT = "#4A90D9"


def _preview_cols(file_type: str) -> list[str]:
    if file_type == "DICOM":
        return [
            "PatientID",
            "Modality",
            "SeriesInstanceUID",
            "BodyPartExamined",
            "ReferencedSeriesUID"
        ]
    elif file_type == "NIFTI":
        return [
            "filepath",
            "Modality",
            "size",
            "spacing",
            "reference_scan"
        ]
    return []


def _interlacer_html_path(ds_name: str) -> Path | None:
    """Return path to forest .html for this dataset, or None if not found."""
    if not _INTERLACER_DIR.is_dir():
        return None
    for candidate in (ds_name, ds_name.replace(" ", "-"), ds_name.replace("-", " ")):
        path = _INTERLACER_DIR / f"{candidate}_forest.html"
        if path.is_file():
            return path
    return None


def _collection_picker(summary: pd.DataFrame, query_dataset: str | None) -> str | None:
    """Search box + selectbox for picking a collection."""
    options = summary["dataset"].sort_values().tolist()
    search = st.text_input(
        "Search collections",
        placeholder="Type to filter collection names",
        key="ds_search",
    )

    filtered = options
    if search:
        filtered = [n for n in options if search.lower() in n.lower()]

    if not filtered:
        st.info("No collections match your search. Try a different term.")
        return None

    if query_dataset and query_dataset in filtered:
        default_idx = filtered.index(query_dataset)
    else:
        default_idx = 0

    selected = st.selectbox(
        "Collection",
        filtered,
        index=default_idx,
        key="ds_selectbox",
    )

    if selected:
        st.query_params["dataset"] = selected
    return selected


def _card_header(ds_name: str, row: pd.Series, description: str | None) -> None:
    """Render collection title, description, and KPI metrics."""
    st.markdown(f"### {ds_name}")

    if description:
        st.markdown(description)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("File Type", row["format"])
    c2.metric("Source", row["source"])
    c3.metric("Images", f"{int(row['images']):,}")
    c4.metric("Size", f"{row['size_gb']:.2f} GB")

    info_parts = []
    if row.get("modalities"):
        info_parts.append(f"**Modalities:** {row['modalities']}")
    if row.get("body_parts"):
        info_parts.append(f"**Body parts:** {row['body_parts']}")
    if info_parts:
        st.caption(" | ".join(info_parts))


def _tab_modality(ds_df: pd.DataFrame) -> None:
    """Content for the Modality distribution tab."""
    if "Modality" not in ds_df.columns or ds_df["Modality"].dropna().empty:
        st.info("No modality data available for this collection.")
        return

    mod_counts = (
        ds_df["Modality"]
        .value_counts()
        .rename_axis("Modality")
        .reset_index(name="count")
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        fig = px.pie(
            mod_counts,
            names="Modality",
            values="count",
            hole=0.45,
        )
        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", y=-0.15),
            height=340,
        )
        fig.update_traces(textinfo="percent+label", textposition="outside")
        st.plotly_chart(fig, width='stretch')
    with c2:
        bar_df = mod_counts.sort_values("count", ascending=True)
        fig = px.bar(
            bar_df,
            x="count",
            y="Modality",
            orientation="h",
            text="count",
            color_discrete_sequence=[ACCENT],
        )
        fig.update_layout(
            xaxis_title="Images",
            yaxis_title="",
            yaxis=dict(dtick=1),
            showlegend=False,
            height=max(300, len(bar_df) * 38),
            margin=dict(t=10, b=10, l=10, r=10),
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width='stretch')


def _tab_interlacer(ds_name: str) -> None:
    """Content for the Series Hierarchy tab."""
    html_path = _interlacer_html_path(ds_name)
    if html_path is None:
        st.info(
            "No series hierarchy visualization is available for this collection yet."
        )
        return

    st.caption("Drag nodes to rearrange; scroll to zoom.")
    try:
        html_content = html_path.read_text(encoding="utf-8", errors="replace")
        st.components.v1.html(html_content, height=560, scrolling=True)
    except Exception as e:
        st.warning(f"Could not load forest HTML: {e}")


def _tab_sample_rows(ds_df: pd.DataFrame, file_type: str) -> None:
    """Content for the Sample Rows tab."""
    preview_cols = [col for col in _preview_cols(file_type) if col in ds_df.columns]
    if not preview_cols:
        st.info("No preview columns available for this file type.")
        return

    st.dataframe(
        ds_df[preview_cols].head(200),
        width='stretch',
        hide_index=True,
        height=420,
    )
    if len(ds_df) > 200:
        st.caption(f"Showing first 200 of {len(ds_df):,} rows.")


def _dataset_card(ds_name: str, ds_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    row = summary[summary["dataset"] == ds_name].iloc[0]

    store = get_store()
    description = store.description(ds_name)

    st.divider()

    with st.container(border=True):
        _card_header(ds_name, row, description)

        tab_labels = [
            ":material/bar_chart: Modality",
            ":material/account_tree: Series hierarchy",
            ":material/table_rows: Sample rows",
        ]
        tab_mod, tab_inter, tab_sample = st.tabs(tab_labels)

        with tab_mod:
            _tab_modality(ds_df)

        with tab_inter:
            _tab_interlacer(ds_name)

        with tab_sample:
            _tab_sample_rows(ds_df, row["format"])

    st.page_link(
        "pages/home.py",
        label="Back to Home",
        icon=":material/home:",
        width='stretch',
    )


def run() -> None:
    st.header("Collections")
    st.caption(
        "Select a collection below to explore its metadata card."
    )

    summary = dataset_summary().sort_values("dataset", ascending=True)
    if summary.empty:
        st.warning("No collections summary available.")
        return

    query_dataset = st.query_params.get("dataset")
    if isinstance(query_dataset, list):
        query_dataset = query_dataset[0] if query_dataset else None
    query_dataset = unquote(query_dataset) if query_dataset else None
    if query_dataset not in set(summary["dataset"].tolist()):
        query_dataset = None

    selected_dataset = _collection_picker(summary, query_dataset)

    if selected_dataset:
        ds_df = load_index_dataset(selected_dataset)
        if ds_df.empty:
            st.warning(
                f"No index data found for **{selected_dataset}**. "
                "The collection may still be indexing."
            )
        else:
            _dataset_card(selected_dataset, ds_df, summary)
    else:
        if not query_dataset:
            st.info("Select a collection above to view its card.")


run()
