"""Landing page -- hero section, key stats, and top-level collection table."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from pages.data import dataset_summary, get_store, load_all_indexes, load_summary

ACCENT = "#4A90D9"


def _hero() -> None:
    store = get_store()
    n_collections = len(store.collections)
    st.markdown(
        f"""
        <div style="padding: 2rem 0 1rem 0; text-align: center; color: var(--text-color, inherit);">
            <h1 style="font-size: 2.6rem; margin-bottom: .25rem; color: inherit;">
                Med-ImageNet Explorer
            </h1>
            <p style="font-size: 1.1rem; opacity: .72; max-width: 620px; margin: 0 auto; color: inherit;">
                Browse, visualise, and export metadata across
                <strong>{n_collections} indexed medical imaging collections</strong>
                &mdash; DICOM &amp; NIfTI datasets from TCIA, Zenodo, S3, and more.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _kpi_row(df: pd.DataFrame) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)

    n_datasets = df["dataset"].nunique()
    n_images = len(df)
    n_patients = df["PatientID"].nunique() if "PatientID" in df.columns else 0
    n_modalities = df["Modality"].nunique() if "Modality" in df.columns else 0

    col1.metric("Collections", f"{n_datasets:,}")
    col2.metric("Total Images", f"{n_images:,}")
    col3.metric("Unique Patients", f"{n_patients:,}")
    col4.metric("Modalities", f"{n_modalities:,}")

    summary_data = load_summary()
    total_gb = sum(info.get("Size", 0) for info in summary_data.values())
    col5.metric("Total Size", f"{total_gb:,.1f} GB")


def _modality_snapshot(df: pd.DataFrame) -> None:
    if "Modality" not in df.columns or df["Modality"].dropna().empty:
        return

    st.subheader("Modality snapshot")
    mod_counts = df["Modality"].value_counts().reset_index()
    mod_counts.columns = ["Modality", "count"]

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
            height=320,
        )
        fig.update_traces(textinfo="percent+label", textposition="outside")
        st.plotly_chart(fig, width='stretch')

    with c2:
        top = mod_counts.head(12).sort_values("count", ascending=True)
        fig = px.bar(
            top,
            x="count",
            y="Modality",
            orientation="h",
            text="count",
            color_discrete_sequence=[ACCENT],
        )
        fig.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            yaxis=dict(dtick=1),
            xaxis_title="Images",
            yaxis_title="",
            height=320,
            showlegend=False,
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width='stretch')


def _top_datasets_table() -> None:
    st.subheader("Collections at a glance")
    st.caption("Click a row to open that collection in the Collections page.")

    summary = dataset_summary().sort_values("dataset", ascending=True)
    table = summary.copy()
    table["collection"] = table["dataset"]

    col_config = {
        "collection": st.column_config.TextColumn(
            "Collection",
            width="medium",
        ),
        "body_parts": st.column_config.TextColumn("Body Parts"),
        "modalities": st.column_config.TextColumn("Modalities"),
        "images": st.column_config.NumberColumn("Images", format="%d"),
        "size_gb": st.column_config.NumberColumn("Size (GB)", format="%.2f"),
        "format": st.column_config.TextColumn("File Type", width="small"),
        "source": st.column_config.TextColumn("Source", width="small"),
    }

    event = st.dataframe(
        table[list(col_config.keys())],
        width='stretch',
        hide_index=True,
        height=440,
        column_config=col_config,
        on_select="rerun",
        selection_mode="single-row",
        key="home_collections_table",
    )

    rows = event.selection.rows if event and event.selection else []
    if rows:
        selected_dataset = str(table.iloc[rows[0]]["dataset"])
        st.query_params["dataset"] = selected_dataset
        st.switch_page("pages/datasets.py")


def _quick_links() -> None:
    st.divider()
    st.subheader("Quick links")

    c1, = st.columns(1)
    with c1:
        st.page_link(
            "pages/datasets.py",
            label="Collections",
            icon=":material/dataset:",
            width='stretch',
        )


def run() -> None:
    df = load_all_indexes()
    _hero()
    _kpi_row(df)
    st.markdown("")
    _modality_snapshot(df)
    _top_datasets_table()
    _quick_links()


run()
