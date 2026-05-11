"""Landing page -- hero section, key stats, and top-level collection table."""

from __future__ import annotations

from urllib.parse import quote

import pandas as pd
import plotly.express as px
import streamlit as st
from pages.data import (
    dataset_summary,
    get_store,
    home_summary_kpis,
    modality_distribution_from_summary,
)

ACCENT = "#4A90D9"
# Med-ImageNet package / CLI / pipelines (this dashboard only reads published indexes).
GITHUB_MED_IMAGENET = "https://github.com/bhklab/med-imagenet"
DOCS_MED_IMAGENET = "https://bhklab.github.io/med-imagenet/dev/"
USER_FEEDBACK_FORM = "https://forms.gle/aGy5eR6YMpB6sFQE8"

# Modalities below this fraction of total estimated counts are merged into "Other" for charts.
_MIN_MODALITY_PIE_SHARE = 0.02


def _collapse_small_modalities(mod_counts: pd.DataFrame, *, min_share: float) -> pd.DataFrame:
    """Drop very small modality slices by merging them into a single *Other* row."""
    if mod_counts.empty or "Modality" not in mod_counts.columns or "count" not in mod_counts.columns:
        return mod_counts
    total = float(mod_counts["count"].sum())
    if total <= 0:
        return mod_counts
    cutoff = total * min_share
    keep = mod_counts[mod_counts["count"] >= cutoff].copy()
    drop = mod_counts[mod_counts["count"] < cutoff]
    if keep.empty or drop.empty:
        return mod_counts
    other = int(drop["count"].sum())
    if other <= 0:
        return keep.reset_index(drop=True)
    merged = pd.concat(
        [keep, pd.DataFrame([{"Modality": "Other", "count": other}])],
        ignore_index=True,
    )
    return merged.sort_values("count", ascending=False).reset_index(drop=True)


def _med_imagenet_tooling_strip() -> None:
    """Native Streamlit layout so light/dark follows Settings → Theme (no custom HTML/CSS)."""
    with st.container(border=True):
        st.subheader("Welcome to the early preview")
        st.link_button(
            "Click here to give us your feedback!",
            USER_FEEDBACK_FORM,
            type="primary",
            icon=":material/assignment_add:",
            help="Opens the Med-ImageNet feedback form in a new tab.",
            width='stretch',
            key="home_feedback_survey",
        )
        st.caption(
            "**Med-ImageNet** tooling is **CLI-only** for now (for example the **`imgnet`** command). "
            "Use the documentation for setup and usage. "
            "This dashboard **browses** published indexes; richer in-app workflows will follow over time."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.link_button(
                "Documentation and Installation Guide",
                DOCS_MED_IMAGENET,
                icon=":material/menu_book:",
                type="secondary",
                width='stretch',
            )
            
        with c2:
            st.link_button(
                "GitHub Repository",
                GITHUB_MED_IMAGENET,
                icon=":material/terminal:",
                width='stretch',
            )


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
                Browse and visualise metadata across
                <strong>{n_collections} indexed medical imaging collections</strong>
                <br>DICOM &amp; NIfTI datasets from TCIA, Zenodo, S3, and more.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _kpi_row(kpis: dict[str, int | float]) -> None:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Collections", f"{int(kpis['collections']):,}")
    col2.metric("Total Images", f"{int(kpis['total_images']):,}")
    col3.metric("Modalities", f"{int(kpis['modalities']):,}")
    col4.metric("Total Size", f"{float(kpis['total_size_gb']):,.1f} GB")


def _modality_snapshot(mod_counts: pd.DataFrame) -> None:
    if mod_counts.empty or "Modality" not in mod_counts.columns:
        return

    plot_df = _collapse_small_modalities(mod_counts, min_share=_MIN_MODALITY_PIE_SHARE)

    st.subheader("Modality snapshot")
    st.caption(
        "Counts are estimated from collection summaries (split evenly across listed modalities per collection). "
        f"Modalities under **{_MIN_MODALITY_PIE_SHARE:.0%}** of the total are grouped as **Other**."
    )

    # Slightly wider pie column + taller plot so the donut and outside labels are not cramped.
    c1, c2 = st.columns([1.25, 1.75])
    with c1:
        fig = px.pie(
            plot_df,
            names="Modality",
            values="count",
            hole=0.42,
        )
        fig.update_layout(
            margin=dict(t=20, b=72, l=8, r=8),
            legend=dict(orientation="h", y=-0.22, yanchor="top"),
            height=400,
            autosize=True,
        )
        fig.update_traces(textinfo="percent+label", textposition="outside")
        st.plotly_chart(fig, width='stretch', key="home_modality_pie")

    with c2:
        top = plot_df.head(12).sort_values("count", ascending=True)
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
            height=400,
            showlegend=False,
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width='stretch')


def _collections_page_href(dataset: str) -> str:
    """URL to the Collections page with ``dataset`` (from ``st.context.url``, never a fake host)."""
    # Keep common filename-style characters readable for the link label (regex captures query value).
    enc = quote(str(dataset), safe="/-_.~")
    base = st.context.url
    if base:
        base = base.split("#")[0].rstrip("/")
        low = base.lower()
        for suf in ("/home",):
            if low.endswith(suf):
                base = base[: -len(suf)].rstrip("/")
                break
        return f"{base}/collections?dataset={enc}"
    # No browser URL in context (e.g. some tests): same-origin root-relative path only.
    return f"/collections?dataset={enc}"


def _top_datasets_table() -> None:
    st.subheader("Collections at a glance")
    st.caption("Click a **collection** name to open it on the Collections page.")

    summary = dataset_summary().sort_values("dataset", ascending=True)
    table = summary.copy()
    # LinkColumn cell values are URLs; label is taken from the `dataset=` query (see display_text regex).
    table["collection"] = table["dataset"].astype(str).map(_collections_page_href)

    col_config = {
        "collection": st.column_config.LinkColumn(
            "Collection",
            help="Open this collection on the Collections page",
            display_text=r"dataset=(.+)$",
            validate=r"^(https?://|/)",
            width="medium",
        ),
        "body_parts": st.column_config.TextColumn("Body Parts"),
        "modalities": st.column_config.TextColumn("Modalities"),
        "images": st.column_config.NumberColumn("Images", format="%d"),
        "size_gb": st.column_config.NumberColumn("Size (GB)", format="%.2f"),
        "format": st.column_config.TextColumn("File Type", width="small"),
        "source": st.column_config.TextColumn("Source", width="small"),
    }

    display_cols = list(col_config.keys())
    st.dataframe(
        table[display_cols],
        width='stretch',
        hide_index=True,
        height=440,
        column_config=col_config,
        key="home_collections_table",
    )




def run() -> None:
    _med_imagenet_tooling_strip()
    _hero()
    _kpi_row(home_summary_kpis())
    st.markdown("")
    _modality_snapshot(modality_distribution_from_summary())
    _top_datasets_table()


run()
