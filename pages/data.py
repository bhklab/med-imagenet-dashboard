"""Shared data loading and helpers for all pages.

Uses the ``IndexedDatasets`` store from the *imgnet* package so that
downloading, path resolution, and per-collection access are handled
centrally.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from imgnet.collections.store import IndexedDatasets
from imgnet.collections.source import FileType


@st.cache_resource(show_spinner="Connecting to indexed datasets …")
def get_store() -> IndexedDatasets:
    """Return a shared ``IndexedDatasets`` instance (created once)."""
    return IndexedDatasets(force_download=True)


@st.cache_data(show_spinner="Loading index data …")
def load_all_indexes() -> pd.DataFrame:
    """Read every collection index, tag with dataset name, normalise columns."""
    store = get_store()
    frames: list[pd.DataFrame] = []

    for collection in store.collections:
        try:
            df = store.index(collection)
        except Exception:
            continue

        df.insert(0, "dataset", collection)
        fmt = "NIfTI" if store.file_type(collection) == FileType.NIFTI else "DICOM"
        df.insert(1, "format", fmt)

        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner="Loading collections summary …")
def load_summary() -> dict:
    """Return the ``collections_summary.json`` via the store."""
    return get_store().summary()


def dataset_summary() -> pd.DataFrame:
    """One row per collection — same columns as ``IndexedDatasets.display_summary``.

    Columns: dataset, body_parts, modalities, series_count, size_gb,
    format (file type), source.
    """
    summary_data = load_summary()

    rows: list[dict] = []
    for collection, info in summary_data.items():

        rows.append(
            {
                "dataset": collection,
                "body_parts": ", ".join(info.get("BodyPartsExamined", [])),
                "modalities": ", ".join(info.get("Modalities", [])),
                "images": info.get("Images", 0),
                "size_gb": info.get("Size", 0.0),
                "format": info.get("File Type", "N/A"),
                "source": info.get("Source", "N/A"),
            }
        )

    return pd.DataFrame(rows)


def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filter widgets and return the filtered DataFrame."""
    st.sidebar.header("Filters")

    datasets = sorted(df["dataset"].unique())
    selected_datasets = st.sidebar.multiselect(
        "Datasets", datasets, default=[], placeholder="All datasets"
    )
    if selected_datasets:
        df = df[df["dataset"].isin(selected_datasets)]

    fmt = st.sidebar.multiselect(
        "Format", ["DICOM", "NIfTI"], default=[], placeholder="All formats"
    )
    if fmt:
        df = df[df["format"].isin(fmt)]

    if "Modality" in df.columns:
        modalities = sorted(df["Modality"].dropna().unique())
        selected_mod = st.sidebar.multiselect(
            "Modality", modalities, default=[], placeholder="All modalities"
        )
        if selected_mod:
            df = df[df["Modality"].isin(selected_mod)]

    return df


class PageFilterResult:
    """Container for the outputs of ``apply_page_filters``."""

    def __init__(self, df: pd.DataFrame, selected_formats: list[str]) -> None:
        self.df = df
        self.selected_formats = selected_formats


def apply_page_filters(
    df: pd.DataFrame,
    default_datasets: list[str] | None = None,
) -> PageFilterResult:
    """Render filter widgets in the main page (not sidebar) and return the
    filtered DataFrame together with the raw filter selections.

    Use on Metrics and Manifest so the sidebar can show the Voxel area instead.
    """
    st.subheader("Filters")
    c1, c2, c3 = st.columns(3)

    with c1:
        datasets = sorted(df["dataset"].unique())
        defaults = [d for d in (default_datasets or []) if d in datasets]
        selected_datasets = st.multiselect(
            "Datasets", datasets, default=defaults, placeholder="All datasets"
        )
    with c2:
        fmt = st.multiselect(
            "Format", ["DICOM", "NIfTI"], default=[], placeholder="All formats"
        )
    with c3:
        if "Modality" in df.columns:
            modalities = sorted(df["Modality"].dropna().unique())
            selected_mod = st.multiselect(
                "Modality", modalities, default=[], placeholder="All modalities"
            )
        else:
            selected_mod = []

    if selected_datasets:
        df = df[df["dataset"].isin(selected_datasets)]
    if fmt:
        df = df[df["format"].isin(fmt)]
    if selected_mod:
        df = df[df["Modality"].isin(selected_mod)]

    return PageFilterResult(df=df, selected_formats=fmt)


