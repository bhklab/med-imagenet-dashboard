"""Shared data loading and helpers for all pages.

Uses the ``IndexedDatasets`` store from the *imgnet* package so that
downloading, path resolution, and per-collection access are handled
centrally.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
import streamlit as st

from imgnet.collections.store import IndexedDatasets
from imgnet.collections.source import FileType


@st.cache_resource(show_spinner="Connecting to indexed datasets …")
def get_store() -> IndexedDatasets:
    """Return a shared ``IndexedDatasets`` instance (created once)."""
    return IndexedDatasets(force_download=True)


def _tag_index_frame(store: IndexedDatasets, collection: str, df: pd.DataFrame) -> pd.DataFrame:
    """Insert ``dataset`` and ``format`` columns (mutates a copy)."""
    out = df.copy()
    out.insert(0, "dataset", collection)
    fmt = "NIfTI" if store.file_type(collection) == FileType.NIFTI else "DICOM"
    out.insert(1, "format", fmt)
    return out


@st.cache_data(show_spinner="Loading index …")
def load_index_dataset(collection: str) -> pd.DataFrame:
    """Load one collection index (cached per collection)."""
    store = get_store()
    try:
        df = store.index(collection)
    except Exception:
        return pd.DataFrame()
    return _tag_index_frame(store, collection, df)


def load_indexes_datasets(collections: list[str]) -> pd.DataFrame:
    """Concatenate indexes for the given collection names (each slice may hit ``load_index_dataset`` cache)."""
    frames = [load_index_dataset(name) for name in collections]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_all_indexes() -> pd.DataFrame:
    """Read every collection index into one frame (not cached as a whole — prefer ``load_index_dataset``)."""
    store = get_store()
    return load_indexes_datasets(list(store.collections))


def modality_distribution_from_summary() -> pd.DataFrame:
    """Approximate modality image counts from ``collections_summary`` only (no index I/O)."""
    summary_data = load_summary()
    counts: defaultdict[str, float] = defaultdict(float)
    for info in summary_data.values():
        imgs = float(info.get("Images", 0) or 0)
        raw = info.get("Modalities") or []
        if isinstance(raw, str):
            mods = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
        else:
            mods = [str(m).strip() for m in raw if str(m).strip()]
        if not mods or imgs <= 0:
            continue
        share = imgs / len(mods)
        for m in mods:
            counts[m] += share
    if not counts:
        return pd.DataFrame(columns=["Modality", "count"])
    rows = [
        {"Modality": name, "count": max(1, int(round(val)))}
        for name, val in sorted(counts.items(), key=lambda x: -x[1])
    ]
    return pd.DataFrame(rows)


def home_summary_kpis() -> dict[str, int | float]:
    """Dashboard KPIs from summary JSON only (avoids loading per-image indexes on the home page)."""
    summary_data = load_summary()
    n_collections = len(summary_data)
    n_images = sum(int(info.get("Images", 0) or 0) for info in summary_data.values())
    total_gb = float(sum(float(info.get("Size", 0) or 0) for info in summary_data.values()))
    seen: set[str] = set()
    for info in summary_data.values():
        raw = info.get("Modalities") or []
        if isinstance(raw, str):
            for part in raw.replace(";", ",").split(","):
                p = part.strip()
                seen.add(p)
        else:
            for m in raw:
                seen.add(str(m))
    n_modalities = len(seen)
    return {
        "collections": n_collections,
        "total_images": n_images,
        "modalities": n_modalities,
        "total_size_gb": total_gb,
    }


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


