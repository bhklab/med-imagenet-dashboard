#!/usr/bin/env python3
"""Render print-quality poster panels for the Med-ImageNet Explorer.

Four standalone figures, each saved as PNG (raster) + PDF (vector):

  1. hero      — title + modality dot-grid visualization (the "wow" opener)
  2. kpis      — four stat cards (collections, images, size, modalities)
  3. modality  — 10×10 waffle chart of image modalities
  4. sources   — lollipop chart of images per data source

All numeric values come from the med-imagenet stack (``imgnet.collections.store.IndexedDatasets``)
via the same indexing and summary logic as ``pages/data.py``. Requires a working store / downloaded
indexes (same as running the Streamlit app).

Usage:
  pixi run python scripts/render_home_poster.py                   # all four panels
  pixi run python scripts/render_home_poster.py --panel hero
  pixi run python scripts/render_home_poster.py --use-local-cache # use indexes on disk when possible
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

# -------------------------- Design tokens ----------------------------------
BG = "#FFFFFF"
SOFT_BG = "#F8FAFC"
TEXT = "#0F172A"
MUTED = "#64748B"
# KPI card upper labels (Collections, …) — darker than MUTED for readability
KPI_LABEL = "#1E293B"
BORDER = "#E2E8F0"

BRAND_1 = "#1E3A8A"
BRAND_2 = "#2563EB"
BRAND_3 = "#0EA5E9"
BRAND_4 = "#14B8A6"
BRAND_5 = "#F59E0B"
BRAND_6 = "#E11D48"
BRAND_7 = "#7C3AED"
BRAND_8 = "#10B981"
BRAND_9 = "#8B5CF6"
BRAND_10 = "#F97316"

MODALITY_PALETTE = [
    BRAND_1,
    BRAND_2,
    BRAND_3,
    BRAND_5,
    BRAND_4,
    BRAND_7,
    BRAND_6,
    BRAND_8,
    BRAND_9,
    BRAND_10,
    "#64748B",
    "#94A3B8",
]

KPI_ACCENTS = [BRAND_2, BRAND_3, BRAND_4, BRAND_5]

PANELS = ("hero", "kpis", "modality", "sources")


def _is_flair_modality(name: object) -> bool:
    """True if modality label should be excluded (FLAIR sequences)."""
    s = str(name).strip().upper()
    return s.startswith("FLAIR")


def _drop_flair_modalities(pairs: list[tuple[str, int]]) -> list[tuple[str, int]]:
    out = [(n, c) for n, c in pairs if not _is_flair_modality(n)]
    return sorted(out, key=lambda x: -x[1])


# -------------------------- Data from imgnet / Med-ImageNet -----------------
def load_poster_data(*, force_download: bool = True) -> tuple[dict[str, float | int], list[tuple[str, int]], list[tuple[str, int]]]:
    """Pull metrics and series from IndexedDatasets (same sources as ``pages/data.py``).

    Modalities: prefer row-level ``Modality`` counts from concatenated indexes; if indexes are
    unavailable, approximate by splitting each collection's ``Modalities`` list from the summary
    and allocating image counts evenly across listed modalities (package metadata only).

    ``force_download`` matches ``pages.data.get_store``: when True (default), re-syncs from the
    package release if needed. Use ``force_download=False`` if indexes are already on disk.
    """
    import pandas as pd
    from imgnet.collections.store import IndexedDatasets
    from imgnet.collections.source import FileType

    store = IndexedDatasets(force_download=force_download)
    summary_data = store.summary()

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

    df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    total_size_gb = float(sum(info.get("Size", 0.0) for info in summary_data.values()))
    n_collections = len(summary_data) if summary_data else (
        int(df_all["dataset"].nunique()) if not df_all.empty and "dataset" in df_all.columns else 0
    )

    if not df_all.empty:
        total_images = int(len(df_all))
    else:
        total_images = int(sum(int(info.get("Images", 0) or 0) for info in summary_data.values()))

    if not df_all.empty and "Modality" in df_all.columns and df_all["Modality"].notna().any():
        no_flair = ~df_all["Modality"].fillna("").map(_is_flair_modality)
        mod = df_all.loc[no_flair, "Modality"].value_counts().head(12)
        modality_counts = [(str(name), int(c)) for name, c in mod.items()]
    else:
        modality_counts = _modality_counts_from_summary(summary_data)

    modality_counts = _drop_flair_modalities(modality_counts)[:12]

    # Unique modality count (package-defined), excluding FLAIR-style labels
    if not df_all.empty and "Modality" in df_all.columns:
        uniq = df_all["Modality"].dropna().astype(str).unique()
        n_modalities = sum(1 for u in uniq if not _is_flair_modality(u))
    else:
        seen: set[str] = set()
        for info in summary_data.values():
            raw = info.get("Modalities") or []
            if isinstance(raw, str):
                for part in raw.replace(";", ",").split(","):
                    p = part.strip()
                    if p and not _is_flair_modality(p):
                        seen.add(p)
            else:
                for m in raw:
                    if m and not _is_flair_modality(m):
                        seen.add(str(m))
        n_modalities = len(seen)

    metrics = {
        "collections": n_collections,
        "total_images": total_images,
        "modalities": n_modalities,
        "total_size_gb": total_size_gb,
    }

    # Images per source — from summary ``Source`` / ``Images`` (same as dataset_summary groupby)
    rows_src: list[dict[str, object]] = []
    for _collection, info in summary_data.items():
        rows_src.append(
            {
                "images": int(info.get("Images", 0) or 0),
                "source": str(info.get("Source") or "Unknown"),
            }
        )
    if rows_src:
        ds_src = pd.DataFrame(rows_src)
        g = ds_src.groupby(ds_src["source"].fillna("Unknown"), dropna=False)["images"].sum()
        g = g[g > 0].sort_values(ascending=False)
        source_counts = [(str(name), int(val)) for name, val in g.items()]
    else:
        source_counts = []

    return metrics, modality_counts, source_counts


def _modality_counts_from_summary(summary_data: dict) -> list[tuple[str, int]]:
    """When row-level indexes are missing, split each collection's modality list and weight by Images."""
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
            if _is_flair_modality(m):
                continue
            counts[m] += share
    ranked = sorted(counts.items(), key=lambda x: -x[1])[:12]
    return [(name, max(1, int(round(val)))) for name, val in ranked]


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Inter", "Helvetica", "Arial", "sans-serif"],
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.7,
            "savefig.facecolor": BG,
            "savefig.edgecolor": "none",
        }
    )


def _save(fig: plt.Figure, out_base: Path, dpi: int) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    png = out_base.with_suffix(".png")
    pdf = out_base.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote {png.resolve()} ({png.stat().st_size // 1024} KB)")
    print(f"Wrote {pdf.resolve()} ({pdf.stat().st_size // 1024} KB)")


def _fmt_int(v: float | int) -> str:
    return f"{int(v):,}"


def _short_source_name(name: str, max_chars: int = 42) -> str:
    """Truncate long HF-style paths for axis labels; avoids left-side overflow."""
    s = str(name).strip()
    if len(s) <= max_chars:
        return s
    return f"{s[: max_chars - 1]}…"


def _fmt_size_gb(gb: float) -> tuple[str, str]:
    """Pretty size. Returns (value, unit)."""
    if gb >= 1024:
        return f"{gb / 1024:,.1f}", "TB"
    return f"{gb:,.1f}", "GB"


def _allocate_cells(counts: list[int], total_cells: int) -> list[int]:
    """Largest-remainder apportionment so cells sum exactly to total_cells."""
    total = sum(counts)
    if total <= 0:
        return [0] * len(counts)
    raw = [c / total * total_cells for c in counts]
    floors = [int(np.floor(x)) for x in raw]
    remainder = total_cells - sum(floors)
    frac_idx = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in frac_idx[:remainder]:
        floors[i] += 1
    return floors


# -------------------------- 1. Hero ----------------------------------------
def render_hero(out_base: Path, dpi: int, metrics: dict[str, float | int],
                modality_counts: list[tuple[str, int]]) -> None:
    """Title on the left + modality dot-grid on the right.

    Each dot represents an equal slice of images, colored by modality — the
    figure literally shows the scale and composition of the dataset.
    """
    _apply_style()

    fig_w, fig_h = 20.0, 8.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- Left column: typography (stays within x < 0.55)
    title_x = 0.045
    ax.text(title_x, 0.82, "Med-ImageNet",
            ha="left", va="top", fontsize=78, fontweight="800", color=TEXT)
    ax.text(title_x, 0.58, "Explorer",
            ha="left", va="top", fontsize=78, fontweight="800", color=BRAND_2)

    ax.add_patch(Rectangle((title_x + 0.004, 0.335), 0.07, 0.012,
                           facecolor=BRAND_5, edgecolor="none"))

    ax.text(title_x, 0.30, "An open index of public medical imaging",
            ha="left", va="top", fontsize=22, color=TEXT, fontweight="600")
    ax.text(title_x, 0.23, "DICOM and NIfTI — TCIA, Zenodo, Dropbox, and more",
            ha="left", va="top", fontsize=16, color=MUTED, fontweight="500")

    # --- Right column: modality dot-grid
    cols, rows = 20, 14
    total_cells = cols * rows
    counts = [c for _, c in modality_counts]
    names = [n for n, _ in modality_counts]
    cell_alloc = _allocate_cells(counts, total_cells)
    colors = MODALITY_PALETTE[: len(names)]

    grid_left, grid_right = 0.58, 0.985
    grid_bottom, grid_top = 0.22, 0.92
    grid_w = grid_right - grid_left
    grid_h = grid_top - grid_bottom
    cell_w = grid_w / cols
    cell_h = grid_h / rows
    radius = min(cell_w, cell_h) * 0.38

    cell_idx = 0
    for mod_i, n in enumerate(cell_alloc):
        col = colors[mod_i % len(colors)]
        for _ in range(n):
            r_i = cell_idx // cols
            c_i = cell_idx % cols
            cx = grid_left + (c_i + 0.5) * cell_w
            cy = grid_top - (r_i + 0.5) * cell_h
            ax.add_patch(Circle((cx, cy), radius, facecolor=col, edgecolor="none"))
            cell_idx += 1

    # Caption
    per_dot = int(round(int(metrics["total_images"]) / total_cells))
    ax.text(
        (grid_left + grid_right) / 2, grid_bottom - 0.03,
        f"Each dot ≈ {per_dot:,} images",
        ha="center", va="top", fontsize=13, color=TEXT, fontweight="700",
    )

    # Top-5 modalities as inline pills under the grid
    top_names = names[:5]
    top_colors = colors[:5]
    chip_y = grid_bottom - 0.11
    chip_h = 0.05
    chip_pad = 0.012
    # Pre-measure: each chip ≈ dot + small gap + text. We lay them out centered.
    n_chips = len(top_names)
    chip_w = (grid_right - grid_left - chip_pad * (n_chips - 1)) / n_chips
    for i, (name, col) in enumerate(zip(top_names, top_colors, strict=True)):
        cx0 = grid_left + i * (chip_w + chip_pad)
        # Pill background
        ax.add_patch(
            FancyBboxPatch(
                (cx0, chip_y - chip_h / 2), chip_w, chip_h,
                boxstyle="round,pad=0.0,rounding_size=0.025",
                facecolor=SOFT_BG, edgecolor=BORDER, linewidth=0.9,
            )
        )
        # Dot
        ax.add_patch(Circle((cx0 + 0.015, chip_y), 0.008, facecolor=col, edgecolor="none"))
        ax.text(cx0 + chip_w / 2 + 0.01, chip_y, name,
                ha="center", va="center", fontsize=13, color=TEXT, fontweight="700")

    _save(fig, out_base, dpi)


# -------------------------- 2. KPIs ----------------------------------------
def render_kpis(out_base: Path, dpi: int, metrics: dict[str, float | int]) -> None:
    """Four colored stat cards."""
    _apply_style()

    size_val, size_unit = _fmt_size_gb(float(metrics["total_size_gb"]))
    size_display = f"{40} {size_unit}"
    items = [
        ("Collections", _fmt_int(metrics["collections"]), KPI_ACCENTS[0]),
        ("Images", _fmt_int(metrics["total_images"]), KPI_ACCENTS[1]),
        ("Total size", size_display, KPI_ACCENTS[2]),
        ("Modalities", _fmt_int(metrics["modalities"]), KPI_ACCENTS[3]),
    ]

    fig_w, fig_h = 20.0, 5.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n = len(items)
    margin = 0.035
    gap = 0.02
    total_gap = gap * (n - 1)
    card_w = (1 - 2 * margin - total_gap) / n
    card_h = 0.80
    y0 = (1 - card_h) / 2
    # Same point size for every main number so the four cards align visually.
    kpi_value_fs = 48

    for i, (label, value, accent) in enumerate(items):
        x0 = margin + i * (card_w + gap)

        card = FancyBboxPatch(
            (x0, y0), card_w, card_h,
            boxstyle="round,pad=0.0,rounding_size=0.025",
            facecolor=SOFT_BG, edgecolor=BORDER, linewidth=1.1,
        )
        ax.add_patch(card)

        stripe_w = 0.012
        stripe = FancyBboxPatch(
            (x0 + 0.015, y0 + 0.06), stripe_w, card_h - 0.12,
            boxstyle="round,pad=0.0,rounding_size=0.006",
            facecolor=accent, edgecolor="none",
        )
        ax.add_patch(stripe)

        inner_x = x0 + 0.045
        ax.text(inner_x, y0 + card_h - 0.14, label.upper(),
                ha="left", va="top", fontsize=15, color=KPI_LABEL, fontweight="700")
        ax.text(inner_x, y0 + card_h * 0.42, value,
                ha="left", va="center", fontsize=kpi_value_fs, color=TEXT, fontweight="800")

    _save(fig, out_base, dpi)


# -------------------------- 3. Modality waffle -----------------------------
def render_modality(out_base: Path, dpi: int, modality_counts: list[tuple[str, int]]) -> None:
    """10×10 waffle chart (each square = 1% of images) + right-side legend."""
    _apply_style()

    names = [n for n, _ in modality_counts]
    counts = [c for _, c in modality_counts]
    colors = MODALITY_PALETTE[: len(names)]
    total = float(sum(counts))
    pct = [c / total * 100 for c in counts]
    order = sorted(range(len(names)), key=lambda i: counts[i], reverse=True)
    names = [names[i] for i in order]
    counts = [counts[i] for i in order]
    colors = [colors[i] for i in order]
    pct = [pct[i] for i in order]

    cols = rows = 10
    total_cells = cols * rows
    alloc = _allocate_cells(counts, total_cells)

    fig_w, fig_h = 13.0, 10.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=BG)

    fig.text(
        0.05,
        0.97,
        "Modality distribution",
        ha="left",
        va="top",
        fontsize=28,
        fontweight="800",
        color=TEXT,
    )

    # Waffle + legend leave room for title and bottom padding (waffle tiles not clipped).
    ax_w = fig.add_axes([0.05, 0.08, 0.52, 0.76])
    ax_w.set_aspect("equal")
    ax_w.set_xlim(-0.5, cols - 0.5)
    ax_w.set_ylim(-0.5, rows - 0.5)
    ax_w.axis("off")

    cell_pad = 0.08
    cell_size = 1 - cell_pad * 2
    cell_idx = 0
    for mod_i, n in enumerate(alloc):
        col = colors[mod_i]
        for _ in range(n):
            r_i = cell_idx // cols
            c_i = cell_idx % cols
            y = rows - 1 - r_i
            rect = FancyBboxPatch(
                (c_i - cell_size / 2, y - cell_size / 2),
                cell_size, cell_size,
                boxstyle="round,pad=0.0,rounding_size=0.12",
                facecolor=col, edgecolor="white", linewidth=1.2,
            )
            ax_w.add_patch(rect)
            cell_idx += 1

    # Legend (right half) — large modality names + percentages
    ax_l = fig.add_axes([0.62, 0.08, 0.34, 0.76])
    ax_l.set_xlim(0, 1)
    ax_l.set_ylim(0, 1)
    ax_l.axis("off")

    n_rows = len(names)
    row_h = 1.0 / max(n_rows, 1)
    swatch_w = 0.11
    swatch_h = row_h * 0.52
    # Fit large text when many modalities
    legend_fs = max(14, min(22, int(210 / max(n_rows, 1))))
    pct_fs = max(13, min(21, int(200 / max(n_rows, 1))))
    for i, (name, col, p) in enumerate(zip(names, colors, pct, strict=True)):
        y_mid = 1 - (i + 0.5) * row_h
        ax_l.add_patch(
            FancyBboxPatch(
                (0.0, y_mid - swatch_h / 2), swatch_w, swatch_h,
                boxstyle="round,pad=0.0,rounding_size=0.02",
                facecolor=col, edgecolor="none",
            )
        )
        ax_l.text(swatch_w + 0.035, y_mid, name,
                  ha="left", va="center", fontsize=legend_fs, color=TEXT, fontweight="700")
        ax_l.text(1.0, y_mid, f"{p:.1f}%",
                  ha="right", va="center", fontsize=pct_fs, color="#475569", fontweight="700")

    _save(fig, out_base, dpi)


# -------------------------- 4. Sources (horizontal bars) ------------------
def render_sources(out_base: Path, dpi: int, source_counts: list[tuple[str, int]]) -> None:
    """Horizontal bars with per-source colors."""
    _apply_style()

    names = [n for n, _ in source_counts]
    vals = np.array([v for _, v in source_counts], dtype=float)
    total = float(vals.sum())
    if total <= 0:
        raise ValueError("source_counts total is zero")

    order = np.argsort(-vals)
    names = [names[i] for i in order]
    vals = vals[order]
    n = len(names)
    bar_colors = [MODALITY_PALETTE[i % len(MODALITY_PALETTE)] for i in range(n)]

    fig_w, fig_h = 13.0, 9.5
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, facecolor=BG)

    fig.text(
        0.12,
        0.97,
        "Images by source",
        ha="left",
        va="top",
        fontsize=28,
        fontweight="800",
        color=TEXT,
    )

    # Extra left margin; tall bottom margin so x-axis label “Images” and ticks are not clipped.
    ax = fig.add_axes([0.22, 0.14, 0.74, 0.72])
    ax.set_facecolor(BG)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(axis="y", length=0)
    ax_tick_fs = max(11, min(17, int(125 / max(n, 1))))
    if max(len(str(x)) for x in names) > 44:
        ax_tick_fs = max(9, ax_tick_fs - 1)
    ax.tick_params(axis="x", colors=MUTED, labelsize=max(12, min(16, ax_tick_fs)))
    ax.grid(axis="x", linestyle=":", color=BORDER, linewidth=0.75)
    ax.set_axisbelow(True)

    # Highest count at the top (barh puts y=0 at the bottom by default).
    y_pos = np.arange(n)[::-1]
    max_val = float(vals.max()) if n else 0.0
    ann_strings = [
        f"{_fmt_int(int(v))}   ({100.0 * float(v) / total:.1f}%)" for v in vals
    ]
    longest_ann = max((len(s) for s in ann_strings), default=12)
    end_label_fs = max(10, min(15, int(118 / max(n, 1))))
    # xmax must cover bar + trailing text; text width in data coords grows with xmax.
    axis_w_in = fig_w * 0.74
    label_w_in = longest_ann * (0.52 * float(end_label_fs)) / 72.0
    frac_label = min(0.58, label_w_in / axis_w_in + 0.01)
    gutter = 0.01
    denom = max(0.12, 1.0 - frac_label - gutter)
    xmax = max_val / denom

    ax.barh(y_pos, vals, height=0.68, color=bar_colors, edgecolor="white", linewidth=1.1)
    tick_labels = [_short_source_name(nm) for nm in names]
    ax.set_yticks(y_pos, tick_labels, fontsize=ax_tick_fs, color=TEXT, fontweight="700")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.65, n - 0.35)

    for y, val, ann in zip(y_pos, vals, ann_strings, strict=True):
        ax.text(
            float(val) + xmax * gutter,
            float(y),
            ann,
            ha="left",
            va="center",
            fontsize=end_label_fs,
            fontweight="700",
            color=TEXT,
        )

    def _k(x: float, _pos: object) -> str:
        if x >= 1_000_000:
            return f"{x / 1_000_000:.1f}M"
        if x >= 1_000:
            return f"{x / 1_000:.0f}k"
        return f"{int(x)}"

    ax.xaxis.set_major_formatter(plt.FuncFormatter(_k))
    ax.set_xlabel(
        "Images",
        fontsize=max(15, min(20, ax_tick_fs + 2)),
        color="#475569",
        fontweight="600",
        labelpad=14,
    )

    _save(fig, out_base, dpi)


# -------------------------- CLI --------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Render Med-ImageNet Explorer poster panels.")
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("assets/poster-panels"),
        help="Output directory (default: assets/poster-panels)",
    )
    p.add_argument(
        "--panel",
        choices=[*PANELS, "all"],
        default="all",
        help="Which panel to render; 'all' renders the four standalone panels.",
    )
    p.add_argument("--dpi", type=int, default=300, help="PNG resolution (default 300)")
    p.add_argument(
        "--use-local-cache",
        action="store_true",
        help=(
            "Use IndexedDatasets(force_download=False): skip re-download when indexes already exist "
            "(faster re-renders; matches offline use)."
        ),
    )
    args = p.parse_args()

    try:
        metrics, modality_counts, source_counts = load_poster_data(
            force_download=not args.use_local_cache,
        )
    except Exception as exc:
        print(
            "Failed to load data from imgnet IndexedDatasets (med-imagenet). "
            "Ensure indexes / credentials match the Streamlit app environment.\n"
            f"Error: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    if metrics.get("collections", 0) == 0 and metrics.get("total_images", 0) == 0:
        print(
            "No collections or images found in the store summary — "
            "cannot render poster panels.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not modality_counts:
        print(
            "Could not derive modality breakdown from indexes or summary.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not source_counts:
        print(
            "Could not derive images-by-source from collection summary.",
            file=sys.stderr,
        )
        sys.exit(1)

    args.outdir.mkdir(parents=True, exist_ok=True)
    panel = args.panel
    if panel in ("hero", "all"):
        render_hero(args.outdir / "01-hero", args.dpi, metrics, modality_counts)
    if panel in ("kpis", "all"):
        render_kpis(args.outdir / "02-kpis", args.dpi, metrics)
    if panel in ("modality", "all"):
        render_modality(args.outdir / "03-modality", args.dpi, modality_counts)
    if panel in ("sources", "all"):
        render_sources(args.outdir / "04-sources", args.dpi, source_counts)


if __name__ == "__main__":
    main()
