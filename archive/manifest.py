"""Manifest page -- visual ValidQuery builder for med-imagenet."""

from __future__ import annotations

import io
import json

import streamlit as st

from imgnet.query import ValidQuery
from imgnet.query.models import Rule
from pages.data import apply_page_filters, get_store, load_all_indexes

COMPARISONS = ["==", "!=", ">", "<", ">=", "<="]

SAMPLE_RULES: list[dict] = [
    {"modality": "CT", "tag": "Modality", "comparison": "==", "value": "CT"},
    {"modality": "RTSTRUCT", "tag": "ROINames", "comparison": "==", "value": "lung.*"},
]


def _init_rules() -> None:
    if "manifest_rules" not in st.session_state:
        st.session_state.manifest_rules: list[dict] = []


def _load_sample_rules() -> None:
    st.session_state.manifest_rules = [r.copy() for r in SAMPLE_RULES]


def _add_rule() -> None:
    st.session_state.manifest_rules.append(
        {"modality": "", "tag": "", "comparison": "==", "value": ""}
    )


def _remove_rule(idx: int) -> None:
    st.session_state.manifest_rules.pop(idx)


def _clear_rules() -> None:
    st.session_state.manifest_rules = []


def _build_rules_dict(
    rule_rows: list[dict],
) -> dict[str, Rule | list[Rule]] | None:
    """Convert the UI rule rows into the dict[modality, Rule|list[Rule]] format."""
    if not rule_rows:
        return None
    grouped: dict[str, list[Rule]] = {}
    for row in rule_rows:
        mod = row["modality"].strip()
        tag = row["tag"].strip()
        val = row["value"].strip()
        if not (mod and tag and val):
            continue
        rule = Rule(tag=tag, value=val, comparison=row["comparison"])
        grouped.setdefault(mod, []).append(rule)
    if not grouped:
        return None
    return {
        mod: rules[0] if len(rules) == 1 else rules
        for mod, rules in grouped.items()
    }


def run() -> None:
    st.header("Query builder")
    st.caption(
        "Select collections and modalities, add optional filter rules, "
        "then run the query to build a downloadable manifest."
    )

    store = get_store()

    # ── Filters ──────────────────────────────────────────────────────────
    prefill = st.session_state.pop("manifest_prefill_collection", None)
    prefill_list = [prefill] if prefill else None

    raw_df = load_all_indexes()
    filter_result = apply_page_filters(raw_df, default_datasets=prefill_list)
    filtered_df = filter_result.df
    selected_formats = filter_result.selected_formats

    selected_collections = sorted(filtered_df["dataset"].unique().tolist())
    all_collections = sorted(raw_df["dataset"].unique().tolist())
    collections_param = (
        selected_collections
        if set(selected_collections) != set(all_collections)
        else "all"
    )

    if "Modality" in filtered_df.columns:
        selected_modalities = sorted(
            filtered_df["Modality"].dropna().unique().tolist()
        )
        all_modalities = sorted(raw_df["Modality"].dropna().unique().tolist())
        modalities_param = (
            selected_modalities
            if set(selected_modalities) != set(all_modalities)
            else "all"
        )
    else:
        selected_modalities = []
        modalities_param = "all"

    formats = sorted(filtered_df["format"].unique().tolist()) if "format" in filtered_df.columns else []
    fmt_label = ", ".join(formats) if formats else "all"

    st.info(
        f"**{len(selected_collections)}** collection(s)  ·  "
        f"**{fmt_label}** format(s)  ·  "
        f"**{len(selected_modalities)}** modality(ies) selected"
    )

    # ── Rules ────────────────────────────────────────────────────────────
    _init_rules()
    st.subheader("Filter rules")
    with st.expander(":material/help: How do rules work?", expanded=False):
        st.markdown("""
Rules filter series **per modality** using DICOM (or index) tag values.

- **Tag** — DICOM tag or index column name (e.g. `Modality`, `ROINames`, `SeriesInstanceUID`, `BodyPartExamined`).
- **Value** — For `==` and `!=`, value supports **regex** (e.g. `lung.*`, `1.2.3.4.*`). For `>`, `<`, `>=`, `<=`, value must be numeric.
- **Operators** — `==` / `!=` (string or regex); `>`, `<`, `>=`, `<=` (numeric only).

| Modality | Tag            | Op  | Value   | Meaning                    |
|----------|----------------|-----|---------|----------------------------|
| CT       | Modality       | ==  | CT      | Keep CT series             |
| RTSTRUCT | ROINames       | ==  | lung.\\* | ROI names matching regex   |
| CT       | SeriesInstanceUID | == | 1.2.3.4 | Exact series UID          |

Leave rules empty to include all series for the selected modalities and collections.
        """)

    _col_widths = [2, 3, 1.5, 3, 0.8]

    col_rule_title, col_sample, col_clear = st.columns([2, 1, 1])
    with col_sample:
        st.button(
            "Load sample rules",
            on_click=_load_sample_rules,
            help="Insert example rules (Modality, ROINames)",
        )
    with col_clear:
        st.button(
            "Clear all",
            on_click=_clear_rules,
            disabled=len(st.session_state.manifest_rules) == 0,
        )

    mod_options = selected_modalities or ["CT", "MR", "RTSTRUCT", "PT", "SEG"]

    _skip_cols = {"dataset", "format"}
    tag_options = sorted(
        c for c in filtered_df.columns if c not in _skip_cols and filtered_df[c].notna().any()
    )

    if st.session_state.manifest_rules:
        hdr = st.columns(_col_widths)
        hdr[0].markdown("**Modality**")
        hdr[1].markdown("**Tag**")
        hdr[2].markdown("**Op**")
        hdr[3].markdown("**Value**")

    for i, row in enumerate(st.session_state.manifest_rules):
        cols = st.columns(_col_widths)
        with cols[0]:
            row["modality"] = st.selectbox(
                "Modality",
                mod_options,
                index=(
                    mod_options.index(row["modality"])
                    if row["modality"] in mod_options
                    else 0
                ),
                key=f"rule_mod_{i}",
                label_visibility="collapsed",
            )
        with cols[1]:
            cur_tag = row["tag"]
            opts = tag_options if cur_tag in tag_options or not cur_tag else [cur_tag] + tag_options
            row["tag"] = st.selectbox(
                "Tag",
                opts,
                index=opts.index(cur_tag) if cur_tag in opts else 0,
                key=f"rule_tag_{i}",
                label_visibility="collapsed",
            )
        with cols[2]:
            row["comparison"] = st.selectbox(
                "Op",
                COMPARISONS,
                index=COMPARISONS.index(row["comparison"]),
                key=f"rule_cmp_{i}",
                label_visibility="collapsed",
            )
        with cols[3]:
            row["value"] = st.text_input(
                "Value",
                value=row["value"],
                placeholder="e.g. lung.* or 1",
                key=f"rule_val_{i}",
                label_visibility="collapsed",
            )
        with cols[4]:
            st.button(
                ":material/close:",
                key=f"rule_del_{i}",
                on_click=_remove_rule,
                args=(i,),
                width='stretch',
            )

        # Inline validation
        missing = []
        if not row["modality"].strip():
            missing.append("modality")
        if not row["tag"].strip():
            missing.append("tag")
        if not row["value"].strip():
            missing.append("value")
        if missing:
            st.caption(f":material/warning: Missing {', '.join(missing)} — this rule will be skipped.")

    st.button(":material/add: Add rule", on_click=_add_rule)

    st.divider()

    # ── Build query dict ─────────────────────────────────────────────────
    fmt_map = {"DICOM": "dicom", "NIfTI": "nifti"}
    if len(selected_formats) == 1:
        file_type_param = fmt_map.get(selected_formats[0], "all")
    else:
        file_type_param = "all"

    query_dict: dict = {
        "collections": collections_param,
        "modalities": modalities_param,
        "file_type": file_type_param,
    }
    rules = _build_rules_dict(st.session_state.manifest_rules)
    if rules is not None:
        query_dict["rules"] = {
            mod: [r.model_dump() for r in (v if isinstance(v, list) else [v])]
            for mod, v in rules.items()
        }

    # ── Live query preview ──────────────────────────────────────────────
    preview_query = None
    try:
        preview_query = ValidQuery(**query_dict)
    except Exception:
        pass

    with st.expander("Query preview", expanded=bool(rules or preview_query)):
        tab_json, tab_token = st.tabs(["ValidQuery JSON", "Shareable token"])
        with tab_json:
            st.code(json.dumps(query_dict, indent=2), language="json")
        with tab_token:
            if preview_query:
                st.code(preview_query.to_token(), language="text")
                st.caption("Share or reuse this token to reproduce the exact same query.")
            else:
                st.info("Token will appear once the query configuration is valid.")

    # ── Execute ──────────────────────────────────────────────────────────
    if st.button("Run query", type="primary", width='stretch'):
        if preview_query is None:
            st.error("**Validation error:** check your filters and rules above.")
            return

        with st.spinner("Running query..."):
            try:
                result_df = preview_query.process(store=store)
            except Exception as exc:
                st.error(f"**Query error:** {exc}")
                return

        st.session_state.manifest_result = result_df
        st.session_state.manifest_query = preview_query
        st.toast(f"Query returned {len(result_df):,} rows.", icon=":material/check_circle:")

    # ── Results ──────────────────────────────────────────────────────────
    if "manifest_result" not in st.session_state:
        st.info("Configure filters and rules above, then click **Run query** to generate a manifest.")
        return

    result_df = st.session_state.manifest_result

    st.divider()
    st.subheader("Results")

    r1, r2, r3 = st.columns([1, 1, 1])
    r1.metric("Rows", f"{len(result_df):,}")
    r2.metric("Columns", f"{len(result_df.columns):,}")

    buf = io.BytesIO()
    result_df.to_csv(buf, index=False)
    with r3:
        st.download_button(
            ":material/download: Download CSV",
            data=buf.getvalue(),
            file_name="manifest.csv",
            mime="text/csv",
            width='stretch',
        )

    st.dataframe(
        result_df.head(500),
        width='stretch',
        hide_index=True,
        height=500,
    )
    if len(result_df) > 500:
        st.caption(
            f"Showing first 500 of {len(result_df):,} rows. "
            "Download to get the full file."
        )


run()
