from __future__ import annotations

import json

from imgnet.collections.store import IndexedDatasets
from imgnet.query import ValidQuery

_store: IndexedDatasets | None = None


def _get_store() -> IndexedDatasets:
    global _store
    if _store is None:
        _store = IndexedDatasets()
    return _store


def execute_query(query_json: str) -> dict:
    """Validate and execute a Med-ImageNet query.

    Accepts a JSON string representing a ValidQuery object, validates it
    through Pydantic, runs the query against the IndexedDatasets store,
    and returns matching series.

    Args:
        query_json: A JSON string with keys ``collections``,
            ``modalities``, and optionally ``rules`` that conform to the
            ValidQuery schema.

    Returns:
        A dict with ``status`` and either query results or an error message.
    """
    try:
        raw = json.loads(query_json)
    except json.JSONDecodeError as exc:
        return {"status": "error", "error": f"Invalid JSON: {exc}"}

    try:
        query = ValidQuery(**raw)
    except Exception as exc:
        return {"status": "error", "error": f"Validation error: {exc}"}

    try:
        store = _get_store()
        result_df = query.process(store=store)
    except Exception as exc:
        return {"status": "error", "error": f"Processing error: {exc}"}

    return {
        "status": "success",
        "total_rows": len(result_df),
        "columns": result_df.columns.tolist(),
        "preview": result_df.head(20).to_dict(orient="records"),
    }


def list_collections(body_part: str = "", modality: str = "") -> dict:
    """List available Med-ImageNet collections with summary metadata.

    Reads the collections summary from the IndexedDatasets store and
    optionally filters by body part and/or modality.

    Args:
        body_part: Filter collections that examine this body part
            (e.g. ``"LUNG"``, ``"BREAST"``).  Case-insensitive.
            Pass an empty string to skip filtering.
        modality: Filter collections that contain this imaging modality
            (e.g. ``"CT"``, ``"MR"``).  Case-insensitive.
            Pass an empty string to skip filtering.

    Returns:
        A dict mapping ``"collections"`` to a list of summary dicts.
    """
    store = _get_store()
    data = store.summary()

    bp_filter = body_part.strip().upper()
    mod_filter = modality.strip().upper()

    results = []
    for name, info in data.items():
        if bp_filter:
            body_parts_upper = [bp.upper() for bp in info.get("BodyPartsExamined", [])]
            if not any(bp_filter in bp for bp in body_parts_upper):
                continue
        if mod_filter:
            modalities_upper = [m.upper() for m in info.get("Modalities", [])]
            if mod_filter not in modalities_upper:
                continue

        results.append(
            {
                "name": name,
                "modalities": info.get("Modalities", []),
                "body_parts": info.get("BodyPartsExamined", []),
                "series_count": info.get("SeriesCount", 0),
                "size_gb": info.get("Size", 0),
            }
        )

    return {"total": len(results), "collections": results}
