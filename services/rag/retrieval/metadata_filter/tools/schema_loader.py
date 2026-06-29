"""Filterable field extraction from a Milvus collection."""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Vector dtype substrings — fields with these are skipped
_VECTOR_DTYPES = ("FLOAT_VECTOR", "BINARY_VECTOR", "SPARSE_FLOAT_VECTOR")


def load_fields_from_collection(
    collection,
    *,
    excluded_fields: Optional[List[str]] = None,
    fetch_values: bool = False,
    value_limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Extract filterable field definitions from an injected pymilvus Collection.

    The caller creates and loads the Collection; this function never opens
    its own Milvus connection.

    Args:
        collection: pymilvus Collection object (caller creates and loads it).
        excluded_fields: Field names to omit from the result.
        fetch_values: If True, query Milvus for distinct VARCHAR values and
            attach them as ``examples``. Adds one query per VARCHAR field.
        value_limit: Max rows to scan when fetching distinct values.

    Returns:
        List of field dicts: {name, dtype, description, [examples]}
    """
    excluded = set(excluded_fields or [])
    fields: List[Dict[str, Any]] = []

    try:
        for f in collection.schema.fields:
            dtype_str = str(f.dtype).split(".")[-1]
            if any(v in dtype_str for v in _VECTOR_DTYPES):
                continue
            if f.is_primary:
                continue
            if f.name in excluded:
                continue
            fields.append({
                "name": f.name,
                "dtype": dtype_str,
                "description": f.description or f"Field: {f.name}",
            })
    except Exception as e:
        logger.error(f"Failed to read schema from collection '{collection.name}': {e}")
        return []

    if fetch_values:
        for field in fields:
            if field["dtype"] == "VARCHAR":
                try:
                    rows = collection.query(
                        expr="", output_fields=[field["name"]], limit=value_limit
                    )
                    values = sorted({r.get(field["name"]) for r in rows if r.get(field["name"])})
                    if values:
                        field["examples"] = values[:10]
                except Exception as e:
                    logger.warning(f"Could not fetch values for '{field['name']}': {e}")

    logger.info(f"Loaded {len(fields)} filterable fields from '{collection.name}'")
    return fields


def get_default_fields() -> List[Dict[str, Any]]:
    """Built-in fallback fields used when no collection is available."""
    return [
        {
            "name": "document_id",
            "dtype": "VARCHAR",
            "description": "Unique identifier of the document",
            "examples": ["uuid-1234", "uuid-5678"],
        },
        {
            "name": "source",
            "dtype": "VARCHAR",
            "description": "Source or file name/URI of the document",
            "examples": ["document.pdf", "report.docx"],
        },
        {
            "name": "page",
            "dtype": "INT64",
            "description": "Page number of the chunk",
            "examples": [1, 2, 5],
        },
        {
            "name": "ingested_at",
            "dtype": "VARCHAR",
            "description": "ISO timestamp when the document was ingested",
            "examples": ["2026-06-29T12:00:00Z"],
        },
    ]
