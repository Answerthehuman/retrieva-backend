# metadata_filter

Extracts a Milvus filter expression from a natural language query using an LLM.

Internal Retrieva module — no separate install step, import directly from
`services.rag.retrieval.metadata_filter`.

## What it does

1. Takes the user's query (and a list of filterable field definitions)
2. Calls the LLM to produce a Milvus filter expression
3. Returns the expression, or `None` if no filter applies

It's used by `agents/tools.py::search_knowledge_base` to automatically derive
metadata filters from the user's question before hitting Milvus, merged with
any caller-supplied filter override.

## Usage

### Fixed field list (simplest)

```python
from services.rag.retrieval.metadata_filter import extract_filter

expr = await extract_filter(
    "show me proposals from last year",
    llm=get_gemini(),
    fields=[
        {"name": "file_name", "dtype": "VARCHAR", "description": "Source file name"},
        {
            "name": "section_type",
            "dtype": "VARCHAR",
            "description": "Type of section",
            "examples": ["introduction", "table", "summary"],
        },
        {"name": "page", "dtype": "INT64", "description": "Page number"},
    ],
)
```

### Built-in defaults

Falls back to a generic field set matching Retrieva's actual Milvus schema
(`document_id`, `source`, `page`, `ingested_at`) when no `fields` are given:

```python
expr = await extract_filter("show me proposals", llm=get_gemini())
```

### Full result (expression + cleaned query + confidence)

```python
from services.rag.retrieval.metadata_filter import extract_filters_from_query, get_default_fields

result = await extract_filters_from_query("show me proposals", get_default_fields(), llm=llm)
# {"milvus_expression": '...', "cleaned_query": '...', "confidence": 0.9}
```

### Validating an expression before use

```python
from services.rag.retrieval.metadata_filter import validate_expression

if validate_expression(expr):
    ...  # safe to pass to Milvus
```

---

## Available utilities

| Function | Description |
|---|---|
| `extract_filter(query, *, llm, fields=None)` | Async. Returns just the expression string, or `None`. |
| `extract_filters_from_query(query, available_fields, *, llm)` | Async. Returns the full `{milvus_expression, cleaned_query, confidence}` dict. |
| `get_default_fields()` | Built-in fallback fields matching Retrieva's Milvus schema. |
| `load_fields_from_collection(collection, ...)` | Reads filterable fields from a live pymilvus `Collection` schema. Not currently wired into the live query path — available for dynamic-schema use cases. |
| `build_milvus_expression(filters)` / `validate_expression(expr)` | Build/validate a Milvus expression from a structured filter-dict list. |
| `FieldCache` | Thin Redis wrapper for caching field definitions. Not currently wired into the live query path. |

Field dict shape:
```python
{
    "name": str,  # Milvus field name
    "dtype": str,  # "VARCHAR", "INT64", etc.
    "description": str,  # shown to LLM
    "examples": List[str],  # sample values shown to LLM (optional)
}
```

---

## Refreshing the Milvus syntax docs

The LLM prompt includes Milvus filter syntax documentation bundled with this
module (`tools/milvus_filter_docs.md`). It's read once per process and cached
in memory — no network call at runtime. To refresh it from the official Milvus
docs:

```bash
python -m services.rag.retrieval.metadata_filter.tools.fetch_milvus_docs
```

Commit the updated `milvus_filter_docs.md` file afterwards.
