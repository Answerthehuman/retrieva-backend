# metadata_filter

Extracts a Milvus filter expression from a natural language query using an LLM.

The node never opens its own Milvus or Redis connections — the caller creates
clients once and passes them in.

## What it does

1. Reads the last message from state as the query
2. Resolves filterable field definitions (from fixed list, Redis cache, or Milvus schema)
3. Calls the LLM to produce a Milvus filter expression
4. Returns the expression to state as `filters`

## Install

```toml
[tool.poetry.dependencies]
pyzo-ai-core = {git = "git@github.com:Pyzo-AI/pyzo-ai-core.git", branch = "main"}
```

## Import

```python
from pyzo_ai_core.nodes.metadata_filter import MetadataFilterNode
```

---

## Usage modes

### Mode A — fixed schema (simplest)

Pass field definitions directly. No external services needed.

```python
node = MetadataFilterNode(
    llm=get_gemini(),
    fields=[
        {"name": "file_name",    "dtype": "VARCHAR", "description": "Source file name"},
        {"name": "section_type", "dtype": "VARCHAR", "description": "Type of section",
         "examples": ["introduction", "table", "summary"]},
        {"name": "page",         "dtype": "INT64",   "description": "Page number"},
    ],
)
```

### Mode B — Milvus schema + Redis cache 

Schema is loaded from the collection once, cached in Redis, and served from
cache on every subsequent call. Pass `fetch_values=True` to also query Milvus
for distinct values per VARCHAR field and attach them as `examples` in the prompt.

```python
from pymilvus import Collection, connections
import redis

connections.connect(uri=os.environ["MILVUS_URI"])
collection = Collection("compass_docs")
collection.load()

redis_client = redis.Redis(host="localhost", port=6380, decode_responses=True)

node = MetadataFilterNode(
    llm=get_gemini(),
    collection=collection,
    redis_client=redis_client,
    redis_key="myapp:fields:compass_docs",
    fetch_values=True,      # query Milvus for example values on cache miss
    excluded_fields=["ingested_at", "chunk_index"],
)
```

After ingesting new documents, update the cache so the LLM sees fresh examples:

```python
node.cache.update("file_name", ["new_report.pdf", "contract_2026.docx"])
```

### Mode C — Milvus schema, no cache

Schema is re-read from Milvus on every call. Simpler setup, higher latency.

```python
node = MetadataFilterNode(
    llm=get_gemini(),
    collection=collection,
    fetch_values=True,
)
```

### No args (built-in defaults)

Falls back to a generic field set (project_name, document_type, file_type, source, section_type).

```python
node = MetadataFilterNode(llm=get_gemini())
```

---

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `llm` | any | LangChain-compatible LLM with `invoke()` |
| `fields` | `List[dict]` | (Mode A) Fixed field definitions |
| `collection` | `pymilvus.Collection` | (Mode B/C) Loaded collection to read schema from |
| `excluded_fields` | `List[str]` | Field names to skip when loading from Milvus |
| `fetch_values` | `bool` | Query Milvus for distinct VARCHAR values as examples (default `False`) |
| `redis_client` | `redis.Redis` | (Mode B) Caller-created Redis client |
| `redis_key` | `str` | (Mode B) Redis key for this node's field cache |

Field dict shape:
```python
{
    "name":        str,           # Milvus field name
    "dtype":       str,           # "VARCHAR", "INT64", etc.
    "description": str,           # shown to LLM
    "examples":    List[str],     # sample values shown to LLM (optional)
}
```

Both `"dtype"` and `"type"` keys are accepted.

---

## State contract

| | Fields |
|---|---|
| **Reads** | `messages` |
| **Writes** | `filters` |

```python
filters: str | None   # e.g. 'file_name like "%proposal%" AND page > 0'
```

---

## FieldCache — keeping examples fresh

`FieldCache` is the thin Redis wrapper used internally. You can access it via
`node.cache` to update example values from your ingestion pipeline:

```python
from pyzo_ai_core.nodes.metadata_filter import FieldCache

# Standalone usage (e.g. in ingestion pipeline)
cache = FieldCache(redis_client, "myapp:fields:compass_docs")
cache.update("file_name", ["2026_annual_report.pdf"])
cache.clear()   # force reload from Milvus on next call
```

---

## Loading schema without the node

```python
from pyzo_ai_core.nodes.metadata_filter import load_fields_from_collection

fields = load_fields_from_collection(
    collection,
    excluded_fields=["id", "ingested_at"],
    fetch_values=True,
)
```

---

## Standalone filter extraction

```python
from pyzo_ai_core.nodes.metadata_filter import extract_filter, extract_filters_from_query

# Returns just the expression string
expr = extract_filter("show me proposals from last year", llm=llm, fields=fields)

# Returns full dict with confidence + cleaned_query
result = extract_filters_from_query("show me proposals", fields, llm=llm)
# {"milvus_expression": '...', "cleaned_query": '...', "confidence": 0.9}
```

---

## Refreshing the Milvus syntax docs

The LLM prompt includes Milvus filter syntax documentation bundled with the package
(`tools/milvus_filter_docs.md`). It is read once per process and cached in memory —
no network call at runtime. To refresh it from the official Milvus docs:

```bash
python -m pyzo_ai_core.nodes.metadata_filter.tools.fetch_milvus_docs
```

Commit the updated `milvus_filter_docs.md` file afterwards.
