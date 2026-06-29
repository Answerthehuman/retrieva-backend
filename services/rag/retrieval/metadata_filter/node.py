"""LangGraph node for metadata filter extraction."""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def extract_filter(
    query: str,
    *,
    llm=None,
    fields: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """
    Standalone filter extraction — no LangGraph state dependency.

    Args:
        query: The user's question.
        llm: LangChain-compatible LLM with invoke() method.
        fields: Filterable field definitions. Falls back to built-in defaults.

    Returns:
        Milvus filter expression string, or None.
    """
    from .tools import extract_filters_from_query, get_default_fields

    result = extract_filters_from_query(query, fields or get_default_fields(), llm=llm)
    filter_expr = result.get("milvus_expression")
    logger.info(f"Extracted filter: {filter_expr!r}")
    return filter_expr


async def metadata_filter_node(
    state: Dict[str, Any],
    *,
    llm=None,
    fields: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    LangGraph node — extracts a Milvus filter expression from the last message.

    Input state:
        messages (List[BaseMessage]): conversation history

    Output state:
        filters (str | None): Milvus filter expression

    Args:
        llm: LangChain-compatible LLM with invoke() method
        fields: Filterable field definitions. Falls back to built-in defaults.
    """
    messages = state.get("messages", [])
    if not messages:
        return {"filters": None}

    query = messages[-1].content
    filters = extract_filter(query, llm=llm, fields=fields)
    return {"filters": filters}


class MetadataFilterNode:
    """
    Class wrapper for metadata_filter_node — inject dependencies once, reuse as a node.

    Three modes depending on what you pass:

    **Mode A — fixed schema** (no external services):
        Pass ``fields`` directly. Used as-is on every call.

    **Mode B — Milvus schema + Redis cache** (production):
        Pass ``collection`` + ``redis_client`` + ``redis_key``.
        Schema is loaded from the collection on first call and cached in Redis.
        Subsequent calls are served from cache. Call ``node.cache.update(field_name,
        new_values)`` from your ingestion pipeline to keep examples fresh.

    **Mode C — Milvus schema, no cache**:
        Pass only ``collection``. Schema is re-read from Milvus on every call.

    Args:
        llm: LangChain-compatible LLM with invoke() method.
        fields: (Mode A) Explicit field list. When given, collection/redis args
            are ignored.
        collection: (Mode B/C) pymilvus Collection object. Caller creates and
            loads it; the node never opens its own Milvus connection.
        excluded_fields: Field names to omit when loading from Milvus.
        fetch_values: If True, query Milvus for distinct VARCHAR values and
            attach them as ``examples`` (adds one query per VARCHAR field on
            cache miss).
        redis_client: (Mode B) redis.Redis-compatible client. Caller creates once.
        redis_key: (Mode B) Redis key for this node's field cache, e.g.
            ``"myapp:fields:compass_docs"``. Required when redis_client is given.
    """

    def __init__(
        self,
        *,
        llm=None,
        fields: Optional[List[Dict[str, Any]]] = None,
        collection=None,
        excluded_fields: Optional[List[str]] = None,
        fetch_values: bool = False,
        redis_client=None,
        redis_key: Optional[str] = None,
    ):
        self.llm = llm
        self._fields = fields
        self._collection = collection
        self._excluded_fields = excluded_fields
        self._fetch_values = fetch_values

        self._cache = None
        if redis_client is not None and redis_key is not None:
            from .tools.field_cache import FieldCache
            self._cache = FieldCache(redis_client, redis_key)
        elif redis_client is not None:
            logger.warning(
                "MetadataFilterNode: redis_client given without redis_key — "
                "Redis caching disabled."
            )

    @property
    def cache(self):
        """The FieldCache instance, or None if Redis is not configured."""
        return self._cache

    def _resolve_fields(self) -> List[Dict[str, Any]]:
        # Mode A: caller-supplied fixed schema
        if self._fields is not None:
            return self._fields

        # Mode B/C: dynamic from Milvus collection
        if self._collection is not None:
            if self._cache is not None:
                cached = self._cache.get()
                if cached is not None:
                    logger.debug("MetadataFilterNode: fields served from Redis cache")
                    return cached

            from .tools.schema_loader import load_fields_from_collection
            fields = load_fields_from_collection(
                self._collection,
                excluded_fields=self._excluded_fields,
                fetch_values=self._fetch_values,
            )

            if self._cache is not None and fields:
                self._cache.set(fields)

            return fields

        # Fallback: built-in defaults
        from .tools.schema_loader import get_default_fields
        return get_default_fields()

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return await metadata_filter_node(state, llm=self.llm, fields=self._resolve_fields())
