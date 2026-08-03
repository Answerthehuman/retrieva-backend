"""Natural-language to Milvus filter-expression extraction."""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def extract_filter(
    query: str,
    *,
    llm=None,
    fields: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """
    Extract a Milvus filter expression from a natural language query.

    Args:
        query: The user's question.
        llm: LangChain-compatible LLM with an ainvoke() method.
        fields: Filterable field definitions. Falls back to built-in defaults.

    Returns:
        Milvus filter expression string, or None.
    """
    from .tools import extract_filters_from_query, get_default_fields

    result = await extract_filters_from_query(query, fields or get_default_fields(), llm=llm)
    filter_expr = result.get("milvus_expression")
    logger.info(f"Extracted filter: {filter_expr!r}")
    return filter_expr
