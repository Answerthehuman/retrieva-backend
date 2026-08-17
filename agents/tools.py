"""Tools available to the Retrieva agent."""

import logging
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from agents.state import AgentState

logger = logging.getLogger(__name__)


def build_tools(*, retriever, reranker, llm, settings) -> list:
    """Build the tool list bound into the agent graph. Called once at graph-build time."""
    from core.utils.bm25 import build_bm25_loader
    from services.rag.retrieval.metadata_filter import (
        extract_filter,
        get_default_fields,
        validate_expression,
    )
    from services.rag.retrieval.orchestrator import retrieve

    bm25_loader = build_bm25_loader(settings)

    @tool(response_format="content_and_artifact")
    async def search_knowledge_base(
        query: str,
        state: Annotated[AgentState, InjectedState],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Search the knowledge base for facts, data, or document content relevant to `query`.
        Use this whenever answering requires information from ingested documents.
        Do NOT use for greetings, small talk, or questions about your own capabilities.
        You may call this again with a reformulated/narrower query if the first results
        are insufficient — but avoid unnecessary repeated calls."""
        collection_name = state["collection_name"]

        try:
            extracted = await extract_filter(query, llm=llm, fields=get_default_fields())
        except Exception as e:
            logger.warning(f"Metadata filter extraction failed: {e}")
            extracted = None

        if extracted and not validate_expression(extracted):
            logger.warning(f"Discarding malformed filter expression: {extracted!r}")
            extracted = None

        base = state.get("base_filters")
        parts = [f"({f})" for f in (base, extracted) if f]
        combined_filters = " and ".join(parts) if parts else None

        try:
            docs, context_str = await retrieve(
                retrieval_plan=[{"query": query, "collection": collection_name}],
                standalone_query=query,
                retriever=retriever,
                filters=combined_filters,
                bm25_loader=bm25_loader,
                reranker=reranker,
                output_fields=["content", "document_summary", "source", "page", "id"],
            )
        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}", exc_info=True)
            return f"Search failed: {e}", []

        if not docs:
            return "No relevant documents found.", []

        return context_str, docs

    return [search_knowledge_base]
