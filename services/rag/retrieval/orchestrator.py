import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_FIELDS = [
    "id",
    "content",
    "document_id",
    "document_summary",
    "source",
    "page",
    "chunk_index",
    "total_chunks",
    "chunk_size",
    "ingested_at",
]


def build_context(documents: list[dict[str, Any]]) -> str:
    """Format retrieved documents into a context string for an LLM prompt."""
    if not documents:
        return "No context found."
    parts = []
    for i, doc in enumerate(documents, 1):
        text = doc.get("content", "").strip()
        source = doc.get("file_name") or doc.get("source", "Unknown")
        page = doc.get("page", "")

        page_str = f" (Page {page})" if page else ""
        parts.append(f"--- Document {i} [{source}{page_str}] ---\n{text}\n")
    return "\n".join(parts)


async def retrieve(
    retrieval_plan: list[dict[str, Any]],
    standalone_query: str,
    *,
    retriever,
    filters: str | None = None,
    collection_name: str | None = None,
    bm25_loader: Callable | None = None,
    reranker=None,
    rerank_top_k: int = 10,
    rerank_batch_size: int = 20,
    output_fields: list[str] | None = None,
    build_context_str: bool = True,
) -> tuple[list[dict[str, Any]], str]:
    """
    Standalone retrieval workflow.

    Supports two modes:
    - **Parallel** (when retrieval_plan is non-empty): runs all vector_store steps in
      parallel, merges, deduplicates, and optionally reranks.
    - **Single-query** (fallback): uses standalone_query against collection_name.
    """
    _output_fields = output_fields or DEFAULT_OUTPUT_FIELDS
    vs_steps = [s for s in retrieval_plan if s.get("type", "vector_store") in ("vector_store", "")]

    if collection_name:
        for step in vs_steps:
            if not step.get("collection") or step["collection"] == "all":
                step["collection"] = collection_name

    if vs_steps:
        logger.info(f"Parallel retrieval: {len(vs_steps)} steps, query='{standalone_query}'")
        documents = await retriever.search_parallel(
            vs_steps, output_fields=_output_fields, bm25_loader=bm25_loader, filters=filters
        )
        if reranker and documents:
            rerank_query = standalone_query or vs_steps[0].get("query", "")
            documents = await reranker.rerank(
                rerank_query, documents, top_k=rerank_top_k, batch_size=rerank_batch_size
            )
    else:
        if not standalone_query:
            logger.warning("No retrieval plan and no standalone_query — returning empty")
            return [], ""

        if not collection_name:
            logger.error("collection_name is required for single-query retrieval")
            return [], ""

        try:
            documents = await retriever.search_semantic(
                collection_name, standalone_query, output_fields=_output_fields, filters=filters
            )
            if reranker and documents:
                documents = await reranker.rerank(
                    standalone_query, documents, top_k=rerank_top_k, batch_size=rerank_batch_size
                )
            logger.info(
                f"Single-query retrieval: query='{standalone_query}', {len(documents)} documents"
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            return [], ""

    context = build_context(documents) if build_context_str else ""
    return documents, context
