"""Provider factory — lazily creates and caches LLM, embedding, retriever, and reranker singletons.

Usage:
    from core.providers import get_llm, get_embeddings, get_retriever, get_reranker

All functions return the same instance on repeated calls (module-level cache).
"""
import logging
from typing import Optional

from .config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# ── Module-level singletons ───────────────────────────────────────────────────
_llm = None
_embeddings = None
_retriever = None
_reranker = None


def get_llm(settings: Optional[Settings] = None):
    """Return a LangChain-compatible chat LLM (Gemini or OpenAI)."""
    global _llm
    if _llm is not None:
        return _llm

    s = settings or get_settings()

    if s.llm_provider == "openai":
        from pyzo_ai_core.providers import get_openai
        _llm = get_openai(model=s.llm_model, temperature=s.llm_temperature)
        logger.info("LLM provider: OpenAI (%s)", s.llm_model)
    else:
        from pyzo_ai_core.providers import get_gemini
        _llm = get_gemini(model=s.llm_model, temperature=s.llm_temperature)
        logger.info("LLM provider: Gemini (%s)", s.llm_model)

    return _llm


def get_embeddings(settings: Optional[Settings] = None):
    """Return a LangChain-compatible embedding model (Gemini or OpenAI)."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    s = settings or get_settings()

    if s.embedding_provider == "openai":
        from pyzo_ai_core.providers import get_openai_embeddings
        _embeddings = get_openai_embeddings(model=s.embedding_model)
        logger.info("Embedding provider: OpenAI (%s)", s.embedding_model)
    else:
        from pyzo_ai_core.providers import get_gemini_embeddings
        _embeddings = get_gemini_embeddings(model=s.embedding_model)
        logger.info("Embedding provider: Gemini (%s)", s.embedding_model)

    return _embeddings


def get_retriever(settings: Optional[Settings] = None):
    """Return an AsyncRetriever connected to Milvus."""
    global _retriever
    if _retriever is not None:
        return _retriever

    from pyzo_ai_core.nodes.rag_retrieval import AsyncRetriever

    s = settings or get_settings()
    _retriever = AsyncRetriever(
        uri=s.milvus_uri,
        embedding_function=get_embeddings(s),
        top_k=s.retrieval_top_k,
    )
    logger.info("AsyncRetriever connected to %s", s.milvus_uri)
    return _retriever


def get_reranker(settings: Optional[Settings] = None):
    """Return a Reranker instance (lazy-loads the model on first rerank call)."""
    global _reranker
    if _reranker is not None:
        return _reranker

    s = settings or get_settings()
    if not s.rerank_enabled:
        return None

    from pyzo_ai_core.nodes.rag_retrieval import Reranker

    _reranker = Reranker(model_name=s.rerank_model, top_k=s.rerank_top_k)
    logger.info("Reranker ready: %s (top_k=%d)", s.rerank_model, s.rerank_top_k)
    return _reranker


async def close_retriever():
    """Shutdown hook — close the Milvus connection."""
    global _retriever
    if _retriever is not None:
        await _retriever.close()
        _retriever = None
        logger.info("AsyncRetriever connection closed")
