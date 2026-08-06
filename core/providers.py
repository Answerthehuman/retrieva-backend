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
_agent_graph = None


def get_llm(settings: Optional[Settings] = None):
    """Return a LangChain-compatible chat LLM. Claude by default."""
    global _llm
    if _llm is not None:
        return _llm

    s = settings or get_settings()

    if s.llm_provider == "openai":
        from core.llm.openai import get_openai
        _llm = get_openai(model=s.llm_model, temperature=s.llm_temperature)
        logger.info("LLM provider: OpenAI (%s)", s.llm_model)
    elif s.llm_provider == "gemini":
        from core.llm.gemini import get_gemini
        _llm = get_gemini(model=s.llm_model, temperature=s.llm_temperature)
        logger.info("LLM provider: Gemini (%s)", s.llm_model)
    else:
        from core.llm.anthropic import get_anthropic
        # temperature is dropped inside the factory for models that reject it.
        _llm = get_anthropic(
            model=s.llm_model,
            temperature=s.llm_temperature,
            max_tokens=s.llm_max_tokens,
            thinking=s.llm_thinking_enabled,
            effort=s.llm_effort,
            api_key=s.anthropic_api_key,
        )
        logger.info("LLM provider: Anthropic (%s)", s.llm_model)

    return _llm


def get_embeddings(settings: Optional[Settings] = None):
    """Return a LangChain-compatible embedding model. Ollama by default."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    s = settings or get_settings()

    if s.embedding_provider == "openai":
        from core.llm.openai import get_openai_embeddings
        _embeddings = get_openai_embeddings(model=s.embedding_model)
        logger.info("Embedding provider: OpenAI (%s)", s.embedding_model)
    elif s.embedding_provider == "gemini":
        from core.llm.gemini import get_gemini_embeddings
        _embeddings = get_gemini_embeddings(model=s.embedding_model)
        logger.info("Embedding provider: Gemini (%s)", s.embedding_model)
    else:
        from core.llm.ollama import expected_dimension, get_ollama_embeddings

        # A dim/model mismatch silently produces an unusable collection —
        # Milvus rejects the insert only once vectors of the wrong width
        # arrive, long after the schema was created. Warn loudly at wiring time.
        expected = expected_dimension(s.embedding_model)
        if expected is not None and expected != s.embedding_dim:
            logger.warning(
                "EMBEDDING_DIM=%d does not match '%s' (expects %d). "
                "Set EMBEDDING_DIM=%d and re-ingest into a fresh collection.",
                s.embedding_dim, s.embedding_model, expected, expected,
            )

        _embeddings = get_ollama_embeddings(
            model=s.embedding_model, base_url=s.ollama_base_url
        )
        logger.info(
            "Embedding provider: Ollama (%s, dim=%d) at %s",
            s.embedding_model, s.embedding_dim, s.ollama_base_url,
        )

    return _embeddings


def get_retriever(settings: Optional[Settings] = None):
    """Return an AsyncRetriever connected to Milvus."""
    global _retriever
    if _retriever is not None:
        return _retriever

    from services.rag.retrieval import AsyncRetriever

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

    from services.rag.retrieval import Reranker

    _reranker = Reranker(model_name=s.rerank_model, top_k=s.rerank_top_k)
    logger.info("Reranker ready: %s (top_k=%d)", s.rerank_model, s.rerank_top_k)
    return _reranker


def get_agent_graph(settings: Optional[Settings] = None):
    """Return the compiled Retrieva agent graph (built once, tools bound once)."""
    global _agent_graph
    if _agent_graph is not None:
        return _agent_graph

    from agents.graph import build_agent_graph
    from agents.tools import build_tools

    s = settings or get_settings()
    tools = build_tools(retriever=get_retriever(s), reranker=get_reranker(s), llm=get_llm(s), settings=s)
    _agent_graph = build_agent_graph(llm=get_llm(s), tools=tools)
    logger.info("Agent graph compiled with %d tool(s)", len(tools))
    return _agent_graph


async def close_retriever():
    """Shutdown hook — close the Milvus connection."""
    global _retriever
    if _retriever is not None:
        await _retriever.close()
        _retriever = None
        logger.info("AsyncRetriever connection closed")
