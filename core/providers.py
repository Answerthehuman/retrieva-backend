"""Provider factory — lazily creates and caches LLM, embedding, retriever, and reranker singletons.

Usage:
    from core.providers import get_llm, get_embeddings, get_retriever, get_reranker

All functions return the same instance on repeated calls (module-level cache).
"""
import logging
from typing import Any, List, Optional, Tuple

from .config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# ── Module-level singletons ───────────────────────────────────────────────────
_llm = None
_llm_providers = None
_embeddings = None
_retriever = None
_reranker = None
_agent_graph = None


_SUPPORTED_LLM_PROVIDERS = ("gemini", "anthropic", "openai")


def _resolve_provider_order(s: Settings) -> List[str]:
    """Ordered, de-duplicated provider names from settings."""
    order = [p.strip().lower() for p in s.llm_fallback_order.split(",") if p.strip()]

    # Legacy LLM_PROVIDER wins the top slot so pre-existing .env files keep
    # their intended primary rather than silently switching.
    if s.llm_provider:
        primary = s.llm_provider.strip().lower()
        if primary in _SUPPORTED_LLM_PROVIDERS:
            order = [primary] + [p for p in order if p != primary]
            logger.info(
                "LLM_PROVIDER=%s is deprecated; promoting it to the head of the "
                "chain. Set LLM_FALLBACK_ORDER instead.", primary,
            )

    unknown = [p for p in order if p not in _SUPPORTED_LLM_PROVIDERS]
    if unknown:
        logger.warning("Ignoring unknown LLM provider(s): %s", ", ".join(unknown))

    seen, result = set(), []
    for p in order:
        if p in _SUPPORTED_LLM_PROVIDERS and p not in seen:
            seen.add(p)
            result.append(p)
    return result


def _build_provider(name: str, s: Settings):
    """Construct one provider's chat model, or None if it has no API key."""
    # A legacy LLM_MODEL only makes sense for the provider it was written for.
    def _model_for(provider: str, default: str) -> str:
        if s.llm_model and s.llm_provider and s.llm_provider.strip().lower() == provider:
            return s.llm_model
        return default

    if name == "gemini":
        if not s.google_api_key:
            return None
        from core.llm.gemini import get_gemini
        return get_gemini(
            model=_model_for("gemini", s.gemini_model),
            temperature=s.llm_temperature,
            google_api_key=s.google_api_key,
        )

    if name == "anthropic":
        if not s.anthropic_api_key:
            return None
        from core.llm.anthropic import get_anthropic
        # temperature is dropped inside the factory for models that reject it.
        return get_anthropic(
            model=_model_for("anthropic", s.anthropic_model),
            temperature=s.llm_temperature,
            max_tokens=s.llm_max_tokens,
            thinking=s.llm_thinking_enabled,
            effort=s.llm_effort,
            api_key=s.anthropic_api_key,
        )

    if name == "openai":
        if not s.openai_api_key:
            return None
        from core.llm.openai import get_openai
        return get_openai(
            model=_model_for("openai", s.openai_model),
            temperature=s.llm_temperature,
            api_key=s.openai_api_key,
        )

    return None


def get_llm_providers(settings: Optional[Settings] = None) -> List[Tuple[str, Any]]:
    """Ordered (name, model) pairs for every *configured* provider.

    Providers without an API key are skipped rather than raising, so an
    unused entry in LLM_FALLBACK_ORDER costs nothing.
    """
    global _llm_providers
    if _llm_providers is not None:
        return _llm_providers

    s = settings or get_settings()
    built, skipped = [], []
    for name in _resolve_provider_order(s):
        model = _build_provider(name, s)
        if model is None:
            skipped.append(name)
        else:
            built.append((name, model))

    if skipped:
        logger.info("LLM providers skipped (no API key): %s", ", ".join(skipped))
    if not built:
        logger.error(
            "No LLM provider is configured. Set at least one of GOOGLE_API_KEY, "
            "ANTHROPIC_API_KEY, or OPENAI_API_KEY for providers listed in "
            "LLM_FALLBACK_ORDER=%s.", s.llm_fallback_order,
        )
    else:
        logger.info(
            "LLM chain: %s", " → ".join(f"{n}" for n, _ in built),
        )

    _llm_providers = built
    return _llm_providers


def _chain_with_fallbacks(runnables: List[Any]):
    """First runnable, falling back through the rest on error."""
    if not runnables:
        raise RuntimeError(
            "No LLM provider is configured. Set GOOGLE_API_KEY, ANTHROPIC_API_KEY, "
            "or OPENAI_API_KEY (see LLM_FALLBACK_ORDER)."
        )
    if len(runnables) == 1:
        return runnables[0]
    return runnables[0].with_fallbacks(runnables[1:])


def get_llm(settings: Optional[Settings] = None):
    """Chat LLM for plain (non-tool) calls, with automatic provider fallback.

    Used by document summarization, metadata-filter extraction, and vision OCR.
    The agent binds tools instead — see get_llm_with_tools().
    """
    global _llm
    if _llm is not None:
        return _llm

    providers = get_llm_providers(settings)
    _llm = _chain_with_fallbacks([m for _, m in providers])
    return _llm


def get_llm_with_tools(tools: List, settings: Optional[Settings] = None):
    """Tool-bound chat LLM with provider fallback.

    Tools are bound to each provider *before* chaining: RunnableWithFallbacks
    has no .bind_tools(), so binding must happen on the concrete models.
    """
    providers = get_llm_providers(settings)
    return _chain_with_fallbacks([m.bind_tools(tools) for _, m in providers])


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

    # sentence-transformers is an optional extra (it pulls torch + ~3 GB of
    # Linux CUDA packages). Without it Reranker.rerank() catches the ImportError
    # and silently returns the unranked order — correct, but easy to mistake for
    # "reranking is on and just not helping". Say so once, loudly, at startup.
    from importlib.util import find_spec

    if find_spec("sentence_transformers") is None:
        logger.warning(
            "RERANK_ENABLED=true but sentence-transformers is not installed — "
            "results will NOT be reranked. Install it with "
            "`poetry install --extras rerank`, or rebuild the image with "
            "`docker compose build --build-arg INSTALL_RERANK=true`."
        )
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
    _agent_graph = build_agent_graph(llm_with_tools=get_llm_with_tools(tools, s), tools=tools)
    logger.info("Agent graph compiled with %d tool(s)", len(tools))
    return _agent_graph


async def close_retriever():
    """Shutdown hook — close the Milvus connection."""
    global _retriever
    if _retriever is not None:
        await _retriever.close()
        _retriever = None
        logger.info("AsyncRetriever connection closed")
