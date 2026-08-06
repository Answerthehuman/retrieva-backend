"""Health / readiness endpoint.

Reports per-dependency status rather than a bare "ok" so the UI (and any
orchestrator) can tell the difference between "up" and "up but unusable".
"""
import asyncio
import logging

from fastapi import APIRouter
from sqlalchemy import text

from core.config.settings import get_settings
from core.db.database import DATABASE_URL, engine

logger = logging.getLogger(__name__)

router = APIRouter()


def _check_database() -> dict:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "engine": DATABASE_URL.split("://", 1)[0]}
    except Exception as e:
        return {"status": "error", "engine": DATABASE_URL.split("://", 1)[0], "detail": str(e)[:200]}


def _check_llm(settings) -> dict:
    """Configuration check only — deliberately does not spend a token."""
    provider = settings.llm_provider
    key = {
        "anthropic": settings.anthropic_api_key,
        "gemini": settings.google_api_key,
        "openai": settings.openai_api_key,
    }.get(provider)

    env_var = {
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "openai": "OPENAI_API_KEY",
    }.get(provider, "the provider's API key")

    if not key:
        return {
            "status": "unconfigured",
            "provider": provider,
            "model": settings.llm_model,
            "detail": f"{env_var} is not set.",
        }
    return {"status": "ok", "provider": provider, "model": settings.llm_model}


async def _check_embeddings(settings) -> dict:
    """Verify the embedding backend is reachable and the model is present.

    Ollama is a separate service that must have the model pulled — a missing
    model otherwise only surfaces on the first ingest, mid-upload.
    """
    provider = settings.embedding_provider
    base = {
        "provider": provider,
        "model": settings.embedding_model,
        "dim": settings.embedding_dim,
    }

    if provider != "ollama":
        # Gemini/OpenAI embeddings ride on the same API key checked above.
        return {**base, "status": "ok"}

    try:
        import httpx

        async with httpx.AsyncClient(timeout=5) as http:
            resp = await http.get(f"{settings.ollama_base_url}/api/tags")
            resp.raise_for_status()
            installed = [m.get("name", "") for m in resp.json().get("models", [])]

        wanted = settings.embedding_model.split(":", 1)[0]
        if not any(name.split(":", 1)[0] == wanted for name in installed):
            return {
                **base,
                "status": "error",
                "detail": (
                    f"Model '{settings.embedding_model}' is not pulled. "
                    f"Run: ollama pull {settings.embedding_model}"
                ),
                "installed": installed,
            }

        # Guard the dim/schema mismatch that would otherwise fail at insert time.
        from core.llm.ollama import expected_dimension

        expected = expected_dimension(settings.embedding_model)
        if expected is not None and expected != settings.embedding_dim:
            return {
                **base,
                "status": "error",
                "detail": (
                    f"EMBEDDING_DIM={settings.embedding_dim} but "
                    f"'{settings.embedding_model}' produces {expected}-dim vectors."
                ),
            }

        return {**base, "status": "ok", "url": settings.ollama_base_url}
    except Exception as e:
        return {
            **base,
            "status": "error",
            "url": settings.ollama_base_url,
            "detail": str(e)[:200],
        }


async def _check_milvus(settings) -> dict:
    try:
        from pymilvus import connections, utility

        def _ping():
            alias = "healthcheck"
            connections.connect(alias=alias, uri=settings.milvus_uri, timeout=3)
            try:
                return utility.list_collections(using=alias)
            finally:
                connections.disconnect(alias)

        collections = await asyncio.wait_for(asyncio.to_thread(_ping), timeout=6)
        return {"status": "ok", "uri": settings.milvus_uri, "collections": collections}
    except Exception as e:
        return {"status": "error", "uri": settings.milvus_uri, "detail": str(e)[:200]}


@router.get("/health")
async def health_check():
    settings = get_settings()

    database, milvus, embeddings = await asyncio.gather(
        asyncio.to_thread(_check_database),
        _check_milvus(settings),
        _check_embeddings(settings),
    )
    llm = _check_llm(settings)

    checks = {
        "database": database,
        "milvus": milvus,
        "llm": llm,
        "embeddings": embeddings,
    }
    # "degraded" == the service is reachable but can't fully serve requests.
    overall = "ok" if all(c["status"] == "ok" for c in checks.values()) else "degraded"

    return {
        "service": "Retrieva",
        "status": overall,
        "version": "0.1.0",
        "checks": checks,
        "config": {
            "embedding_model": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
            "collection": settings.milvus_default_collection,
            "hybrid_search": settings.hybrid_search_enabled,
            "rerank_enabled": settings.rerank_enabled,
            "retrieval_top_k": settings.retrieval_top_k,
            "rerank_top_k": settings.rerank_top_k,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
    }
