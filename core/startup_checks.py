"""Startup consistency checks.

The embedding dimension is fixed when a Milvus collection is created. If the
configured model later disagrees with the stored schema, inserts fail deep in
the ingest path with an opaque pymilvus error, and — worse — searches against a
mismatched index return nonsense rather than an error at all.

Checking once at boot turns that into a single clear message naming the fix.
"""

import logging

from core.config.settings import Settings

logger = logging.getLogger(__name__)


class EmbeddingDimensionMismatch(RuntimeError):
    """Configured embedding dimension disagrees with an existing collection."""


def check_embedding_dimension(settings: Settings, *, strict: bool = True) -> str | None:
    """Compare configured EMBEDDING_DIM against the live Milvus collection.

    Args:
        settings: Active settings.
        strict: Raise on mismatch. When False, the problem is logged and
            returned — used by /health, which must stay reachable precisely
            when something is misconfigured.

    Returns:
        A description of the problem, or None when consistent (or unverifiable).
    """
    # 1. Does the configured model's known width match the configured dim?
    from core.llm.ollama import expected_dimension

    if settings.embedding_provider == "ollama":
        expected = expected_dimension(settings.embedding_model)
        if expected is not None and expected != settings.embedding_dim:
            message = (
                f"EMBEDDING_DIM={settings.embedding_dim} does not match "
                f"'{settings.embedding_model}', which produces {expected}-dim vectors. "
                f"Set EMBEDDING_DIM={expected}."
            )
            if strict:
                raise EmbeddingDimensionMismatch(message)
            return message

    # 2. Does the configured dim match what the collection was created with?
    try:
        from pymilvus import Collection, connections, utility

        if not connections.has_connection("default"):
            connections.connect(alias="default", uri=settings.milvus_uri)

        collection_name = settings.milvus_default_collection
        if not utility.has_collection(collection_name):
            # Nothing ingested yet — the first ingest will create the
            # collection at the configured dim, which is by definition correct.
            return None

        collection = Collection(collection_name)
        stored_dim = None
        for field in collection.schema.fields:
            if field.name == "embedding":
                stored_dim = (field.params or {}).get("dim")
                break

        if stored_dim is None:
            return None

        if int(stored_dim) != int(settings.embedding_dim):
            message = (
                f"Embedding dimension mismatch: collection '{collection_name}' was created "
                f"with dim={stored_dim}, but EMBEDDING_DIM={settings.embedding_dim} "
                f"(model '{settings.embedding_model}'). Inserts will fail and searches "
                f"would be meaningless.\n"
                f"Fix by either:\n"
                f"  • reverting EMBEDDING_DIM to {stored_dim} and using the original model, or\n"
                f"  • re-ingesting into a fresh collection:\n"
                f"      docker compose exec backend python -c "
                f'"from pymilvus import connections,utility; '
                f"connections.connect(uri='{settings.milvus_uri}'); "
                f"utility.drop_collection('{collection_name}')\"\n"
                f"    then re-upload your documents."
            )
            if strict:
                raise EmbeddingDimensionMismatch(message)
            return message

    except EmbeddingDimensionMismatch:
        raise
    except Exception as e:
        # Milvus being unreachable is a separate, already-reported problem —
        # it must not masquerade as a dimension mismatch or block startup.
        logger.debug("Skipped embedding-dimension check (Milvus unreachable: %s)", e)
        return None

    return None


def run_startup_checks(settings: Settings) -> list[str]:
    """Run all non-fatal startup checks, returning any problems found.

    Deliberately non-fatal: a crash here takes the container down in a restart
    loop and makes /health unreachable — exactly when it is most needed. Report
    loudly, serve /health, and let the operator act.
    """
    problems: list[str] = []

    mismatch = check_embedding_dimension(settings, strict=False)
    if mismatch:
        problems.append(mismatch)
        logger.error("STARTUP CHECK FAILED: %s", mismatch)

    return problems
