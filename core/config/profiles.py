"""Runtime resource/quality profiles.

Retrieva runs on hardware ranging from a RAM-starved laptop VM (~2 GB free) to
a proper workstation. Rather than one config that is either too heavy to boot
or too weak to be good, the operator picks a tier at launch and the expensive
features switch themselves on or off.

⚠️ **The embedding model is deliberately NOT part of a profile.**

``FieldSchema(name="embedding", dtype=FLOAT_VECTOR, dim=dim)`` fixes the vector
width when the Milvus collection is created. Varying the model per profile
would mean every profile switch either silently invalidated the index or hard
-failed on insert. So the embedding model is chosen once (it must run in the
smallest profile) and held constant; the levers that scale with available RAM
are reranking, contextual retrieval, and query expansion — all of which are
either stateless or affect only newly-ingested documents.
"""

from typing import Any

LITE = "lite"
STANDARD = "standard"
FULL = "full"

VALID_PROFILES = (LITE, STANDARD, FULL)

#: Feature defaults per tier. These apply only where the operator has not set
#: the corresponding env var explicitly — an explicit setting always wins, so
#: a profile is a sensible starting point rather than a straitjacket.
PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    LITE: {
        # Cross-encoder reranking needs sentence-transformers + torch (~200 MB
        # resident). Off here; retrieval must stay usable without it.
        "rerank_enabled": False,
        # Each expansion is an extra LLM round trip and N extra searches.
        "multi_query_enabled": False,
        # One LLM call per chunk at ingest — by far the most expensive feature.
        "contextual_retrieval_enabled": False,
        "retrieval_top_k": 30,
        "rerank_top_k": 10,
        "agent_max_tool_calls": 4,
        # Smaller batches keep peak memory down during ingest on a small VM.
        "ingest_batch_size": 16,
    },
    STANDARD: {
        "rerank_enabled": True,
        "multi_query_enabled": True,
        "contextual_retrieval_enabled": False,
        "retrieval_top_k": 50,
        "rerank_top_k": 10,
        "agent_max_tool_calls": 6,
        "ingest_batch_size": 32,
    },
    FULL: {
        "rerank_enabled": True,
        "multi_query_enabled": True,
        "contextual_retrieval_enabled": True,
        "retrieval_top_k": 100,
        "rerank_top_k": 20,
        "agent_max_tool_calls": 10,
        "ingest_batch_size": 64,
    },
}

#: Human-readable summary, surfaced by /health so the UI can explain *why*
#: reranking is off rather than leaving degraded quality unexplained.
PROFILE_DESCRIPTIONS = {
    LITE: "Minimal footprint (~2 GB). Reranking and query expansion disabled.",
    STANDARD: "Balanced (~4 GB). Reranking and multi-query enabled.",
    FULL: "Maximum quality (8 GB+). All retrieval features enabled.",
}


def normalize_profile(name: str) -> str:
    """Return a valid profile name, falling back to LITE.

    An unrecognised profile falls back rather than raising: the safest
    behaviour on a misconfigured host is to boot in the *smallest* tier, not to
    refuse to start or to over-commit memory it may not have.
    """
    candidate = (name or "").strip().lower()
    return candidate if candidate in VALID_PROFILES else LITE


def profile_defaults(name: str) -> dict[str, Any]:
    """Feature defaults for a profile."""
    return dict(PROFILE_DEFAULTS[normalize_profile(name)])
