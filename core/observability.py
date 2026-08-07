"""Langfuse tracing — captures every agent run, tool call, and LLM generation.

Host-agnostic by design: the SDK talks to Langfuse Cloud or a self-hosted
instance through the same code path, differing only in ``LANGFUSE_HOST``.

Tracing is **opt-in and fail-open**. With no keys configured it is silently
inert, and any error inside the tracing layer is swallowed rather than allowed
to break a chat turn — observability must never take down the thing it observes.
"""
import logging
from typing import Any, Dict, List, Optional

from core.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_handler = None
_resolved = False


def _build_handler(s: Settings):
    """Construct the Langfuse LangChain callback handler, or None if unusable."""
    # getattr rather than attribute access: callers may pass a partial or stub
    # settings object (test harnesses do), and tracing must degrade to "off"
    # for those rather than raising and taking the caller down with it.
    if not getattr(s, "langfuse_enabled", False):
        logger.info("Langfuse tracing disabled (LANGFUSE_ENABLED=false).")
        return None

    public_key = getattr(s, "langfuse_public_key", None)
    secret_key = getattr(s, "langfuse_secret_key", None)
    host = getattr(s, "langfuse_host", "https://cloud.langfuse.com")

    if not (public_key and secret_key):
        logger.info(
            "Langfuse tracing inactive — LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY "
            "are not set. Chat and ingestion are unaffected."
        )
        return None

    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        # The SDK reads credentials from the client singleton, which the handler
        # then picks up; constructing it explicitly keeps configuration in
        # settings rather than relying on ambient env vars.
        Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        handler = CallbackHandler()
        logger.info("Langfuse tracing active → %s", host)
        return handler
    except Exception as e:
        # Never let an observability misconfiguration break the request path.
        logger.warning("Langfuse tracing unavailable (%s) — continuing without it.", e)
        return None


def get_langfuse_handler(settings: Optional[Settings] = None):
    """Return the shared callback handler, or None when tracing is off.

    Cached after the first call: building the handler opens an HTTP client and
    the result is stable for the process lifetime.
    """
    global _handler, _resolved
    if _resolved:
        return _handler

    _handler = _build_handler(settings or get_settings())
    _resolved = True
    return _handler


def build_callbacks(settings: Optional[Settings] = None) -> List[Any]:
    """Callback list for LangChain/LangGraph ``config``. Empty when tracing is off."""
    handler = get_langfuse_handler(settings)
    return [handler] if handler else []


def trace_metadata(
    *,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build the LangChain ``config`` metadata Langfuse reads trace fields from.

    Langfuse picks up these reserved ``langfuse_*`` metadata keys and promotes
    them to first-class trace attributes, which is what makes traces filterable
    by session and user in the UI rather than being one anonymous blob per run.
    """
    metadata: Dict[str, Any] = {}
    if session_id:
        metadata["langfuse_session_id"] = session_id
    if user_id:
        metadata["langfuse_user_id"] = user_id
    if tags:
        metadata["langfuse_tags"] = tags
    if name:
        metadata["langfuse_trace_name"] = name
    metadata.update({k: v for k, v in extra.items() if v is not None})
    return metadata


def flush() -> None:
    """Flush buffered events. Call on shutdown so short-lived runs aren't lost."""
    if not _resolved or _handler is None:
        return
    try:
        from langfuse import get_client

        get_client().flush()
        logger.info("Flushed pending Langfuse events.")
    except Exception as e:
        logger.warning("Langfuse flush failed: %s", e)
