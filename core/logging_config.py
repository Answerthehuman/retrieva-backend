"""Application logging configuration.

Without this, nothing configures the root logger, so Python's default level
(WARNING) silently swallowed every ``logger.info`` in the codebase — including
startup diagnostics like provider wiring and Langfuse status. Those lines were
being written and thrown away, which made ``docker compose logs`` misleading
rather than merely quiet.
"""

import logging
import sys

#: Libraries that are chatty at INFO and drown out application logs.
_NOISY_LOGGERS = {
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "urllib3": logging.WARNING,
    "openai": logging.WARNING,
    "anthropic": logging.WARNING,
    "google": logging.WARNING,
    "google_genai": logging.WARNING,
    "pymilvus": logging.WARNING,
    # Langfuse logs each queued span at INFO; useful when debugging tracing,
    # far too noisy otherwise.
    "langfuse": logging.WARNING,
    "opentelemetry": logging.WARNING,
    # Access logs are already emitted by uvicorn.access; this one duplicates.
    "uvicorn.access": logging.INFO,
}


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging. Safe to call more than once.

    Args:
        level: Root log level name (DEBUG/INFO/WARNING/ERROR). Unrecognised
            values fall back to INFO rather than raising — a typo'd LOG_LEVEL
            should not prevent the app from starting.
    """
    resolved = getattr(logging, str(level).upper(), None)
    if not isinstance(resolved, int):
        resolved = logging.INFO

    root = logging.getLogger()

    # force=True replaces any handler uvicorn/pytest installed first. Without
    # it, a pre-existing handler wins and the level change appears to do
    # nothing — the exact failure this module exists to fix.
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    root.setLevel(resolved)

    for name, noisy_level in _NOISY_LOGGERS.items():
        # Never quiet a library below the root level: if the operator asked for
        # DEBUG they want the noise too.
        logging.getLogger(name).setLevel(max(noisy_level, resolved))
