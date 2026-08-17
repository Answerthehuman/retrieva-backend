"""FastAPI application entry point for Retrieva."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config.settings import get_settings
from core.db.database import Base, engine
from core.logging_config import configure_logging
from core.observability import flush as flush_langfuse
from core.observability import get_langfuse_handler
from core.providers import close_retriever, get_embeddings, get_llm, get_retriever
from core.startup_checks import run_startup_checks

from .routes.chat import router as chat_router
from .routes.health import router as health_router
from .routes.ingest import router as ingest_router

settings = get_settings()

# Must run before any logger is used, or the module-level loggers created at
# import time keep the default WARNING level and startup logs vanish.
configure_logging(settings.log_level)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Retrieva application...")

    # Startup must not be fatal on dependency problems: a crash here takes the
    # whole container down in a restart loop and makes /health unreachable,
    # which is exactly when you most need it to answer. Degraded state is
    # reported by /health and surfaces per-request instead.
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables initialized")
    except Exception as e:
        logger.error("❌ Database initialization failed: %s", e)

    # Warm the provider singletons so the first request isn't slow. Missing API
    # keys or an unreachable Milvus are logged, not raised.
    logger.info("🔄 Initializing AI providers...")
    for name, factory in (
        ("LLM", get_llm),
        ("embeddings", get_embeddings),
        ("retriever", get_retriever),
    ):
        try:
            factory(settings)
        except Exception as e:
            logger.warning("⚠️  %s unavailable at startup: %s", name, e)

    # Resolve tracing once at startup so its status is visible in the boot log
    # rather than only surfacing on the first chat turn.
    get_langfuse_handler(settings)

    # Consistency checks (e.g. embedding dim vs the live Milvus collection).
    # Reported loudly but not fatal — a restart loop would make /health
    # unreachable exactly when it is needed to diagnose the problem.
    app.state.startup_problems = run_startup_checks(settings)

    logger.info("Active profile: %s", settings.retrieva_profile)

    logger.info("✅ Startup complete")

    yield

    logger.info("🛑 Stopping Retrieva application...")
    try:
        await close_retriever()
    except Exception as e:
        logger.warning("Error closing retriever: %s", e)

    # Langfuse batches events in the background; without an explicit flush the
    # final turns of a short-lived container are dropped on exit.
    flush_langfuse()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Retrieva API",
        description="Generic hybrid RAG API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(ingest_router)
    return app


app = create_app()
