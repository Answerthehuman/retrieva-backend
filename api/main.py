"""FastAPI application entry point for Retrieva."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.health import router as health_router
from .routes.chat import router as chat_router
from .routes.ingest import router as ingest_router
from core.config.settings import get_settings
from core.db.database import Base, engine
from core.providers import close_retriever, get_llm, get_embeddings, get_retriever

logger = logging.getLogger(__name__)
settings = get_settings()


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
    logger.info("✅ Startup complete")

    yield

    logger.info("🛑 Stopping Retrieva application...")
    try:
        await close_retriever()
    except Exception as e:
        logger.warning("Error closing retriever: %s", e)


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
