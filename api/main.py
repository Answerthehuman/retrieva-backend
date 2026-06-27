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
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables initialized")
    
    # Initialize provider singletons
    logger.info("🔄 Initializing AI providers...")
    get_llm(settings)
    get_embeddings(settings)
    get_retriever(settings)
    logger.info("✅ AI providers ready")
    
    yield
    
    logger.info("🛑 Stopping Retrieva application...")
    await close_retriever()


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
