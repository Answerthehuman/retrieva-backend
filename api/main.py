"""FastAPI application entry point for Retrieva."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.health import router as health_router
from .routes.chat import router as chat_router
from ..core.config.settings import get_settings
from ..core.db.database import Base, engine

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Retrieva application...")
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables initialized")
    yield
    logger.info("🛑 Stopping Retrieva application...")


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
    return app


app = create_app()
