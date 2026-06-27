from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Database ──────────────────────────────────────────────────────────────
    database_url: Optional[str] = Field(None, env="DATABASE_URL")
    postgres_user: str = Field("postgres", env="POSTGRES_USER")
    postgres_password: str = Field("postgres", env="POSTGRES_PASSWORD")
    postgres_db: str = Field("retrieva", env="POSTGRES_DB")
    postgres_host: str = Field("postgres", env="POSTGRES_HOST")
    postgres_port: int = Field(5432, env="POSTGRES_PORT")

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_host: str = Field("redis", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")

    # ── Milvus ────────────────────────────────────────────────────────────────
    milvus_host: str = Field("milvus", env="MILVUS_HOST")
    milvus_port: int = Field(19530, env="MILVUS_PORT")
    milvus_uri: str = Field("http://localhost:19530", env="MILVUS_URI")
    milvus_default_collection: str = Field("retrieva_docs", env="MILVUS_DEFAULT_COLLECTION")

    # ── LLM Provider ─────────────────────────────────────────────────────────
    llm_provider: str = Field("gemini", env="LLM_PROVIDER")  # "gemini" | "openai"
    llm_model: str = Field("gemini-2.5-flash", env="LLM_MODEL")
    llm_temperature: float = Field(0.0, env="LLM_TEMPERATURE")

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_provider: str = Field("gemini", env="EMBEDDING_PROVIDER")  # "gemini" | "openai"
    embedding_model: str = Field("models/gemini-embedding-2", env="EMBEDDING_MODEL")
    embedding_dim: int = Field(3072, env="EMBEDDING_DIM")

    # ── API Keys ──────────────────────────────────────────────────────────────
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    hybrid_search_enabled: bool = Field(True, env="HYBRID_SEARCH_ENABLED")
    rerank_enabled: bool = Field(False, env="RERANK_ENABLED")
    rerank_model: str = Field("BAAI/bge-reranker-v2-m3", env="RERANK_MODEL")
    retrieval_top_k: int = Field(50, env="RETRIEVAL_TOP_K")
    rerank_top_k: int = Field(10, env="RERANK_TOP_K")

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(600, env="CHUNK_SIZE")
    chunk_overlap: int = Field(120, env="CHUNK_OVERLAP")

    # ── App ───────────────────────────────────────────────────────────────────
    app_host: str = Field("0.0.0.0", env="APP_HOST")
    app_port: int = Field(9090, env="APP_PORT")

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
