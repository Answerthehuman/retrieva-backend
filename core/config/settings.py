from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: Optional[str] = Field(None, env="DATABASE_URL")
    postgres_user: str = Field("postgres", env="POSTGRES_USER")
    postgres_password: str = Field("postgres", env="POSTGRES_PASSWORD")
    postgres_db: str = Field("retrieva", env="POSTGRES_DB")
    postgres_host: str = Field("postgres", env="POSTGRES_HOST")
    postgres_port: int = Field(5432, env="POSTGRES_PORT")

    redis_host: str = Field("redis", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")

    milvus_host: str = Field("milvus", env="MILVUS_HOST")
    milvus_port: int = Field(19530, env="MILVUS_PORT")
    milvus_default_collection: str = Field("retrieva_docs", env="MILVUS_DEFAULT_COLLECTION")

    llm_provider: str = Field("openai", env="LLM_PROVIDER")
    embedding_provider: str = Field("huggingface", env="EMBEDDING_PROVIDER")
    embedding_model: str = Field("BAAI/bge-large-en-v1.5", env="EMBEDDING_MODEL")

    hybrid_search_enabled: bool = Field(True, env="HYBRID_SEARCH_ENABLED")
    rerank_enabled: bool = Field(False, env="RERANK_ENABLED")
    retrieval_top_k: int = Field(10, env="RETRIEVAL_TOP_K")
    rerank_top_k: int = Field(5, env="RERANK_TOP_K")

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
