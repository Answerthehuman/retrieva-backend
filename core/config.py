from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "retrieva"
    postgres_host: str = "postgres"
    postgres_port: int = 5432

    redis_host: str = "redis"
    redis_port: int = 6379

    milvus_host: str = "milvus"
    milvus_port: int = 19530
    milvus_default_collection: str = "retrieva_docs"

    app_host: str = "0.0.0.0"
    app_port: int = 9090

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
