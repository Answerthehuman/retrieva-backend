from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from ..config.settings import get_settings

settings = get_settings()
backend_dir = Path(__file__).resolve().parents[2]
sqlite_db_path = backend_dir / "retrieva.db"

if settings.database_url:
    DATABASE_URL = settings.database_url
elif settings.postgres_host in {"postgres", "localhost", "127.0.0.1"}:
    DATABASE_URL = f"sqlite:///{sqlite_db_path}"
else:
    DATABASE_URL = (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}@"
        f"{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )

engine_kwargs = {"pool_pre_ping": True, "future": True}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
