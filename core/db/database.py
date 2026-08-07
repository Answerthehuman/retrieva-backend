import logging
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from ..config.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()
backend_dir = Path(__file__).resolve().parents[2]
sqlite_db_path = backend_dir / "retrieva.db"

# Resolution order is explicit on purpose. A previous version fell back to
# SQLite whenever postgres_host was "postgres"/"localhost"/"127.0.0.1" — i.e.
# exactly the hostnames where a real Postgres actually lives — so Postgres was
# never used, including inside Docker, and data lived in a container-local
# SQLite file that vanished on restart. SQLite is now opt-in via USE_SQLITE.
if settings.database_url:
    DATABASE_URL = settings.database_url
elif settings.use_sqlite:
    DATABASE_URL = f"sqlite:///{sqlite_db_path}"
else:
    DATABASE_URL = (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}@"
        f"{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )

engine_kwargs = {"pool_pre_ping": True, "future": True}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs.update(pool_size=10, max_overflow=20, pool_recycle=1800)

# Log the backend (credentials stripped) — silently using the wrong database is
# the exact failure this module used to have.
_scheme = DATABASE_URL.split("://", 1)[0]
_target = DATABASE_URL.rsplit("@", 1)[-1] if "@" in DATABASE_URL else sqlite_db_path
logger.info("Database backend: %s (%s)", _scheme, _target)

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
