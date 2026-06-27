# Retrieva

Retrieva is a hybrid Retrieval‑Augmented Generation (RAG) system. It combines a FastAPI backend, PostgreSQL, Redis, and Milvus vector search to deliver hybrid semantic search and streamed chat responses.

## Architecture

- FastAPI backend with SSE chat streaming
- SQLAlchemy + PostgreSQL session and message store
- Redis for cache / hybrid search metadata
- Milvus for vector retrieval
- Docker Compose for local orchestration

## Quickstart

### Prerequisites
- Python 3.11+
- Docker & Docker‑Compose
- Poetry (install via script below)

### Installation & setup
```bash
# 1. Clone the repository
git clone <repository‑url>
cd Retrieva/backend

# 2. Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# 3. Install project dependencies and activate a virtual environment
poetry install
poetry shell
```

### Environment configuration
```bash
cp .env.example .env   # edit the file if you need to change defaults
```

### Docker infrastructure
```bash
# Start PostgreSQL, Redis, Milvus, and any other services
docker compose up -d
```

### Database migrations (Alembic)
```bash
# Initialise Alembic (run once)
alembic init alembic

# Generate the initial migration based on the current models
alembic revision --autogenerate -m "Initial schema"

# Apply the migration to create tables
alembic upgrade head
```

### Running the API
```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 9090
# In a separate terminal you can start Celery workers if needed
# celery -A services.workers.celery_app worker --loglevel=info
# celery -A services.workers.celery_app beat   --loglevel=info
```

Open `http://localhost:9090/docs` in a browser to explore the OpenAPI UI.

### Ingest & query example commands
```bash
python cli.py ingest --source filesystem --collection my_docs --path ./data
python cli.py query --collection my_docs "your question here"
```

### Documentation
- `POETRY_SETUP.md` – detailed Poetry and Alembic instructions
- `QUICKSTART.md` – this quick‑start guide
- `DEPLOYMENT_GUIDE.md` – production deployment notes
- `ARCHITECTURE.md` – system design overview
- `MISSING_COMPONENTS.md` – roadmap for future features

## Notes
This scaffold provides a working foundation for building a production‑ready RAG stack. Adjust the configuration in `.env` as needed for your environment.
