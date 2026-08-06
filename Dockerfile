FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.3.1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# Build deps for psycopg2 / pymupdf wheels that may need compiling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==${POETRY_VERSION}"

# Dependency layer — cached until the lockfile changes.
# poetry.lock is the single source of truth for dependencies.
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-interaction --no-ansi

# Application source. The code lives directly under backend/, so these are the
# real top-level packages — there is no `retrieva/` package.
COPY api ./api
COPY agents ./agents
COPY core ./core
COPY services ./services
COPY shared ./shared
COPY scripts ./scripts
COPY main.py cli.py ./

EXPOSE 9090

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "9090"]
