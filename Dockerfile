# syntax=docker/dockerfile:1

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — builder. Compiles/collects dependencies into a self-contained venv.
# Nothing from this stage ships except the venv itself, so build toolchains,
# Poetry, and pip caches never reach the final image.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # PyPI reads can stall on large wheels; the defaults (15s, 5 retries) are
    # not enough on a slow link and surface as a ReadTimeoutError mid-build.
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10 \
    POETRY_VERSION=2.3.1 \
    POETRY_REQUESTS_TIMEOUT=180 \
    # Build into a project-local venv so the whole tree can be copied across.
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

WORKDIR /app

# Build-only toolchain — needed if any dependency lacks a manylinux wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==${POETRY_VERSION}"

# Dependency layer — cached until the lockfile changes.
# poetry.lock is the single source of truth for dependencies.
COPY pyproject.toml poetry.lock ./

# Reranking is opt-in. Left false, torch and its 14 Linux-only CUDA packages
# (~3 GB) are never downloaded — matching RERANK_ENABLED=false, the default.
# Set true to enable it; torch then comes from PyTorch's CPU index (~200 MB)
# instead of the default PyPI build that bundles the whole CUDA stack.
ARG INSTALL_RERANK=false

RUN poetry install --no-root --only main --no-ansi \
    && if [ "$INSTALL_RERANK" = "true" ]; then \
         echo "Installing CPU-only torch + sentence-transformers…" \
         && /app/.venv/bin/pip install --no-cache-dir \
              --index-url https://download.pytorch.org/whl/cpu torch \
         && /app/.venv/bin/pip install --no-cache-dir sentence-transformers ; \
       fi \
    # Strip artifacts that are pure dead weight at runtime.
    && find /app/.venv -type d -name '__pycache__' -prune -exec rm -rf {} + \
    && find /app/.venv -type d -name 'tests' -prune -exec rm -rf {} + \
    && find /app/.venv -type f -name '*.pyc' -delete \
    && find /app/.venv -type f -name '*.pyo' -delete

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime. Only the interpreter, the venv, and the app source.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# curl only, for the container healthcheck. psycopg2-binary bundles its own
# libpq, so libpq-dev is a build-time concern only and is not needed here.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

COPY --from=builder /app/.venv /app/.venv

# Application source. The code lives directly under backend/, so these are the
# real top-level packages — there is no `retrieva/` package.
COPY api ./api
COPY agents ./agents
COPY core ./core
COPY services ./services
COPY shared ./shared
COPY scripts ./scripts
COPY main.py cli.py ./

# BM25 sparse-retrieval stats (vocabulary + IDF scores) are written here at
# ingest time and are the source of truth for hybrid search — Redis is only a
# cache in front of them. The app source above is copied as root, so this must
# be created and chowned before dropping privileges, or every ingest logs
# "Permission denied: '/app/data'" and silently loses its BM25 vocabulary.
# Mount a volume at this path to persist stats across container rebuilds.
RUN mkdir -p /app/data/bm25_stats && chown -R appuser:appuser /app/data

USER appuser

EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:9090/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "9090"]
