# Retrieva — developer entry points.
#
# Profiles trade RAM for retrieval quality. `lite` is the default and the
# design target on a 5.9 GB host; move up as hardware allows.
#   make up-lite       ~2 GB   no reranking, no query expansion
#   make up-standard   ~4 GB   + CPU reranking + multi-query
#   make up-full       8 GB+   + contextual retrieval
#
# NOTE: the embedding model is deliberately identical across profiles — its
# dimension is fixed in the Milvus schema, so varying it would invalidate the
# index. See docs/profiles.md.

.DEFAULT_GOAL := help
COMPOSE := docker compose

.PHONY: help test test-cov lint fmt check up-lite up-standard up-full down logs ps \
        build build-rerank health eval eval-baseline clean

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ── Quality ──────────────────────────────────────────────────────────────────

test:  ## Run the test suite (never makes real API calls)
	poetry run pytest

test-cov:  ## Run tests with a coverage report
	poetry run pytest --cov=. --cov-report=term-missing --cov-report=html

lint:  ## Check formatting and lint rules
	poetry run ruff check .
	poetry run ruff format --check .

fmt:  ## Auto-fix lint and formatting
	poetry run ruff check --fix .
	poetry run ruff format .

check: lint test  ## Lint + test, the pre-commit gate

# ── Stack ────────────────────────────────────────────────────────────────────
# --env-file is order-sensitive: the profile overlay comes last so it wins.

up-lite:  ## Start the stack in lite profile (~2 GB)
	$(COMPOSE) --env-file .env --env-file .env.lite up -d

up-standard:  ## Start in standard profile (~4 GB, needs a rerank-enabled image)
	$(COMPOSE) --env-file .env --env-file .env.standard up -d

up-full:  ## Start in full profile (8 GB+, needs a rerank-enabled image)
	$(COMPOSE) --env-file .env --env-file .env.full up -d

down:  ## Stop the stack (volumes preserved)
	$(COMPOSE) down

logs:  ## Tail backend logs
	$(COMPOSE) logs -f backend

ps:  ## Show container status
	$(COMPOSE) ps

build:  ## Rebuild the backend image (no reranking)
	$(COMPOSE) build backend

build-rerank:  ## Rebuild with CPU-only torch + sentence-transformers (~200 MB)
	$(COMPOSE) build --build-arg INSTALL_RERANK=true backend

health:  ## Pretty-print /health
	@curl -s http://localhost:9090/health | python -m json.tool

# ── Evaluation ───────────────────────────────────────────────────────────────

eval:  ## Run retrieval eval against the committed baseline
	poetry run python scripts/eval_retrieval.py --compare evals/results/baseline.json

eval-baseline:  ## Record a new baseline (overwrites — commit the result)
	poetry run python scripts/eval_retrieval.py --save evals/results/baseline.json

clean:  ## Remove caches and coverage artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
