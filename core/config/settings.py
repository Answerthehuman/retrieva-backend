from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Runtime profile ───────────────────────────────────────────────────────
    # Selects a resource/quality tier at launch: "lite" (~2GB, minimal),
    # "standard" (~4GB), or "full" (8GB+). Feature flags below whose value is
    # left unset are derived from this — see apply_profile_defaults().
    #
    # ⚠️ The embedding model is deliberately NOT profile-dependent. Its
    # dimension is baked into the Milvus collection at creation time, so
    # varying it per run would invalidate the index or fail on insert. Pick one
    # that runs in "lite" and keep it constant. See docs/profiles.md.
    retrieva_profile: str = Field("lite", env="RETRIEVA_PROFILE")

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field("INFO", env="LOG_LEVEL")

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str | None = Field(None, env="DATABASE_URL")
    # Opt-in SQLite for zero-setup local dev. Off by default so that a
    # configured Postgres is actually used instead of being silently ignored.
    use_sqlite: bool = Field(False, env="USE_SQLITE")
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

    # ── LLM Providers (ordered fallback chain) ───────────────────────────────
    # Comma-separated, highest priority first. Every augmentation call (agent
    # generation, metadata-filter extraction, document summarization, PDF vision
    # OCR) runs through this chain: if a provider errors — no credits, rate
    # limit, transient 5xx — the next one is tried automatically.
    #
    # Providers with no API key configured are skipped when the chain is built,
    # so listing one costs nothing until you supply its key.
    llm_fallback_order: str = Field("gemini,anthropic,openai", env="LLM_FALLBACK_ORDER")

    # Per-provider model ids. Each provider needs its own — a single LLM_MODEL
    # cannot describe a chain whose members are different model families.
    gemini_model: str = Field("gemini-3.5-flash-lite", env="GEMINI_MODEL")
    anthropic_model: str = Field("claude-opus-5", env="ANTHROPIC_MODEL")
    openai_model: str = Field("gpt-4o-mini", env="OPENAI_MODEL")

    # Legacy single-provider settings. If llm_provider is set it is promoted to
    # the head of the chain, so existing .env files keep their intended primary
    # instead of silently changing behaviour. Prefer LLM_FALLBACK_ORDER.
    llm_provider: str | None = Field(None, env="LLM_PROVIDER")
    llm_model: str | None = Field(None, env="LLM_MODEL")
    # Ignored on Claude 5 / Opus 4.7+ models, which reject sampling parameters
    # with a 400. Retained for the Gemini/OpenAI fallback paths.
    llm_temperature: float = Field(0.0, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(16000, env="LLM_MAX_TOKENS")
    # Adaptive thinking. Off by default — the agent makes many short
    # tool-routing calls where thinking is spend without benefit.
    llm_thinking_enabled: bool = Field(False, env="LLM_THINKING_ENABLED")
    # low | medium | high | xhigh | max. None leaves the server default (high).
    llm_effort: str | None = Field(None, env="LLM_EFFORT")

    # ── Embeddings ────────────────────────────────────────────────────────────
    # Anthropic has no embeddings endpoint, so vectors come from a local
    # open-source model served by Ollama.
    embedding_provider: str = Field(
        "ollama", env="EMBEDDING_PROVIDER"
    )  # "ollama" | "gemini" | "openai"
    embedding_model: str = Field("nomic-embed-text", env="EMBEDDING_MODEL")
    # MUST match the model's vector width — it is baked into the Milvus schema
    # at collection-creation time. nomic-embed-text=768, bge-m3=1024.
    embedding_dim: int = Field(768, env="EMBEDDING_DIM")
    ollama_base_url: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")

    # ── API Keys ──────────────────────────────────────────────────────────────
    anthropic_api_key: str | None = Field(None, env="ANTHROPIC_API_KEY")
    google_api_key: str | None = Field(None, env="GOOGLE_API_KEY")
    openai_api_key: str | None = Field(None, env="OPENAI_API_KEY")

    # ── Observability (Langfuse) ─────────────────────────────────────────────
    # Opt-in and fail-open: with no keys set, tracing is inert and the app is
    # unaffected. Works against Langfuse Cloud or a self-hosted instance — only
    # LANGFUSE_HOST differs between them.
    langfuse_enabled: bool = Field(True, env="LANGFUSE_ENABLED")
    langfuse_public_key: str | None = Field(None, env="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(None, env="LANGFUSE_SECRET_KEY")
    # EU cloud: https://cloud.langfuse.com — US: https://us.cloud.langfuse.com
    # Self-hosted: http://langfuse-web:3000 (see docker-compose.langfuse.yml)
    langfuse_host: str = Field("https://cloud.langfuse.com", env="LANGFUSE_HOST")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    # Values below marked "profile-derived" are overwritten by the active
    # profile's default UNLESS explicitly set via env — see model_post_init.
    hybrid_search_enabled: bool = Field(True, env="HYBRID_SEARCH_ENABLED")
    rerank_enabled: bool = Field(False, env="RERANK_ENABLED")  # profile-derived
    rerank_model: str = Field("BAAI/bge-reranker-v2-m3", env="RERANK_MODEL")
    retrieval_top_k: int = Field(50, env="RETRIEVAL_TOP_K")  # profile-derived
    rerank_top_k: int = Field(10, env="RERANK_TOP_K")  # profile-derived
    # Retrieve for several paraphrases of the query and fuse the results.
    multi_query_enabled: bool = Field(False, env="MULTI_QUERY_ENABLED")  # profile-derived
    # LLM-generated context per chunk at ingest time. Highest quality gain,
    # highest cost — one LLM call per chunk.
    contextual_retrieval_enabled: bool = Field(
        False, env="CONTEXTUAL_RETRIEVAL_ENABLED"
    )  # profile-derived

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent_max_tool_calls: int = Field(4, env="AGENT_MAX_TOOL_CALLS")  # profile-derived

    # ── Ingestion ─────────────────────────────────────────────────────────────
    # Chunks embedded per batch. Smaller batches keep peak memory lower on a
    # constrained VM at the cost of ingest throughput.
    ingest_batch_size: int = Field(32, env="INGEST_BATCH_SIZE")  # profile-derived

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(600, env="CHUNK_SIZE")
    chunk_overlap: int = Field(120, env="CHUNK_OVERLAP")

    # ── App ───────────────────────────────────────────────────────────────────
    app_host: str = Field("0.0.0.0", env="APP_HOST")
    app_port: int = Field(9090, env="APP_PORT")

    class Config:
        env_file = ".env"
        case_sensitive = False

    def model_post_init(self, __context) -> None:
        """Apply profile defaults to any field the operator didn't set explicitly.

        `model_fields_set` contains only fields that came from the environment,
        an .env file, or an explicit constructor argument — not those left at
        their declared default. That distinction is what makes a profile a
        starting point rather than an override: setting RERANK_ENABLED=true in
        .env still wins over a profile that would disable it.
        """
        from .profiles import normalize_profile, profile_defaults

        resolved = normalize_profile(self.retrieva_profile)
        # Pydantic models are not frozen here, but assigning through
        # object.__setattr__ avoids re-triggering validation mid-init.
        object.__setattr__(self, "retrieva_profile", resolved)

        explicitly_set = self.model_fields_set
        for key, value in profile_defaults(resolved).items():
            if key not in explicitly_set:
                object.__setattr__(self, key, value)

    def profile_summary(self) -> dict:
        """Active profile plus the feature flags it produced, for /health."""
        from .profiles import PROFILE_DESCRIPTIONS

        return {
            "profile": self.retrieva_profile,
            "description": PROFILE_DESCRIPTIONS.get(self.retrieva_profile, ""),
            "features": {
                "rerank_enabled": self.rerank_enabled,
                "multi_query_enabled": self.multi_query_enabled,
                "contextual_retrieval_enabled": self.contextual_retrieval_enabled,
                "hybrid_search_enabled": self.hybrid_search_enabled,
                "retrieval_top_k": self.retrieval_top_k,
                "rerank_top_k": self.rerank_top_k,
                "agent_max_tool_calls": self.agent_max_tool_calls,
                "ingest_batch_size": self.ingest_batch_size,
            },
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
