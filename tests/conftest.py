"""Shared pytest fixtures.

**Invariant: this suite never makes a real API call.** Provider keys are
scrubbed from the environment before any test runs (see `_no_live_api_calls`),
so even a test that accidentally constructs a real client cannot reach a
provider and spend credits. Every LLM is a scripted fake.
"""

import json
import os
import re

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

# ── Credit safety ────────────────────────────────────────────────────────────

_PROVIDER_KEY_VARS = (
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
)


@pytest.fixture(autouse=True, scope="session")
def _no_live_api_calls():
    """Make it impossible for the suite to reach a provider.

    Two things are required, and missing either one leaks credentials:

    1. Scrub the provider env vars.
    2. **Stop Settings reading the developer's .env file.** `Settings.Config`
       declares `env_file = ".env"`, so pydantic-settings loads real API keys
       straight off disk regardless of the environment. Without this, the
       provider chain builds successfully in tests and a stray call would spend
       real money.

    autouse + session scope so it cannot be forgotten.
    """
    from core.config.settings import Settings

    saved = {k: os.environ.pop(k, None) for k in _PROVIDER_KEY_VARS}

    # Detach Settings from the on-disk .env for the whole session.
    original_env_file = getattr(Settings.Config, "env_file", None)
    Settings.Config.env_file = None
    # pydantic-settings v2 mirrors the legacy Config into model_config; set both
    # so neither path can resurrect the file.
    original_model_config_env = None
    if hasattr(Settings, "model_config") and isinstance(Settings.model_config, dict):
        original_model_config_env = Settings.model_config.get("env_file")
        Settings.model_config["env_file"] = None

    # Belt and braces: keep Langfuse inert even if a key leaks in another way.
    os.environ["LANGFUSE_ENABLED"] = "false"
    # Tests must never touch a real Postgres.
    os.environ["USE_SQLITE"] = "true"

    yield

    Settings.Config.env_file = original_env_file
    if original_model_config_env is not None:
        Settings.model_config["env_file"] = original_model_config_env
    for key, value in saved.items():
        if value is not None:
            os.environ[key] = value


def test_credit_safety_is_active():
    """Guard the guard.

    If this fails, the suite can reach a real provider and the "no credits"
    promise is broken — that is a stop-the-line failure, not a flaky test.
    """
    from core.config.settings import Settings

    settings = Settings()
    assert not settings.anthropic_api_key, "ANTHROPIC_API_KEY leaked into tests"
    assert not settings.google_api_key, "GOOGLE_API_KEY leaked into tests"
    assert not settings.openai_api_key, "OPENAI_API_KEY leaked into tests"


@pytest.fixture(autouse=True)
def _reset_provider_singletons():
    """Clear cached provider/graph singletons between tests.

    `core.providers` caches the LLM, embeddings, retriever, reranker, and
    compiled graph at module level. Without this reset, the first test's stubs
    leak into every later test and failures become order-dependent.
    """
    import core.observability as obs
    import core.providers as providers

    def _clear():
        providers._llm = None
        providers._llm_providers = None
        providers._embeddings = None
        providers._retriever = None
        providers._reranker = None
        providers._agent_graph = None
        obs._handler = None
        obs._resolved = False

    _clear()
    yield
    _clear()


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """`get_settings()` is lru_cached; drop it so per-test env changes apply."""
    from core.config.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ── Fake LLM ─────────────────────────────────────────────────────────────────


class ScriptedChatModel(FakeMessagesListChatModel):
    """Fake chat model that actually streams, including tool-call chunks.

    The stock fake only implements non-streaming `_generate`, which would make
    `astream_events` skip `on_chat_model_stream` entirely — so the SSE mapping
    under test would never run. Real providers stream, so this must too.
    """

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002 - signature parity
        return self

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        response = self.responses[self.i]
        self.i = self.i + 1 if self.i < len(self.responses) - 1 else 0

        if response.tool_calls:
            for idx, tc in enumerate(response.tool_calls):
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "name": tc["name"],
                                "args": json.dumps(tc["args"]),
                                "id": tc["id"],
                                "index": idx,
                            }
                        ],
                    )
                )
        else:
            emitted = False
            for token in re.split(r"(\s)", response.content):
                if token:
                    emitted = True
                    yield ChatGenerationChunk(message=AIMessageChunk(content=token))
            if not emitted:
                # Real providers always emit at least one chunk, even for an
                # empty completion. Yielding none makes langchain-core raise
                # "No generations found in stream" before our code is reached,
                # which would be an artefact of the fake rather than behaviour
                # worth testing.
                yield ChatGenerationChunk(message=AIMessageChunk(content=""))


@pytest.fixture
def scripted_llm():
    """Factory: `scripted_llm([AIMessage(...), ...])`."""

    def _make(responses: list[AIMessage]) -> ScriptedChatModel:
        return ScriptedChatModel(responses=responses)

    return _make


# ── Stub retrieval layer ─────────────────────────────────────────────────────


class StubRetriever:
    """Returns canned documents keyed on the real `content` field.

    Uses `content` (not `text`) deliberately — a reranker reading the wrong key
    was a real bug that silently disabled reranking, and these fixtures must
    reflect the true document shape so that class of bug stays catchable.
    """

    def __init__(self, documents=None):
        self.documents = (
            documents
            if documents is not None
            else [
                {
                    "id": "doc-1",
                    "score": 0.9,
                    "content": "Retrieva is a hybrid RAG system.",
                    "source": "readme.md",
                    "page": 1,
                },
                {
                    "id": "doc-2",
                    "score": 0.8,
                    "content": "It uses Milvus for vector search.",
                    "source": "readme.md",
                    "page": 2,
                },
            ]
        )
        self.calls = []

    async def search_parallel(self, steps, *, output_fields=None, bm25_loader=None, filters=None):
        self.calls.append({"steps": steps, "filters": filters})
        return self.documents


@pytest.fixture
def stub_retriever():
    return StubRetriever()


@pytest.fixture
def stub_settings():
    """Minimal settings object for components that only read a few fields."""

    class StubSettings:
        hybrid_search_enabled = False
        agent_max_tool_calls = 4
        milvus_default_collection = "test_collection"
        rerank_enabled = False
        retrieval_top_k = 30
        rerank_top_k = 10
        langfuse_enabled = False
        langfuse_public_key = None
        langfuse_secret_key = None
        langfuse_host = "https://cloud.langfuse.com"

    return StubSettings()
