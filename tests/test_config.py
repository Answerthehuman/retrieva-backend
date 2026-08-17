"""Settings, runtime profiles, prompts, and the collection-name sanitiser."""

import pytest

from agents.prompts import AVAILABLE_MODES, BASE_SYSTEM_PROMPT, build_system_prompt
from core.config.profiles import FULL, LITE, STANDARD, normalize_profile, profile_defaults
from core.config.settings import Settings


class TestProfiles:
    def test_known_profiles_pass_through(self):
        assert normalize_profile("lite") == LITE
        assert normalize_profile("standard") == STANDARD
        assert normalize_profile("full") == FULL

    @pytest.mark.parametrize("value", ["FULL", " full ", "Full"])
    def test_case_and_whitespace_tolerated(self, value):
        assert normalize_profile(value) == FULL

    @pytest.mark.parametrize("value", ["", "enormous", None])
    def test_unknown_falls_back_to_lite(self, value):
        """Falling back to the *smallest* tier is the safe default: an unknown
        profile on a constrained host must not over-commit memory."""
        assert normalize_profile(value) == LITE

    def test_resource_tiers_are_ordered(self):
        """Each tier must be at least as generous as the one below it."""
        lite, standard, full = (profile_defaults(p) for p in (LITE, STANDARD, FULL))
        for key in ("retrieval_top_k", "agent_max_tool_calls", "ingest_batch_size"):
            assert lite[key] <= standard[key] <= full[key], f"{key} not monotonic across tiers"

    def test_expensive_features_off_in_lite(self):
        lite = profile_defaults(LITE)
        assert lite["rerank_enabled"] is False
        assert lite["multi_query_enabled"] is False
        assert lite["contextual_retrieval_enabled"] is False

    def test_full_enables_everything(self):
        full = profile_defaults(FULL)
        assert full["rerank_enabled"] is True
        assert full["multi_query_enabled"] is True
        assert full["contextual_retrieval_enabled"] is True

    def test_defaults_are_copies(self):
        """Callers must not be able to mutate the shared profile table."""
        first = profile_defaults(LITE)
        first["retrieval_top_k"] = 9999
        assert profile_defaults(LITE)["retrieval_top_k"] != 9999


class TestSettingsProfileApplication:
    def test_profile_drives_features(self):
        assert Settings(_env_file=None, retrieva_profile="lite").rerank_enabled is False
        assert Settings(_env_file=None, retrieva_profile="full").rerank_enabled is True

    def test_explicit_value_beats_profile(self):
        """A profile is a starting point, not a straitjacket — an operator who
        sets a value explicitly must win."""
        s = Settings(_env_file=None, retrieva_profile="lite", rerank_enabled=True)
        assert s.rerank_enabled is True

        s2 = Settings(_env_file=None, retrieva_profile="full", rerank_enabled=False)
        assert s2.rerank_enabled is False

    def test_explicit_numeric_override(self):
        s = Settings(_env_file=None, retrieva_profile="lite", retrieval_top_k=999)
        assert s.retrieval_top_k == 999

    def test_env_var_beats_profile(self, monkeypatch):
        monkeypatch.setenv("RERANK_ENABLED", "true")
        assert Settings(_env_file=None, retrieva_profile="lite").rerank_enabled is True

    def test_invalid_profile_normalised_on_instance(self):
        assert Settings(_env_file=None, retrieva_profile="bogus").retrieva_profile == "lite"

    def test_profile_summary_shape(self):
        summary = Settings(_env_file=None, retrieva_profile="standard").profile_summary()
        assert summary["profile"] == "standard"
        assert summary["description"]
        for key in ("rerank_enabled", "multi_query_enabled", "retrieval_top_k"):
            assert key in summary["features"]

    def test_embedding_model_is_not_profile_dependent(self):
        """The embedding dimension is baked into the Milvus collection, so it
        must be identical across profiles — otherwise switching profiles would
        invalidate the index or fail on insert."""
        models = {
            Settings(_env_file=None, retrieva_profile=p).embedding_model
            for p in (LITE, STANDARD, FULL)
        }
        dims = {
            Settings(_env_file=None, retrieva_profile=p).embedding_dim
            for p in (LITE, STANDARD, FULL)
        }
        assert len(models) == 1, f"embedding model varies across profiles: {models}"
        assert len(dims) == 1, f"embedding dim varies across profiles: {dims}"


class TestSystemPrompts:
    def test_no_mode_returns_base(self):
        assert build_system_prompt(None) == BASE_SYSTEM_PROMPT

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_blank_returns_base(self, value):
        assert build_system_prompt(value) == BASE_SYSTEM_PROMPT

    def test_unknown_mode_degrades_to_base(self):
        """A stale client sending a retired mode must not break the request."""
        assert build_system_prompt("telepathy") == BASE_SYSTEM_PROMPT

    @pytest.mark.parametrize("mode", AVAILABLE_MODES)
    def test_each_mode_extends_base(self, mode):
        prompt = build_system_prompt(mode)
        assert BASE_SYSTEM_PROMPT in prompt
        assert len(prompt) > len(BASE_SYSTEM_PROMPT)
        assert "Active mode" in prompt

    @pytest.mark.parametrize("mode", AVAILABLE_MODES)
    def test_mode_lookup_is_case_insensitive(self, mode):
        assert build_system_prompt(mode.upper()) == build_system_prompt(mode)

    def test_modes_are_distinct(self):
        prompts = {m: build_system_prompt(m) for m in AVAILABLE_MODES}
        assert len(set(prompts.values())) == len(AVAILABLE_MODES), "two modes share a prompt"

    def test_base_prompt_requests_markdown(self):
        """The frontend renders markdown; the model must be told to emit it."""
        assert "markdown" in BASE_SYSTEM_PROMPT.lower()


class TestCollectionNameSanitiser:
    """Swagger's 'Try it out' pre-fills every optional string field with the
    literal text `string`. Submitting that created a real Milvus collection
    named "string" that chat never searched — a document went missing that way.
    """

    @pytest.mark.parametrize("value", ["string", "  string  ", "", "   ", None])
    def test_placeholder_and_blank_become_none(self, value):
        from api.routes.ingest import _sanitize_collection_name

        assert _sanitize_collection_name(value) is None

    @pytest.mark.parametrize("value", ["my_docs", "  my_docs  ", "strings", "String_Theory"])
    def test_real_names_preserved_and_trimmed(self, value):
        from api.routes.ingest import _sanitize_collection_name

        assert _sanitize_collection_name(value) == value.strip()


class TestEmbeddingDimensions:
    def test_known_models(self):
        from core.llm.ollama import expected_dimension

        assert expected_dimension("nomic-embed-text") == 768
        assert expected_dimension("bge-m3") == 1024

    def test_tag_suffix_tolerated(self):
        from core.llm.ollama import expected_dimension

        assert expected_dimension("bge-m3:latest") == 1024

    def test_unknown_model_returns_none(self):
        """Unknown means 'cannot verify', not 'wrong' — the caller decides."""
        from core.llm.ollama import expected_dimension

        assert expected_dimension("some-new-model") is None
