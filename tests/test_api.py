"""HTTP contract tests.

These exercise the real FastAPI app with no live LLM, Milvus, or Postgres —
enough to catch request/response shape breakage, which is what actually breaks
the frontend.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Module-scoped: entering TestClient runs the app's full startup
    (provider warm-up + Milvus consistency check), which costs seconds. These
    are read-only contract tests, so one app instance per module is plenty —
    per-test startup made the suite ~4x slower for no added coverage."""
    from api.main import app

    # raise_server_exceptions=False so a handler raising produces a 500
    # response to assert on, rather than propagating into the test.
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestHealth:
    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_shape(self, client):
        body = client.get("/health").json()
        for key in ("service", "status", "version", "checks", "config"):
            assert key in body, f"missing top-level key: {key}"
        assert body["service"] == "Retrieva"
        assert body["status"] in {"ok", "degraded"}

    def test_reports_every_subsystem(self, client):
        checks = client.get("/health").json()["checks"]
        for subsystem in ("database", "milvus", "llm", "embeddings"):
            assert subsystem in checks
            assert "status" in checks[subsystem]

    def test_reports_active_profile(self, client):
        """The UI uses this to explain why reranking is off in lite."""
        body = client.get("/health").json()
        assert "profile" in body
        assert body["profile"]["profile"] in {"lite", "standard", "full"}
        assert "features" in body["profile"]

    def test_llm_unconfigured_without_keys(self, client):
        """The credit-safety fixture strips provider keys, so the chain must
        report itself unconfigured rather than pretending to work."""
        llm = client.get("/health").json()["checks"]["llm"]
        assert llm["status"] == "unconfigured"
        assert "chain" in llm


class TestChatValidation:
    def test_empty_message_rejected(self, client):
        r = client.post("/chat/sessions/abc/messages", json={"message": ""})
        assert r.status_code == 422

    def test_missing_message_rejected(self, client):
        assert client.post("/chat/sessions/abc/messages", json={}).status_code == 422

    def test_mode_is_accepted(self, client):
        """Unknown fields would 422; this proves `mode` is part of the schema."""
        r = client.post(
            "/chat/sessions/abc/messages",
            json={"message": "hi", "mode": "insights"},
        )
        assert r.status_code != 422

    def test_unknown_mode_not_rejected_at_validation(self, client):
        """Unknown modes degrade to normal chat server-side rather than 422 —
        a stale client must not break."""
        r = client.post(
            "/chat/sessions/abc/messages",
            json={"message": "hi", "mode": "telepathy"},
        )
        assert r.status_code != 422


class TestOpenAPIContract:
    """The frontend is written against this schema; drift breaks it silently."""

    def test_send_message_fields(self, client):
        schema = client.get("/openapi.json").json()
        props = schema["components"]["schemas"]["SendMessageRequest"]["properties"]
        for field in ("message", "collection_name", "filters", "mode"):
            assert field in props, f"SendMessageRequest lost field: {field}"

    def test_ingest_endpoint_present(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        assert "/ingest/upload" in paths

    def test_documented_routes_present(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        for route in ("/health", "/chat/sessions", "/ingest/upload"):
            assert route in paths, f"route disappeared from the API: {route}"


class TestIngestValidation:
    def test_upload_requires_a_file(self, client):
        assert client.post("/ingest/upload").status_code == 422
