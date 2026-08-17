"""Agent graph wiring and the astream_events → SSE mapping.

Ported from scripts/agent_smoke_test.py. Covers the two shapes that matter:
a turn with no tool call (chit-chat) and a turn that searches then answers.
"""

import json

import pytest
from langchain_core.messages import AIMessage

from agents.graph import build_agent_graph
from agents.tools import build_tools
from services.rag.pipeline.rag_pipeline import RAGPipeline


def _make_pipeline(fake_llm, retriever, settings) -> RAGPipeline:
    tools = build_tools(retriever=retriever, reranker=None, llm=None, settings=settings)
    graph = build_agent_graph(llm=fake_llm, tools=tools)

    # Bypass __init__: it resolves real providers, which the credit-safety
    # fixture has deliberately made unavailable.
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.settings = settings
    pipeline.graph = graph
    return pipeline, tools


async def _collect(pipeline, query_text, **kwargs):
    """Run a turn and tally the SSE events it emits."""
    counts = {
        "retrieval_start": 0,
        "retrieval_complete": 0,
        "generation_start": 0,
        "generation_complete": 0,
        "tokens": [],
        "documents": [],
    }
    async for line in pipeline.query(query_text=query_text, collection_name="test", **kwargs):
        assert line.startswith("data: "), f"malformed SSE line: {line!r}"
        payload = json.loads(line[6:])
        if payload.get("type") == "token":
            counts["tokens"].append(payload["content"])
        else:
            fmt = payload.get("format")
            if fmt in counts:
                counts[fmt] += 1
            if fmt == "retrieval_complete":
                counts["documents"].extend(payload.get("documents", []))
    return counts


class TestInjectedState:
    def test_state_hidden_from_llm_schema(self, stub_retriever, stub_settings):
        """`state` is injected by LangGraph and must not appear in the tool schema.

        If it leaked, the LLM would try to fabricate the whole agent state as a
        tool argument.
        """
        tools = build_tools(
            retriever=stub_retriever, reranker=None, llm=None, settings=stub_settings
        )
        schema = tools[0].tool_call_schema.model_json_schema()
        assert "state" not in schema["properties"]
        assert "query" in schema["properties"]


class TestChitChat:
    async def test_no_retrieval_for_greeting(self, scripted_llm, stub_retriever, stub_settings):
        llm = scripted_llm([AIMessage(content="Hi there! How can I help?")])
        pipeline, _ = _make_pipeline(llm, stub_retriever, stub_settings)

        counts = await _collect(pipeline, "hi, how are you?")

        assert counts["retrieval_start"] == 0
        assert counts["retrieval_complete"] == 0
        assert counts["generation_start"] == 1
        assert counts["generation_complete"] == 1
        assert "".join(counts["tokens"]) == "Hi there! How can I help?"
        # The retriever must not have been touched at all.
        assert stub_retriever.calls == []


class TestToolCallThenAnswer:
    async def test_search_then_answer(self, scripted_llm, stub_retriever, stub_settings):
        llm = scripted_llm(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_knowledge_base",
                            "args": {"query": "what is Retrieva"},
                            "id": "call_1",
                        }
                    ],
                ),
                AIMessage(content="Retrieva is a hybrid RAG system (readme.md, p.1)."),
            ]
        )
        pipeline, _ = _make_pipeline(llm, stub_retriever, stub_settings)

        counts = await _collect(pipeline, "what is Retrieva?")

        assert counts["retrieval_start"] == 1
        assert counts["retrieval_complete"] == 1
        assert counts["generation_start"] == 1
        assert counts["generation_complete"] == 1
        assert "".join(counts["tokens"]) == "Retrieva is a hybrid RAG system (readme.md, p.1)."
        assert len(stub_retriever.calls) == 1

    async def test_documents_reach_the_client(self, scripted_llm, stub_retriever, stub_settings):
        """Retrieved docs must ride the retrieval_complete event as citations."""
        llm = scripted_llm(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "search_knowledge_base", "args": {"query": "q"}, "id": "c1"}
                    ],
                ),
                AIMessage(content="Answer."),
            ]
        )
        pipeline, _ = _make_pipeline(llm, stub_retriever, stub_settings)

        counts = await _collect(pipeline, "what is Retrieva?")

        assert len(counts["documents"]) == 2
        assert {d["id"] for d in counts["documents"]} == {"doc-1", "doc-2"}

    async def test_vectors_stripped_from_sse(self, scripted_llm, stub_settings):
        """Embeddings must never be serialised to the client — huge and useless."""
        from tests.conftest import StubRetriever

        retriever = StubRetriever(
            [
                {
                    "id": "doc-1",
                    "content": "text",
                    "source": "a.md",
                    "page": 1,
                    "embedding": [0.1] * 768,
                    "sparse_vector": {1: 0.5},
                    "dense_vector": [0.2] * 768,
                }
            ]
        )
        llm = scripted_llm(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "search_knowledge_base", "args": {"query": "q"}, "id": "c1"}
                    ],
                ),
                AIMessage(content="Answer."),
            ]
        )
        pipeline, _ = _make_pipeline(llm, retriever, stub_settings)

        counts = await _collect(pipeline, "q")

        doc = counts["documents"][0]
        for banned in ("embedding", "sparse_vector", "dense_vector"):
            assert banned not in doc, f"{banned} leaked into the SSE payload"
        assert doc["content"] == "text"


class TestTerminalEventGuarantee:
    async def test_always_emits_terminal_events(self, scripted_llm, stub_retriever, stub_settings):
        """A turn that searches but never produces text must still close the stream.

        This is the realistic shape of the failure: the agent calls a tool and
        then emits nothing usable. Without the fallback the frontend spinner
        hangs forever waiting for generation_complete.
        """
        llm = scripted_llm(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "search_knowledge_base", "args": {"query": "q"}, "id": "c1"}
                    ],
                ),
                AIMessage(content=""),
            ]
        )
        pipeline, _ = _make_pipeline(llm, stub_retriever, stub_settings)

        counts = await _collect(pipeline, "...")

        assert counts["generation_start"] == 1
        assert counts["generation_complete"] == 1
        # The fallback message, not silence.
        assert "".join(counts["tokens"]).strip() != ""


class TestModeThreading:
    @pytest.mark.parametrize("mode", ["summarise", "insights", "analyse", "explain"])
    async def test_mode_does_not_alter_user_message(
        self, scripted_llm, stub_retriever, stub_settings, mode
    ):
        """A mode must ride the system prompt, never the user's text.

        If it were prepended to the message it would appear in the transcript
        and be replayed as user text on later turns.
        """
        captured = {}

        llm = scripted_llm([AIMessage(content="ok")])
        original_astream = llm._astream

        async def spy(messages, *args, **kwargs):
            captured["messages"] = messages
            async for chunk in original_astream(messages, *args, **kwargs):
                yield chunk

        llm._astream = spy
        pipeline, _ = _make_pipeline(llm, stub_retriever, stub_settings)

        await _collect(pipeline, "Summarise the report", mode=mode)

        msgs = captured["messages"]
        human = [m for m in msgs if m.type == "human"]
        system = [m for m in msgs if m.type == "system"]

        assert human[-1].content == "Summarise the report", "user message was mutated"
        assert "Active mode" in system[0].content
