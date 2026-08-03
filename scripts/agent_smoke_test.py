"""
Smoke test for the Retrieva agent graph and its SSE event mapping.

Exercises, with the LLM/retriever mocked (no live Milvus/Redis/API keys needed):
  - graph compilation and tool binding (agents/graph.py, agents/tools.py)
  - InjectedState hides `state` from the LLM-visible tool schema
  - RAGPipeline.query()'s astream_events -> SSE mapping, for both a no-tool
    (chit-chat) turn and a one-tool-call-then-answer turn
  - the reranker field-name fix, in isolation

Run: poetry run python scripts/agent_smoke_test.py
"""
import asyncio
import json
import sys

import re

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGenerationChunk

from agents.graph import build_agent_graph
from agents.tools import build_tools
from services.rag.pipeline.rag_pipeline import RAGPipeline


class ScriptedChatModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel + real token/tool-call streaming (_astream) and a
    no-op bind_tools. The base fake only implements non-streaming _generate, which
    would make astream_events skip on_chat_model_stream entirely — real streaming
    providers (ChatGoogleGenerativeAI, ChatOpenAI) DO stream, so this exercises the
    same code path RAGPipeline.query() relies on in production."""

    def bind_tools(self, tools, **kwargs):
        return self

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        response = self.responses[self.i]
        self.i = self.i + 1 if self.i < len(self.responses) - 1 else 0

        if response.tool_calls:
            for idx, tc in enumerate(response.tool_calls):
                chunk = AIMessageChunk(content="", tool_call_chunks=[{
                    "name": tc["name"], "args": json.dumps(tc["args"]), "id": tc["id"], "index": idx,
                }])
                yield ChatGenerationChunk(message=chunk)
        else:
            for token in re.split(r"(\s)", response.content):
                if token:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=token))


class StubSettings:
    hybrid_search_enabled = False
    agent_max_tool_calls = 4
    milvus_default_collection = "test_collection"


class StubRetriever:
    async def search_parallel(self, steps, *, output_fields=None, bm25_loader=None, filters=None):
        return [
            {"id": "doc-1", "score": 0.9, "content": "Retrieva is a hybrid RAG system.",
             "source": "readme.md", "page": 1},
            {"id": "doc-2", "score": 0.8, "content": "It uses Milvus for vector search.",
             "source": "readme.md", "page": 2},
        ]


def _make_pipeline(fake_llm) -> RAGPipeline:
    tools = build_tools(retriever=StubRetriever(), reranker=None, llm=None, settings=StubSettings())
    # tool.args reflects the full pydantic schema (includes injected params); the LLM-visible
    # schema used by bind_tools()/convert_to_openai_tool() is tool.tool_call_schema, which
    # InjectedState correctly prunes `state` from.
    assert "state" not in tools[0].tool_call_schema.model_json_schema()["properties"], (
        "InjectedState param leaked into the LLM-visible tool schema"
    )

    graph = build_agent_graph(llm=fake_llm, tools=tools)

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.settings = StubSettings()
    pipeline.graph = graph
    return pipeline


async def _tally(pipeline: RAGPipeline, query_text: str) -> dict:
    counts = {"retrieval_start": 0, "retrieval_complete": 0, "generation_start": 0,
              "generation_complete": 0, "tokens": []}

    async for sse_line in pipeline.query(query_text=query_text, collection_name="test_collection"):
        line = sse_line.strip()
        assert line.startswith("data: "), f"malformed SSE line: {line!r}"
        payload = json.loads(line[6:])
        if payload.get("type") == "token":
            counts["tokens"].append(payload["content"])
        else:
            fmt = payload.get("format")
            if fmt in counts:
                counts[fmt] += 1

    return counts


async def case_chit_chat():
    fake_llm = ScriptedChatModel(responses=[AIMessage(content="Hi there! How can I help?")])
    pipeline = _make_pipeline(fake_llm)
    counts = await _tally(pipeline, "hi, how are you?")

    assert counts["retrieval_start"] == 0, counts
    assert counts["retrieval_complete"] == 0, counts
    assert counts["generation_start"] == 1, counts
    assert counts["generation_complete"] == 1, counts
    assert "".join(counts["tokens"]) == "Hi there! How can I help?", counts
    print("PASS: chit-chat (no tool call) ->", counts)


async def case_tool_call_then_answer():
    fake_llm = ScriptedChatModel(responses=[
        AIMessage(content="", tool_calls=[
            {"name": "search_knowledge_base", "args": {"query": "what is Retrieva"}, "id": "call_1"},
        ]),
        AIMessage(content="Retrieva is a hybrid RAG system (readme.md, p.1)."),
    ])
    pipeline = _make_pipeline(fake_llm)
    counts = await _tally(pipeline, "what is Retrieva?")

    assert counts["retrieval_start"] == 1, counts
    assert counts["retrieval_complete"] == 1, counts
    assert counts["generation_start"] == 1, counts
    assert counts["generation_complete"] == 1, counts
    assert "".join(counts["tokens"]) == "Retrieva is a hybrid RAG system (readme.md, p.1).", counts
    print("PASS: tool call then answer ->", counts)


async def case_reranker_fix():
    from services.rag.retrieval.reranker import Reranker

    class StubModel:
        def predict(self, pairs, batch_size=20):
            # Higher score for the doc whose content mentions "Milvus"
            return [1.0 if "Milvus" in text else 0.5 for _, text in pairs]

    reranker = Reranker()
    reranker._model = StubModel()  # skip real model load

    docs = [
        {"id": "a", "content": "Retrieva is a hybrid RAG system."},
        {"id": "b", "content": "It uses Milvus for vector search."},
    ]
    reranked = await reranker.rerank("vector search", docs)
    assert reranked[0]["id"] == "b", reranked  # the Milvus doc should sort first
    assert all("rerank_score" in d for d in reranked), reranked
    print("PASS: reranker field-name fix ->", [d["id"] for d in reranked])


async def main():
    await case_chit_chat()
    await case_tool_call_then_answer()
    await case_reranker_fix()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        sys.exit(1)
