import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agents.prompts import SYSTEM_PROMPT
from agents.state import history_to_messages
from core.config.settings import get_settings
from core.providers import get_agent_graph
from core.utils.sse import format_sse_event

logger = logging.getLogger(__name__)


def _strip_vectors(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {k: v for k, v in doc.items() if k not in ("dense_vector", "sparse_vector", "embedding")}
        for doc in documents
    ]


class RAGPipeline:
    """Thin adapter around the compiled LangGraph agent — preserves the query() call
    signature used by cli.py and api/routes/chat.py, translating the agent's
    astream_events stream into the same SSE event contract as before."""

    def __init__(self):
        self.settings = get_settings()
        self.graph = get_agent_graph(self.settings)

    async def query(
        self,
        query_text: str,
        collection_name: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        filters: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Run the agent end-to-end, yielding SSE events.

        Args:
            query_text: The user's query.
            collection_name: Milvus collection to search.
            chat_history: Previous messages [{"role": "user", "content": "..."}, ...]
            filters: Optional caller-supplied Milvus filter expression, merged with
                any filter the agent's search tool extracts from the query itself.

        Yields:
            JSON-encoded Server-Sent Events (SSE) strings.
        """
        target_collection = collection_name or self.settings.milvus_default_collection

        messages = (
            [SystemMessage(content=SYSTEM_PROMPT)]
            + history_to_messages(chat_history)
            + [HumanMessage(content=query_text)]
        )
        initial_state = {
            "messages": messages,
            "collection_name": target_collection,
            "base_filters": filters,
        }
        config = {"recursion_limit": self.settings.agent_max_tool_calls * 2 + 1}

        current_chunks: List[str] = []
        running_accum = None

        async for event in self.graph.astream_events(initial_state, config=config, version="v2"):
            kind = event["event"]
            node = event.get("metadata", {}).get("langgraph_node")

            if kind == "on_tool_start" and event["name"] == "search_knowledge_base":
                query_arg = event.get("data", {}).get("input", {}).get("query")
                yield format_sse_event("event", {"format": "retrieval_start", "query": query_arg})

            elif kind == "on_tool_end" and event["name"] == "search_knowledge_base":
                tool_message = event.get("data", {}).get("output")
                docs = getattr(tool_message, "artifact", None) or []
                yield format_sse_event(
                    "event", {"format": "retrieval_complete", "documents": _strip_vectors(docs)}
                )

            elif kind == "on_chat_model_stream" and node == "agent":
                chunk = event["data"]["chunk"]
                running_accum = chunk if running_accum is None else running_accum + chunk
                if chunk.content:
                    current_chunks.append(chunk.content)

            elif kind == "on_chat_model_end" and node == "agent":
                # The merged chunk's .tool_calls only resolves once every
                # tool_call_chunk has arrived — this is how we know, after the
                # fact, whether this turn was a tool decision or the final answer.
                if running_accum is not None and not running_accum.tool_calls and current_chunks:
                    yield format_sse_event("event", {"format": "generation_start"})
                    for piece in current_chunks:
                        yield format_sse_event("token", {"format": "markdown", "content": piece})
                    yield format_sse_event("event", {"format": "generation_complete"})
                current_chunks = []
                running_accum = None
