"""Graph state for the Retrieva agent."""
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    messages: full conversation, including tool calls/results. Merged via add_messages
        so nodes can return just the new message(s) rather than the whole list.
    collection_name: Milvus collection to search — request-scoped, injected into tools
        via InjectedState so the compiled graph/tools can be built once as a singleton.
    base_filters: caller-supplied Milvus filter expression override (e.g. from a UI
        filter picker), never chosen by the LLM. Merged with LLM-extracted filters
        inside the search tool.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    collection_name: str
    base_filters: Optional[str]


def history_to_messages(chat_history: Optional[List[Dict[str, Any]]]) -> List[BaseMessage]:
    """Convert [{"role": "user"|"assistant", "content": str}, ...] to LangChain messages."""
    messages: List[BaseMessage] = []
    if not chat_history:
        return messages
    for msg in chat_history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages
