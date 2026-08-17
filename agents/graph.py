"""LangGraph agent graph — a dynamic tool-calling ReAct-style loop."""

from typing import Literal

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agents.state import AgentState


def build_agent_graph(*, tools: list, llm=None, llm_with_tools=None):
    """Compile the agent graph: agent <-> tools, looping until the LLM answers without a tool call.

    Pass either `llm` (a chat model; tools are bound here) or `llm_with_tools`
    (already bound). The latter exists because the provider fallback chain is a
    RunnableWithFallbacks, which has no .bind_tools() — tools must be bound to
    each concrete provider before the chain is assembled.
    """
    if llm_with_tools is None:
        if llm is None:
            raise ValueError("build_agent_graph requires either llm or llm_with_tools")
        llm_with_tools = llm.bind_tools(tools)

    async def agent_node(state: AgentState) -> dict:
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()
