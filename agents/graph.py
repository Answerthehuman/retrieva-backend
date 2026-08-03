"""LangGraph agent graph — a dynamic tool-calling ReAct-style loop."""
from typing import List, Literal

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agents.state import AgentState


def build_agent_graph(*, llm, tools: List):
    """Compile the agent graph: agent <-> tools, looping until the LLM answers without a tool call."""
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
