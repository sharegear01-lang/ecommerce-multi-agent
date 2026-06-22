"""
LangGraph StateGraph construction and compilation.

Builds the hub-and-spoke multi-agent graph:
  START → router → shopping_guide ─┐
                  → order_agent    ─┼→ router → END
                  → aftersales_agent┘

All agents use the Command API to simultaneously update state
and specify the next node. The router is the central hub.
"""

import logging
from typing import Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.state.schema import AgentState
from src.agents.router_agent import RouterAgent
from src.agents.shopping_guide import ShoppingGuideAgent
from src.agents.order_agent import OrderAgent
from src.agents.aftersales_agent import AftersalesAgent

logger = logging.getLogger(__name__)


def build_graph(
    router: RouterAgent,
    shopping_guide: ShoppingGuideAgent,
    order_agent: OrderAgent,
    aftersales_agent: AftersalesAgent,
    checkpointer: Optional[MemorySaver] = None,
):
    """
    Build and compile the multi-agent StateGraph.

    Architecture:
        START → router (intent detection + dispatch)
                ├──→ shopping_guide (INQUIRY) ──→ back to router
                ├──→ order_agent (ORDER) ──────→ back to router
                └──→ aftersales_agent (AFTERSALES) → back to router
        router → END (when intent is goodbye or task_chain is empty)

    Args:
        router: The RouterAgent instance for intent classification and routing.
        shopping_guide: The ShoppingGuideAgent instance.
        order_agent: The OrderAgent instance.
        aftersales_agent: The AftersalesAgent instance.
        checkpointer: Optional LangGraph checkpointer for persistence.

    Returns:
        Compiled LangGraph application.
    """
    logger.info("Building StateGraph...")

    # Create the graph with AgentState schema
    workflow = StateGraph(AgentState)

    # Register all agent nodes
    workflow.add_node("router", router)
    workflow.add_node("shopping_guide", shopping_guide)
    workflow.add_node("order_agent", order_agent)
    workflow.add_node("aftersales_agent", aftersales_agent)

    # Entry point: always start at the router
    workflow.add_edge(START, "router")

    # All specialized agents return to router via Command(goto="router").
    # The router then decides the next step (another agent or END).
    # No static edges from agents — purely Command-driven routing.

    # Conditional edge from router: it returns Command(goto="X") or dict with final message.
    # When router returns a dict (not a Command), we check if we should end.
    # The Command-based routing handles the happy path; the conditional edge
    # catches returned dicts (e.g., when task_chain is empty and we go idle).

    # Compile with optional checkpointer
    if checkpointer:
        app = workflow.compile(checkpointer=checkpointer)
    else:
        app = workflow.compile()

    logger.info("StateGraph compiled successfully")
    return app
