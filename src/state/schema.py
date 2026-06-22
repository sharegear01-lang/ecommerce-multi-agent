"""
Shared AgentState TypedDict — the single state object flowing through
every node in the LangGraph StateGraph.

Uses LangGraph's `add_messages` reducer for chat history accumulation
and plain overwrite semantics for other fields.
"""

from typing import TypedDict, Annotated, Optional, Any
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """
    Shared state for the multi-agent system.

    Fields:
        messages: Accumulated chat history (HumanMessage, AIMessage, etc.).
                  Uses add_messages reducer — new messages are appended.
        current_state: Current state machine state tag.
                       One of: IDLE, INQUIRY, ORDER, AFTERSALES, CROSS_AGENT.
        intent: Detected user intent label (e.g., "product_inquiry", "return_request").
        context: Cross-agent payload dictionary. Source agent writes data here;
                 target agent reads it. Overwritten each time.
        task_chain: Ordered list of pending tasks for cross-agent workflows.
                    Router pops the first task and dispatches to the appropriate agent.
        conflict_log: List of conflicting recommendations for arbitration.
                      Each entry is a dict: {"issue": str, "recommendations": list}.
        user_id: Session user identifier.
        session_id: Session identifier for multi-turn tracking.
    """

    # Chat history — appended via LangGraph's add_messages reducer
    messages: Annotated[list, add_messages]

    # State machine tracking
    current_state: str  # IDLE | INQUIRY | ORDER | AFTERSALES | CROSS_AGENT

    # Intent classification result
    intent: str

    # Cross-agent payload (overwritten by each agent)
    context: dict

    # Pending cross-agent task chain
    task_chain: list

    # Conflict resolution log
    conflict_log: list

    # Session identifiers
    user_id: str
    session_id: str


# Default initial state for a new session
def get_initial_state(user_id: str = "default_user", session_id: str = "") -> AgentState:
    """Return a fresh AgentState for a new conversation session."""
    import uuid
    return AgentState(
        messages=[],
        current_state="IDLE",
        intent="",
        context={},
        task_chain=[],
        conflict_log=[],
        user_id=user_id,
        session_id=session_id or str(uuid.uuid4())[:8],
    )
