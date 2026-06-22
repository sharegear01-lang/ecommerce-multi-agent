"""
Inter-Agent Communication Protocol.

Defines the JSON message envelope used for structured communication
between agents in the multi-agent system. Uses Pydantic for validation.
"""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    """
    Standardized message envelope for inter-agent communication.

    All agents use this format when passing context and task information
    to each other, ensuring information is transferred without loss.
    """

    task_type: str = Field(
        description="Type of task (e.g., product_inquiry, order_create, return_request, exchange_request)",
    )
    source_agent: Literal["router", "shopping_guide", "order_agent", "aftersales_agent"] = Field(
        description="Agent that sent this message",
    )
    target_agent: Literal["router", "shopping_guide", "order_agent", "aftersales_agent"] = Field(
        description="Intended recipient agent",
    )
    context_data: dict = Field(
        default_factory=dict,
        description="Arbitrary payload (product_ids, order_id, user_preferences, etc.)",
    )
    priority: Literal["low", "medium", "high", "urgent"] = Field(
        default="medium",
        description="Message priority level",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO 8601 timestamp of message creation",
    )
    session_id: str = Field(
        default="",
        description="Session identifier for tracking",
    )
    reply_to: Optional[str] = Field(
        default=None,
        description="Message ID this is a reply to (for chained messages)",
    )

    def to_context_dict(self) -> dict:
        """Serialize the message into a dict suitable for AgentState.context."""
        return self.model_dump()

    @classmethod
    def from_context_dict(cls, data: dict) -> "AgentMessage":
        """Deserialize from a context dict back to an AgentMessage."""
        return cls(**data)

    def summary(self) -> str:
        """Human-readable one-line summary."""
        return (
            f"[{self.priority.upper()}] {self.source_agent} → {self.target_agent}: "
            f"{self.task_type} (session: {self.session_id})"
        )
