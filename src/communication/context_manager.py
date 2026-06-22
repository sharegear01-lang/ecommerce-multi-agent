"""
Context Manager — handles serialization and deserialization of cross-agent
context data, preventing information loss when control passes between agents.
"""

import logging
from typing import Optional, Any
from datetime import datetime

from src.communication.protocol import AgentMessage

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages the shared context dict in AgentState.

    Responsibilities:
    1. Package agent output into the context dict for the next agent to consume.
    2. Extract relevant data from context for the current agent's use.
    3. Version and tag context entries to track data lineage.
    """

    CONTEXT_VERSION_KEY = "_context_version"
    CONTEXT_HISTORY_KEY = "_context_history"

    def __init__(self):
        self.version = 0

    def pack_context(
        self,
        current_context: dict,
        source_agent: str,
        task_type: str,
        data: dict,
        session_id: str = "",
        priority: str = "medium",
    ) -> dict:
        """
        Package data into context for the next agent.

        Args:
            current_context: The existing AgentState.context dict.
            source_agent: Name of the agent writing this context.
            task_type: Type of task being handed off.
            data: The payload data for the next agent.
            session_id: Session identifier.
            priority: Message priority.

        Returns:
            Updated context dict with new data merged in.
        """
        self.version += 1

        # Create an inter-agent message envelope
        message = AgentMessage(
            task_type=task_type,
            source_agent=source_agent,
            target_agent="router",  # Always goes through router
            context_data=data,
            priority=priority,
            session_id=session_id,
        )

        # Build the new context
        new_context = {**current_context}
        new_context.update(data)  # Merge payload fields directly
        new_context["_last_message"] = message.model_dump()
        new_context[self.CONTEXT_VERSION_KEY] = self.version

        # Track history
        history = current_context.get(self.CONTEXT_HISTORY_KEY, [])
        history.append({
            "version": self.version,
            "source": source_agent,
            "task_type": task_type,
            "timestamp": datetime.now().isoformat(),
        })
        new_context[self.CONTEXT_HISTORY_KEY] = history[-10:]  # Keep last 10 entries

        logger.debug(
            "Context packed by %s: task=%s, version=%d, keys=%s",
            source_agent, task_type, self.version, list(data.keys()),
        )
        return new_context

    def unpack_context(
        self,
        context: dict,
        expected_agent: str,
    ) -> dict:
        """
        Extract relevant data from context for the current agent.

        Args:
            context: The AgentState.context dict.
            expected_agent: The agent that should be consuming this context.

        Returns:
            Dict of relevant data fields for the agent.
        """
        # Return all user-facing data, stripping internal bookkeeping
        data = {
            k: v for k, v in context.items()
            if not k.startswith("_")
        }
        logger.debug(
            "Context unpacked for %s: keys=%s",
            expected_agent, list(data.keys()),
        )
        return data

    def get_last_message(self, context: dict) -> Optional[dict]:
        """Retrieve the last inter-agent message from context."""
        return context.get("_last_message")

    def get_history(self, context: dict) -> list:
        """Retrieve the context history trail."""
        return context.get(self.CONTEXT_HISTORY_KEY, [])

    def clear_context(self, context: dict) -> dict:
        """Reset the context while preserving version tracking."""
        return {
            self.CONTEXT_VERSION_KEY: context.get(self.CONTEXT_VERSION_KEY, 0),
            self.CONTEXT_HISTORY_KEY: context.get(self.CONTEXT_HISTORY_KEY, []),
        }
