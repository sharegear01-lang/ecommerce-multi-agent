"""
Structured logging setup for the MAS.
Provides agent-specific loggers with state transition tracking.
"""

import logging
import sys
from typing import Optional

from config import settings


class MASFormatter(logging.Formatter):
    """Custom formatter with agent and state context."""

    def format(self, record: logging.LogRecord) -> str:
        agent_name = getattr(record, "agent_name", "-")
        state = getattr(record, "state", "-")
        base = super().format(record)
        return f"[{agent_name}][{state}] {base}"


def setup_logger(
    name: str,
    agent_name: Optional[str] = None,
    level: Optional[str] = None,
) -> logging.Logger:
    """
    Create a configured logger instance.

    Args:
        name: Logger name (usually __name__).
        agent_name: Optional agent identifier for log prefix.
        level: Log level override (defaults to settings.LOG_LEVEL).

    Returns:
        Configured logging.Logger instance.
    """
    log_level = level or settings.LOG_LEVEL
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(MASFormatter(
            fmt="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)

    # Attach agent_name as a log record filter
    if agent_name:
        old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.agent_name = agent_name
            return record

        logging.setLogRecordFactory(record_factory)

    return logger


def get_agent_logger(agent_name: str) -> logging.Logger:
    """
    Get a logger pre-configured with an agent name.

    Args:
        agent_name: Name of the agent (e.g., 'Router', 'ShoppingGuide').

    Returns:
        Logger with agent context.
    """
    logger = logging.getLogger(f"mas.agent.{agent_name.lower()}")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(MASFormatter(
            fmt="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)

    return logger
