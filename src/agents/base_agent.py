"""
Base Agent — abstract foundation for all specialized agents.

Provides the common RAG-augmented chain pattern, shared utility methods,
and the interface contract that all agents must fulfill.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.rag.retriever import Retriever
from src.state.schema import AgentState

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base for all agents in the system.

    Each subclass must define:
    - agent_name: unique identifier string
    - system_prompt: the system-level instruction template
    - _build_chain(): constructs the LangChain runnable
    """

    agent_name: str = "base"

    # Maximum chat history messages to include (to stay within context window)
    MAX_HISTORY_MESSAGES = 10

    def __init__(self, llm, retriever: Retriever):
        """
        Args:
            llm: A LangChain chat model instance (e.g., ChatOpenAI).
            retriever: Domain-specific retriever for RAG.
        """
        self.llm = llm
        self.retriever = retriever
        self.chain = self._build_chain()
        logger.info("Agent '%s' initialized", self.agent_name)

    @abstractmethod
    def _build_chain(self) -> Runnable:
        """
        Build the LangChain chain for this agent.

        Typical pattern:
            prompt | rag_context | llm | output_parser

        Returns:
            A LangChain Runnable that takes state dict and returns response dict.
        """
        ...

    def __call__(self, state: AgentState) -> dict:
        """
        Process the current state and return a partial state update.

        This is the LangGraph node interface. Each agent receives the full
        shared state, processes it, and returns a dict of fields to update.

        Args:
            state: The current AgentState.

        Returns:
            Dict of state fields to update (merged into shared state).
        """
        logger.info(
            "[%s] Processing — state: %s, intent: %s, task_chain: %s",
            self.agent_name,
            state.get("current_state", "?"),
            state.get("intent", "?"),
            state.get("task_chain", []),
        )

        try:
            result = self._process(state)
            return result
        except Exception as e:
            logger.error("[%s] Error: %s", self.agent_name, str(e), exc_info=True)
            return {
                "messages": [
                    AIMessage(content=f"[{self.agent_name}] 处理请求时出现错误，请稍后重试。")
                ],
            }

    @abstractmethod
    def _process(self, state: AgentState) -> dict:
        """
        Core processing logic — implemented by each agent.

        Args:
            state: The current AgentState.

        Returns:
            Dict of state updates.
        """
        ...

    def _get_user_message(self, state: AgentState) -> str:
        """Extract the last user message content from state."""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        return ""

    def _get_history_messages(self, state: AgentState) -> list:
        """Get recent chat history (last N messages) for context."""
        messages = state.get("messages", [])
        return messages[-self.MAX_HISTORY_MESSAGES:]

    def _format_rag_context(self, docs: list) -> str:
        """Format retrieved documents into a context string."""
        return self.retriever.format_context(docs)

    def _build_base_prompt(self) -> ChatPromptTemplate:
        """
        Build a standard prompt template with system message, RAG context,
        and chat history.
        """
        return ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("system", "参考资料:\n{rag_context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

    def _build_base_chain(self) -> Runnable:
        """
        Build a standard RAG chain: retrieve → format → prompt → LLM.

        Subclasses can use this directly or override for custom chains.
        """
        prompt = self._build_base_prompt()

        def retrieve_and_format(state: dict) -> dict:
            """Retrieve relevant docs based on the user question."""
            question = state.get("question", "")
            docs = self._retrieve_for_agent(question)
            return {
                "rag_context": self._format_rag_context(docs),
                "question": question,
                "history": state.get("history", []),
            }

        chain = (
            RunnableLambda(retrieve_and_format)
            | prompt
            | self.llm
        )
        return chain

    def _retrieve_for_agent(self, query: str) -> list:
        """
        Retrieve domain-specific documents. Override in subclasses
        to use the appropriate retriever method.
        """
        return self.retriever.retrieve_all(query)

    def _run_chain(self, state: AgentState, question: str) -> str:
        """
        Run the agent's chain and return the LLM response text.

        Args:
            state: Current AgentState.
            question: User question or processed prompt.

        Returns:
            LLM response string.
        """
        history = self._get_history_messages(state)

        try:
            response = self.chain.invoke({
                "question": question,
                "history": history,
            })
            # Handle different response types from different LangChain versions
            if hasattr(response, "content"):
                return response.content
            return str(response)
        except Exception:
            # Fallback: try without chain if chain fails
            logger.warning("Chain invoke failed, trying direct LLM call")

        # Direct fallback
        prompt = (
            f"{self.system_prompt}\n\n"
            f"用户问题: {question}\n"
            f"请用中文回复，基于你的专业知识。"
        )
        resp = self.llm.invoke(prompt)
        if hasattr(resp, "content"):
            return resp.content
        return str(resp)
