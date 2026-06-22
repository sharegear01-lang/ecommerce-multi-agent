"""
Router Agent — the central coordinator.

Responsibilities:
1. Intent classification from user messages (LLM + keyword fallback).
2. State determination and agent dispatch via LangGraph Command API.
3. Cross-agent task chain management.
4. Conflict detection and triggering arbitration.
"""

import json
import logging
import re
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

from src.agents.base_agent import BaseAgent
from src.state.schema import AgentState
from src.state.transitions import (
    INTENT_TO_STATE,
    STATE_TO_AGENT,
    TASK_TO_AGENT,
    classify_intent_keywords,
)
from src.communication.context_manager import ContextManager

logger = logging.getLogger(__name__)

# Valid intent values
VALID_INTENTS = frozenset({
    "product_inquiry", "product_recommend", "promotion_inquiry",
    "spec_question", "price_inquiry", "compare_products",
    "order_create", "order_status", "order_track", "inventory_check",
    "payment_inquiry", "logistics_inquiry", "cancel_order",
    "return_request", "exchange_request", "refund_inquiry",
    "dispute", "complaint", "warranty_inquiry", "satisfaction_feedback",
    "greeting", "goodbye", "help", "unknown",
})


class RouterAgent(BaseAgent):
    """
    Central routing agent.

    Acts as the hub in the hub-and-spoke architecture. All specialized agents
    return control here; the Router decides the next step based on intent,
    task chain, and conflict status.

    State machine logic:
    1. If task_chain is non-empty → dispatch next task (cross-agent flow).
    2. If current_state is NOT IDLE and task_chain is empty →
       agent has finished; transition to IDLE (end of turn).
    3. If current_state IS IDLE → new turn; classify user message and route.
    """

    agent_name: str = "Router"

    CLASSIFY_PROMPT: str = """你是一个电商意图分类器。分析用户消息，输出一个JSON对象。

有效的意图值: product_inquiry, product_recommend, promotion_inquiry, spec_question, price_inquiry, compare_products, order_create, order_status, order_track, inventory_check, payment_inquiry, logistics_inquiry, cancel_order, return_request, exchange_request, refund_inquiry, dispute, complaint, warranty_inquiry, satisfaction_feedback, greeting, goodbye, help, unknown

输出格式（严格JSON，不要markdown代码块）:
{"intent": "<intent>", "confidence": 0.0-1.0, "reason": "简要说明"}
"""

    def __init__(self, llm, retriever, conflict_resolver=None):
        super().__init__(llm, retriever)
        self.context_manager = ContextManager()
        self.conflict_resolver = conflict_resolver

    def _build_chain(self):
        """Router uses custom intent classification, not a standard chain."""
        return None

    def _process(self, state: AgentState) -> dict:
        """
        Process the state and determine routing.

        Returns a Command with both state update AND routing target,
        or a plain dict update with a final message.
        """
        current_state = state.get("current_state", "IDLE")

        # --- 1. Handle cross-agent task chain (highest priority) ---
        task_chain = list(state.get("task_chain", []))
        if task_chain:
            return self._dispatch_next_task(state, task_chain)

        # --- 2. If NOT idle and no task chain → agent just finished; go idle ---
        if current_state != "IDLE":
            logger.info("Agent finished — returning to IDLE (was: %s)", current_state)
            return self._goto_idle(state)

        # --- 3. IDLE state: new turn — get user message ---
        user_msg = self._get_user_message(state)
        if not user_msg:
            return self._goto_idle(state)

        # --- 4. Check for conflicts ---
        conflict_log = state.get("conflict_log", [])
        if conflict_log and self.conflict_resolver:
            return self._handle_conflict(state, conflict_log)

        # --- 5. Classify intent ---
        intent, confidence, reason = self._classify_intent(user_msg)
        logger.info("Intent: %s (confidence: %.2f, reason: %s)", intent, confidence, reason)

        # --- 6. Handle special termination intents ---
        if intent in ("goodbye",):
            return {
                "current_state": "IDLE",
                "intent": intent,
                "messages": [AIMessage(content="感谢您的咨询，如有需要随时联系我。再见！")],
            }

        if intent in ("greeting", "help"):
            return {
                "current_state": "IDLE",
                "intent": intent,
                "messages": [
                    AIMessage(content=(
                        "您好！我是电商智能助手，可以帮您：\n"
                        "商品咨询：推荐商品、解答规格参数、说明促销活动\n"
                        "订单服务：指导下单、查询库存、追踪物流\n"
                        "售后服务：退换货政策、纠纷处理、保修咨询\n\n"
                        "请问有什么可以帮您的？"
                    ))
                ],
            }

        # --- 7. Route to specialized agent ---
        target_state = INTENT_TO_STATE.get(intent, "IDLE")
        target_agent = STATE_TO_AGENT.get(target_state, "router")

        if target_state == "IDLE" or target_agent == "router":
            return {
                "current_state": "IDLE",
                "intent": intent,
                "messages": [
                    AIMessage(content="抱歉，我没有完全理解您的需求。能否换个方式描述一下？")
                ],
            }

        logger.info("Routing to agent: %s (state: %s)", target_agent, target_state)
        return Command(
            update={
                "current_state": target_state,
                "intent": intent,
            },
            goto=target_agent,
        )

    def _classify_intent(self, text: str) -> tuple:
        """
        Classify user intent.

        Strategy:
        1. Keyword fast-path (covers ~90% of e-commerce queries).
        2. If no keyword match, use LLM with plain JSON prompt
           (not structured_output — deepseek-v4-flash doesn't support it).
        3. Fallback to "unknown".

        Returns:
            Tuple of (intent: str, confidence: float, reason: str).
        """
        # Step 1: Keyword fast path
        keyword_intent = classify_intent_keywords(text)
        if keyword_intent:
            logger.debug("Keyword match: %s", keyword_intent)
            return keyword_intent, 0.85, "关键词匹配"

        # Step 2: LLM classification (plain JSON, no structured_output)
        try:
            llm_intent, llm_confidence = self._llm_classify(text)
            if llm_intent and llm_intent in VALID_INTENTS:
                return llm_intent, llm_confidence, "LLM分类"
        except Exception as e:
            logger.warning("LLM intent classification failed: %s", e)

        # Step 3: Fallback
        return "unknown", 0.3, "无法确定意图"

    def _llm_classify(self, text: str) -> tuple:
        """
        Use LLM (plain chat) to classify intent.
        Returns (intent_str, confidence) or (None, 0) on failure.
        """
        prompt = f"{self.CLASSIFY_PROMPT}\n\n用户消息: {text}"
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        # Try to extract JSON from the response
        json_match = re.search(r'\{[^{}]*"intent"[^{}]*\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("intent"), data.get("confidence", 0.5)

        # Try parsing the whole content as JSON
        try:
            data = json.loads(content.strip())
            return data.get("intent"), data.get("confidence", 0.5)
        except json.JSONDecodeError:
            pass

        return None, 0.0

    def _dispatch_next_task(self, state: AgentState, task_chain: list) -> dict:
        """
        Dispatch the next task in a cross-agent task chain.

        Pops the first task, maps it to the responsible agent,
        and routes there while preserving context.
        """
        next_task = task_chain[0]
        remaining_tasks = task_chain[1:]

        target_agent = TASK_TO_AGENT.get(next_task)
        target_state = "CROSS_AGENT" if remaining_tasks else self._task_to_state(next_task)

        logger.info(
            "Dispatching task '%s' → agent '%s' (remaining: %s)",
            next_task, target_agent, remaining_tasks,
        )

        if target_agent == "router":
            # Unknown task — skip it
            if remaining_tasks:
                return Command(
                    update={"task_chain": remaining_tasks},
                    goto="router",
                )
            return self._goto_idle(state)

        return Command(
            update={
                "current_state": target_state,
                "intent": next_task,
                "task_chain": remaining_tasks,
            },
            goto=target_agent,
        )

    def _handle_conflict(self, state: AgentState, conflict_log: list) -> dict:
        """
        Handle conflicting recommendations using the conflict resolver.

        The resolver evaluates each recommendation and selects the winner.
        """
        if not self.conflict_resolver:
            # No resolver — take the last recommendation
            return self._goto_idle(state)

        resolution = self.conflict_resolver.resolve(conflict_log)
        logger.info("Conflict resolved: %s", resolution.get("decision"))

        # Add resolution message
        decision_msg = (
            f"🔍 **冲突仲裁结果**\n\n"
            f"问题: {resolution.get('issue', 'N/A')}\n"
            f"方案A: {resolution.get('option_a', {}).get('summary', 'N/A')}\n"
            f"方案B: {resolution.get('option_b', {}).get('summary', 'N/A')}\n\n"
            f"**最终决定**: {resolution.get('decision', 'N/A')}\n"
            f"理由: {resolution.get('reasoning', 'N/A')}"
        )

        return {
            "current_state": "IDLE",
            "conflict_log": [],
            "context": self.context_manager.clear_context(state.get("context", {})),
            "messages": [AIMessage(content=decision_msg)],
        }

    def _goto_idle(self, state: AgentState) -> dict:
        """Return to IDLE state."""
        return {
            "current_state": "IDLE",
        }

    def _task_to_state(self, task: str) -> str:
        """Map a task type to its corresponding state."""
        task_state_map = {
            "product_inquiry": "INQUIRY",
            "product_recommend": "INQUIRY",
            "new_order": "ORDER",
            "order_create": "ORDER",
            "return_process": "AFTERSALES",
            "exchange_process": "AFTERSALES",
            "refund_process": "AFTERSALES",
        }
        return task_state_map.get(task, "IDLE")
