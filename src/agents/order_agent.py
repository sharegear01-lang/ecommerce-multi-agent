"""
Order Agent — handles order placement, inventory checks, logistics tracking,
and payment inquiries.

This agent is the primary contact for the ORDER state.
"""

import logging
import uuid
from datetime import datetime

from langchain_core.messages import AIMessage
from langgraph.types import Command

from src.agents.base_agent import BaseAgent
from src.state.schema import AgentState
from src.communication.context_manager import ContextManager

logger = logging.getLogger(__name__)


class OrderAgent(BaseAgent):
    """
    Order management and logistics agent.

    Domain: ORDER state.
    Handles order creation simulation, inventory checks, order status
    tracking, and shipping/logistics inquiries.
    """

    agent_name: str = "OrderAgent"

    system_prompt: str = """
你是一个专业的电商订单处理员（Order Agent）。

你的职责：
1. **下单引导**：帮用户确认商品、地址、支付方式，完成下单流程。
2. **库存查询**：查询商品库存状态。
3. **物流追踪**：提供订单物流状态信息。
4. **支付协助**：解答支付方式相关问题。

工作规范：
- 下单前必须确认：商品、数量、收货地址、支付方式。
- 库存信息必须基于参考资料中的数据。
- 物流信息要提供具体的时效估计。
- 引导用户分步完成操作，不要跳过必要步骤。

回复格式：
1. 明确当前订单处理步骤。
2. 提供具体信息（库存量、价格、预计配送时间等）。
3. 引导用户完成下一步操作。

请用严谨、细致、可靠的语言回复。
"""

    # Simulated order database
    _orders = {}

    def __init__(self, llm, retriever):
        super().__init__(llm, retriever)
        self.context_manager = ContextManager()

    def _build_chain(self):
        """Build the RAG-augmented order agent chain."""
        return self._build_base_chain()

    def _process(self, state: AgentState) -> dict:
        """
        Process order-related requests.

        Handles:
        - New order creation
        - Inventory checks
        - Order status tracking
        - Logistics inquiries
        """
        user_msg = self._get_user_message(state)
        context = state.get("context", {})
        intent = state.get("intent", "order_create")
        task_chain = list(state.get("task_chain", []))

        # Retrieve relevant policies for shipping/payment
        policy_docs = self.retriever.retrieve_policies(query=user_msg)
        policy_context = self._format_rag_context(policy_docs)

        # Check if this is a cross-agent order (from exchange flow)
        recommended = context.get("recommended_products", [])
        is_exchange_order = bool(context.get("exchange_request") and recommended)

        # Build question based on intent
        if intent == "order_create" and recommended:
            # Ordering from exchange flow
            product_list = "\n".join(
                f"- {p['id']}: {p['name']}" for p in recommended
            )
            question = (
                f"用户想从推荐列表中下单。\n"
                f"推荐的商品:\n{product_list}\n"
                f"用户消息: {user_msg}\n"
                f"请引导用户完成下单（确认商品、地址、支付）。"
            )
        elif intent == "order_create":
            # Fresh order
            # Retrieve relevant products
            product_docs = self.retriever.retrieve_products(user_msg, top_k=2)
            product_context = self._format_rag_context(product_docs)
            question = (
                f"用户想下单购买商品。\n"
                f"用户消息: {user_msg}\n"
                f"相关商品信息:\n{product_context}\n"
                f"配送支付政策:\n{policy_context}\n"
                f"请引导用户完成下单流程。"
            )
        elif intent in ("order_status", "order_track", "logistics_inquiry"):
            question = (
                f"用户查询订单/物流状态。\n"
                f"用户消息: {user_msg}\n"
                f"配送政策:\n{policy_context}\n"
                f"请告知配送时效、物流查询方式等信息。"
            )
        elif intent == "cancel_order":
            question = (
                f"用户想取消订单。\n"
                f"用户消息: {user_msg}\n"
                f"请告知取消订单的流程和注意事项。"
            )
        else:
            question = f"{user_msg}\n\n配送政策参考:\n{policy_context}"

        # Generate response
        history = self._get_history_messages(state)
        try:
            response = self.chain.invoke({
                "question": question,
                "history": history,
                "rag_context": "",
            })
            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)
        except Exception:
            response_text = self._run_chain(state, question)

        # Simulate order creation if user confirms
        updated_context = dict(context)
        order_info = None
        if intent == "order_create" and ("确认" in user_msg or "下单" in user_msg or "购买" in user_msg):
            order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
            product_ref = recommended[0] if recommended else {"id": "PROD-001", "name": "未知商品"}
            order_info = {
                "order_id": order_id,
                "product": product_ref,
                "status": "待付款",
                "created_at": datetime.now().isoformat(),
                "estimated_delivery": "3-5个工作日",
            }
            OrderAgent._orders[order_id] = order_info
            updated_context["current_order"] = order_info
            response_text += (
                f"\n\n📋 **订单已生成**\n"
                f"订单号: **{order_id}**\n"
                f"商品: {product_ref['name']}\n"
                f"状态: 待付款\n"
                f"预计配送: 3-5个工作日\n"
                f"请尽快完成支付，超时订单将自动取消。"
            )

        # Route back to Router
        return Command(
            update={
                "messages": [AIMessage(content=response_text)],
                "context": updated_context,
                "task_chain": task_chain,
            },
            goto="router",
        )

    def _retrieve_for_agent(self, query: str) -> list:
        """Retrieve shipping/payment policies and product inventory."""
        policies = self.retriever.retrieve_policies(query)
        products = self.retriever.retrieve_products(query, top_k=1)
        return policies + products
