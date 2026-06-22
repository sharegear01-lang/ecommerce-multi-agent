"""
Shopping Guide Agent — handles product recommendations, spec explanations,
and promotion inquiries.

This agent is the primary contact for the INQUIRY state.
"""

import logging
from typing import Optional

from langchain_core.messages import AIMessage
from langgraph.types import Command

from src.agents.base_agent import BaseAgent
from src.state.schema import AgentState
from src.communication.context_manager import ContextManager
from src.communication.protocol import AgentMessage

logger = logging.getLogger(__name__)


class ShoppingGuideAgent(BaseAgent):
    """
    Shopping guide and product recommendation agent.

    Domain: INQUIRY state.
    Uses RAG to retrieve product information, promotions, and specs,
    then generates natural-language recommendations grounded in real data.
    """

    agent_name: str = "ShoppingGuide"

    system_prompt: str = """
你是一个专业的电商导购员（Shopping Guide Agent）。

你的职责：
1. **商品推荐**：根据用户需求，从商品库中推荐最合适的产品。
2. **规格解答**：准确说明商品的规格参数、功能特点。
3. **促销说明**：介绍当前促销活动、优惠信息。
4. **对比分析**：当用户比较多个商品时，客观分析各自优劣。

工作规范：
- 务必引用参考资料中的具体商品信息（名称、价格、规格）。
- 如果用户需求不明确，主动询问使用场景、预算范围等。
- 推荐理由要具体，不要笼统。
- 提及当前促销和优惠信息。
- 如果参考资料中没有匹配的商品，如实告知并建议用户关注新品。

回复格式：
1. 理解用户需求，简要重述。
2. 列出推荐商品（名称、价格、核心卖点）。
3. 说明推荐理由。
4. 提及促销信息（如有）。
5. 引导下一步操作（如"需要下单请告诉我"）。

请用热情、专业、耐心的语气回复。
"""

    def __init__(self, llm, retriever):
        super().__init__(llm, retriever)
        self.context_manager = ContextManager()

    def _build_chain(self):
        """Build the RAG-augmented shopping guide chain."""
        return self._build_base_chain()

    def _process(self, state: AgentState) -> dict:
        """
        Handle product inquiry or recommendation.

        Can also detect cross-agent needs: if the user is in an exchange flow
        (context contains return info), may set up a task chain for ordering.
        """
        user_msg = self._get_user_message(state)
        context = state.get("context", {})
        intent = state.get("intent", "product_recommend")
        task_chain = list(state.get("task_chain", []))

        # Check if this is part of an exchange flow
        is_exchange_flow = bool(context.get("exchange_request") or context.get("return_request"))

        # Retrieve relevant products
        docs = self.retriever.retrieve_products(user_msg, top_k=3)
        rag_context = self._format_rag_context(docs)

        # Build enriched prompt
        if is_exchange_flow:
            exchange_info = context.get("exchange_request", {})
            question = (
                f"用户正在办理换货，原商品信息: {exchange_info}\n"
                f"用户想换的新商品要求: {user_msg}\n"
                f"请根据参考资料推荐合适的替代商品，并说明换货差价。"
            )
        else:
            question = user_msg

        # Generate response
        history = self._get_history_messages(state)
        try:
            response = self.chain.invoke({
                "question": question,
                "history": history,
                "rag_context": rag_context,
            })
            if hasattr(response, "content"):
                response_text = response.content
            else:
                response_text = str(response)
        except Exception as e:
            logger.error("Chain invoke failed: %s", e)
            response_text = self._run_chain(state, question)

        # Update context with recommended products for potential cross-agent use
        updated_context = dict(context)
        if docs:
            product_ids = []
            for d in docs:
                if d.metadata.get("product_id"):
                    product_ids.append({
                        "id": d.metadata["product_id"],
                        "name": d.metadata.get("brand", "") + " " + d.page_content.split("\n")[0].replace("商品名称: ", ""),
                    })
            updated_context["recommended_products"] = product_ids[:3]

        # If in exchange flow and user shows intent to order, set up task chain
        if is_exchange_flow and ("下单" in user_msg or "买" in user_msg or "换" in user_msg):
            if not task_chain:
                task_chain = ["new_order"]
                logger.info("Exchange flow detected — adding 'new_order' to task chain")

        # Route back to Router
        logger.info("Shopping guide responding — routing back to router")
        return Command(
            update={
                "messages": [AIMessage(content=response_text)],
                "context": updated_context,
                "task_chain": task_chain,
            },
            goto="router",
        )

    def _retrieve_for_agent(self, query: str) -> list:
        """Retrieve products relevant to the query."""
        return self.retriever.retrieve_products(query)
