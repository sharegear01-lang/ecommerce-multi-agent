"""
After-sales Agent — handles return requests, exchange processing, dispute
resolution, warranty inquiries, and satisfaction follow-ups.

This agent is the primary contact for the AFTERSALES state.
It is also the most likely to trigger cross-agent workflows (e.g.,
exchange → product inquiry → new order).
"""

import logging
from datetime import datetime

from langchain_core.messages import AIMessage
from langgraph.types import Command

from src.agents.base_agent import BaseAgent
from src.state.schema import AgentState
from src.communication.context_manager import ContextManager

logger = logging.getLogger(__name__)


class AftersalesAgent(BaseAgent):
    """
    After-sales service agent.

    Domain: AFTERSALES state.
    Handles returns, exchanges, refunds, disputes, warranty claims,
    and satisfaction surveys. May trigger cross-agent exchange flows.
    """

    agent_name: str = "Aftersales"

    system_prompt: str = """
你是一个专业的电商售后服务专员（After-sales Agent）。

你的职责：
1. **退换货处理**：根据退换货政策，判断是否符合条件并引导流程。
2. **退款咨询**：说明退款方式、时效。
3. **纠纷处理**：耐心倾听用户投诉，给出合理解决方案。
4. **保修咨询**：解答保修政策，指引维修流程。
5. **满意度回访**：在服务结束后询问用户满意度。

工作规范：
- 处理退换货时必须依据参考资料中的退换货政策。
- 判断退货原因：质量问题（商家承担运费）vs 个人原因（用户承担运费）。
- 涉及换货时，如果用户想换不同商品，应引导至导购Agent（设置跨Agent任务链）。
- 保持耐心和同理心，即使是投诉也要积极解决。

退换货政策要点：
- 7天无理由退货（商品完好，不影响二次销售）。
- 15天内可换货（质量问题、描述不符、规格不合适）。
- 质量问题商家承担运费，个人原因买家承担运费。
- 退款3-5个工作日到账，原路返回。
- 换货流程5-7个工作日完成。

回复格式：
1. 理解用户售后需求，表达同理心。
2. 根据政策说明处理方案。
3. 提供具体操作步骤。
4. 涉及换不同商品时，告知将转接导购员。
5. 询问是否需要其他帮助。

请用耐心、友善、专业的语气回复，让用户感受到被重视。
"""

    def __init__(self, llm, retriever):
        super().__init__(llm, retriever)
        self.context_manager = ContextManager()

    def _build_chain(self):
        """Build the RAG-augmented after-sales agent chain."""
        return self._build_base_chain()

    def _process(self, state: AgentState) -> dict:
        """
        Process after-sales requests.

        Detects cross-agent needs: if a user wants to exchange for a
        different product, sets task_chain = ["product_inquiry", "new_order"]
        and passes return/exchange context to subsequent agents.
        """
        user_msg = self._get_user_message(state)
        context = state.get("context", {})
        intent = state.get("intent", "return_request")
        task_chain = list(state.get("task_chain", []))

        # Retrieve relevant policies
        if "return" in intent or "退款" in user_msg or "退货" in user_msg:
            policy_docs = self.retriever.retrieve_policies(policy_type="return_policy")
            exchange_docs = self.retriever.retrieve_policies(policy_type="exchange_policy")
            policy_docs.extend(exchange_docs)
        elif "exchange" in intent or "换" in user_msg:
            policy_docs = self.retriever.retrieve_policies(policy_type="exchange_policy")
        elif "warranty" in intent or "保修" in user_msg or "维修" in user_msg:
            policy_docs = self.retriever.retrieve_policies(policy_type="warranty_policy")
        elif "refund" in intent or "退款" in user_msg:
            policy_docs = self.retriever.retrieve_policies(policy_type="return_policy")
        else:
            policy_docs = self.retriever.retrieve_policies(query=user_msg)

        rag_context = self._format_rag_context(policy_docs)

        # Check for cross-agent exchange scenario
        # User wants to exchange for a DIFFERENT product (not same replacement)
        cross_agent_exchange = self._detect_cross_agent_exchange(user_msg, intent)

        # Determine if this is a new exchange flow
        is_new_exchange = not bool(
            context.get("exchange_request") or context.get("exchange_approved")
        )

        # Build enriched question
        if cross_agent_exchange:
            question = (
                f"用户售后请求: {user_msg}\n"
                f"退换货政策:\n{rag_context}\n\n"
                f"重要：用户想换**不同**的商品（不是同款换新）。请：\n"
                f"1. 先确认换货资格（是否符合15天换货条件）\n"
                f"2. 了解用户想换什么类型的商品\n"
                f"3. 告知会转接导购员帮其选择新商品\n"
                f"4. 记录原订单信息和用户偏好\n"
                f"{'如果换货资格已确认，请批准换货申请。' if not is_new_exchange else '请判断是否符合换货条件。'}"
            )
        else:
            question = (
                f"用户售后请求: {user_msg}\n"
                f"参考资料:\n{rag_context}\n"
                f"请根据退换货/保修/退款政策给出处理方案。"
            )

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
        except Exception:
            response_text = self._run_chain(state, question)

        # Handle cross-agent exchange flow
        updated_context = dict(context)
        if cross_agent_exchange:
            # Determine if exchange is approved
            if self._is_exchange_approved(response_text, user_msg, is_new_exchange):
                # Set up cross-agent task chain: product inquiry → new order
                updated_context["exchange_request"] = {
                    "approved": True,
                    "original_product": context.get("current_order", {}).get("product", "未知商品"),
                    "original_order_id": context.get("current_order", {}).get("order_id", "未知"),
                    "user_preference": user_msg,
                    "approved_at": datetime.now().isoformat(),
                }
                task_chain = ["product_inquiry", "new_order"]
                response_text += (
                    "\n\n🔄 换货申请已审核通过！正在为您转接导购员，帮您挑选合适的新商品..."
                )
                logger.info("Cross-agent exchange triggered: task_chain=%s", task_chain)

            # Add conflict log entry if there's a potential conflict
            # (e.g., system detects the user might also be eligible for refund)
            updated_context["conflict_entry"] = {
                "issue": "exchange_vs_refund",
                "aftersales_recommendation": "exchange" if cross_agent_exchange else "refund",
                "user_intent": intent,
            }

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
        """Retrieve after-sales policies and relevant FAQ."""
        policies = self.retriever.retrieve_policies(query=query)
        faq = self.retriever.retrieve_faq(query)
        return policies + faq

    def _detect_cross_agent_exchange(self, user_msg: str, intent: str) -> bool:
        """
        Detect if the user wants to exchange for a DIFFERENT product
        (not just a same-item replacement), requiring cross-agent flow.
        """
        cross_indicators = [
            "换个", "换一个", "换别的", "换其他", "换不同",
            "换更", "换高", "换低", "换贵", "换便宜",
            "升级", "降级", "换成", "更换为",
            "不是同款", "不同型号", "别的型号",
        ]
        user_lower = user_msg.lower()
        return any(indicator in user_lower for indicator in cross_indicators)

    def _is_exchange_approved(
        self,
        response_text: str,
        user_msg: str,
        is_new: bool,
    ) -> bool:
        """
        Determine if the exchange should be approved.
        In a real system, this would check against actual order data.
        """
        # Positive indicators in the response
        approved_keywords = ["符合", "可以", "通过", "批准", "同意", "满足条件", "办理"]
        denied_keywords = ["不符合", "不能", "拒绝", "无法", "超过期限", "不在"]

        has_approved = any(kw in response_text for kw in approved_keywords)
        has_denied = any(kw in response_text for kw in denied_keywords)

        if has_denied:
            return False
        if has_approved:
            return True
        # Default: approve exchanges for demonstration
        return is_new and ("换" in user_msg or "exchange" in user_msg.lower())
