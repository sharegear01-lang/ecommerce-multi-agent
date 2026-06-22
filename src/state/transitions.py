"""
Intent-to-State mapping and state-to-agent routing tables.

These pure-function mappings keep the routing logic testable
and separated from the graph construction and agent implementations.
"""

from typing import Optional

# Intent → State machine state mapping
# The Router classifies the user's message into one of these intents,
# then this table determines which state the system should enter.
INTENT_TO_STATE: dict[str, str] = {
    # Shopping Guide domain → INQUIRY
    "product_inquiry": "INQUIRY",
    "product_recommend": "INQUIRY",
    "promotion_inquiry": "INQUIRY",
    "spec_question": "INQUIRY",
    "price_inquiry": "INQUIRY",
    "compare_products": "INQUIRY",

    # Order domain → ORDER
    "order_create": "ORDER",
    "order_status": "ORDER",
    "order_track": "ORDER",
    "inventory_check": "ORDER",
    "payment_inquiry": "ORDER",
    "logistics_inquiry": "ORDER",
    "cancel_order": "ORDER",

    # After-sales domain → AFTERSALES
    "return_request": "AFTERSALES",
    "exchange_request": "AFTERSALES",
    "refund_inquiry": "AFTERSALES",
    "dispute": "AFTERSALES",
    "complaint": "AFTERSALES",
    "warranty_inquiry": "AFTERSALES",
    "satisfaction_feedback": "AFTERSALES",

    # Meta / idle
    "greeting": "IDLE",
    "goodbye": "IDLE",
    "unknown": "IDLE",
    "help": "IDLE",
}

# State → target agent node name in the graph
STATE_TO_AGENT: dict[str, str] = {
    "IDLE": "router",
    "INQUIRY": "shopping_guide",
    "ORDER": "order_agent",
    "AFTERSALES": "aftersales_agent",
    "CROSS_AGENT": "router",  # Router handles cross-agent dispatch
}

# Task type → agent node name (for cross-agent task chain dispatch)
TASK_TO_AGENT: dict[str, str] = {
    "product_inquiry": "shopping_guide",
    "product_recommend": "shopping_guide",
    "new_order": "order_agent",
    "order_create": "order_agent",
    "return_process": "aftersales_agent",
    "exchange_process": "aftersales_agent",
    "refund_process": "aftersales_agent",
}

# Keyword-based fallback intent detection (runs before LLM classifier)
# Used as a fast path for common patterns and as fallback when LLM is unreliable.
# ORDER MATTERS: after-sales keywords checked first to avoid false matches
# (e.g., "退货" must be caught before "买" in "我买的要退货")
KEYWORD_INTENTS: list[tuple[list[str], str]] = [
    # --- After-sales first (high specificity) ---
    (["退换货", "退货退款", "退换"], "return_request"),
    (["退货", "退款", "退了", "退掉"], "return_request"),
    (["换货", "换一个", "换别的", "换个", "换更高", "换更贵"], "exchange_request"),
    (["售后", "保修", "维修", "坏了", "故障", "质量"], "warranty_inquiry"),
    (["投诉", "差评", "投诉"], "complaint"),
    # --- Product inquiry ---
    (["推荐", "有什么", "哪些", "哪个好", "帮我选", "介绍一下"], "product_recommend"),
    (["多少钱", "价格", "优惠", "促销", "打折", "便宜"], "price_inquiry"),
    (["规格", "参数", "配置", "功能", "特点"], "spec_question"),
    # --- Order (lower priority, avoid short ambiguous words) ---
    (["我要买", "我想买", "下单", "购买", "订购", "帮我下单", "加入购物车"], "order_create"),
    (["订单", "物流", "快递", "发货", "到哪了", "配送", "查物流"], "order_status"),
    # --- Meta ---
    (["你好", "hi", "hello", "在吗"], "greeting"),
    (["再见", "拜拜", "bye", "谢谢", "感谢"], "goodbye"),
]


def classify_intent_keywords(text: str) -> Optional[str]:
    """
    Fast keyword-based intent classification.
    Returns the intent string if matched, or None if no match found.

    Args:
        text: User input text (lowercased).

    Returns:
        Intent string or None.
    """
    text_lower = text.lower()
    for keywords, intent in KEYWORD_INTENTS:
        if any(kw in text_lower for kw in keywords):
            return intent
    return None


def get_state_for_intent(intent: str) -> str:
    """
    Map an intent string to the corresponding state machine state.

    Args:
        intent: Classified intent string.

    Returns:
        State string (IDLE | INQUIRY | ORDER | AFTERSALES).
    """
    return INTENT_TO_STATE.get(intent, "IDLE")


def get_agent_for_state(state: str) -> str:
    """
    Map a state to the graph node name of the responsible agent.

    Args:
        state: State string.

    Returns:
        Agent node name for the graph.
    """
    return STATE_TO_AGENT.get(state, "router")


def get_agent_for_task(task: str) -> str:
    """
    Map a cross-agent task type to the responsible agent node.

    Args:
        task: Task type string from the task_chain.

    Returns:
        Agent node name for the graph.
    """
    return TASK_TO_AGENT.get(task, "router")
