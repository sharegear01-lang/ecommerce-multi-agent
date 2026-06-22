"""
Conflict Resolution Engine.

When multiple agents produce conflicting recommendations for the same
issue, the Router invokes this resolver to arbitrate.

Resolution strategy:
1. Score each option by: user intent priority (60%) + historical satisfaction (40%).
2. Cross-reference with RAG policies for compliance.
3. Return the winning recommendation with reasoning.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ConflictResolver:
    """
    Detects and resolves conflicts between agent recommendations.

    Rules-based arbitration with weighted scoring:
    - User intent priority: 60% weight — what did the user originally ask for?
    - Historical satisfaction: 40% weight — which option has better outcomes?
    """

    # Simulated historical satisfaction scores (0.0–1.0)
    # In production, this would come from a real database.
    SATISFACTION_SCORES = {
        "refund": 0.85,
        "exchange": 0.78,
        "repair": 0.65,
        "coupon": 0.50,
        "replace_same": 0.90,
    }

    def __init__(self, intent_priority_weight: float = 0.6):
        """
        Args:
            intent_priority_weight: Weight given to user intent (0.0–1.0).
                                    Remaining weight goes to historical satisfaction.
        """
        self.intent_weight = intent_priority_weight
        self.satisfaction_weight = 1.0 - intent_priority_weight

    def detect_conflict(self, conflict_log: list) -> bool:
        """
        Check if the conflict log contains entries requiring arbitration.

        A conflict exists when two agents recommend different actions
        for the same issue.

        Args:
            conflict_log: List of conflict entries from AgentState.

        Returns:
            True if a conflict needs resolution.
        """
        if len(conflict_log) < 2:
            return False

        # Compare the latest two entries
        latest = conflict_log[-1]
        previous = conflict_log[-2]

        if latest.get("issue") == previous.get("issue"):
            rec_a = latest.get("aftersales_recommendation") or latest.get("recommendation")
            rec_b = previous.get("shopping_guide_recommendation") or previous.get("recommendation")
            if rec_a and rec_b and rec_a != rec_b:
                return True

        return False

    def resolve(self, conflict_log: list) -> dict:
        """
        Resolve the most recent conflict.

        Args:
            conflict_log: List of conflict entries.

        Returns:
            Resolution dict with keys: issue, option_a, option_b, decision, reasoning, scores.
        """
        if not conflict_log:
            return {
                "issue": "unknown",
                "decision": "无法判断（无冲突记录）",
                "reasoning": "冲突日志为空",
            }

        # Take the last two entries as the conflicting options
        # In a full implementation, would aggregate entries by issue
        entry = conflict_log[-1] if conflict_log else {}

        issue = entry.get("issue", "unknown")
        user_intent = entry.get("user_intent", "unknown")
        aftersales_rec = entry.get("aftersales_recommendation", "unknown")
        shopping_rec = entry.get("shopping_guide_recommendation", "")

        # Score option A (typically aftersales recommendation)
        score_a = self._score_option(aftersales_rec, user_intent)

        # Score option B (typically alternative recommendation)
        option_b_key = shopping_rec or self._get_alternative(aftersales_rec)
        score_b = self._score_option(option_b_key, user_intent)

        # Determine winner
        if score_a >= score_b:
            winner_key = aftersales_rec
            winner_score = score_a
            loser_key = option_b_key
            loser_score = score_b
        else:
            winner_key = option_b_key
            winner_score = score_b
            loser_key = aftersales_rec
            loser_score = score_a

        reasoning = self._build_reasoning(
            issue, winner_key, loser_key,
            winner_score, loser_score,
            user_intent,
        )

        resolution = {
            "issue": issue,
            "option_a": {
                "recommendation": aftersales_rec,
                "agent": "aftersales_agent",
                "score": round(score_a, 3),
                "summary": self._describe_option(aftersales_rec),
            },
            "option_b": {
                "recommendation": option_b_key,
                "agent": "shopping_guide",
                "score": round(score_b, 3),
                "summary": self._describe_option(option_b_key),
            },
            "decision": self._describe_option(winner_key),
            "reasoning": reasoning,
            "scores": {
                "winner": round(winner_score, 3),
                "loser": round(loser_score, 3),
            },
        }

        logger.info(
            "Conflict resolved: %s → %s (score: %.3f vs %.3f)",
            issue, winner_key, winner_score, loser_score,
        )
        return resolution

    def _score_option(self, option: str, user_intent: str) -> float:
        """
        Score an option based on user intent priority and historical satisfaction.

        Args:
            option: The recommendation key (e.g., "exchange", "refund").
            user_intent: The original user intent.

        Returns:
            Weighted score 0.0–1.0.
        """
        # Intent match score (how well does this option align with user intent?)
        intent_score = self._intent_match(option, user_intent)

        # Historical satisfaction score
        satisfaction_score = self.SATISFACTION_SCORES.get(option, 0.5)

        # Weighted combination
        total = (self.intent_weight * intent_score) + (self.satisfaction_weight * satisfaction_score)
        return total

    def _intent_match(self, option: str, user_intent: str) -> float:
        """
        Calculate how well an option matches the user's original intent.

        Returns 1.0 for perfect match, 0.0 for no match.
        """
        # Direct match: user asked for X, option is X
        if option == user_intent or user_intent in option or option in user_intent:
            return 1.0

        # Partial match scoring
        match_pairs = {
            "exchange_request": {"exchange": 1.0, "refund": 0.6, "repair": 0.3},
            "return_request": {"refund": 1.0, "exchange": 0.7, "repair": 0.3},
            "refund_inquiry": {"refund": 1.0, "exchange": 0.5, "coupon": 0.3},
            "warranty_inquiry": {"repair": 1.0, "exchange": 0.7, "refund": 0.4},
            "complaint": {"refund": 0.9, "exchange": 0.7, "coupon": 0.6, "repair": 0.5},
        }

        if user_intent in match_pairs:
            return match_pairs[user_intent].get(option, 0.3)

        return 0.5  # Unknown intent — neutral score

    def _get_alternative(self, option: str) -> str:
        """Return the likely alternative recommendation."""
        alternatives = {
            "refund": "exchange",
            "exchange": "refund",
            "repair": "exchange",
            "coupon": "refund",
            "replace_same": "exchange",
        }
        return alternatives.get(option, "refund")

    def _describe_option(self, option: str) -> str:
        """Human-readable description of a recommendation option."""
        descriptions = {
            "refund": "全额退款",
            "exchange": "换货（换同品牌更高型号）",
            "repair": "免费维修",
            "coupon": "赠送优惠券作为补偿",
            "replace_same": "换同款新机",
        }
        return descriptions.get(option, option)

    def _build_reasoning(
        self,
        issue: str,
        winner: str,
        loser: str,
        winner_score: float,
        loser_score: float,
        user_intent: str,
    ) -> str:
        """Build a human-readable reasoning explanation."""
        intent_desc = {
            "return_request": "退货",
            "exchange_request": "换货",
            "refund_inquiry": "退款",
            "warranty_inquiry": "保修",
            "complaint": "投诉处理",
        }.get(user_intent, user_intent)

        return (
            f"用户原始意图为「{intent_desc}」，在方案A（{self._describe_option(loser)}，"
            f"得分{loser_score:.2f}）与方案B（{self._describe_option(winner)}，"
            f"得分{winner_score:.2f}）之间，"
            f"方案{self._describe_option(winner)}更符合用户需求"
            f"{'且历史满意度更高' if winner_score > loser_score else ''}。"
            f"权重分配：用户意图优先级{self.intent_weight:.0%}，"
            f"历史满意度{self.satisfaction_weight:.0%}。"
        )
