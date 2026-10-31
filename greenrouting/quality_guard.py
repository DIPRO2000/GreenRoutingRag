"""
Quality Guard for GreenRouting.
Assesses the factual groundedness, confidence, and completeness of generated answers.
Triggers automatic fallback retry to a capable model when lightweight inference is inadequate.
"""
import re
from typing import Dict, Any, List, Optional
from greenrouting.config import RoutingConfig

REFUSAL_UNCERTAINTY_PHRASES = [
    "i do not know", "i don't know", "not enough information",
    "the context does not provide", "the provided text does not mention",
    "cannot be determined", "unclear from the context", "as an ai",
    "i cannot answer", "information is missing"
]


class QualityGuard:
    """Evaluates answer groundedness and decides whether an escalation retry is warranted."""

    def __init__(self, routing_config: Optional[RoutingConfig] = None):
        self.config = routing_config or RoutingConfig()

    def evaluate_answer(self, query: str, answer: str, context_chunks: List[Dict[str, Any]],
                        model_used: str, is_lightweight: bool) -> Dict[str, Any]:
        """
        Evaluate answer quality:
        - Groundedness (token overlap with context)
        - Refusal/uncertainty detection
        - Completeness/substantiveness
        Returns quality_score, should_retry, and rationale.
        """
        clean_ans = answer.strip()
        ans_lower = clean_ans.lower()

        # 1. Check for immediate refusal or uncertainty
        has_refusal = any(phrase in ans_lower for phrase in REFUSAL_UNCERTAINTY_PHRASES)

        # 2. Extract significant context words
        context_text = " ".join(c.get("text_content", "") for c in context_chunks).lower()
        ans_words = [w for w in re.findall(r"\b[a-z]{3,}\b", ans_lower)
                     if w not in {"the", "and", "that", "this", "with", "from", "based", "provided", "according", "document"}]
        
        if not ans_words:
            groundedness_score = 0.0
        else:
            supported_words = sum(1 for w in ans_words if w in context_text)
            groundedness_score = supported_words / len(ans_words)

        # 3. Substantiveness factor
        word_count = len(clean_ans.split())
        if word_count < 6:
            length_penalty = 0.4
        elif word_count < 15:
            length_penalty = 0.8
        else:
            length_penalty = 1.0

        # Composite quality score
        base_score = groundedness_score * length_penalty
        if has_refusal:
            base_score = min(base_score, 0.35)

        quality_score = round(float(base_score), 4)

        # Retry logic: only retry if lightweight model produced sub-par quality
        # and retry is enabled in configuration
        should_retry = (
            self.config.enable_retry_on_low_quality and
            is_lightweight and
            (quality_score < self.config.min_acceptable_quality_score or has_refusal)
        )

        reasons = []
        if has_refusal:
            reasons.append("uncertainty/refusal statement detected")
        if groundedness_score < 0.50:
            reasons.append(f"low context grounding overlap ({groundedness_score:.2f})")
        if word_count < 12:
            reasons.append(f"brief response length ({word_count} words)")
        if not reasons:
            reasons.append(f"strong context alignment ({groundedness_score:.2f})")

        explanation = (
            f"Quality score: {quality_score:.2f} (Threshold: {self.config.min_acceptable_quality_score:.2f}). "
            f"Assessment: {'; '.join(reasons)}. "
            f"{'Escalating retry required.' if should_retry else 'Answer verified acceptable.'}"
        )

        return {
            "quality_score": quality_score,
            "groundedness_score": round(groundedness_score, 4),
            "has_refusal": has_refusal,
            "word_count": word_count,
            "should_retry": should_retry,
            "explanation": explanation
        }
