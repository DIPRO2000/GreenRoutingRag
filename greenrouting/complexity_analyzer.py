
"""
Query Complexity Analyzer for GreenRouting.

Computes an explainable complexity score using four heuristic factors:
length, reasoning keywords, syntactic complexity, and domain depth.
"""

import re
from typing import Dict, Any, List, Optional

from greenrouting.config import RoutingConfig


REASONING_KEYWORDS = {
    "compare", "contrast", "difference", "relationship", "correlate",
    "correlates", "analyze", "analysis", "synthesize", "synthesizing",
    "evaluate", "evaluation", "trade-off", "tradeoff", "trade-offs",
    "impact", "implications", "mechanism", "why", "how does",
    "explain why", "cause", "consequence", "justify",
    "advantages and disadvantages", "pros and cons",
    "step-by-step", "derive"
}

TECHNICAL_TERMS = {
    "architecture", "algorithm", "parameter", "hyperparameter",
    "complexity", "latency", "throughput", "bandwidth",
    "consumption", "photovoltaic", "emission", "emissions",
    "carbon", "greenhouse", "thermodynamic", "gradient",
    "convergence", "variance", "stochastic", "embedding",
    "vector", "semantic", "inference", "quantization",
    "microarchitecture"
}

EXPECTED_WEIGHTS = {
    "length",
    "reasoning_keywords",
    "syntactic_complexity",
    "domain_depth"
}


class ComplexityAnalyzer:
    """Evaluates query complexity using four linguistic heuristics."""

    def __init__(self, routing_config: Optional[RoutingConfig] = None):
        self.config = routing_config or RoutingConfig()
        self._validate_weights()

    def _validate_weights(self) -> None:
        """Ensure all four weights are valid and sum to 1."""

        weights = self.config.weights

        if not isinstance(weights, dict):
            raise ValueError("Routing weights must be a dictionary.")

        if set(weights.keys()) != EXPECTED_WEIGHTS:
            raise ValueError(
                f"Weights must contain exactly these keys: "
                f"{sorted(EXPECTED_WEIGHTS)}"
            )

        for name, value in weights.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"Weight '{name}' must be numeric.")

            if value < 0:
                raise ValueError(f"Weight '{name}' cannot be negative.")

        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError("Routing weights must sum to 1.0.")

    @staticmethod
    def _match_phrases(text: str, keywords: set) -> List[str]:
        """Match complete words or phrases, not arbitrary substrings."""

        matched = []

        for keyword in sorted(keywords):
            pattern = (
                r"(?<!\w)"
                + re.escape(keyword).replace(r"\ ", r"\s+")
                + r"(?!\w)"
            )

            if re.search(pattern, text):
                matched.append(keyword)

        return matched

    def analyze(
        self,
        query: str,
        context_chunks: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:

        # 1. Validate query
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        clean_q = query.strip()
        clean_lower = clean_q.lower()

        words = re.findall(r"\b[A-Za-z0-9_-]+\b", clean_q)
        word_count = len(words)

        # 2. Length factor
        length_score = min(
            1.0,
            max(0.0, (word_count - 5) / 20.0)
        )

        # 3. Reasoning keyword factor
        matched_reasoning = self._match_phrases(
            clean_lower, REASONING_KEYWORDS
        )

        reasoning_score = min(
            1.0,
            len(matched_reasoning) * 0.35
        )

        # 4. Syntactic complexity factor
        conjunctions = len(re.findall(
            r"\b(and|or|but|because|although|whereas|while|if|since|unless)\b",
            clean_lower
        ))

        clause_separators = len(re.findall(r"[,;:]", clean_q))

        syntactic_score = min(
            1.0,
            conjunctions * 0.25 + clause_separators * 0.15
        )

        # 5. Domain depth factor
        matched_tech = self._match_phrases(
            clean_lower, TECHNICAL_TERMS
        )

        domain_score = min(
            1.0,
            len(matched_tech) * 0.30
        )

        # 6. Weighted aggregate score
        weights = self.config.weights

        raw_score = (
            weights["length"] * length_score
            + weights["reasoning_keywords"] * reasoning_score
            + weights["syntactic_complexity"] * syntactic_score
            + weights["domain_depth"] * domain_score
        )

        final_score = round(float(raw_score), 4)

        # 7. Human-readable explanation
        reasons = []

        if matched_reasoning:
            reasons.append(
                "analytical indicators detected "
                f"({', '.join(matched_reasoning[:3])})"
            )

        if matched_tech:
            reasons.append(
                "technical domain terms "
                f"({', '.join(matched_tech[:3])})"
            )

        if word_count > 18:
            reasons.append(f"lengthy query ({word_count} words)")
        elif word_count <= 8:
            reasons.append(f"concise query ({word_count} words)")

        if conjunctions >= 2:
            reasons.append(
                f"multiple conjunctions detected ({conjunctions})"
            )

        if not reasons:
            reasons.append("standard informational query")

        is_complex = (
            final_score >= self.config.complexity_threshold
        )

        rationale_summary = (
            f"{'High' if is_complex else 'Low'} complexity score "
            f"({final_score:.2f} "
            f"{'>=' if is_complex else '<'} threshold "
            f"{self.config.complexity_threshold:.2f}): "
            f"{'; '.join(reasons)}."
        )

        return {
            "complexity_score": final_score,
            "is_complex": is_complex,
            "rationale": rationale_summary,
            "factors": {
                "length_score": round(length_score, 3),
                "reasoning_score": round(reasoning_score, 3),
                "syntactic_score": round(syntactic_score, 3),
                "domain_score": round(domain_score, 3),
                "word_count": word_count,
                "matched_reasoning_keywords": matched_reasoning,
                "matched_domain_terms": matched_tech
            }
        }
