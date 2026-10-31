"""
Query Complexity Analyzer for GreenRouting.
Computes multi-dimensional linguistic, heuristic, and semantic complexity scores
to inform energy-aware model routing decisions.
"""
import re
from typing import Dict, Any, List, Optional
from greenrouting.config import RoutingConfig


# Keywords indicative of multi-hop, analytical, comparative, or causal reasoning
REASONING_KEYWORDS = {
    "compare", "contrast", "difference", "relationship", "correlate", "correlates",
    "analyze", "analysis", "synthesize", "synthesizing", "evaluate", "evaluation",
    "trade-off", "tradeoff", "trade-offs", "impact", "implications", "mechanism",
    "why", "how does", "explain why", "cause", "consequence", "justify",
    "advantages and disadvantages", "pros and cons", "step-by-step", "derive"
}

# Technical and scientific domain terms frequently found in advanced corpora
TECHNICAL_TERMS = {
    "architecture", "algorithm", "parameter", "hyperparameter", "complexity",
    "latency", "throughput", "bandwidth", "consumption", "photovoltaic",
    "emission", "emissions", "carbon", "greenhouse", "thermodynamic",
    "gradient", "convergence", "variance", "stochastic", "embedding",
    "vector", "semantic", "inference", "quantization", "microarchitecture"
}


class ComplexityAnalyzer:
    """Evaluates question complexity based on syntactic, lexical, and semantic heuristics."""

    def __init__(self, routing_config: Optional[RoutingConfig] = None):
        self.config = routing_config or RoutingConfig()

    def analyze(self, query: str, context_chunks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Compute an explainable complexity score between 0.0 (simple) and 1.0 (complex).
        Returns the overall score, sub-scores, and a detailed justification.
        """
        clean_q = query.strip()
        words = re.findall(r"\b[A-Za-z0-9_-]+\b", clean_q)
        word_count = len(words)
        clean_lower = clean_q.lower()

        # 1. Length Factor: 0.0 for <= 5 words, scaling to 1.0 at >= 25 words
        length_score = min(1.0, max(0.0, (word_count - 5) / 20.0))

        # 2. Reasoning Keyword Factor
        matched_reasoning = [kw for kw in REASONING_KEYWORDS if kw in clean_lower]
        reasoning_score = min(1.0, len(matched_reasoning) * 0.35)

        # 3. Syntactic & Clause Complexity
        # Checks multiple clauses, conjunctions, subordinate markers
        conjunctions = len(re.findall(r"\b(and|or|but|because|although|whereas|while|if|since|unless)\b", clean_lower))
        clause_separators = len(re.findall(r"[,;:]", clean_q))
        syntactic_score = min(1.0, (conjunctions * 0.25) + (clause_separators * 0.15))

        # 4. Domain & Technical Term Density
        matched_tech = [t for t in TECHNICAL_TERMS if t in clean_lower]
        domain_score = min(1.0, len(matched_tech) * 0.30)

        # 5. Weighted Aggregate Score
        w = self.config.weights
        total_weight = sum(w.values())
        raw_score = (
            w.get("length", 0.20) * length_score +
            w.get("reasoning_keywords", 0.35) * reasoning_score +
            w.get("syntactic_complexity", 0.25) * syntactic_score +
            w.get("domain_depth", 0.20) * domain_score
        ) / total_weight

        # Boost slightly if multiple context chunks exist and query is multi-aspect
        if context_chunks and len(context_chunks) > 2 and reasoning_score > 0.3:
            raw_score = min(1.0, raw_score + 0.10)

        final_score = round(float(raw_score), 4)

        # Formulate human-readable explanation
        reasons = []
        if matched_reasoning:
            reasons.append(f"analytical indicators detected ({', '.join(matched_reasoning[:3])})")
        if matched_tech:
            reasons.append(f"technical domain terms ({', '.join(matched_tech[:3])})")
        if word_count > 18:
            reasons.append(f"lengthy multi-clause query ({word_count} words)")
        elif word_count <= 8:
            reasons.append(f"concise factual lookup ({word_count} words)")
        if conjunctions >= 2:
            reasons.append(f"multi-part conjunctions ({conjunctions})")

        if not reasons:
            reasons.append("standard informational query")

        is_complex = final_score >= self.config.complexity_threshold
        rationale_summary = (
            f"{'High' if is_complex else 'Low'} complexity score ({final_score:.2f} "
            f"{'>=' if is_complex else '<'} threshold {self.config.complexity_threshold:.2f}): "
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
