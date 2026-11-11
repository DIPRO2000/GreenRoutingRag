"""Deterministic retrieval sizing for energy-aware RAG queries."""

from dataclasses import dataclass
import re
from typing import Set


@dataclass(frozen=True)
class RetrievalDecision:
    """The selected context size and the explanation shown to the user."""

    top_k: int
    complexity_score: float
    reason: str


_ANALYTICAL_MARKERS: Set[str] = {
    "analyze",
    "analysis",
    "compare",
    "comparison",
    "contrast",
    "difference",
    "differences",
    "implication",
    "implications",
    "tradeoff",
    "tradeoffs",
    "trade-off",
    "trade-offs",
    "versus",
    "evaluate",
    "evaluation",
}


def choose_top_k(
    query: str,
    available_chunks: int,
    default_top_k: int,
    min_top_k: int = 2,
    max_top_k: int = 5,
) -> RetrievalDecision:
    """Choose a bounded context size using transparent local query signals."""
    available = max(0, int(available_chunks))
    lower_bound = max(1, int(min_top_k))
    upper_bound = max(lower_bound, int(max_top_k))
    default = max(lower_bound, min(upper_bound, int(default_top_k)))

    if not query or not query.strip():
        selected = min(default, available) if available else 0
        return RetrievalDecision(selected, 0.0, f"Empty query; using the configured default top_k={selected}.")

    words = re.findall(r"[\w-]+", query.lower())
    marker_hits = sum(1 for word in words if word in _ANALYTICAL_MARKERS)
    length_signal = min(len(words) / 20.0, 0.4)
    complexity = min(1.0, 0.2 + length_signal + marker_hits * 0.15)

    if marker_hits >= 2 or len(words) >= 16:
        selected = upper_bound
        reason = "Analytical query detected; using a larger context for comparison and evidence coverage."
    elif marker_hits == 1 or len(words) >= 9:
        selected = min(upper_bound, lower_bound + 1)
        reason = "Moderate-complexity query detected; using an intermediate context size."
    else:
        selected = lower_bound
        reason = "Simple factual query detected; using a small context to reduce retrieval and prompt work."

    selected = min(selected, available) if available else 0
    return RetrievalDecision(selected, round(complexity, 4), reason)
