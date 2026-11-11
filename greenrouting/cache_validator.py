"""
Quality-Aware Cache Validator for GreenRouting.
Ensures semantic cache entries are not accepted based on vector similarity alone.
Validates document context compatibility, content freshness, source support, and answer grounding.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from greenrouting.config import CachingConfig

logger = logging.getLogger(__name__)


class CacheValidationStatus:
    HIT_VALID = "HIT_VALID"
    HIT_REJECTED_CONTEXT_MISMATCH = "HIT_REJECTED_CONTEXT_MISMATCH"
    HIT_REJECTED_STALE = "HIT_REJECTED_STALE"
    HIT_REJECTED_UNSUPPORTED = "HIT_REJECTED_UNSUPPORTED"
    HIT_REJECTED_LOW_GROUNDING = "HIT_REJECTED_LOW_GROUNDING"
    MISS = "MISS"


class CacheQualityValidator:
    """Rigorous multi-layer validation engine for semantic cache hits."""

    def __init__(self, caching_config: Optional[CachingConfig] = None):
        self.config = caching_config or CachingConfig()

    def validate_cache_candidate(
        self,
        candidate_entry: Dict[str, Any],
        similarity_score: float,
        current_doc_ids: Optional[List[str]] = None,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[bool, str, str]:
        """
        Validate whether a candidate semantic cache entry can be safely returned.
        Returns: (is_valid: bool, status_code: str, reason: str)
        """
        # 1. Similarity Threshold
        if similarity_score < self.config.semantic_similarity_threshold:
            return (
                False,
                CacheValidationStatus.MISS,
                f"Similarity ({similarity_score:.3f}) below threshold ({self.config.semantic_similarity_threshold:.3f})."
            )

        # 2. Inactive / Flagged Check
        if not candidate_entry.get("is_active", 1):
            return (
                False,
                CacheValidationStatus.HIT_REJECTED_STALE,
                "Cache entry has been marked inactive or invalidated."
            )

        # 3. Document Context Compatibility (Do not serve answers across unrelated documents)
        cached_doc_ids = set(candidate_entry.get("source_doc_ids", []))
        if self.config.enforce_document_context_match and current_doc_ids:
            active_doc_set = set(current_doc_ids)
            # If the query is scoped to active documents and none overlap with cached doc IDs
            if cached_doc_ids and not cached_doc_ids.intersection(active_doc_set):
                return (
                    False,
                    CacheValidationStatus.HIT_REJECTED_CONTEXT_MISMATCH,
                    f"Context mismatch: Cached answer originates from document(s) {cached_doc_ids}, "
                    f"which does not match currently active document(s) {active_doc_set}."
                )

        # 4. Staleness / Age Check
        created_at_str = candidate_entry.get("created_at")
        if created_at_str:
            try:
                # Support standard SQLite timestamp formats
                created_dt = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                if created_dt.tzinfo is None:
                    now = datetime.now(timezone.utc).replace(tzinfo=None)
                age_hours = (now - created_dt).total_seconds() / 3600.0
                if age_hours > self.config.max_cache_age_hours:
                    return (
                        False,
                        CacheValidationStatus.HIT_REJECTED_STALE,
                        f"Cache entry expired: Age is {age_hours:.1f} hours "
                        f"(max allowed: {self.config.max_cache_age_hours} hours)."
                    )
            except Exception as e:
                logger.debug("Date parsing skipped: %s", e)

        # 5. Quality & Grounding Score Check
        cached_grounding = candidate_entry.get("grounding_score", 1.0)
        if cached_grounding < self.config.min_cache_grounding_score:
            return (
                False,
                CacheValidationStatus.HIT_REJECTED_LOW_GROUNDING,
                f"Cached answer has sub-standard grounding score ({cached_grounding:.2f} < "
                f"{self.config.min_cache_grounding_score:.2f})."
            )

        # 6. Source Support Compatibility with currently retrieved chunks
        if retrieved_chunks:
            retrieved_text = " ".join(c.get("text_content", "") for c in retrieved_chunks).lower()
            ans_text = candidate_entry.get("answer_text", "").lower()
            ans_keywords = [w for w in ans_text.split() if len(w) > 4 and w.isalnum()]
            if ans_keywords:
                supported_kw = sum(1 for w in ans_keywords if w in retrieved_text)
                ratio = supported_kw / len(ans_keywords)
                # If less than 20% of keywords appear in current context, it's likely unsupported
                if ratio < 0.20:
                    return (
                        False,
                        CacheValidationStatus.HIT_REJECTED_UNSUPPORTED,
                        f"Unsupported claims: Only {ratio*100:.1f}% of key terms in the cached answer "
                        f"are corroborated by currently retrieved context chunks."
                    )

        # All checks passed!
        return (
            True,
            CacheValidationStatus.HIT_VALID,
            f"Valid semantic match (Similarity: {similarity_score:.3f} >= {self.config.semantic_similarity_threshold:.3f}), "
            f"context verified, grounding confirmed ({cached_grounding:.2f})."
        )
