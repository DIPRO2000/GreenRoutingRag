"""
Semantic Caching Layer for GreenRouting.
Leverages vector similarity and multi-dimensional quality validation to eliminate redundant
LLM inferences and conserve compute energy.
"""
import uuid
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from greenrouting.config import CachingConfig
from greenrouting.database import DatabaseManager
from greenrouting.vector_store import EmbeddingService
from greenrouting.cache_validator import CacheQualityValidator, CacheValidationStatus

logger = logging.getLogger(__name__)


class SemanticCache:
    """Manages storage, retrieval, cosine matching, and quality validation of semantic cache entries."""

    def __init__(self, db_manager: DatabaseManager, embedding_service: EmbeddingService,
                 caching_config: Optional[CachingConfig] = None):
        self.db = db_manager
        self.embedding_service = embedding_service
        self.config = caching_config or CachingConfig()
        self.validator = CacheQualityValidator(self.config)

    def lookup(
        self,
        query: str,
        current_doc_ids: Optional[List[str]] = None,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Lookup semantic cache for a query:
        1. If caching disabled, return MISS immediately.
        2. Embed query.
        3. Compute cosine similarity against all active cached queries.
        4. Select top candidate.
        5. Run CacheQualityValidator (context match, grounding, freshness, support).
        6. Return complete validation outcome.
        """
        if not self.config.enabled:
            return {
                "status": CacheValidationStatus.MISS,
                "cached_answer": None,
                "similarity_score": 0.0,
                "validation_reason": "Semantic caching is disabled in configuration.",
                "candidate": None
            }

        entries = self.db.get_active_cache_entries()
        if not entries:
            return {
                "status": CacheValidationStatus.MISS,
                "cached_answer": None,
                "similarity_score": 0.0,
                "validation_reason": "Semantic cache is empty.",
                "candidate": None
            }

        # Embed input query
        query_vec = self.embedding_service.embed_query(query)
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 1e-9:
            query_vec = query_vec / q_norm

        best_score = -1.0
        best_entry = None

        for entry in entries:
            emb_list = entry.get("query_embedding")
            if not emb_list:
                continue
            cached_vec = np.array(emb_list, dtype=np.float32)
            c_norm = np.linalg.norm(cached_vec)
            if c_norm > 1e-9:
                cached_vec = cached_vec / c_norm

            sim = float(np.dot(query_vec, cached_vec))
            if sim > best_score:
                best_score = sim
                best_entry = entry

        if not best_entry or best_score < self.config.semantic_similarity_threshold:
            return {
                "status": CacheValidationStatus.MISS,
                "cached_answer": None,
                "similarity_score": round(max(0.0, best_score), 4),
                "validation_reason": f"No semantically similar query found (Best similarity: {max(0.0, best_score):.3f} < threshold {self.config.semantic_similarity_threshold:.3f}).",
                "candidate": None
            }

        # Rigorous Quality & Context Validation
        is_valid, status_code, reason = self.validator.validate_cache_candidate(
            candidate_entry=best_entry,
            similarity_score=best_score,
            current_doc_ids=current_doc_ids,
            retrieved_chunks=retrieved_chunks
        )

        if is_valid:
            # Update access count and timestamp in persistent storage
            self.db.update_cache_access(best_entry["cache_id"])
            return {
                "status": CacheValidationStatus.HIT_VALID,
                "cached_answer": best_entry["answer_text"],
                "similarity_score": round(best_score, 4),
                "validation_reason": reason,
                "candidate": best_entry
            }
        else:
            return {
                "status": status_code,
                "cached_answer": None,
                "similarity_score": round(best_score, 4),
                "validation_reason": reason,
                "candidate": best_entry
            }

    def store(
        self,
        query: str,
        answer: str,
        source_doc_ids: List[str],
        source_chunks: List[Dict[str, Any]],
        model_used: str,
        grounding_score: float
    ) -> Optional[str]:
        """Store verified answer into semantic cache."""
        if not self.config.enabled:
            return None

        # Do not cache answers with substandard grounding
        if grounding_score < self.config.min_cache_grounding_score:
            logger.info("Skipping cache storage: grounding score %f below threshold.", grounding_score)
            return None

        cache_id = f"cache_{uuid.uuid4().hex[:10]}"
        query_vec = self.embedding_service.embed_query(query).tolist()

        self.db.insert_cache_entry(
            cache_id=cache_id,
            query_text=query,
            query_embedding=query_vec,
            answer_text=answer,
            source_doc_ids=source_doc_ids,
            source_chunks=source_chunks,
            model_used=model_used,
            grounding_score=grounding_score
        )
        return cache_id
