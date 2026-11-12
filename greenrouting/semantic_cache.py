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
from greenrouting.cache_storage import QualityAwareCacheStore

logger = logging.getLogger(__name__)


class SemanticCache:
    """Manages storage, retrieval, cosine matching, and quality validation of semantic cache entries."""

    def __init__(self, db_manager: DatabaseManager, embedding_service: EmbeddingService,
                 caching_config: Optional[CachingConfig] = None,
                 cache_storage: Optional[QualityAwareCacheStore] = None):
        self.db = db_manager
        self.embedding_service = embedding_service
        self.config = caching_config or CachingConfig()
        self.validator = CacheQualityValidator(self.config)
        self.storage = cache_storage or QualityAwareCacheStore(
            db_manager,
            redis_url=self.config.redis_url,
        )

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

        entries = self.storage.get_active_cache_entries()
        if not entries:
            return {
                "status": CacheValidationStatus.MISS,
                "cached_answer": None,
                "similarity_score": 0.0,
                "validation_reason": "Semantic cache is empty.",
                "candidate": None
            }
        # Embed the input query
        query_vec = self.embedding_service.embed_query(query)
        q_norm = np.linalg.norm(query_vec)

        if q_norm > 1e-9:
            query_vec = query_vec / q_norm

        # Calculate similarity with every cached entry
        candidates = []
        best_score = -1.0

        for entry in entries:
            emb_list = entry.get("query_embedding")

            if not emb_list:
                continue

            cached_vec = np.array(emb_list, dtype=np.float32)

            # Skip embeddings with incompatible dimensions
            if cached_vec.shape != query_vec.shape:
                continue

            c_norm = np.linalg.norm(cached_vec)

            if c_norm > 1e-9:
                cached_vec = cached_vec / c_norm

            sim = float(np.dot(query_vec, cached_vec))

            best_score = max(best_score, sim)

            # Keep only candidates above the similarity threshold
            if sim >= self.config.semantic_similarity_threshold:
                candidates.append((sim, entry))

        # No sufficiently similar candidate was found
        if not candidates:
            return {
                "status": CacheValidationStatus.MISS,
                "cached_answer": None,
                "similarity_score": round(max(0.0, best_score), 4),
                "validation_reason": (
                    f"No semantically similar query found "
                    f"(Best similarity: {max(0.0, best_score):.3f} "
                    f"< threshold "
                    f"{self.config.semantic_similarity_threshold:.3f})."
                ),
                "candidate": None
            }

        # Try candidates from highest to lowest similarity
        candidates.sort(key=lambda item: item[0], reverse=True)

        best_rejected = None

        for similarity, candidate in candidates:

            # Validate document context, freshness and answer quality
            is_valid, status_code, reason = (
                self.validator.validate_cache_candidate(
                    candidate_entry=candidate,
                    similarity_score=similarity,
                    current_doc_ids=current_doc_ids,
                    retrieved_chunks=retrieved_chunks
                )
            )

            if is_valid:
                # Update access count only for a valid cache hit
                self.storage.update_cache_access(candidate["cache_id"])

                return {
                    "status": CacheValidationStatus.HIT_VALID,
                    "cached_answer": candidate["answer_text"],
                    "similarity_score": round(similarity, 4),
                    "validation_reason": reason,
                    "candidate": candidate
                }

            # Remember the highest-similarity rejected candidate
            if best_rejected is None:
                best_rejected = {
                    "status": status_code,
                    "similarity_score": similarity,
                    "validation_reason": reason,
                    "candidate": candidate
                }

        # Similar candidates existed, but all failed validation
        return {
            "status": best_rejected["status"],
            "cached_answer": None,
            "similarity_score": round(
                best_rejected["similarity_score"], 4
            ),
            "validation_reason": best_rejected["validation_reason"],
            "candidate": best_rejected["candidate"]
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

        self.storage.insert_cache_entry(
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
