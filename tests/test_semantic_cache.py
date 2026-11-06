"""
Unit tests for Semantic Cache integration in GreenRouting.
"""
import numpy as np
import pytest
from pathlib import Path
from greenrouting.config import CachingConfig
from greenrouting.database import DatabaseManager
from greenrouting.vector_store import EmbeddingService
from greenrouting.semantic_cache import SemanticCache
from greenrouting.cache_validator import CacheValidationStatus


@pytest.fixture
def cache_system(tmp_path):
    db_path = tmp_path / "test_cache.db"
    db_manager = DatabaseManager(db_path)
    embedding_service = EmbeddingService()
    config = CachingConfig(semantic_similarity_threshold=0.85)
    cache = SemanticCache(db_manager, embedding_service, config)
    return cache, db_manager


def test_semantic_cache_hit_and_miss(cache_system):
    cache, _ = cache_system

    # 1. Miss on empty cache
    res = cache.lookup("What is solar energy efficiency?")
    assert res["status"] == CacheValidationStatus.MISS
    assert res["cached_answer"] is None

    # 2. Store answer
    doc_ids = ["doc_green"]
    chunks = [{"text_content": "Solar panels convert sunlight to electricity with 20% efficiency."}]
    cache.store(
        query="What is the efficiency of solar panels?",
        answer="Solar panels convert sunlight into electricity at approximately 20% efficiency.",
        source_doc_ids=doc_ids,
        source_chunks=chunks,
        model_used="qwen2.5:0.5b",
        grounding_score=0.92
    )

    # 3. Hit on semantically similar query with matching document context
    lookup_hit = cache.lookup(
        query="What is solar panel efficiency?",
        current_doc_ids=doc_ids,
        retrieved_chunks=chunks
    )
    assert lookup_hit["status"] == CacheValidationStatus.HIT_VALID
    assert "20% efficiency" in lookup_hit["cached_answer"]
    assert lookup_hit["similarity_score"] >= 0.85

    # 4. Miss on semantically unrelated query
    lookup_miss = cache.lookup(
        query="Explain deep ocean volcanic hydrothermal vents.",
        current_doc_ids=doc_ids,
        retrieved_chunks=chunks
    )
    assert lookup_miss["status"] == CacheValidationStatus.MISS
    assert lookup_miss["cached_answer"] is None


def test_cache_tries_next_candidate_when_best_candidate_is_invalid(cache_system, monkeypatch):
    cache, _ = cache_system

    # Give each question a controlled embedding so the similarity
    # scores are predictable.
    embeddings = {
        "wrong document question": np.array([1.0, 0.0], dtype=np.float32),
        "valid document question": np.array([0.95, 0.31225], dtype=np.float32),
        "new user question": np.array([1.0, 0.0], dtype=np.float32)
    }

    def fake_embed_query(query):
        return embeddings[query]

    monkeypatch.setattr(
        cache.embedding_service,
        "embed_query",
        fake_embed_query
    )

    # Store the highest-similarity answer, but from the wrong document.
    cache.store(
        query="wrong document question",
        answer="This answer belongs to another document.",
        source_doc_ids=["doc_wrong"],
        source_chunks=[],
        model_used="test-model",
        grounding_score=0.95
    )

    # Store a slightly less similar answer from the correct document.
    cache.store(
        query="valid document question",
        answer="This is the correct solar energy answer.",
        source_doc_ids=["doc_valid"],
        source_chunks=[],
        model_used="test-model",
        grounding_score=0.95
    )

    # Search using the correct document context.
    result = cache.lookup(
        query="new user question",
        current_doc_ids=["doc_valid"]
    )

    # The cache should reject the first candidate and try the second.
    assert result["status"] == CacheValidationStatus.HIT_VALID
    assert result["cached_answer"] == "This is the correct solar energy answer."