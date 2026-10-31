"""
Unit tests for Semantic Cache integration in GreenRouting.
"""
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
