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
def test_semantic_cache_disabled(cache_system):
    cache, db = cache_system

    # Create a cache with caching disabled
    disabled_cache = SemanticCache(
        db_manager=db,
        embedding_service=cache.embedding_service,
        caching_config=CachingConfig(enabled=False)
    )

    # Lookup should return MISS
    result = disabled_cache.lookup(
        query="What is green computing?"
    )

    assert result["status"] == CacheValidationStatus.MISS
    assert result["cached_answer"] is None

    # Storing an answer should be skipped
    cache_id = disabled_cache.store(
        query="What is green computing?",
        answer="Green computing reduces environmental impact.",
        source_doc_ids=["doc_green"],
        source_chunks=[],
        model_used="test-model",
        grounding_score=0.95
    )

    assert cache_id is None

    # Database should remain empty
    assert db.get_active_cache_entries() == []

def test_cache_access_count_increases_on_valid_hit(cache_system):
    cache, db = cache_system

    # Store an answer
    cache.store(
        query="What is green computing?",
        answer="Green computing reduces environmental impact.",
        source_doc_ids=["doc_green"],
        source_chunks=[],
        model_used="test-model",
        grounding_score=0.95
    )

    # Initially, access count should be 1
    entries = db.get_active_cache_entries()
    assert entries[0]["access_count"] == 1

    # First successful lookup
    result = cache.lookup(
        query="What is green computing?",
        current_doc_ids=["doc_green"]
    )

    assert result["status"] == CacheValidationStatus.HIT_VALID

    # Access count should now be 2
    entries = db.get_active_cache_entries()
    assert entries[0]["access_count"] == 2

    # Second successful lookup
    cache.lookup(
        query="What is green computing?",
        current_doc_ids=["doc_green"]
    )

    # Access count should now be 3
    entries = db.get_active_cache_entries()
    assert entries[0]["access_count"] == 3

def test_cache_persistence_after_reinitialization(tmp_path):
    db_path = tmp_path / "persistent_cache.db"

    # Create the first cache instance
    db1 = DatabaseManager(db_path)
    embedding_service = EmbeddingService()
    config = CachingConfig()

    cache1 = SemanticCache(db1, embedding_service, config)

    # Store an answer
    cache1.store(
        query="What is green computing?",
        answer="Green computing reduces environmental impact.",
        source_doc_ids=["doc_green"],
        source_chunks=[],
        model_used="test-model",
        grounding_score=0.95
    )

    # Simulate application reinitialization:
    # Create a new database manager and cache instance
    db2 = DatabaseManager(db_path)

    cache2 = SemanticCache(
        db2,
        embedding_service,
        config
    )

    # Retrieve the previously stored answer
    result = cache2.lookup(
        query="What is green computing?",
        current_doc_ids=["doc_green"]
    )

    assert result["status"] == CacheValidationStatus.HIT_VALID
    assert result["cached_answer"] == (
        "Green computing reduces environmental impact."
    )
