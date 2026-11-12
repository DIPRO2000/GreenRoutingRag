import json

import numpy as np

from greenrouting.cache_storage import QualityAwareCacheStore
from greenrouting.cache_validator import CacheValidationStatus
from greenrouting.config import CachingConfig
from greenrouting.database import DatabaseManager
from greenrouting.semantic_cache import SemanticCache


class FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.values = {}
        self.fail_commands = False

    def ping(self):
        return True

    def exists(self, key):
        self._check()
        return int(key in self.values)

    def set(self, key, value):
        self._check()
        self.values[key] = value

    def hset(self, name, mapping):
        self._check()
        self.hashes.setdefault(name, {}).update(mapping)

    def hget(self, name, key):
        self._check()
        return self.hashes.get(name, {}).get(key)

    def hgetall(self, name):
        self._check()
        return dict(self.hashes.get(name, {}))

    def delete(self, *keys):
        self._check()
        for key in keys:
            self.values.pop(key, None)
            self.hashes.pop(key, None)

    def _check(self):
        if self.fail_commands:
            raise ConnectionError("Redis connection lost")


class FixedEmbeddingService:
    def embed_query(self, _query):
        return np.array([1.0, 0.0], dtype=np.float32)


def test_quality_aware_cache_uses_redis_and_validates_context(tmp_path):
    db = DatabaseManager(tmp_path / "quality-cache.db")
    redis_client = FakeRedis()
    storage = QualityAwareCacheStore(db, redis_client=redis_client)
    cache = SemanticCache(
        db,
        FixedEmbeddingService(),
        CachingConfig(semantic_similarity_threshold=0.8),
        cache_storage=storage,
    )
    chunks = [{"doc_id": "doc-green", "text_content": "Solar panels convert sunlight into electricity."}]
    cache.store(
        query="What do solar panels do?",
        answer="Solar panels convert sunlight into electricity.",
        source_doc_ids=["doc-green"],
        source_chunks=chunks,
        model_used="small-model",
        grounding_score=0.95,
    )

    rejected = cache.lookup("What do solar panels do?", current_doc_ids=["doc-other"])
    accepted = cache.lookup("What do solar panels do?", current_doc_ids=["doc-green"])

    assert storage.backend_name == "Redis"
    assert rejected["status"] == CacheValidationStatus.HIT_REJECTED_CONTEXT_MISMATCH
    assert accepted["status"] == CacheValidationStatus.HIT_VALID
    assert accepted["cached_answer"] == "Solar panels convert sunlight into electricity."
    redis_entry = json.loads(next(iter(redis_client.hgetall(storage.HASH_KEY).values())))
    assert redis_entry["access_count"] == 2
    assert db.get_active_cache_entries()[0]["access_count"] == 2

    db.delete_document("doc-green")
    storage.invalidate_document("doc-green")
    assert storage.get_active_cache_entries() == []


def test_redis_hydrates_from_sqlite_and_falls_back_after_connection_loss(tmp_path):
    db = DatabaseManager(tmp_path / "cache-migration.db")
    db.insert_cache_entry(
        cache_id="legacy-entry",
        query_text="legacy query",
        query_embedding=[1.0, 0.0],
        answer_text="legacy answer",
        source_doc_ids=["doc-1"],
        source_chunks=[],
        model_used="test-model",
        grounding_score=0.9,
    )
    redis_client = FakeRedis()
    storage = QualityAwareCacheStore(db, redis_client=redis_client)

    assert storage.get_active_cache_entries()[0]["cache_id"] == "legacy-entry"
    assert redis_client.hgetall(storage.HASH_KEY)

    redis_client.fail_commands = True
    entries = storage.get_active_cache_entries()

    assert storage.backend_name == "SQLite fallback"
    assert entries[0]["answer_text"] == "legacy answer"


def test_cache_snapshot_clear_and_restore_stays_in_sync_with_redis(tmp_path):
    db = DatabaseManager(tmp_path / "cache-benchmark.db")
    redis_client = FakeRedis()
    storage = QualityAwareCacheStore(db, redis_client=redis_client)
    storage.insert_cache_entry(
        cache_id="saved-entry",
        query_text="saved query",
        query_embedding=[1.0],
        answer_text="saved answer",
        source_doc_ids=["doc-1"],
        source_chunks=[],
        model_used="test-model",
        grounding_score=0.9,
    )
    snapshot = storage.snapshot_cache_rows()

    storage.clear_all_cache()
    assert storage.get_active_cache_entries() == []

    storage.restore_cache_rows(snapshot)

    assert storage.get_active_cache_entries()[0]["cache_id"] == "saved-entry"
