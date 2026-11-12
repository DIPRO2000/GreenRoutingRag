"""Redis-backed cache storage with SQLite durability and offline fallback."""
import json
import logging
from typing import Any, Dict, List, Optional

from greenrouting.database import DatabaseManager

logger = logging.getLogger(__name__)


class QualityAwareCacheStore:
    """Keep SQLite as the durable source and use Redis for active cache reads."""

    HASH_KEY = "greenrouting:semantic_cache:entries"
    READY_KEY = "greenrouting:semantic_cache:sqlite_sync_complete"

    def __init__(self, database: DatabaseManager, redis_url: Optional[str] = None, redis_client: Any = None):
        self.database = database
        self.redis = None
        if redis_client is not None:
            self._connect(redis_client)
        elif redis_url:
            try:
                import redis

                client = redis.Redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=0.2,
                    socket_timeout=0.5,
                )
                self._connect(client)
            except Exception as exc:
                logger.info("Redis unavailable (%s); using SQLite cache storage.", type(exc).__name__)

    @property
    def backend_name(self) -> str:
        return "Redis" if self.redis is not None else "SQLite fallback"

    def _connect(self, client: Any) -> None:
        try:
            client.ping()
            self.redis = client
        except Exception as exc:
            logger.info("Redis unavailable (%s); using SQLite cache storage.", type(exc).__name__)

    def _disable_redis(self, exc: Exception) -> None:
        logger.warning("Redis cache operation failed (%s); switching to SQLite.", type(exc).__name__)
        self.redis = None

    @staticmethod
    def _decode_database_row(row: Dict[str, Any]) -> Dict[str, Any]:
        entry = dict(row)
        entry["query_embedding"] = json.loads(entry["query_embedding_json"])
        entry["source_doc_ids"] = json.loads(entry["source_doc_ids"])
        entry["source_chunks"] = json.loads(entry["source_chunks_json"])
        return entry

    def _ensure_redis_ready(self) -> None:
        if not self.redis.exists(self.READY_KEY):
            rows = self.database.snapshot_cache_rows()
            mapping = {
                row["cache_id"]: json.dumps(self._decode_database_row(row), ensure_ascii=False)
                for row in rows
            }
            self.redis.delete(self.HASH_KEY)
            if mapping:
                self.redis.hset(self.HASH_KEY, mapping=mapping)
            self.redis.set(self.READY_KEY, "1")

    def get_active_cache_entries(self) -> List[Dict[str, Any]]:
        if self.redis is None:
            return self.database.get_active_cache_entries()
        try:
            self._ensure_redis_ready()
            entries = [json.loads(payload) for payload in self.redis.hgetall(self.HASH_KEY).values()]
            return [entry for entry in entries if entry.get("is_active", 1)]
        except Exception as exc:
            self._disable_redis(exc)
            return self.database.get_active_cache_entries()

    def insert_cache_entry(self, **entry: Any) -> None:
        self.database.insert_cache_entry(**entry)
        if self.redis is None:
            return
        try:
            self._ensure_redis_ready()
            stored_entry = next(
                item for item in self.database.get_active_cache_entries()
                if item["cache_id"] == entry["cache_id"]
            )
            self.redis.hset(
                self.HASH_KEY,
                mapping={entry["cache_id"]: json.dumps(stored_entry, ensure_ascii=False)},
            )
        except Exception as exc:
            self._disable_redis(exc)

    def update_cache_access(self, cache_id: str) -> None:
        self.database.update_cache_access(cache_id)
        self._sync_entry_from_database(cache_id)

    def invalidate_cache_entry(self, cache_id: str) -> None:
        self.database.invalidate_cache_entry(cache_id)
        if self.redis is None:
            return
        try:
            self._ensure_redis_ready()
            payload = self.redis.hget(self.HASH_KEY, cache_id)
            if payload:
                entry = json.loads(payload)
                entry["is_active"] = 0
                self.redis.hset(self.HASH_KEY, mapping={cache_id: json.dumps(entry, ensure_ascii=False)})
        except Exception as exc:
            self._disable_redis(exc)

    def invalidate_document(self, doc_id: str) -> None:
        """Mirror SQLite's document-linked cache invalidation into Redis."""
        if self.redis is None:
            return
        try:
            self._ensure_redis_ready()
            for cache_id, payload in self.redis.hgetall(self.HASH_KEY).items():
                entry = json.loads(payload)
                if doc_id in entry.get("source_doc_ids", []):
                    entry["is_active"] = 0
                    self.redis.hset(
                        self.HASH_KEY,
                        mapping={cache_id: json.dumps(entry, ensure_ascii=False)},
                    )
        except Exception as exc:
            self._disable_redis(exc)

    def clear_all_cache(self) -> None:
        self.database.clear_all_cache()
        if self.redis is not None:
            try:
                self.redis.delete(self.HASH_KEY, self.READY_KEY)
            except Exception as exc:
                self._disable_redis(exc)

    def snapshot_cache_rows(self) -> List[Dict[str, Any]]:
        return self.database.snapshot_cache_rows()

    def restore_cache_rows(self, rows: List[Dict[str, Any]]) -> None:
        self.database.restore_cache_rows(rows)
        if self.redis is not None:
            try:
                self.redis.delete(self.HASH_KEY, self.READY_KEY)
                self._ensure_redis_ready()
            except Exception as exc:
                self._disable_redis(exc)

    def _sync_entry_from_database(self, cache_id: str) -> None:
        if self.redis is None:
            return
        try:
            self._ensure_redis_ready()
            stored_entry = next(
                (item for item in self.database.get_active_cache_entries() if item["cache_id"] == cache_id),
                None,
            )
            payload = self.redis.hget(self.HASH_KEY, cache_id)
            if stored_entry is not None:
                self.redis.hset(
                    self.HASH_KEY,
                    mapping={cache_id: json.dumps(stored_entry, ensure_ascii=False)},
                )
            elif payload:
                entry = json.loads(payload)
                entry["is_active"] = 0
                self.redis.hset(self.HASH_KEY, mapping={cache_id: json.dumps(entry, ensure_ascii=False)})
        except Exception as exc:
            self._disable_redis(exc)
