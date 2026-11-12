"""
SQLite Database Operations for GreenRouting.
Manages documents, semantic cache, query logs, and benchmark results.
"""
import sqlite3
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Handles thread-safe SQLite operations for GreenRouting."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_tables()

    @contextmanager
    def get_connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()
        finally:
            conn.close()

    def init_tables(self) -> None:
        """Create schema if it does not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    doc_hash TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    total_chunks INTEGER NOT NULL,
                    total_characters INTEGER NOT NULL,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Document Chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text_content TEXT NOT NULL,
                    page_number INTEGER,
                    character_count INTEGER NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
                )
            """)

            # Semantic Cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    cache_id TEXT PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    query_embedding_json TEXT NOT NULL,
                    answer_text TEXT NOT NULL,
                    source_doc_ids TEXT NOT NULL,
                    source_chunks_json TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    grounding_score REAL NOT NULL,
                    access_count INTEGER DEFAULT 1,
                    is_active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Query Logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT NOT NULL,
                    answer_text TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    cache_status TEXT NOT NULL, -- HIT, MISS, INVALIDATED
                    cache_validation_reason TEXT,
                    complexity_score REAL,
                    routing_reason TEXT,
                    retry_count INTEGER DEFAULT 0,
                    grounding_score REAL,
                    latency_seconds REAL NOT NULL,
                    energy_kwh REAL,
                    emissions_kg REAL,
                    energy_source TEXT, -- DIRECT_SENSOR, MODELED_TDP, SIMULATED
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Benchmark Results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    run_id TEXT NOT NULL,
                    benchmark_name TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    total_queries INTEGER NOT NULL,
                    cache_hits INTEGER NOT NULL,
                    lightweight_count INTEGER NOT NULL,
                    capable_count INTEGER NOT NULL,
                    retry_count INTEGER NOT NULL,
                    avg_latency_sec REAL NOT NULL,
                    total_energy_kwh REAL NOT NULL,
                    total_emissions_kg REAL NOT NULL,
                    avg_grounding_score REAL NOT NULL,
                    avg_quality_score REAL NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, mode)
                )
            """)
            benchmark_columns = cursor.execute("PRAGMA table_info(benchmark_results)").fetchall()
            primary_key = [row["name"] for row in sorted(benchmark_columns, key=lambda row: row["pk"]) if row["pk"]]
            if primary_key != ["run_id", "mode"]:
                cursor.execute("ALTER TABLE benchmark_results RENAME TO benchmark_results_legacy")
                cursor.execute("""
                    CREATE TABLE benchmark_results (
                        run_id TEXT NOT NULL,
                        benchmark_name TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        total_queries INTEGER NOT NULL,
                        cache_hits INTEGER NOT NULL,
                        lightweight_count INTEGER NOT NULL,
                        capable_count INTEGER NOT NULL,
                        retry_count INTEGER NOT NULL,
                        avg_latency_sec REAL NOT NULL,
                        total_energy_kwh REAL NOT NULL,
                        total_emissions_kg REAL NOT NULL,
                        avg_grounding_score REAL NOT NULL,
                        avg_quality_score REAL NOT NULL DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (run_id, mode)
                    )
                """)
                legacy_columns = {row["name"] for row in cursor.execute("PRAGMA table_info(benchmark_results_legacy)")}
                # Older releases never stored answer quality, so do not infer it
                # from grounding; preserve that history as unavailable (0).
                quality_expr = "avg_quality_score" if "avg_quality_score" in legacy_columns else "0"
                cursor.execute(f"""
                    INSERT OR IGNORE INTO benchmark_results
                    (run_id, benchmark_name, mode, total_queries, cache_hits, lightweight_count,
                     capable_count, retry_count, avg_latency_sec, total_energy_kwh,
                     total_emissions_kg, avg_grounding_score, avg_quality_score, created_at)
                    SELECT run_id, benchmark_name, mode, total_queries, cache_hits, lightweight_count,
                           capable_count, retry_count, avg_latency_sec, total_energy_kwh,
                           total_emissions_kg, avg_grounding_score, {quality_expr}, created_at
                    FROM benchmark_results_legacy
                """)
                cursor.execute("DROP TABLE benchmark_results_legacy")
            elif "avg_quality_score" not in {row["name"] for row in benchmark_columns}:
                cursor.execute("ALTER TABLE benchmark_results ADD COLUMN avg_quality_score REAL NOT NULL DEFAULT 0")
            conn.commit()

    # --- Document helpers ---
    def save_document(self, doc_id: str, filename: str, doc_hash: str, file_type: str, total_chunks: int, total_characters: int) -> None:
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO documents (doc_id, filename, doc_hash, file_type, total_chunks, total_characters)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (doc_id, filename, doc_hash, file_type, total_chunks, total_characters))
            conn.commit()

    def save_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        with self.get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO document_chunks (chunk_id, doc_id, chunk_index, text_content, page_number, character_count)
                VALUES (:chunk_id, :doc_id, :chunk_index, :text_content, :page_number, :character_count)
            """, chunks)
            conn.commit()

    def get_all_documents(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents ORDER BY uploaded_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Return all persisted chunks in stable document/chunk order."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT chunk_id, doc_id, chunk_index, text_content, page_number, character_count
                FROM document_chunks
                ORDER BY doc_id, chunk_index
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Return every stored chunk for one document, in original chunk order."""
        if not doc_id:
            return []
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, doc_id, chunk_index, text_content, page_number, character_count
                FROM document_chunks
                WHERE doc_id = ?
                ORDER BY chunk_index ASC
                """,
                (doc_id,),
            )
            return [dict(row) for row in rows]

    def get_document_by_hash(self, doc_hash: str) -> Optional[Dict[str, Any]]:
        """Return the first indexed document with this content hash, if any."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM documents
                WHERE doc_hash = ?
                ORDER BY uploaded_at ASC, rowid ASC
                LIMIT 1
                """,
                (doc_hash,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_document(self, doc_id: str) -> Dict[str, int]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM document_chunks WHERE doc_id = ?", (doc_id,))
            removed_chunk_count = int(cursor.fetchone()[0])

            cursor.execute("SELECT cache_id, source_doc_ids FROM semantic_cache WHERE is_active = 1")
            invalidated_cache_count = 0
            for cache_id, source_doc_ids_json in cursor.fetchall():
                try:
                    source_doc_ids = json.loads(source_doc_ids_json)
                except (TypeError, json.JSONDecodeError):
                    source_doc_ids = []
                if doc_id in source_doc_ids:
                    cursor.execute(
                        "UPDATE semantic_cache SET is_active = 0 WHERE cache_id = ?",
                        (cache_id,),
                    )
                    invalidated_cache_count += 1

            cursor.execute("DELETE FROM document_chunks WHERE doc_id = ?", (doc_id,))
            cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            conn.commit()
            return {
                "removed_chunk_count": removed_chunk_count,
                "invalidated_cache_count": invalidated_cache_count,
            }

    # --- Semantic Cache helpers ---
    def insert_cache_entry(self, cache_id: str, query_text: str, query_embedding: List[float],
                           answer_text: str, source_doc_ids: List[str], source_chunks: List[Dict[str, Any]],
                           model_used: str, grounding_score: float) -> None:
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO semantic_cache 
                (cache_id, query_text, query_embedding_json, answer_text, source_doc_ids, source_chunks_json, model_used, grounding_score, access_count, is_active, created_at, last_accessed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, (
                cache_id,
                query_text,
                json.dumps(query_embedding),
                answer_text,
                json.dumps(source_doc_ids),
                json.dumps(source_chunks),
                model_used,
                grounding_score
            ))
            conn.commit()

    def get_active_cache_entries(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM semantic_cache WHERE is_active = 1")
            entries = []
            for row in cursor.fetchall():
                d = dict(row)
                d["query_embedding"] = json.loads(d["query_embedding_json"])
                d["source_doc_ids"] = json.loads(d["source_doc_ids"])
                d["source_chunks"] = json.loads(d["source_chunks_json"])
                entries.append(d)
            return entries

    def update_cache_access(self, cache_id: str) -> None:
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE semantic_cache
                SET access_count = access_count + 1, last_accessed_at = CURRENT_TIMESTAMP
                WHERE cache_id = ?
            """, (cache_id,))
            conn.commit()

    def invalidate_cache_entry(self, cache_id: str) -> None:
        with self.get_connection() as conn:
            conn.execute("UPDATE semantic_cache SET is_active = 0 WHERE cache_id = ?", (cache_id,))
            conn.commit()

    def clear_all_cache(self) -> None:
        with self.get_connection() as conn:
            conn.execute("DELETE FROM semantic_cache")
            conn.commit()

    def snapshot_cache_rows(self) -> List[Dict[str, Any]]:
        """Return raw cache rows, including inactive entries, for isolated benchmarks."""
        with self.get_connection() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM semantic_cache")]

    def restore_cache_rows(self, rows: List[Dict[str, Any]]) -> None:
        """Replace cache contents with a previously captured exact snapshot."""
        columns = (
            "cache_id, query_text, query_embedding_json, answer_text, source_doc_ids, "
            "source_chunks_json, model_used, grounding_score, access_count, is_active, "
            "created_at, last_accessed_at"
        )
        placeholders = ", ".join("?" for _ in columns.split(", "))
        with self.get_connection() as conn:
            conn.execute("DELETE FROM semantic_cache")
            if rows:
                conn.executemany(
                    f"INSERT INTO semantic_cache ({columns}) VALUES ({placeholders})",
                    [tuple(row[column.strip()] for column in columns.split(", ")) for row in rows],
                )
            conn.commit()

    # --- Query Logging ---
    def log_query(self, log_entry: Dict[str, Any]) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO query_logs 
                (query_text, answer_text, mode, model_used, cache_status, cache_validation_reason, 
                 complexity_score, routing_reason, retry_count, grounding_score, latency_seconds, 
                 energy_kwh, emissions_kg, energy_source)
                VALUES (:query_text, :answer_text, :mode, :model_used, :cache_status, :cache_validation_reason,
                        :complexity_score, :routing_reason, :retry_count, :grounding_score, :latency_seconds,
                        :energy_kwh, :emissions_kg, :energy_source)
            """, log_entry)
            conn.commit()
            return cursor.lastrowid

    def get_recent_query_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM query_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    # --- Benchmark Results ---
    def save_benchmark_result(self, result: Dict[str, Any]) -> None:
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO benchmark_results
                (run_id, benchmark_name, mode, total_queries, cache_hits, lightweight_count, capable_count,
                 retry_count, avg_latency_sec, total_energy_kwh, total_emissions_kg, avg_grounding_score, avg_quality_score)
                VALUES (:run_id, :benchmark_name, :mode, :total_queries, :cache_hits, :lightweight_count, :capable_count,
                        :retry_count, :avg_latency_sec, :total_energy_kwh, :total_emissions_kg, :avg_grounding_score, :avg_quality_score)
            """, result)
            conn.commit()

    def get_all_benchmark_results(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM benchmark_results ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]
