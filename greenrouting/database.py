"""
SQLite Database Operations for GreenRouting / EcoWave.
Manages documents, semantic cache, query logs, and benchmark results.
Supports transparent energy telemetry, stage latency breakdowns, simulation tracking, and baseline comparisons.
"""
import sqlite3
import json
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Handles thread-safe SQLite operations for GreenRouting / EcoWave."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_tables()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init_tables(self) -> None:
        """Create schema if it does not exist and migrate existing schema."""
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
                    query_id TEXT,
                    query_text TEXT NOT NULL,
                    answer_text TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    cache_status TEXT NOT NULL,
                    cache_validation_reason TEXT,
                    complexity_score REAL,
                    routing_reason TEXT,
                    retry_count INTEGER DEFAULT 0,
                    grounding_score REAL,
                    quality_score REAL,
                    latency_seconds REAL NOT NULL,
                    stage_latencies_json TEXT,
                    energy_wh REAL,
                    energy_kwh REAL,
                    emissions_g REAL,
                    emissions_kg REAL,
                    energy_source TEXT,
                    simulation_mode INTEGER DEFAULT 0,
                    source_chunks_json TEXT,
                    baseline_energy_wh REAL,
                    baseline_emissions_g REAL,
                    energy_savings_pct REAL,
                    carbon_savings_g REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Benchmark Results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    run_id TEXT PRIMARY KEY,
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Schema migration for existing databases: check for missing columns and add them safely
            cursor.execute("PRAGMA table_info(query_logs)")
            existing_columns = {row["name"] for row in cursor.fetchall()}
            desired_columns = {
                "query_id": "TEXT",
                "simulation_mode": "INTEGER DEFAULT 0",
                "stage_latencies_json": "TEXT",
                "energy_wh": "REAL",
                "emissions_g": "REAL",
                "quality_score": "REAL",
                "source_chunks_json": "TEXT",
                "baseline_energy_wh": "REAL",
                "baseline_emissions_g": "REAL",
                "energy_savings_pct": "REAL",
                "carbon_savings_g": "REAL",
            }
            for col_name, col_type in desired_columns.items():
                if col_name not in existing_columns:
                    try:
                        cursor.execute(f"ALTER TABLE query_logs ADD COLUMN {col_name} {col_type}")
                    except Exception as e:
                        logger.warning("Could not add column %s to query_logs: %s", col_name, e)

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
                    if doc_id in source_doc_ids:
                        cursor.execute("UPDATE semantic_cache SET is_active = 0 WHERE cache_id = ?", (cache_id,))
                        invalidated_cache_count += 1
                except Exception:
                    pass

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

    # --- Query Logging ---
    def log_query(self, log_entry: Dict[str, Any]) -> int:
        """
        Persist a single complete query execution record.
        Ensures query_text is present, prevents duplicate records, and guarantees compatibility.
        """
        query_text = log_entry.get("query_text")
        if not query_text:
            logger.warning("Attempted to log query with missing or empty query_text; skipping.")
            return -1

        query_id = log_entry.get("query_id") or f"query_{uuid.uuid4().hex[:12]}"

        # Stage latencies formatting
        stage_latencies = log_entry.get("stage_latencies") or log_entry.get("stage_latencies_json")
        if isinstance(stage_latencies, dict):
            stage_latencies_json = json.dumps(stage_latencies)
        else:
            stage_latencies_json = str(stage_latencies or "{}")

        # Source chunks formatting
        source_chunks = log_entry.get("source_chunks") or log_entry.get("sources") or log_entry.get("source_chunks_json")
        if isinstance(source_chunks, (list, dict)):
            source_chunks_json = json.dumps(source_chunks)
        else:
            source_chunks_json = str(source_chunks or "[]")

        # Energy & Emissions in both Wh and kWh
        energy_wh = log_entry.get("energy_wh")
        if energy_wh is None and "energy_kwh" in log_entry:
            energy_wh = float(log_entry["energy_kwh"]) * 1000.0
        energy_wh = float(energy_wh or 0.0)

        emissions_g = log_entry.get("emissions_g")
        if emissions_g is None and "emissions_kg" in log_entry:
            emissions_g = float(log_entry["emissions_kg"]) * 1000.0
        emissions_g = float(emissions_g or 0.0)

        energy_kwh = log_entry.get("energy_kwh", energy_wh / 1000.0)
        emissions_kg = log_entry.get("emissions_kg", emissions_g / 1000.0)

        simulation_mode = 1 if log_entry.get("simulation_mode") or log_entry.get("is_simulated") else 0

        params = {
            "query_id": query_id,
            "query_text": query_text,
            "answer_text": log_entry.get("answer_text", log_entry.get("answer", ""))[:1000],
            "mode": log_entry.get("mode", ""),
            "model_used": log_entry.get("model_used", ""),
            "cache_status": str(log_entry.get("cache_status", "")),
            "cache_validation_reason": log_entry.get("cache_validation_reason", ""),
            "complexity_score": float(log_entry.get("complexity_score", 0.0)),
            "routing_reason": log_entry.get("routing_reason", ""),
            "retry_count": int(log_entry.get("retry_count", 0)),
            "grounding_score": float(log_entry.get("grounding_score", 0.0)),
            "quality_score": float(log_entry.get("quality_score", 0.0)),
            "latency_seconds": float(log_entry.get("latency_seconds", 0.0)),
            "stage_latencies_json": stage_latencies_json,
            "energy_wh": round(energy_wh, 8),
            "energy_kwh": round(energy_kwh, 9),
            "emissions_g": round(emissions_g, 6),
            "emissions_kg": round(emissions_kg, 9),
            "energy_source": log_entry.get("energy_source", log_entry.get("measurement_mode", "modeled_tdp_estimate")),
            "simulation_mode": simulation_mode,
            "source_chunks_json": source_chunks_json,
            "baseline_energy_wh": float(log_entry.get("baseline_energy_wh", 0.0)),
            "baseline_emissions_g": float(log_entry.get("baseline_emissions_g", 0.0)),
            "energy_savings_pct": float(log_entry.get("energy_savings_pct", 0.0)),
            "carbon_savings_g": float(log_entry.get("carbon_savings_g", 0.0)),
        }

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO query_logs 
(query_id, query_text, answer_text, mode, model_used, cache_status, cache_validation_reason,
 complexity_score, routing_reason, retry_count, grounding_score, quality_score, latency_seconds,
                 stage_latencies_json, energy_wh, energy_kwh, emissions_g, emissions_kg, energy_source,
                 simulation_mode, source_chunks_json, baseline_energy_wh, baseline_emissions_g,
                 energy_savings_pct, carbon_savings_g)
                VALUES (:query_id, :query_text, :answer_text, :mode, :model_used, :cache_status, :cache_validation_reason,
                        :complexity_score, :routing_reason, :retry_count, :grounding_score, :quality_score, :latency_seconds,
                        :stage_latencies_json, :energy_wh, :energy_kwh, :emissions_g, :emissions_kg, :energy_source,
                        :simulation_mode, :source_chunks_json, :baseline_energy_wh, :baseline_emissions_g,
                        :energy_savings_pct, :carbon_savings_g)
            """, params)
            conn.commit()
            return cursor.lastrowid

    def get_query_logs(
        self,
        limit: int = 100,
        simulation_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve query logs with optional filtering by execution type:
        - "real": only real-model executions (simulation_mode == 0)
        - "simulation": only simulation executions (simulation_mode == 1)
        - "all" or None: all records
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if simulation_filter == "real":
                cursor.execute(
                    "SELECT * FROM query_logs WHERE simulation_mode = 0 ORDER BY timestamp DESC, log_id DESC LIMIT ?",
                    (limit,)
                )
            elif simulation_filter == "simulation":
                cursor.execute(
                    "SELECT * FROM query_logs WHERE simulation_mode = 1 ORDER BY timestamp DESC, log_id DESC LIMIT ?",
                    (limit,)
                )
            else:
                cursor.execute(
                    "SELECT * FROM query_logs ORDER BY timestamp DESC, log_id DESC LIMIT ?",
                    (limit,)
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_query_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Backwards compatible helper returning recent query logs."""
        return self.get_query_logs(limit=limit, simulation_filter="all")

    # --- Benchmark Results ---
    def save_benchmark_result(self, result: Dict[str, Any]) -> None:
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO benchmark_results
                (run_id, benchmark_name, mode, total_queries, cache_hits, lightweight_count, capable_count,
                 retry_count, avg_latency_sec, total_energy_kwh, total_emissions_kg, avg_grounding_score)
                VALUES (:run_id, :benchmark_name, :mode, :total_queries, :cache_hits, :lightweight_count, :capable_count,
                        :retry_count, :avg_latency_sec, :total_energy_kwh, :total_emissions_kg, :avg_grounding_score)
            """, result)
            conn.commit()

    def get_all_benchmark_results(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM benchmark_results ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]
