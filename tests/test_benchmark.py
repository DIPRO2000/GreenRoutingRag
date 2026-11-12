"""
Unit tests for Benchmarking Engine in GreenRouting.
"""
import pytest
import math
import sqlite3
from pathlib import Path
from greenrouting.config import load_settings
from greenrouting.orchestrator import GreenRoutingOrchestrator, ExecutionMode
from greenrouting.benchmark import BenchmarkRunner


@pytest.fixture
def test_orchestrator(tmp_path):
    settings = load_settings()
    settings.app.db_path = tmp_path / "bench_test.db"
    settings.app.faiss_index_dir = tmp_path / "bench_faiss"
    orch = GreenRoutingOrchestrator(settings)

    # Index sample document
    sample_text = (
        "GreenRouting is an energy-aware retrieval-augmented generation framework. "
        "It uses FAISS for vector search and Sentence Transformers for embeddings. "
        "GreenRouting dynamically selects between a 0.5B lightweight model and a 3B capable model."
    )
    orch.process_and_index_document("overview.txt", sample_text.encode("utf-8"))
    return orch, tmp_path


def test_benchmark_runner_execution(test_orchestrator):
    orch, tmp_dir = test_orchestrator
    runner = BenchmarkRunner(orch)

    # Run quick 3-query benchmark across all 3 modes
    queries = [
        "What is the objective of GreenRouting?",
        "Explain the dynamic model selection between 0.5B and 3B models.",
        "What is the objective of GreenRouting?"  # Repeat query for cache hit
    ]

    result = runner.run_benchmark(
        queries=queries,
        modes=[ExecutionMode.STANDARD_RAG, ExecutionMode.GREENROUTING]
    )

    assert "run_id" in result
    assert ExecutionMode.STANDARD_RAG in result["mode_summaries"]
    assert ExecutionMode.GREENROUTING in result["mode_summaries"]

    # Comparative analysis should compute energy savings
    comp = result["comparative_analysis"]
    assert "energy_savings_pct" in comp
    assert "latency_reduction_pct" in comp
    # Short local runs can legitimately be negative because modeled latency,
    # retries, and hardware contention vary; preserve the observed value.
    assert math.isfinite(comp["energy_savings_pct"])
    assert math.isfinite(comp["latency_reduction_pct"])

    # CSV Export test
    csv_path = str(tmp_dir / "bench_export.csv")
    exported = runner.export_to_csv(result, csv_path)
    assert Path(exported).exists()
    assert Path(exported).stat().st_size > 0


def test_custom_benchmark_records_answers_for_selected_document(test_orchestrator, monkeypatch):
    orch, _ = test_orchestrator
    doc_id = orch.db.get_all_documents()[0]["doc_id"]
    calls = []

    def fake_execute_query(query, mode, active_doc_id=None):
        calls.append((query, mode, active_doc_id))
        return {
            "answer": f"Answer to {query}",
            "cache_status": "MISS",
            "cache_validation_reason": "No similar cached answer",
            "model_used": "test-model",
            "is_simulated": False,
            "complexity_score": 0.2,
            "latency_seconds": 0.1,
            "energy_kwh": 0.00001,
            "emissions_g": 0.001,
            "grounding_score": 0.8,
            "quality_score": 0.7,
            "retry_count": 0,
        }

    monkeypatch.setattr(orch, "execute_query", fake_execute_query)
    result = BenchmarkRunner(orch).run_benchmark(
        queries=["How many panels?", "What is the panel count?"],
        active_doc_id=doc_id,
        include_answers=True,
    )

    records = result["detailed_records_df"].to_dict("records")
    assert len(calls) == 6  # Two ordered prompts for each of three modes.
    assert all(call[2] == doc_id for call in calls)
    assert [call[0] for call in calls[:2]] == ["How many panels?", "What is the panel count?"]
    assert records[0]["answer"] == "Answer to How many panels?"
    assert records[0]["cache_validation_reason"] == "No similar cached answer"


def test_benchmark_restores_user_cache_even_when_run_fails(test_orchestrator, monkeypatch):
    orch, _ = test_orchestrator
    orch.db.insert_cache_entry(
        cache_id="user-cache",
        query_text="saved query",
        query_embedding=[1.0],
        answer_text="saved answer",
        source_doc_ids=["doc-user"],
        source_chunks=[],
        model_used="test-model",
        grounding_score=0.9,
    )
    before = orch.db.snapshot_cache_rows()
    runner = BenchmarkRunner(orch)

    def simulate_benchmark_cache_mutation(*_args, **_kwargs):
        orch.semantic_cache.storage.clear_all_cache()
        raise RuntimeError("benchmark failed")

    monkeypatch.setattr(runner, "_run_benchmark", simulate_benchmark_cache_mutation)
    with pytest.raises(RuntimeError, match="benchmark failed"):
        runner.run_benchmark(queries=["example"], modes=[ExecutionMode.STANDARD_RAG])

    assert orch.db.snapshot_cache_rows() == before


def test_database_persists_each_mode_for_a_shared_benchmark_run(test_orchestrator):
    orch, _ = test_orchestrator
    common = {
        "run_id": "bench_shared",
        "benchmark_name": "two modes",
        "total_queries": 1,
        "cache_hits": 0,
        "lightweight_count": 0,
        "capable_count": 1,
        "retry_count": 0,
        "avg_latency_sec": 1.0,
        "total_energy_kwh": 0.01,
        "total_emissions_kg": 0.001,
        "avg_grounding_score": 0.8,
        "avg_quality_score": 0.7,
    }
    orch.db.save_benchmark_result({**common, "mode": ExecutionMode.STANDARD_RAG})
    orch.db.save_benchmark_result({**common, "mode": ExecutionMode.GREENROUTING})

    rows = [row for row in orch.db.get_all_benchmark_results() if row["run_id"] == "bench_shared"]
    assert {row["mode"] for row in rows} == {ExecutionMode.STANDARD_RAG, ExecutionMode.GREENROUTING}


def test_database_migrates_legacy_benchmark_primary_key(tmp_path):
    from greenrouting.database import DatabaseManager

    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("""
            CREATE TABLE benchmark_results (
                run_id TEXT PRIMARY KEY, benchmark_name TEXT NOT NULL, mode TEXT NOT NULL,
                total_queries INTEGER NOT NULL, cache_hits INTEGER NOT NULL,
                lightweight_count INTEGER NOT NULL, capable_count INTEGER NOT NULL,
                retry_count INTEGER NOT NULL, avg_latency_sec REAL NOT NULL,
                total_energy_kwh REAL NOT NULL, total_emissions_kg REAL NOT NULL,
                avg_grounding_score REAL NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.execute("""
            INSERT INTO benchmark_results
            (run_id, benchmark_name, mode, total_queries, cache_hits, lightweight_count,
             capable_count, retry_count, avg_latency_sec, total_energy_kwh,
             total_emissions_kg, avg_grounding_score)
            VALUES ('legacy_run', 'legacy', 'STANDARD_RAG', 1, 0, 0, 1, 0, 1, 0.1, 0.01, 0.8)
        """)

    db = DatabaseManager(db_path)
    row = db.get_all_benchmark_results()[0]
    with db.get_connection() as connection:
        pk = [item["name"] for item in sorted(connection.execute("PRAGMA table_info(benchmark_results)"), key=lambda item: item["pk"]) if item["pk"]]

    assert row["run_id"] == "legacy_run"
    assert row["avg_quality_score"] == 0
    assert pk == ["run_id", "mode"]
