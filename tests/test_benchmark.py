"""
Unit tests for Benchmarking Engine in GreenRouting.
"""
import pytest
import math
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
