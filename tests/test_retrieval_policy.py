"""Tests for deterministic energy-aware retrieval sizing."""

from greenrouting.retrieval_policy import choose_top_k
from greenrouting.config import load_settings
from greenrouting.orchestrator import ExecutionMode, GreenRoutingOrchestrator


def test_short_factual_query_uses_small_context():
    decision = choose_top_k("What is FAISS?", available_chunks=10, default_top_k=3)

    assert decision.top_k == 2
    assert decision.complexity_score < 0.5
    assert "simple" in decision.reason.lower()


def test_analytical_query_uses_larger_context():
    decision = choose_top_k(
        "Compare and analyze the trade-offs between caching and model routing.",
        available_chunks=10,
        default_top_k=3,
    )

    assert decision.top_k == 5
    assert decision.complexity_score >= 0.5
    assert "analytical" in decision.reason.lower()


def test_available_chunk_count_bounds_top_k():
    decision = choose_top_k("Explain this", available_chunks=1, default_top_k=3)

    assert decision.top_k == 1


def test_empty_query_falls_back_to_default_within_bounds():
    decision = choose_top_k("", available_chunks=10, default_top_k=4)

    assert decision.top_k == 4
    assert "default" in decision.reason.lower()


def test_orchestrator_reports_adaptive_retrieval_and_honors_override(tmp_path):
    settings = load_settings()
    settings.app.db_path = tmp_path / "retrieval.db"
    settings.app.faiss_index_dir = tmp_path / "faiss"
    settings.models.enable_simulation_fallback = True
    orchestrator = GreenRoutingOrchestrator(settings)
    orchestrator.process_and_index_document(
        "retrieval.txt",
        b"GreenRouting uses caching and local models to reduce energy consumption. "
        b"It also reports quality and carbon telemetry for each question.",
    )

    adaptive = orchestrator.execute_query(
        "What does GreenRouting reduce?",
        mode=ExecutionMode.STANDARD_RAG,
    )
    override = orchestrator.execute_query(
        "What does GreenRouting reduce?",
        mode=ExecutionMode.STANDARD_RAG,
        top_k=1,
    )

    assert adaptive["retrieval_top_k"] >= 1
    assert adaptive["retrieved_chunk_count"] <= adaptive["retrieval_top_k"]
    assert adaptive["retrieval_reason"]
    assert override["retrieval_top_k"] == 1
