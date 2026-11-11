"""Tests for the additive explainability contract returned by the orchestrator."""

from greenrouting.config import load_settings
from greenrouting.orchestrator import ExecutionMode, GreenRoutingOrchestrator


def _build_orchestrator(tmp_path):
    settings = load_settings()
    settings.app.db_path = tmp_path / "trace.db"
    settings.app.faiss_index_dir = tmp_path / "faiss"
    settings.models.enable_simulation_fallback = True
    orchestrator = GreenRoutingOrchestrator(settings)
    orchestrator.llm_client.generate = lambda **kwargs: {
        "answer": "GreenRouting reduces energy with local caching and model routing.",
        "model_used": kwargs.get("model_name", "test-model"),
        "latency_seconds": 0.01,
        "eval_count": 8,
        "prompt_eval_count": 8,
        "is_simulated": True,
    }
    return orchestrator


def test_no_document_response_contains_complete_decision_trace(tmp_path):
    orchestrator = _build_orchestrator(tmp_path)

    response = orchestrator.execute_query("What is GreenRouting?")

    trace = response["decision_trace"]
    assert set(trace) == {"cache", "retrieval", "routing", "impact"}
    assert trace["cache"]["status"] == "MISS"
    assert trace["retrieval"]["chunk_count"] == 0
    assert trace["routing"]["model"] == "None"
    assert trace["impact"]["energy_kwh"] == 0.0


def test_generated_response_maps_retrieval_routing_and_impact(tmp_path):
    orchestrator = _build_orchestrator(tmp_path)
    orchestrator.process_and_index_document(
        "trace.txt",
        b"GreenRouting reduces energy with caching and local model routing.",
    )

    response = orchestrator.execute_query(
        "What does GreenRouting reduce?",
        mode=ExecutionMode.STANDARD_RAG,
    )

    trace = response["decision_trace"]
    assert trace["cache"]["status"] == "MISS"
    assert trace["retrieval"]["top_k"] == response["retrieval_top_k"]
    assert trace["retrieval"]["chunk_count"] == response["retrieved_chunk_count"]
    assert trace["routing"]["model"] == response["model_used"]
    assert trace["impact"]["measurement_label"] == response["measurement_label"]
    assert response["is_simulated"] is True


def test_cache_hit_trace_explains_llm_bypass(tmp_path):
    orchestrator = _build_orchestrator(tmp_path)
    orchestrator.process_and_index_document(
        "trace.txt",
        b"GreenRouting reduces energy with caching and local model routing.",
    )
    query = "What does GreenRouting reduce?"

    orchestrator.execute_query(query, mode=ExecutionMode.GREENROUTING)
    cached = orchestrator.execute_query(query, mode=ExecutionMode.GREENROUTING)

    trace = cached["decision_trace"]
    assert trace["cache"]["status"] == "HIT_VALID"
    assert "bypassed" in trace["routing"]["reason"].lower()


def test_rag_with_cache_validates_against_actual_retrieved_document(tmp_path, monkeypatch):
    orchestrator = _build_orchestrator(tmp_path)
    indexed = orchestrator.process_and_index_document(
        "scope.txt", b"The project stores relevant document scope with retrieved evidence."
    )
    observed = {}

    def capture_lookup(**kwargs):
        observed.update(kwargs)
        return {
            "status": "MISS",
            "cached_answer": None,
            "validation_reason": "test miss",
        }

    monkeypatch.setattr(orchestrator.semantic_cache, "lookup", capture_lookup)
    orchestrator.execute_query(
        "What does this project store?",
        mode=ExecutionMode.RAG_WITH_CACHE,
    )

    assert observed["current_doc_ids"] == [indexed["doc_id"]]
    assert observed["retrieved_chunks"]
    assert {chunk["doc_id"] for chunk in observed["retrieved_chunks"]} == {indexed["doc_id"]}
