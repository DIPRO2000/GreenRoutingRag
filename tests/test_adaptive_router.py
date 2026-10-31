"""
Unit tests for Adaptive Green Router and Complexity Analyzer in GreenRouting.
"""
import pytest
from greenrouting.config import ModelConfig, RoutingConfig
from greenrouting.complexity_analyzer import ComplexityAnalyzer
from greenrouting.quality_guard import QualityGuard
from greenrouting.adaptive_router import AdaptiveGreenRouter
from greenrouting.llm_interface import OllamaClient


def test_complexity_analyzer_simple_vs_complex():
    analyzer = ComplexityAnalyzer()

    # Simple factual lookup query
    simple_q = "What is the solar panel efficiency?"
    simple_result = analyzer.analyze(simple_q)
    assert simple_result["complexity_score"] < 0.50
    assert simple_result["is_complex"] is False
    assert "Low complexity" in simple_result["rationale"]

    # Complex multi-hop analytical query
    complex_q = (
        "Compare and analyze the trade-offs between renewable solar photovoltaics and wind turbine "
        "energy conversion algorithms, synthesizing their long-term environmental degradation factors."
    )
    complex_result = analyzer.analyze(complex_q)
    assert complex_result["complexity_score"] >= 0.50
    assert complex_result["is_complex"] is True
    assert "High complexity" in complex_result["rationale"]
    assert len(complex_result["factors"]["matched_reasoning_keywords"]) > 0


def test_adaptive_router_execution():
    llm_client = OllamaClient(enable_simulation_fallback=True)
    model_config = ModelConfig(
        lightweight_model="qwen2.5:0.5b",
        capable_model="llama3.2:3b"
    )
    routing_config = RoutingConfig(complexity_threshold=0.50)
    router = AdaptiveGreenRouter(llm_client, model_config, routing_config)

    chunks = [{
        "chunk_id": "c1",
        "doc_id": "d1",
        "text_content": "GreenRouting reduces energy consumption by over 60% compared to standard heavy LLM RAG pipelines."
    }]
    prompt = f"Context: {chunks[0]['text_content']}\n\nQuestion: What percentage does GreenRouting reduce?\n\nAnswer:"

    # Simple query should route to lightweight model
    res = router.route_and_execute(
        query="What percentage does GreenRouting reduce?",
        prompt=prompt,
        context_chunks=chunks
    )

    assert res["initial_model"] == "qwen2.5:0.5b"
    assert res["complexity_score"] < 0.50
    assert res["retry_count"] in (0, 1)
    assert res["quality_score"] > 0.0
    assert len(res["routing_reason"]) > 0


def test_quality_guard_retry_escalation():
    # If lightweight model returns a refusal, retry must trigger
    quality_guard = QualityGuard(RoutingConfig(min_acceptable_quality_score=0.55))
    chunks = [{"text_content": "Relevant information."}]

    assessment = quality_guard.evaluate_answer(
        query="Explain the algorithm.",
        answer="I don't know the answer based on the provided context.",
        context_chunks=chunks,
        model_used="qwen2.5:0.5b",
        is_lightweight=True
    )

    assert assessment["should_retry"] is True
    assert assessment["has_refusal"] is True
    assert assessment["quality_score"] < 0.55
