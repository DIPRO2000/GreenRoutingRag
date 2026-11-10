"""
Unit tests for Adaptive Green Router and Complexity Analyzer in GreenRouting.
"""
from unittest import result

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

def test_router_preserves_initial_answer_when_retry_is_empty(monkeypatch):
    """An empty capable-model response must not replace the initial answer."""

    class MockLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, **kwargs):
            self.calls += 1

            if self.calls == 1:
                return {
                    "answer": "Initial useful answer",
                    "latency_seconds": 1.0,
                    "eval_count": 20,
                    "is_simulated": False
                }

            return {
                "answer": "",
                "latency_seconds": 2.0,
                "eval_count": 0,
                "is_simulated": False
            }

    model_config = ModelConfig(
        lightweight_model="qwen2.5:0.5b",
        capable_model="llama3.2:3b"
    )

    routing_config = RoutingConfig()
    mock_llm = MockLLM()

    router = AdaptiveGreenRouter(
        mock_llm, model_config, routing_config
    )

    # Force the quality guard to request a retry.
    monkeypatch.setattr(
        router.quality_guard,
        "evaluate_answer",
        lambda **kwargs: {
            "should_retry": True,
            "quality_score": 0.2,
            "groundedness_score": 0.2,
            "explanation": "Initial answer requires improvement"
        }
    )

    result = router.route_and_execute(
        query="What is an algorithm?",
        prompt="Explain an algorithm.",
        context_chunks=[]
    )

    assert result["answer"] == "Initial useful answer"
    assert result["initial_model"] == model_config.lightweight_model
    assert result["final_model"] == model_config.lightweight_model
    assert result["retry_count"] == 1
    assert "empty" in result["retry_reason"].lower()


def test_router_preserves_initial_answer_when_retry_raises_exception(monkeypatch):
    """A failed capable-model retry must preserve the initial answer."""

    class MockLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, **kwargs):
            self.calls += 1

            if self.calls == 1:
                return {
                    "answer": "Initial useful answer",
                    "latency_seconds": 1.0,
                    "eval_count": 20,
                    "is_simulated": False
                }

            raise RuntimeError("Capable model unavailable")

    model_config = ModelConfig(
        lightweight_model="qwen2.5:0.5b",
        capable_model="llama3.2:3b"
    )

    routing_config = RoutingConfig()
    mock_llm = MockLLM()

    router = AdaptiveGreenRouter(
        mock_llm, model_config, routing_config
    )

    monkeypatch.setattr(
        router.quality_guard,
        "evaluate_answer",
        lambda **kwargs: {
            "should_retry": True,
            "quality_score": 0.2,
            "groundedness_score": 0.2,
            "explanation": "Initial answer requires improvement"
        }
    )

    result = router.route_and_execute(
        query="What is an algorithm?",
        prompt="Explain an algorithm.",
        context_chunks=[]
    )

    assert result["answer"] == "Initial useful answer"
    assert result["final_model"] == model_config.lightweight_model
    assert result["retry_count"] == 1
    assert "Retry failed" in result["retry_reason"]



def test_complex_query_uses_capable_model_without_retry(monkeypatch):
    """A complex query should directly use the capable model."""

    class MockLLM:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs["model_name"])

            return {
                "answer": "Detailed analytical answer",
                "latency_seconds": 1.0,
                "eval_count": 30,
                "is_simulated": False
            }

    model_config = ModelConfig(
        lightweight_model="qwen2.5:0.5b",
        capable_model="llama3.2:3b"
    )

    routing_config = RoutingConfig()
    mock_llm = MockLLM()

    router = AdaptiveGreenRouter(
        mock_llm, model_config, routing_config
    )

    # Force a complex-query classification.
    monkeypatch.setattr(
        router.complexity_analyzer,
        "analyze",
        lambda *args, **kwargs: {
            "complexity_score": 0.85,
            "is_complex": True,
            "rationale": "High reasoning complexity"
        }
    )

    monkeypatch.setattr(
        router.quality_guard,
        "evaluate_answer",
        lambda **kwargs: {
            "should_retry": False,
            "quality_score": 0.9,
            "groundedness_score": 0.9,
            "explanation": "Answer meets quality requirements"
        }
    )

    result = router.route_and_execute(
        query="Compare two algorithms",
        prompt="Compare two algorithms in detail.",
        context_chunks=[]
    )


    assert mock_llm.calls == [model_config.capable_model]
    assert result["initial_model"] == model_config.capable_model
    assert result["final_model"] == model_config.capable_model
    assert result["retry_count"] == 0
    assert result["is_lightweight"] is False

    # Exactly one generation: directly using the capable model.
    assert len(result["generation_runs"]) == 1

    assert (
        result["generation_runs"][0]["model_name"]
        == model_config.capable_model
    )

def test_simple_query_uses_lightweight_model(monkeypatch):
    """A simple query should initially use the lightweight model."""

    class MockLLM:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs["model_name"])
            return {
                "answer": "Solar panels convert sunlight into electricity.",
                "latency_seconds": 0.5,
                "eval_count": 15,
                "is_simulated": False
            }

    model_config = ModelConfig(
        lightweight_model="qwen2.5:0.5b",
        capable_model="llama3.2:3b"
    )

    routing_config = RoutingConfig(complexity_threshold=0.50)
    mock_llm = MockLLM()

    router = AdaptiveGreenRouter(
        mock_llm, model_config, routing_config
    )

    monkeypatch.setattr(
        router.quality_guard,
        "evaluate_answer",
        lambda **kwargs: {
            "should_retry": False,
            "quality_score": 0.95,
            "groundedness_score": 0.95,
            "explanation": "Answer meets quality requirements"
        }
    )

    result = router.route_and_execute(
        query="What is solar energy?",
        prompt="Explain solar energy.",
        context_chunks=[]
    )

    assert mock_llm.calls == [model_config.lightweight_model]
    assert result["initial_model"] == model_config.lightweight_model
    assert result["final_model"] == model_config.lightweight_model
    assert result["retry_count"] == 0
    assert len(result["generation_runs"]) == 1



def test_low_quality_lightweight_answer_escalates_to_capable_model(monkeypatch):
    """A low-quality lightweight answer should trigger capable-model retry."""

    class MockLLM:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            model = kwargs["model_name"]
            self.calls.append(model)

            return {
                "answer": (
                    "I don't know based on the context."
                    if len(self.calls) == 1
                    else "Solar energy is energy derived from sunlight."
                ),
                "latency_seconds": 0.5,
                "eval_count": 15,
                "is_simulated": False
            }

    model_config = ModelConfig(
        lightweight_model="qwen2.5:0.5b",
        capable_model="llama3.2:3b"
    )

    mock_llm = MockLLM()
    router = AdaptiveGreenRouter(
        mock_llm, model_config, RoutingConfig()
    )

    assessments = iter([
        {
            "should_retry": True,
            "quality_score": 0.2,
            "groundedness_score": 0.2,
            "explanation": "Low-quality answer"
        },
        {
            "should_retry": False,
            "quality_score": 0.9,
            "groundedness_score": 0.9,
            "explanation": "Acceptable answer"
        }
    ])

    monkeypatch.setattr(
        router.quality_guard,
        "evaluate_answer",
        lambda **kwargs: next(assessments)
    )

    result = router.route_and_execute(
        query="What is solar energy?",
        prompt="Explain solar energy.",
        context_chunks=[]
    )

    assert mock_llm.calls == [
        model_config.lightweight_model,
        model_config.capable_model
    ]
    assert result["initial_model"] == model_config.lightweight_model
    assert result["final_model"] == model_config.capable_model
    assert result["retry_count"] == 1
    assert result["answer"] == "Solar energy is energy derived from sunlight."
    assert len(result["generation_runs"]) == 2
