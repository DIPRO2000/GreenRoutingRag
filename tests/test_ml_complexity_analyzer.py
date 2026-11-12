from greenrouting.ml_complexity_analyzer import ScikitLearnComplexityAnalyzer
from greenrouting.adaptive_router import AdaptiveGreenRouter
from greenrouting.config import ModelConfig, RoutingConfig


def test_scikit_classifier_routes_clear_simple_and_analytical_queries():
    analyzer = ScikitLearnComplexityAnalyzer()

    simple = analyzer.analyze("What is solar energy?")
    analytical = analyzer.analyze(
        "Compare and analyze the trade-offs between solar and wind energy."
    )

    assert simple["is_complex"] is False
    assert simple["routing_source"] == "scikit-learn"
    assert analytical["is_complex"] is True
    assert analytical["routing_source"] == "scikit-learn"
    assert "confidence" in analytical["rationale"].lower()


def test_low_confidence_uses_explainable_rule_fallback():
    analyzer = ScikitLearnComplexityAnalyzer(confidence_threshold=1.0)

    result = analyzer.analyze("What is the purpose of a vector index?")

    assert result["routing_source"] == "rule-fallback"
    assert "fallback" in result["rationale"].lower()
    assert result["complexity_score"] == result["rule_complexity_score"]


def test_adaptive_router_uses_classifier_prediction_for_model_selection(monkeypatch):
    class MockLLM:
        def generate(self, **kwargs):
            return {
                "answer": "A short answer grounded in the supplied context.",
                "model_used": kwargs["model_name"],
                "latency_seconds": 0.1,
                "eval_count": 10,
                "is_simulated": False,
            }

    model_config = ModelConfig(
        lightweight_model="small-model",
        capable_model="large-model",
    )
    router = AdaptiveGreenRouter(MockLLM(), model_config, RoutingConfig())
    monkeypatch.setattr(
        router.quality_guard,
        "evaluate_answer",
        lambda **kwargs: {
            "should_retry": False,
            "quality_score": 0.9,
            "groundedness_score": 0.9,
            "explanation": "Good answer",
        },
    )

    result = router.route_and_execute(
        query="What is solar energy?",
        prompt="What is solar energy?",
        context_chunks=[],
    )

    assert result["initial_model"] == model_config.lightweight_model
    assert "Scikit-learn" in result["routing_reason"]
