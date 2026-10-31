"""
Unit tests for RAG Evaluator in GreenRouting.
"""
import pytest
from greenrouting.evaluator import RagEvaluator


def test_evaluator_metrics():
    evaluator = RagEvaluator()

    chunks = [{
        "text_content": "GreenRouting implements quality-aware semantic caching and green model routing."
    }]

    # High quality grounded answer
    grounded_ans = "GreenRouting implements quality-aware caching to route models efficiently."
    metrics = evaluator.evaluate(
        query="What does GreenRouting implement?",
        answer=grounded_ans,
        retrieved_chunks=chunks
    )

    assert 0.0 <= metrics["groundedness"] <= 1.0
    assert 0.0 <= metrics["context_relevance"] <= 1.0
    assert 0.0 <= metrics["answer_relevance"] <= 1.0
    assert 0.0 <= metrics["composite_score"] <= 1.0
    assert metrics["groundedness"] > 0.40


def test_evaluator_empty_answer():
    evaluator = RagEvaluator()
    metrics = evaluator.evaluate(
        query="What is solar power?",
        answer="",
        retrieved_chunks=[]
    )
    assert metrics["groundedness"] == 0.0
    assert metrics["composite_score"] == 0.0
