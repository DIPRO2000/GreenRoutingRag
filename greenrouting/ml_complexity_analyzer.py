"""Experimental scikit-learn query classifier with a rule-based fallback."""
from typing import Any, Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from greenrouting.complexity_analyzer import ComplexityAnalyzer
from greenrouting.config import RoutingConfig


# Small, manually labeled starter data. Expand and evaluate with real benchmark feedback.
STARTER_ROUTING_EXAMPLES = [
    ("What is solar energy?", "simple"),
    ("Define semantic caching.", "simple"),
    ("Which model is lightweight?", "simple"),
    ("How many pages are in this report?", "simple"),
    ("What does FAISS store?", "simple"),
    ("What is a neural network?", "simple"),
    ("State the purpose of an embedding.", "simple"),
    ("List the three ANN architectures.", "simple"),
    ("What does the cache do?", "simple"),
    ("Give the configured emission factor.", "simple"),
    ("Name the retrieval database.", "simple"),
    ("Which file formats can I upload?", "simple"),
    ("How many layers does a single-layer network have?", "simple"),
    ("Find the definition of a neuron.", "simple"),
    ("What is the purpose of the vector index?", "simple"),
    ("State one use of backpropagation.", "simple"),
    ("Compare and analyze trade-offs between solar and wind energy.", "complex"),
    ("Analyze the causes and consequences of model routing errors.", "complex"),
    ("Explain why XOR cannot be solved by a single-layer perceptron.", "complex"),
    ("Evaluate the energy and quality trade-offs and justify a choice.", "complex"),
    ("How do semantic caching and retrieval interact to affect answer quality?", "complex"),
    ("Contrast single-layer and multilayer networks and explain their limitations.", "complex"),
    ("Synthesize the evidence across these sections into a coherent explanation.", "complex"),
    ("Discuss the advantages and disadvantages of using a lightweight model.", "complex"),
    ("Trace the routing steps and identify where quality escalation occurs.", "complex"),
    ("Derive the weight update rule and justify each step.", "complex"),
    ("Investigate how changing the retrieval depth affects quality and energy.", "complex"),
    ("Compare the architectures and explain when each one is appropriate.", "complex"),
    ("Explain the mechanism and implications of increasing the cache threshold.", "complex"),
    ("Assess the trade-offs between local inference speed and answer depth.", "complex"),
    ("Why can retrieval miss evidence, and how should the system respond?", "complex"),
    ("Analyze multiple factors that influence the carbon impact of inference.", "complex"),
]


class ScikitLearnComplexityAnalyzer:
    """Predict query complexity from text, deferring uncertain cases to existing rules."""

    def __init__(
        self,
        routing_config: Optional[RoutingConfig] = None,
        confidence_threshold: float = 0.62,
    ):
        if not 0.5 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.5 and 1.0")
        self.confidence_threshold = confidence_threshold
        self.rule_analyzer = ComplexityAnalyzer(routing_config)
        queries, labels = zip(*STARTER_ROUTING_EXAMPLES)
        self.classifier = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode"),
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        )
        self.classifier.fit(list(queries), list(labels))

    def analyze(
        self,
        query: str,
        context_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        rule_analysis = self.rule_analyzer.analyze(query, context_chunks)
        probabilities = self.classifier.predict_proba([query])[0]
        classes = self.classifier[-1].classes_
        predicted_index = int(probabilities.argmax())
        predicted_class = str(classes[predicted_index])
        confidence = float(probabilities[predicted_index])
        complex_index = list(classes).index("complex")
        complex_probability = float(probabilities[complex_index])

        if confidence < self.confidence_threshold:
            return {
                **rule_analysis,
                "rule_complexity_score": rule_analysis["complexity_score"],
                "ml_confidence": round(confidence, 4),
                "routing_source": "rule-fallback",
                "rationale": (
                    f"ML confidence {confidence:.2f} is below the "
                    f"{self.confidence_threshold:.2f} threshold; using rule-based fallback. "
                    f"{rule_analysis['rationale']}"
                ),
            }

        is_complex = predicted_class == "complex"
        return {
            **rule_analysis,
            "complexity_score": round(complex_probability, 4),
            "is_complex": is_complex,
            "rule_complexity_score": rule_analysis["complexity_score"],
            "ml_confidence": round(confidence, 4),
            "routing_source": "scikit-learn",
            "rationale": (
                f"Scikit-learn TF-IDF classifier predicted {predicted_class} complexity "
                f"with {confidence:.0%} confidence."
            ),
        }
