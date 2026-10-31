"""
Adaptive Green Router for GreenRouting.
Dynamically routes user questions to an optimal model based on question complexity,
transparently explains routing rationale, and automatically retries with a stronger model
when answer quality checks indicate deficiency.
"""
import logging
from typing import Dict, Any, List, Optional
from greenrouting.config import ModelConfig, RoutingConfig
from greenrouting.complexity_analyzer import ComplexityAnalyzer
from greenrouting.quality_guard import QualityGuard
from greenrouting.llm_interface import OllamaClient

logger = logging.getLogger(__name__)


class AdaptiveGreenRouter:
    """Manages energy-aware model routing, quality verification, and transparent explanations."""

    def __init__(self, llm_client: OllamaClient, model_config: ModelConfig, routing_config: RoutingConfig):
        self.llm_client = llm_client
        self.model_config = model_config
        self.routing_config = routing_config
        self.complexity_analyzer = ComplexityAnalyzer(routing_config)
        self.quality_guard = QualityGuard(routing_config)

    def route_and_execute(self, query: str, prompt: str, context_chunks: List[Dict[str, Any]],
                           system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        1. Analyze question complexity.
        2. Select model (Lightweight vs Capable).
        3. Execute LLM generation.
        4. Validate answer quality.
        5. Retry with Capable model if lightweight generation fails quality check.
        6. Return complete routing telemetry.
        """
        # 1. Complexity Analysis
        analysis = self.complexity_analyzer.analyze(query, context_chunks)
        complexity_score = analysis["complexity_score"]
        is_complex = analysis["is_complex"]

        # 2. Initial Model Selection
        if is_complex:
            initial_model = self.model_config.capable_model
            is_initial_lightweight = False
            routing_reason = (
                f"Routed to Capable Model ({initial_model}) due to high query complexity: "
                f"{analysis['rationale']}"
            )
        else:
            initial_model = self.model_config.lightweight_model
            is_initial_lightweight = True
            routing_reason = (
                f"Routed to Lightweight Model ({initial_model}) for energy conservation: "
                f"{analysis['rationale']}"
            )

        # 3. Generate Answer with initial model
        gen_result = self.llm_client.generate(
            prompt=prompt,
            model_name=initial_model,
            system_prompt=system_prompt
        )

        initial_answer = gen_result.get("answer", "")
        latency_total = gen_result.get("latency_seconds", 0.0)
        total_eval_count = gen_result.get("eval_count", 0)

        # 4. Quality Validation
        quality_assessment = self.quality_guard.evaluate_answer(
            query=query,
            answer=initial_answer,
            context_chunks=context_chunks,
            model_used=initial_model,
            is_lightweight=is_initial_lightweight
        )

        final_model = initial_model
        final_answer = initial_answer
        retry_count = 0
        retry_reason = None

        # 5. Conditional Retry Escalation
        if quality_assessment["should_retry"]:
            retry_count = 1
            final_model = self.model_config.capable_model
            retry_reason = (
                f"Escalation Retry: Initial lightweight model ({initial_model}) yielded low quality "
                f"score ({quality_assessment['quality_score']:.2f}). Re-running with capable model ({final_model})."
            )
            logger.info(retry_reason)

            # Re-generate with capable model
            retry_gen = self.llm_client.generate(
                prompt=prompt,
                model_name=final_model,
                system_prompt=system_prompt
            )

            final_answer = retry_gen.get("answer", "")
            latency_total += retry_gen.get("latency_seconds", 0.0)
            total_eval_count += retry_gen.get("eval_count", 0)

            # Re-evaluate final answer quality
            quality_assessment = self.quality_guard.evaluate_answer(
                query=query,
                answer=final_answer,
                context_chunks=context_chunks,
                model_used=final_model,
                is_lightweight=False
            )

        return {
            "answer": final_answer,
            "complexity_score": complexity_score,
            "routing_reason": routing_reason,
            "initial_model": initial_model,
            "final_model": final_model,
            "is_lightweight": (final_model == self.model_config.lightweight_model),
            "retry_count": retry_count,
            "retry_reason": retry_reason,
            "quality_score": quality_assessment["quality_score"],
            "groundedness_score": quality_assessment["groundedness_score"],
            "quality_explanation": quality_assessment["explanation"],
            "latency_seconds": latency_total,
            "eval_count": total_eval_count,
            "is_simulated": gen_result.get("is_simulated", False)
        }
