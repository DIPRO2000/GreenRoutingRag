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
        generation_runs = [{
            "model_name": initial_model,
            "latency_seconds": gen_result.get("latency_seconds", 0.0),
            "eval_count": gen_result.get("eval_count", 0)
        }]

        initial_answer = gen_result.get("answer", "")
        latency_total = gen_result.get("latency_seconds", 0.0)
        total_eval_count = gen_result.get("eval_count", 0)

        # Track whether any generation used simulation fallback
        is_simulated = gen_result.get("is_simulated", False)


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
            retry_model = self.model_config.capable_model

            retry_reason = (
                f"Escalation Retry: Initial lightweight model ({initial_model}) "
                f"yielded low quality score "
                f"({quality_assessment['quality_score']:.2f}). "
                f"Re-running with capable model ({retry_model})."
            )

            logger.info(retry_reason)

            try:
                retry_gen = self.llm_client.generate(
                    prompt=prompt,
                    model_name=retry_model,
                    system_prompt=system_prompt
                )

                generation_runs.append({
                    "model_name": retry_model,
                    "latency_seconds": retry_gen.get("latency_seconds", 0.0),
                    "eval_count": retry_gen.get("eval_count", 0)
                })

                retry_answer = retry_gen.get("answer", "")

                latency_total += retry_gen.get("latency_seconds", 0.0)
                total_eval_count += retry_gen.get("eval_count", 0)

                is_simulated = (
                    is_simulated or retry_gen.get("is_simulated", False)
                )

                # Accept retry only when it returns a non-empty answer
                if isinstance(retry_answer, str) and retry_answer.strip():

                    final_answer = retry_answer
                    final_model = retry_model

                    quality_assessment = self.quality_guard.evaluate_answer(
                        query=query,
                        answer=final_answer,
                        context_chunks=context_chunks,
                        model_used=final_model,
                        is_lightweight=False
                    )

                else:
                    retry_reason += (
                        " Retry returned an empty or invalid answer. "
                        "Preserving the initial answer."
                    )
                    logger.warning(retry_reason)

            except Exception as exc:
                retry_reason += (
                    f" Retry failed: {type(exc).__name__}: {exc}. "
                    "Preserving the initial answer."
                )
                logger.exception("Capable model retry failed")



        return {
            "answer": final_answer,
            "complexity_score": complexity_score,
            "routing_reason": routing_reason,
            "initial_model": initial_model,
            "final_model": final_model,
            "generation_runs": generation_runs,
            "is_lightweight": (final_model == self.model_config.lightweight_model),
            "retry_count": retry_count,
            "retry_reason": retry_reason,
            "quality_score": quality_assessment["quality_score"],
            "groundedness_score": quality_assessment["groundedness_score"],
            "quality_explanation": quality_assessment["explanation"],
            "latency_seconds": latency_total,
            "eval_count": total_eval_count,
            "is_simulated": is_simulated
        }