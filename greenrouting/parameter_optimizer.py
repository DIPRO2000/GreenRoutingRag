"""Offline Optuna search for quality-constrained GreenRouting thresholds."""
from typing import Any, Callable, Dict, List, Optional

from greenrouting.benchmark import BenchmarkRunner
from greenrouting.orchestrator import ExecutionMode


def run_parameter_optimization(
    orchestrator: Any,
    queries: List[str],
    n_trials: int = 6,
    min_quality_score: float = 0.55,
    benchmark_runner_factory: Optional[Callable] = None,
    optuna_module: Any = None,
) -> Dict[str, Any]:
    """Find the lowest-energy tested thresholds meeting the measured quality floor.

    Each trial runs the supplied questions through the real GreenRouting benchmark.
    Settings are restored after the search; recommendations require manual adoption.
    """
    if not queries or any(not str(query).strip() for query in queries):
        raise ValueError("Provide at least one non-empty evaluation question.")
    if not 1 <= n_trials <= 30:
        raise ValueError("Trial count must be between 1 and 30.")
    if not 0.0 <= min_quality_score <= 1.0:
        raise ValueError("Minimum quality score must be between 0 and 1.")

    if optuna_module is None:
        try:
            import optuna as optuna_module
        except ImportError as exc:
            raise RuntimeError(
                "Optuna is not installed. Install project dependencies with `pip install -r requirements.txt`."
            ) from exc

    runner_factory = benchmark_runner_factory or BenchmarkRunner
    runner = runner_factory(orchestrator)
    routing = orchestrator.settings.routing
    caching = orchestrator.settings.caching
    original_routing_threshold = routing.complexity_threshold
    original_cache_threshold = caching.semantic_similarity_threshold
    trial_rows: List[Dict[str, Any]] = []

    def objective(trial: Any) -> float:
        routing_threshold = trial.suggest_float("complexity_threshold", 0.30, 0.70)
        cache_threshold = trial.suggest_float("cache_similarity_threshold", 0.80, 0.98)
        routing.complexity_threshold = routing_threshold
        caching.semantic_similarity_threshold = cache_threshold

        benchmark = runner.run_benchmark(
            queries=queries,
            modes=[ExecutionMode.GREENROUTING],
        )
        summary = benchmark["mode_summaries"][ExecutionMode.GREENROUTING]
        quality = float(summary["avg_quality_score"])
        energy = float(summary["total_energy_kwh"])
        records = benchmark.get("detailed_records_df")
        simulated = bool(
            records is not None
            and not records.empty
            and "is_simulated" in records
            and records["is_simulated"].any()
        )
        trial.set_user_attr("avg_quality_score", quality)
        trial.set_user_attr("total_energy_kwh", energy)
        trial.set_user_attr("quality_shortfall", max(0.0, min_quality_score - quality))
        trial_rows.append({
            "trial": int(trial.number),
            "complexity_threshold": routing_threshold,
            "cache_similarity_threshold": cache_threshold,
            "avg_quality_score": quality,
            "total_energy_kwh": energy,
            "meets_quality_floor": quality >= min_quality_score,
            "contains_simulated_results": simulated,
        })
        return energy

    def constraints(trial: Any) -> List[float]:
        return [trial.user_attrs["quality_shortfall"]]

    try:
        sampler = optuna_module.samplers.TPESampler(
            seed=42,
            n_startup_trials=min(2, n_trials),
            constraints_func=constraints,
        )
        study = optuna_module.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials)
    finally:
        routing.complexity_threshold = original_routing_threshold
        caching.semantic_similarity_threshold = original_cache_threshold

    feasible = [row for row in trial_rows if row["meets_quality_floor"]]
    best_quality = max(trial_rows, key=lambda row: row["avg_quality_score"], default=None)
    return {
        "feasible": bool(feasible),
        "min_quality_score": min_quality_score,
        "recommended": min(feasible, key=lambda row: row["total_energy_kwh"]) if feasible else None,
        "best_quality_trial": best_quality,
        "trials": trial_rows,
        "quality_metric_note": "Quality is the project's heuristic score, not a human-labeled factuality evaluation.",
    }
