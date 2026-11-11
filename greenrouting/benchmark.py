"""
Benchmarking Engine for GreenRouting.
Executes empirical comparison across Standard RAG, RAG with Caching, and Complete GreenRouting.
Measures energy (kWh), carbon emissions (g CO2eq), latency, cache hits, routing, and answer quality.
Exports experimental data to CSV and SQLite.
"""
import uuid
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from greenrouting.orchestrator import GreenRoutingOrchestrator, ExecutionMode

logger = logging.getLogger(__name__)

# Representative benchmark question set with mixed complexity levels and repeated queries
DEFAULT_BENCHMARK_QUERIES = [
    # 1. Simple factual queries (low complexity)
    "What is the primary objective of GreenRouting?",
    "Which vector database is used for retrieval?",
    "What embedding model does the framework default to?",
    
    # 2. Analytical and multi-hop queries (high complexity)
    "Compare and analyze how adaptive green routing and quality-aware caching synergize to reduce energy.",
    "Explain the implications of relying solely on embedding similarity for semantic caching versus multi-layer validation.",
    "Analyze the trade-offs between lightweight model inference speed and capable model answer depth.",

    # 3. Repeated queries (to evaluate semantic caching performance and energy savings)
    "What is the primary objective of GreenRouting?",
    "Which vector database is used for retrieval?",
    "Compare and analyze how adaptive green routing and quality-aware caching synergize to reduce energy."
]


def build_comparison_frame(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Return chart/table fields without mutating raw benchmark summaries."""
    frame = summary_df.copy()
    if frame.empty:
        return frame

    frame["quality_pct"] = (frame["avg_quality_score"] * 100.0).round(2)
    frame["grounding_pct"] = (frame["avg_grounding_score"] * 100.0).round(2)
    frame["carbon_g"] = frame["total_emissions_g"]

    baseline_rows = frame.loc[frame["mode"] == ExecutionMode.STANDARD_RAG]
    baseline = baseline_rows.iloc[0] if not baseline_rows.empty else frame.iloc[0]
    baseline_energy = float(baseline["total_energy_kwh"])
    baseline_carbon = float(baseline["total_emissions_g"])

    if baseline_energy > 0:
        frame["energy_savings_vs_standard_pct"] = (
            ((baseline_energy - frame["total_energy_kwh"]) / baseline_energy) * 100.0
        ).round(2)
    else:
        frame["energy_savings_vs_standard_pct"] = 0.0

    if baseline_carbon > 0:
        frame["carbon_savings_vs_standard_pct"] = (
            ((baseline_carbon - frame["total_emissions_g"]) / baseline_carbon) * 100.0
        ).round(2)
    else:
        frame["carbon_savings_vs_standard_pct"] = 0.0

    return frame


class BenchmarkRunner:
    """Orchestrates reproducible empirical evaluations of energy-aware RAG pipelines."""

    def __init__(self, orchestrator: GreenRoutingOrchestrator):
        self.orchestrator = orchestrator

    def run_benchmark(
        self,
        queries: Optional[List[str]] = None,
        modes: Optional[List[str]] = None,
        active_doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute full benchmark suite across specified modes using real queries and real measurements.
        """
        test_queries = queries or DEFAULT_BENCHMARK_QUERIES
        test_modes = modes or [
            ExecutionMode.STANDARD_RAG,
            ExecutionMode.RAG_WITH_CACHE,
            ExecutionMode.GREENROUTING
        ]

        run_id = f"bench_{uuid.uuid4().hex[:8]}"
        all_query_records: List[Dict[str, Any]] = []
        mode_summaries: Dict[str, Dict[str, Any]] = {}

        for mode in test_modes:
            logger.info("Starting benchmark run for mode: %s", mode)
            # Clear cache between different mode benchmarks to ensure clean comparisons
            self.orchestrator.semantic_cache.db.clear_all_cache()

            total_energy_kwh = 0.0
            total_emissions_g = 0.0
            latencies: List[float] = []
            grounding_scores: List[float] = []
            quality_scores: List[float] = []
            cache_hits = 0
            lightweight_count = 0
            capable_count = 0
            retry_count = 0

            for q_idx, q in enumerate(test_queries, start=1):
                res = self.orchestrator.execute_query(
                    query=q,
                    mode=mode,
                    active_doc_id=active_doc_id
                )

                total_energy_kwh += res.get("energy_kwh", 0.0)
                total_emissions_g += res.get("emissions_g", 0.0)
                lat = res.get("latency_seconds", 0.0)
                latencies.append(lat)
                grounding_scores.append(res.get("grounding_score", 0.0))
                quality_scores.append(res.get("quality_score", 0.0))

                if "HIT_VALID" in res.get("cache_status", ""):
                    cache_hits += 1
                elif res.get("retry_count", 0) > 0:
                    retry_count += 1
                    capable_count += 1
                elif any(k in res.get("model_used", "").lower() for k in ["0.5b", "1b", "tiny"]):
                    lightweight_count += 1
                else:
                    capable_count += 1

                all_query_records.append({
                    "run_id": run_id,
                    "query_index": q_idx,
                    "query": q,
                    "mode": mode,
                    "model_used": res.get("model_used"),
                    "cache_status": res.get("cache_status"),
                    "complexity_score": res.get("complexity_score"),
                    "latency_seconds": lat,
                    "energy_kwh": res.get("energy_kwh"),
                    "emissions_g": res.get("emissions_g"),
                    "grounding_score": res.get("grounding_score"),
                    "quality_score": res.get("quality_score"),
                    "retry_count": res.get("retry_count", 0)
                })

            n = len(test_queries)
            avg_lat = sum(latencies) / n if n > 0 else 0.0
            avg_grounding = sum(grounding_scores) / n if n > 0 else 0.0
            avg_quality = sum(quality_scores) / n if n > 0 else 0.0

            summary = {
                "run_id": run_id,
                "benchmark_name": f"Eval_{len(test_queries)}_Queries",
                "mode": mode,
                "total_queries": n,
                "cache_hits": cache_hits,
                "cache_hit_rate_pct": round((cache_hits / n) * 100.0, 1) if n > 0 else 0.0,
                "lightweight_count": lightweight_count,
                "capable_count": capable_count,
                "retry_count": retry_count,
                "avg_latency_sec": round(avg_lat, 4),
                "total_energy_kwh": round(total_energy_kwh, 8),
                "total_emissions_g": round(total_emissions_g, 5),
                "total_emissions_kg": round(total_emissions_g / 1000.0, 8),
                "avg_grounding_score": round(avg_grounding, 4),
                "avg_quality_score": round(avg_quality, 4)
            }
            mode_summaries[mode] = summary

            # Persist summary in SQLite
            self.orchestrator.db.save_benchmark_result(summary)

        # Compute Comparative Savings against Baseline (STANDARD_RAG)
        baseline = mode_summaries.get(ExecutionMode.STANDARD_RAG)
        greenrouting_res = mode_summaries.get(ExecutionMode.GREENROUTING)

        comparative_analysis = {}
        if baseline and greenrouting_res:
            b_energy = baseline["total_energy_kwh"]
            e_energy = greenrouting_res["total_energy_kwh"]
            energy_savings_pct = round(((b_energy - e_energy) / b_energy) * 100.0, 2) if b_energy > 0 else 0.0

            b_lat = baseline["avg_latency_sec"]
            e_lat = greenrouting_res["avg_latency_sec"]
            latency_reduction_pct = round(((b_lat - e_lat) / b_lat) * 100.0, 2) if b_lat > 0 else 0.0

            comparative_analysis = {
                "energy_savings_pct": energy_savings_pct,
                "latency_reduction_pct": latency_reduction_pct,
                "baseline_energy_kwh": b_energy,
                "greenrouting_energy_kwh": e_energy,
                "baseline_latency_sec": b_lat,
                "greenrouting_latency_sec": e_lat,
                "quality_retained_pct": round((greenrouting_res["avg_quality_score"] / baseline["avg_quality_score"]) * 100.0, 2) if baseline["avg_quality_score"] > 0 else 100.0
            }

        df_records = pd.DataFrame(all_query_records)
        df_summary = pd.DataFrame(list(mode_summaries.values()))

        return {
            "run_id": run_id,
            "mode_summaries": mode_summaries,
            "comparative_analysis": comparative_analysis,
            "detailed_records_df": df_records,
            "summary_df": df_summary
        }

    def export_to_csv(self, benchmark_result: Dict[str, Any], filepath: str) -> str:
        """Export benchmark records to CSV."""
        df = benchmark_result.get("detailed_records_df")
        if df is not None:
            df.to_csv(filepath, index=False)
            return filepath
        raise ValueError("No dataframe available in benchmark result.")
