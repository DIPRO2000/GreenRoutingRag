"""
Energy and Carbon Monitoring Module for GreenRouting.
Estimates energy and emissions from configured power assumptions and elapsed time.
"""
import logging
import math
from typing import Dict, Any, Optional, List
from greenrouting.config import EnergyConfig

logger = logging.getLogger(__name__)


class EnergyMeasurementType:
    MODELED_TDP_ESTIMATE = "MODELED_TDP_ESTIMATE"
    CACHED_ZERO_INFERENCE = "CACHED_ZERO_INFERENCE"


class EnergyTracker:
    """
    Tracks and models compute energy consumption (kWh) and carbon footprint (kg CO2eq).
    Uses a transparent modeled TDP estimate; it does not read hardware sensors.
    """

    def __init__(self, config: Optional[EnergyConfig] = None):
        self.config = config or EnergyConfig()
    def _estimate_model_power(self, model_name: Optional[str]) -> float:
        """Return the configured TDP-based power estimate for a model."""

        m_lower = (model_name or "").lower()

        if any(tag in m_lower for tag in ["0.5b", "1b", "tiny", "mini"]):
            return self.config.cpu_tdp_watts * 0.60

        elif any(tag in m_lower for tag in ["3b", "7b", "mistral", "llama"]):
            return (
                self.config.cpu_tdp_watts
                + self.config.gpu_tdp_watts * 0.65
            )

        return self.config.cpu_tdp_watts


    def track_operation(
        self,
        operation_name: str,
        latency_seconds: float,
        model_name: Optional[str] = None,
        is_cache_hit: bool = False,
        retry_count: int = 0,
        generation_runs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Compute energy (kWh) and emissions (kg CO2eq) for a given operation.
        Explicitly distinguishes modeled inference from cache hits with no LLM inference.
        """
        lat = max(0.001, float(latency_seconds))

        # 1. Zero-inference cache hit
        if is_cache_hit:
            # Model retrieval, cache validation, and evaluation at a low CPU
            # utilization profile. A cache hit avoids inference, not all work.
            retrieval_power_watts = self.config.cpu_tdp_watts * 0.25
            cache_lookup_kwh = (retrieval_power_watts * lat) / (3600.0 * 1000.0)
            cache_emissions_kg = cache_lookup_kwh * self.config.emission_factor_kg_per_kwh

            return {
                "energy_kwh": cache_lookup_kwh,
                "emissions_kg": cache_emissions_kg,
                "emissions_g": round(cache_emissions_kg * 1000.0, 6),
                "latency_seconds": round(lat, 4),
                "measurement_type": EnergyMeasurementType.CACHED_ZERO_INFERENCE,
                "is_direct_measurement": False,
                "measurement_label": "Cache Reuse (Zero LLM Inference)",
                "details": (
                    f"Modeled CPU estimate for {lat:.4f}s of retrieval, cache validation, "
                    "and evaluation; LLM inference was avoided."
                )
            }

        # 2. Query-level Calibrated TDP Engineering Model
        # Formula: E (kWh) = [Power (Watts) * Time (Seconds)] / (3600 * 1000)
        # Emissions (kg CO2eq) = E (kWh) * Emission Factor (kg/kWh)

        # 2. Model-specific TDP energy estimation [Changed]

        generation_details = []

        if generation_runs:
            # Calculate energy separately for every model generation.
            for run in generation_runs:

                run_model = run.get("model_name")
                run_latency = float(run.get("latency_seconds", 0.0))

                if not math.isfinite(run_latency) or run_latency < 0:
                    raise ValueError(
                        "Generation latency must be finite and non-negative."
                    )

                if run_latency == 0:
                    continue

                run_power = self._estimate_model_power(run_model)

                generation_detail = {
                    "model_name": run_model,
                    "latency_seconds": run_latency,
                    "power_watts": run_power,
                    "energy_kwh": (
                        run_power * run_latency / (3600.0 * 1000.0)
                    )
                }
                if run.get("stage"):
                    generation_detail["stage"] = run["stage"]
                generation_details.append(generation_detail)

            if not generation_details:
                raise ValueError("No valid generation records provided.")

            energy_kwh = sum(
                item["energy_kwh"] for item in generation_details
            )

            total_generation_latency = sum(
                item["latency_seconds"] for item in generation_details
            )

            total_power_watts = (
                energy_kwh * 3600.0 * 1000.0
                / total_generation_latency
            )

            lat = total_generation_latency

        else:
            # Backward-compatible single-model calculation.
            total_power_watts = self._estimate_model_power(model_name)

            energy_kwh = (
                total_power_watts * lat / (3600.0 * 1000.0)
            )

        emissions_kg = (
            energy_kwh * self.config.emission_factor_kg_per_kwh
        )

        measurement_type = EnergyMeasurementType.MODELED_TDP_ESTIMATE

        details = (
            f"Modeled TDP estimate using {len(generation_details) if generation_runs else 1} "
            f"generation(s), with total energy {energy_kwh:.8f} kWh "
            f"and grid emission factor "
            f"{self.config.emission_factor_kg_per_kwh} kg CO2/kWh."
        )

        return {
            "energy_kwh": energy_kwh,
            "emissions_kg":emissions_kg,
            "emissions_g": round(emissions_kg * 1000.0, 5),
            "latency_seconds": round(lat, 4),
            "measurement_type": measurement_type,
            "is_direct_measurement": False,
            "measurement_label": "Modeled TDP Estimate",
            "generation_details": generation_details,
            "details": details
        }
