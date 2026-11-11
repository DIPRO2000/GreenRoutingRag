"""
Energy and Carbon Monitoring Module for GreenRouting / EcoWave.
Implements transparent Modeled TDP Estimation for Green RAG systems.
Does not claim direct hardware measurement when running without physical power meters.
"""
import logging
from typing import Dict, Any, Optional
from greenrouting.config import EnergyConfig

logger = logging.getLogger(__name__)


class EnergyMeasurementType:
    DIRECT_HARDWARE = "DIRECT_HARDWARE"
    MODELED_TDP_ESTIMATE = "modeled_tdp_estimate"
    CACHED_ZERO_INFERENCE = "CACHED_ZERO_INFERENCE"


ENERGY_DISCLAIMER = (
    "Energy values are estimates based on configured hardware power assumptions and execution time. "
    "They are not direct measurements from physical power sensors."
)


class EnergyTracker:
    """
    Tracks and models compute energy consumption (Wh / kWh) and carbon emissions (g / kg CO2e).
    Implements calibrated TDP modeling with clear assumptions from config/config.yaml.
    Guarantees transparent reporting: MVP measurement mode is 'modeled_tdp_estimate'.
    """

    def __init__(self, config: Optional[EnergyConfig] = None):
        self.config = config or EnergyConfig()
        self.measurement_mode = "modeled_tdp_estimate"
        self.hardware_sensors_available = False

    def get_stage_power_watts(self, stage: str, model_name: Optional[str] = None) -> float:
        """
        Return the estimated power consumption in Watts for a specific pipeline stage.
        All assumptions are driven by config.yaml.
        """
        stage_cfg = self.config.stage_power
        model_cfg = self.config.model_power
        idle_overhead = getattr(self.config.hardware, "idle_overhead_watts", 0.0)

        if stage == "query_processing":
            return stage_cfg.query_processing_watts + idle_overhead
        elif stage == "cache_lookup":
            return stage_cfg.cache_lookup_watts + idle_overhead
        elif stage in ["retrieval", "retrieval_embedding"]:
            return stage_cfg.retrieval_embedding_watts + idle_overhead
        elif stage in ["routing", "routing_analysis"]:
            return stage_cfg.routing_analysis_watts + idle_overhead
        elif stage in ["model_inference", "inference", "llm_generation"]:
            m_lower = (model_name or "").lower()
            if any(tag in m_lower for tag in ["0.5b", "1b", "tiny", "mini", "lightweight"]):
                base_power = model_cfg.lightweight_model_watts
            elif any(tag in m_lower for tag in ["3b", "7b", "mistral", "llama", "capable"]):
                base_power = model_cfg.capable_model_watts
            else:
                base_power = model_cfg.default_model_watts
            return base_power + idle_overhead
        else:
            return self.config.hardware.cpu_tdp_watts + idle_overhead

    def calculate_energy_wh(self, power_watts: float, duration_seconds: float) -> float:
        """
        Energy (Wh) = Estimated Power (W) × Execution Time (seconds) / 3600
        """
        sec = max(0.0, float(duration_seconds))
        return (power_watts * sec) / 3600.0

    def calculate_emissions_g(self, energy_wh: float) -> float:
        """
        Carbon emissions (g CO2e) = Energy (kWh) × Grid Carbon Intensity (g CO2e/kWh)
        where Energy (kWh) = Energy (Wh) / 1000.
        """
        energy_kwh = energy_wh / 1000.0
        intensity = getattr(self.config, "grid_carbon_intensity_g_per_kwh", 708.0)
        return energy_kwh * intensity

    def track_stages(
        self,
        stage_latencies: Dict[str, float],
        model_used: Optional[str] = None,
        is_cache_hit: bool = False,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Calculate non-overlapping stage-by-stage energy and carbon emissions.
        Guarantees that inference energy is 0.0 for cache hits, and power is never double counted.
        """
        stage_breakdown = {}
        total_energy_wh = 0.0

        for stage_name, lat in stage_latencies.items():
            if is_cache_hit and stage_name in ["model_inference", "inference"]:
                stage_power = 0.0
                stage_energy_wh = 0.0
            else:
                stage_power = self.get_stage_power_watts(stage_name, model_name=model_used)
                stage_energy_wh = self.calculate_energy_wh(stage_power, lat)

            stage_emissions_g = self.calculate_emissions_g(stage_energy_wh)
            stage_breakdown[stage_name] = {
                "latency_seconds": round(lat, 5),
                "estimated_power_watts": round(stage_power, 2),
                "energy_wh": round(stage_energy_wh, 8),
                "emissions_g": round(stage_emissions_g, 6)
            }
            total_energy_wh += stage_energy_wh

        total_emissions_g = self.calculate_emissions_g(total_energy_wh)
        total_energy_kwh = total_energy_wh / 1000.0
        total_emissions_kg = total_emissions_g / 1000.0

        measurement_label = "Modeled TDP Estimate"
        measurement_type = EnergyMeasurementType.MODELED_TDP_ESTIMATE
        if is_cache_hit:
            measurement_label = "Cache Reuse (Zero LLM Inference)"
            measurement_type = EnergyMeasurementType.CACHED_ZERO_INFERENCE

        return {
            "energy_wh": round(total_energy_wh, 6),
            "emissions_g": round(total_emissions_g, 4),
            "energy_kwh": round(total_energy_kwh, 9),
            "emissions_kg": round(total_emissions_kg, 9),
            "stage_breakdown": stage_breakdown,
            "measurement_mode": self.measurement_mode,
            "measurement_type": measurement_type,
            "measurement_label": measurement_label,
            "disclaimer": ENERGY_DISCLAIMER,
            "details": f"Estimated using Modeled TDP assumptions and stage timings. Grid intensity: {self.config.grid_carbon_intensity_g_per_kwh} g CO2e/kWh."
        }

    def calculate_baseline(
        self,
        retrieval_latency: float,
        baseline_inference_latency: float,
        baseline_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Estimate baseline energy and carbon for non-optimized standard RAG (capable model, no cache).
        Uses identical power assumptions and equivalent measurement boundaries.
        """
        b_model = baseline_model or self.config.baseline.model
        ret_power = self.get_stage_power_watts("retrieval")
        inf_power = self.get_stage_power_watts("model_inference", model_name=b_model)

        ret_energy_wh = self.calculate_energy_wh(ret_power, retrieval_latency)
        inf_energy_wh = self.calculate_energy_wh(inf_power, baseline_inference_latency)
        total_baseline_wh = ret_energy_wh + inf_energy_wh
        total_baseline_emissions_g = self.calculate_emissions_g(total_baseline_wh)

        return {
            "baseline_energy_wh": round(total_baseline_wh, 6),
            "baseline_emissions_g": round(total_baseline_emissions_g, 4),
            "baseline_energy_kwh": round(total_baseline_wh / 1000.0, 9),
            "baseline_emissions_kg": round(total_baseline_emissions_g / 1000.0, 9),
            "baseline_model": b_model
        }

    def calculate_savings(
        self,
        baseline_energy_wh: float,
        optimized_energy_wh: float,
        baseline_emissions_g: float,
        optimized_emissions_g: float,
        is_simulated: bool = False
    ) -> Dict[str, Any]:
        """
        Compute energy and carbon savings compared to baseline.
        Energy Savings (%) = ((Baseline Energy - EcoWave Energy) / Baseline Energy) × 100
        Carbon Savings (g CO2e) = Baseline CO2e - EcoWave CO2e
        """
        if baseline_energy_wh <= 0:
            return {
                "energy_savings_wh": 0.0,
                "energy_savings_pct": 0.0,
                "carbon_savings_g": 0.0,
                "is_comparable": False,
                "is_simulated": is_simulated,
                "note": "Baseline energy not available or zero."
            }

        savings_wh = baseline_energy_wh - optimized_energy_wh
        savings_pct = (savings_wh / baseline_energy_wh) * 100.0
        carbon_savings_g = baseline_emissions_g - optimized_emissions_g

        return {
            "energy_savings_wh": round(savings_wh, 6),
            "energy_savings_pct": round(savings_pct, 2),
            "carbon_savings_g": round(carbon_savings_g, 4),
            "is_comparable": True,
            "is_simulated": is_simulated,
            "note": "Simulation results should not be presented as real-world hardware savings." if is_simulated else "Estimated savings based on configured TDP models."
        }

    def track_operation(
        self,
        operation_name: str,
        latency_seconds: float,
        model_name: Optional[str] = None,
        is_cache_hit: bool = False,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Backwards-compatible query-level tracker.
        Returns both Wh and kWh, g and kg, adhering to Modeled TDP Estimate.
        """
        lat = max(0.0001, float(latency_seconds))

        if is_cache_hit:
            retrieval_power_watts = self.config.stage_power.cache_lookup_watts
            energy_wh = self.calculate_energy_wh(retrieval_power_watts, min(lat, 0.05))
            emissions_g = self.calculate_emissions_g(energy_wh)
            energy_kwh = energy_wh / 1000.0
            emissions_kg = emissions_g / 1000.0

            return {
                "energy_wh": round(energy_wh, 8),
                "energy_kwh": round(energy_kwh, 9),
                "emissions_g": round(emissions_g, 6),
                "emissions_kg": round(emissions_kg, 9),
                "latency_seconds": round(lat, 4),
                "measurement_mode": self.measurement_mode,
                "measurement_type": EnergyMeasurementType.CACHED_ZERO_INFERENCE,
                "is_direct_measurement": False,
                "measurement_label": "Cache Reuse (Zero LLM Inference)",
                "disclaimer": ENERGY_DISCLAIMER,
                "details": "LLM generation bypassed. Only in-memory similarity lookup energy estimated."
            }

        # Model inference power
        base_power_watts = self.get_stage_power_watts("model_inference", model_name=model_name)
        effective_duration = lat * (1.0 + 0.40 * retry_count)
        energy_wh = self.calculate_energy_wh(base_power_watts, effective_duration)
        emissions_g = self.calculate_emissions_g(energy_wh)
        energy_kwh = energy_wh / 1000.0
        emissions_kg = emissions_g / 1000.0

        return {
            "energy_wh": round(energy_wh, 6),
            "energy_kwh": round(energy_kwh, 8),
            "emissions_g": round(emissions_g, 5),
            "emissions_kg": round(emissions_kg, 8),
            "latency_seconds": round(lat, 4),
            "measurement_mode": self.measurement_mode,
            "measurement_type": EnergyMeasurementType.MODELED_TDP_ESTIMATE,
            "is_direct_measurement": False,
            "measurement_label": "Modeled TDP Estimate",
            "disclaimer": ENERGY_DISCLAIMER,
            "details": (
                f"Modeled using {base_power_watts:.1f}W TDP power profile over {lat:.2f}s latency "
                f"with grid carbon intensity of {self.config.grid_carbon_intensity_g_per_kwh:.1f} g CO2e/kWh."
            )
        }
