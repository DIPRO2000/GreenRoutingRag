"""
Unit tests for Energy and Carbon Tracker in GreenRouting / EcoWave.
Tests energy formulas, unit conversions, configurable power assumptions,
carbon intensity math, and cache hit energy avoidance.
"""
import pytest
from greenrouting.config import EnergyConfig, HardwarePowerConfig, ModelPowerConfig, StagePowerConfig
from greenrouting.energy_tracker import EnergyTracker, EnergyMeasurementType, ENERGY_DISCLAIMER


def test_energy_tracker_cache_hit():
    tracker = EnergyTracker()
    result = tracker.track_operation(
        operation_name="semantic_cache_hit",
        latency_seconds=0.005,
        is_cache_hit=True
    )

    assert result["measurement_type"] == EnergyMeasurementType.CACHED_ZERO_INFERENCE
    assert result["is_direct_measurement"] is False
    assert result["energy_kwh"] < 1e-6
    assert "Cache Reuse" in result["measurement_label"]


def test_energy_tracker_lightweight_vs_capable_scaling():
    tracker = EnergyTracker(EnergyConfig(cpu_tdp_watts=50.0, gpu_tdp_watts=100.0))

    # Lightweight model: smaller power profile
    light_res = tracker.track_operation(
        operation_name="lightweight_rag",
        latency_seconds=0.5,
        model_name="qwen2.5:0.5b",
        is_cache_hit=False
    )

    # Capable model: higher power profile and typically higher latency
    capable_res = tracker.track_operation(
        operation_name="capable_rag",
        latency_seconds=1.2,
        model_name="llama3.2:3b",
        is_cache_hit=False
    )

    assert capable_res["energy_kwh"] > light_res["energy_kwh"]
    assert capable_res["emissions_kg"] > light_res["emissions_kg"]
    assert light_res["measurement_type"] in [
        EnergyMeasurementType.MODELED_TDP_ESTIMATE,
        EnergyMeasurementType.DIRECT_HARDWARE
    ]


def test_energy_tracker_retry_accounting():
    tracker = EnergyTracker()
    single_run = tracker.track_operation(
        operation_name="rag_query",
        latency_seconds=1.0,
        model_name="llama3.2:3b",
        retry_count=0
    )

    retried_run = tracker.track_operation(
        operation_name="rag_query",
        latency_seconds=1.0,
        model_name="llama3.2:3b",
        retry_count=1
    )

    # Retried run should account for additional power burned during initial attempt
    assert retried_run["energy_kwh"] > single_run["energy_kwh"]


def test_energy_calculation_formula():
    """Verify Energy (Wh) = Estimated Power (W) × Execution Time (seconds) / 3600."""
    tracker = EnergyTracker()
    power_watts = 72.0
    latency_seconds = 5.0
    expected_wh = (72.0 * 5.0) / 3600.0  # 0.1 Wh
    calculated_wh = tracker.calculate_energy_wh(power_watts, latency_seconds)
    assert pytest.approx(calculated_wh, rel=1e-5) == expected_wh
    assert pytest.approx(calculated_wh, rel=1e-5) == 0.1


def test_configurable_power_assumptions():
    """Verify that configurable assumptions from config directly change power estimations."""
    custom_cfg = EnergyConfig(
        hardware=HardwarePowerConfig(cpu_tdp_watts=60.0, gpu_tdp_watts=120.0, idle_overhead_watts=10.0),
        stage_power=StagePowerConfig(cache_lookup_watts=15.0, retrieval_embedding_watts=30.0),
        model_power=ModelPowerConfig(lightweight_model_watts=35.0, capable_model_watts=110.0)
    )
    tracker = EnergyTracker(custom_cfg)

    # Querying stage power reflects customized values
    assert tracker.get_stage_power_watts("cache_lookup") == 15.0 + 10.0
    assert tracker.get_stage_power_watts("retrieval") == 30.0 + 10.0
    assert tracker.get_stage_power_watts("model_inference", model_name="qwen2.5:0.5b") == 35.0 + 10.0
    assert tracker.get_stage_power_watts("model_inference", model_name="llama3.2:3b") == 110.0 + 10.0


def test_energy_unit_conversion():
    """Verify correct conversion between Wh and kWh."""
    tracker = EnergyTracker()
    wh = tracker.calculate_energy_wh(3600.0, 1.0)  # 3600 W * 1s / 3600 = 1.0 Wh
    assert pytest.approx(wh, rel=1e-5) == 1.0

    kwh = wh / 1000.0
    assert pytest.approx(kwh, rel=1e-5) == 0.001


def test_co2e_calculation():
    """Verify carbon calculation: Emissions (g CO2e) = Energy (kWh) × Grid Carbon Intensity (g/kWh)."""
    custom_cfg = EnergyConfig(grid_carbon_intensity_g_per_kwh=500.0)
    tracker = EnergyTracker(custom_cfg)

    energy_wh = 2.0  # 0.002 kWh
    emissions_g = tracker.calculate_emissions_g(energy_wh)
    # 0.002 kWh * 500 g/kWh = 1.0 g CO2e
    assert pytest.approx(emissions_g, rel=1e-5) == 1.0


def test_measurement_mode_and_disclaimer():
    """Verify measurement mode is modeled_tdp_estimate and disclaimer is present."""
    tracker = EnergyTracker()
    telemetry = tracker.track_stages(
        stage_latencies={"retrieval": 0.05, "model_inference": 0.2},
        model_used="llama3.2:3b"
    )
    assert telemetry["measurement_mode"] == "modeled_tdp_estimate"
    assert telemetry["disclaimer"] == ENERGY_DISCLAIMER
    assert "Energy values are estimates based on configured hardware power assumptions" in telemetry["disclaimer"]


def test_cache_hit_zero_inference_power():
    """Verify cache hits incur strictly 0.0W inference power."""
    tracker = EnergyTracker()
    telemetry = tracker.track_stages(
        stage_latencies={"cache_lookup": 0.005, "model_inference": 0.5},
        model_used="Cached Response",
        is_cache_hit=True
    )
    assert telemetry["stage_breakdown"]["model_inference"]["estimated_power_watts"] == 0.0
    assert telemetry["stage_breakdown"]["model_inference"]["energy_wh"] == 0.0
    assert telemetry["measurement_type"] == EnergyMeasurementType.CACHED_ZERO_INFERENCE


def test_baseline_and_savings_calculation():
    """Verify baseline comparison and savings calculation math."""
    tracker = EnergyTracker()
    baseline_wh = 0.10
    optimized_wh = 0.04
    baseline_emissions_g = 0.0708
    optimized_emissions_g = 0.0283

    savings = tracker.calculate_savings(
        baseline_energy_wh=baseline_wh,
        optimized_energy_wh=optimized_wh,
        baseline_emissions_g=baseline_emissions_g,
        optimized_emissions_g=optimized_emissions_g,
        is_simulated=False
    )

    # (0.10 - 0.04) / 0.10 * 100 = 60.0%
    assert pytest.approx(savings["energy_savings_pct"], rel=1e-2) == 60.0
    assert pytest.approx(savings["carbon_savings_g"], rel=1e-4) == 0.0425
    assert savings["is_comparable"] is True


def test_zero_or_missing_baseline_handled_safely():
    """Verify missing or zero baseline values do not crash or produce divide-by-zero."""
    tracker = EnergyTracker()
    savings = tracker.calculate_savings(
        baseline_energy_wh=0.0,
        optimized_energy_wh=0.05,
        baseline_emissions_g=0.0,
        optimized_emissions_g=0.035,
        is_simulated=False
    )
    assert savings["energy_savings_pct"] == 0.0
    assert savings["carbon_savings_g"] == 0.0
    assert savings["is_comparable"] is False
