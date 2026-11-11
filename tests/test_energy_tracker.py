"""
Unit tests for Energy and Carbon Tracker in GreenRouting.
"""
import pytest
from greenrouting.config import EnergyConfig
from greenrouting.energy_tracker import EnergyTracker, EnergyMeasurementType


def test_energy_tracker_cache_hit():
    tracker = EnergyTracker()
    result = tracker.track_operation(
        operation_name="semantic_cache_hit",
        latency_seconds=0.005,
        is_cache_hit=True
    )

    assert result["measurement_type"] == EnergyMeasurementType.CACHED_ZERO_INFERENCE
    assert result["is_direct_measurement"] is False
    expected_energy = (tracker.config.cpu_tdp_watts * 0.25 * 0.005) / 3_600_000
    assert result["energy_kwh"] == pytest.approx(expected_energy)
    assert "Cache Reuse" in result["measurement_label"]
    assert "llm inference was avoided" in result["details"].lower()


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
    assert light_res["measurement_type"] == EnergyMeasurementType.MODELED_TDP_ESTIMATE



def test_energy_tracker_model_specific_retry_accounting():
    """Energy should be calculated separately for each model generation."""

    tracker = EnergyTracker(
        EnergyConfig(
            cpu_tdp_watts=50.0,
            gpu_tdp_watts=100.0,
            emission_factor_kg_per_kwh=0.5
        )
    )

    generation_runs = [
        {
            "model_name": "qwen2.5:0.5b",
            "latency_seconds": 2.0,
            "eval_count": 20
        },
        {
            "model_name": "llama3.2:3b",
            "latency_seconds": 4.0,
            "eval_count": 30
        }
    ]

    result = tracker.track_operation(
        operation_name="rag_query",
        latency_seconds=6.0,
        generation_runs=generation_runs
    )

    # Lightweight power = 50 * 0.60 = 30 W
    # Capable power = 50 + (100 * 0.65) = 115 W
    expected_energy = (
        (30.0 * 2.0) + (115.0 * 4.0)
    ) / 3_600_000

    expected_emissions = expected_energy * 0.5

    assert result["energy_kwh"] == pytest.approx(expected_energy)
    assert result["emissions_kg"] == pytest.approx(expected_emissions)
    assert result["latency_seconds"] == pytest.approx(6.0)
    assert len(result["generation_details"]) == 2

    assert result["generation_details"][0]["power_watts"] == 30.0
    assert result["generation_details"][1]["power_watts"] == 115.0

    assert result["measurement_type"] == (
        EnergyMeasurementType.MODELED_TDP_ESTIMATE
    )
    assert result["is_direct_measurement"] is False
