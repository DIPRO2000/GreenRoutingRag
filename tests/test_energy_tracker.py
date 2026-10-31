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
