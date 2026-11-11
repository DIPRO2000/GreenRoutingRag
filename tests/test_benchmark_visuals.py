"""Tests for benchmark comparison data used by the Streamlit charts."""

import pandas as pd

from greenrouting.benchmark import build_comparison_frame


def test_comparison_frame_exposes_quality_and_carbon_metrics():
    summary = pd.DataFrame(
        [
            {
                "mode": "STANDARD_RAG",
                "total_energy_kwh": 0.01,
                "total_emissions_g": 7.08,
                "avg_quality_score": 0.80,
                "avg_grounding_score": 0.90,
                "avg_latency_sec": 2.0,
                "cache_hit_rate_pct": 0.0,
            },
            {
                "mode": "GREENROUTING",
                "total_energy_kwh": 0.004,
                "total_emissions_g": 2.83,
                "avg_quality_score": 0.78,
                "avg_grounding_score": 0.88,
                "avg_latency_sec": 1.5,
                "cache_hit_rate_pct": 50.0,
            },
        ]
    )

    frame = build_comparison_frame(summary)

    assert list(frame["quality_pct"]) == [80.0, 78.0]
    assert list(frame["carbon_g"]) == [7.08, 2.83]
    assert frame.loc[1, "energy_savings_vs_standard_pct"] == 60.0
    assert frame.loc[1, "carbon_savings_vs_standard_pct"] == 60.03


def test_comparison_frame_handles_zero_baseline_without_infinity():
    summary = pd.DataFrame(
        [
            {
                "mode": "STANDARD_RAG",
                "total_energy_kwh": 0.0,
                "total_emissions_g": 0.0,
                "avg_quality_score": 0.0,
                "avg_grounding_score": 0.0,
                "avg_latency_sec": 0.0,
                "cache_hit_rate_pct": 0.0,
            },
            {
                "mode": "GREENROUTING",
                "total_energy_kwh": 0.001,
                "total_emissions_g": -0.2,
                "avg_quality_score": 0.5,
                "avg_grounding_score": 0.6,
                "avg_latency_sec": 1.0,
                "cache_hit_rate_pct": 20.0,
            },
        ]
    )

    frame = build_comparison_frame(summary)

    assert frame["energy_savings_vs_standard_pct"].notna().all()
    assert frame["carbon_savings_vs_standard_pct"].notna().all()
    assert frame.loc[1, "carbon_g"] == -0.2
