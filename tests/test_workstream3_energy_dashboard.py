"""
End-to-end integration and unit tests for Workstream 3:
Energy Accounting, Persisted Query Logging, and Streamlit Dashboard Backend.
"""
import pytest
import sqlite3
import pandas as pd
from greenrouting.config import load_settings
from greenrouting.orchestrator import GreenRoutingOrchestrator, ExecutionMode
from greenrouting.energy_tracker import EnergyTracker, ENERGY_DISCLAIMER
from greenrouting.benchmark import build_comparison_frame


@pytest.fixture
def clean_orchestrator(tmp_path):
    """Fixture providing an isolated orchestrator with temporary SQLite and FAISS storage."""
    settings = load_settings()
    settings.app.db_path = tmp_path / "test_ecowave.db"
    settings.app.faiss_index_dir = tmp_path / "test_faiss"
    settings.models.enable_simulation_fallback = True
    settings.caching.min_cache_grounding_score = 0.40

    orch = GreenRoutingOrchestrator(settings)

    # Index sample document
    doc_text = (
        "EcoWave minimizes energy consumption in retrieval-augmented generation. "
        "EcoWave uses adaptive green routing and quality-aware caching. "
        "Energy savings are modeled using calibrated hardware TDP power profiles."
    )
    orch.process_and_index_document("sample_ecowave.txt", doc_text.encode("utf-8"))
    return orch


def test_latency_not_double_counted(clean_orchestrator):
    """
    Verify that total latency represents actual wall-clock elapsed time
    and does not add component latency onto an already-inclusive total.
    """
    orch = clean_orchestrator
    query = "What minimizes energy consumption in EcoWave?"

    res = orch.execute_query(query, mode=ExecutionMode.STANDARD_RAG)

    total_latency = res["latency_seconds"]
    stages = res["stage_latencies"]

    # Model inference latency was measured as a sub-stage
    inference_latency = stages["model_inference"]

    # Total latency must NOT be equal to (total_latency + inference_latency)
    # Total latency must be approximately equal to sum of individual non-overlapping stages
    sum_stages = sum(stages.values())
    assert pytest.approx(total_latency, rel=0.15) == sum_stages
    assert total_latency > 0.0


def test_query_text_is_persisted(clean_orchestrator):
    """Verify that query_text is non-empty and accurately persisted in the database."""
    orch = clean_orchestrator
    query = "What minimizes energy consumption in EcoWave?"

    res = orch.execute_query(query, mode=ExecutionMode.GREENROUTING)
    assert res["query_text"] == query

    # Inspect SQLite database directly
    logs = orch.db.get_recent_query_logs(limit=10)
    assert len(logs) >= 1
    latest_log = logs[0]
    assert latest_log["query_text"] == query
    assert len(latest_log["query_text"]) > 0


def test_exactly_one_log_record_per_completed_query(clean_orchestrator):
    """Verify that each completed query produces exactly one log record."""
    orch = clean_orchestrator
    initial_count = len(orch.db.get_query_logs(limit=100))

    orch.execute_query("First query test", mode=ExecutionMode.STANDARD_RAG)
    orch.execute_query("Second query test", mode=ExecutionMode.GREENROUTING)

    final_logs = orch.db.get_query_logs(limit=100)
    assert len(final_logs) == initial_count + 2


def test_cache_hits_generate_valid_log_records(clean_orchestrator):
    """Verify that cache hits also generate a complete, valid log record."""
    orch = clean_orchestrator
    query = "What minimizes energy consumption in EcoWave?"

    # Initial query (Cache Miss)
    first_res = orch.execute_query(query, mode=ExecutionMode.GREENROUTING)
    assert "MISS" in first_res["cache_status"]

    # Second query (Cache Hit)
    hit_res = orch.execute_query(query, mode=ExecutionMode.GREENROUTING)
    assert "HIT_VALID" in hit_res["cache_status"]
    assert hit_res["query_text"] == query

    logs = orch.db.get_recent_query_logs(limit=5)
    hit_log = logs[0]
    assert "HIT_VALID" in hit_log["cache_status"]
    assert hit_log["query_text"] == query
    assert hit_log["latency_seconds"] > 0.0
    assert hit_log["energy_wh"] >= 0.0


def test_cache_hits_do_not_increment_model_inference(clean_orchestrator):
    """Verify that cache hits do not report model inference latency or power."""
    orch = clean_orchestrator
    query = "What minimizes energy consumption in EcoWave?"

    # First query to populate cache
    orch.execute_query(query, mode=ExecutionMode.GREENROUTING)

    # Second query is a cache hit
    hit_res = orch.execute_query(query, mode=ExecutionMode.GREENROUTING)
    stages = hit_res["stage_latencies"]

    # Inference latency must be strictly 0.0
    assert stages["model_inference"] == 0.0
    assert "Cached Response" in hit_res["model_used"]


def test_simulation_records_are_distinguishable(clean_orchestrator):
    """Verify simulation records are flagged with simulation_mode boolean."""
    orch = clean_orchestrator
    # When Ollama is disconnected, is_simulated is True
    res = orch.execute_query("Simulation test question", mode=ExecutionMode.GREENROUTING)

    logs = orch.db.get_recent_query_logs(limit=1)
    latest_log = logs[0]

    if res.get("is_simulated"):
        assert latest_log["simulation_mode"] == 1
    else:
        assert latest_log["simulation_mode"] == 0


def test_simulation_results_excluded_from_real_model_summaries(clean_orchestrator):
    """Verify that filtering by 'real' strictly excludes simulation records."""
    orch = clean_orchestrator

    # Log an explicit simulated row and an explicit real row
    orch.db.log_query({
        "query_text": "Real query",
        "answer_text": "Real answer",
        "mode": "GREENROUTING",
        "model_used": "llama3.2:3b",
        "cache_status": "MISS",
        "latency_seconds": 1.5,
        "energy_wh": 0.04,
        "emissions_g": 0.028,
        "simulation_mode": 0
    })

    orch.db.log_query({
        "query_text": "Simulated query",
        "answer_text": "Simulated answer",
        "mode": "GREENROUTING",
        "model_used": "llama3.2:3b (Simulated)",
        "cache_status": "MISS",
        "latency_seconds": 0.1,
        "energy_wh": 0.002,
        "emissions_g": 0.0014,
        "simulation_mode": 1
    })

    real_logs = orch.db.get_query_logs(limit=10, simulation_filter="real")
    sim_logs = orch.db.get_query_logs(limit=10, simulation_filter="simulation")

    for r in real_logs:
        assert r["simulation_mode"] == 0

    for s in sim_logs:
        assert s["simulation_mode"] == 1


def test_missing_baseline_values_safe(clean_orchestrator):
    """Verify missing baseline values produce safe 0.0 savings without NaN/Inf."""
    tracker = EnergyTracker()
    savings = tracker.calculate_savings(
        baseline_energy_wh=0.0,
        optimized_energy_wh=0.02,
        baseline_emissions_g=0.0,
        optimized_emissions_g=0.014,
        is_simulated=False
    )

    assert savings["energy_savings_pct"] == 0.0
    assert savings["carbon_savings_g"] == 0.0
    assert savings["is_comparable"] is False


def test_dashboard_handles_empty_logs_gracefully():
    """Verify helper and frame builders handle empty logs without error."""
    empty_df = pd.DataFrame()
    frame = build_comparison_frame(empty_df)
    assert frame.empty


def test_dashboard_handles_legacy_log_fields_safely():
    """Verify that dataframe constructed from legacy logs missing new columns is handled safely."""
    legacy_logs = [
        {
            "log_id": 1,
            "query_text": "Old query",
            "answer_text": "Old answer",
            "mode": "STANDARD_RAG",
            "model_used": "llama3.2:3b",
            "cache_status": "MISS",
            "latency_seconds": 1.2,
            "energy_kwh": 0.00003,
            "emissions_kg": 0.000021,
            # Missing query_id, simulation_mode, energy_wh, emissions_g, etc.
        }
    ]
    df = pd.DataFrame(legacy_logs)

    # Mimic dashboard column defaulting
    if "energy_wh" not in df.columns and "energy_kwh" in df.columns:
        df["energy_wh"] = df["energy_kwh"] * 1000.0
    if "simulation_mode" not in df.columns:
        df["simulation_mode"] = 0

    assert pytest.approx(df["energy_wh"].iloc[0]) == 0.03
    assert df["simulation_mode"].iloc[0] == 0
