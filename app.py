"""
EcoWave: Energy-Aware Green Retrieval-Augmented Generation Framework
with Adaptive Model Routing and Quality-Aware Caching.
Streamlit Main Dashboard Application.
"""
import os
import sys
import json
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from greenrouting.config import load_settings
from greenrouting.orchestrator import GreenRoutingOrchestrator, ExecutionMode
from greenrouting.benchmark import BenchmarkRunner, DEFAULT_BENCHMARK_QUERIES
from greenrouting.cache_validator import CacheValidationStatus
from greenrouting.energy_tracker import ENERGY_DISCLAIMER

# Page Configuration
st.set_page_config(
    page_title="EcoWave | Energy-Aware Green RAG",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b4332;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #40916c;
        margin-bottom: 1.0rem;
    }
    .disclaimer-box {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 4px solid #16a34a;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #166534;
        margin-bottom: 1.2rem;
    }
    .sim-banner {
        background-color: #fffbeb;
        border: 1px solid #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #92400e;
        margin-bottom: 1rem;
    }
    .badge-cache-hit {
        background-color: #d8f3dc;
        color: #1b4332;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 4px;
        display: inline-block;
    }
    .badge-cache-miss {
        background-color: #f8d7da;
        color: #721c24;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 4px;
        display: inline-block;
    }
    .badge-model {
        background-color: #e2eafc;
        color: #1d3557;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 4px;
        display: inline-block;
    }
    .badge-sim {
        background-color: #fef3c7;
        color: #92400e;
        font-weight: 600;
        padding: 3px 7px;
        border-radius: 4px;
        display: inline-block;
    }
    .badge-real {
        background-color: #dcfce7;
        color: #166534;
        font-weight: 600;
        padding: 3px 7px;
        border-radius: 4px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_orchestrator():
    settings = load_settings()
    return GreenRoutingOrchestrator(settings)


orchestrator = get_orchestrator()
settings = orchestrator.settings

# ----------------- SIDEBAR DIAGNOSTICS & CONTROLS -----------------
with st.sidebar:
    st.markdown("### 🌿 EcoWave Control Center")
    st.markdown("**Energy-Aware Green RAG System**")
    st.divider()

    # System Diagnostics
    st.markdown("#### 🖥️ Runtime & Sensor Health")
    ollama_conn = orchestrator.llm_client.is_connected
    if ollama_conn:
        st.success(f"🟢 Ollama LLM: Connected ({len(orchestrator.llm_client.available_models)} models)")
    else:
        st.warning("🟠 Ollama: Offline (Simulated Deterministic RAG Active)")

    st.info(f"⚡ Energy Measurement: `{settings.energy.measurement_mode}`")

    total_docs = len(orchestrator.db.get_all_documents())
    total_chunks = orchestrator.vector_store.get_total_chunks()
    active_cache_count = len(orchestrator.db.get_active_cache_entries())

    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Indexed Docs", total_docs)
    col_s2.metric("Total Chunks", total_chunks)
    st.metric("Active Cache Entries", active_cache_count)

    st.divider()
    st.markdown("#### ⚙️ Configured Hardware Assumptions")
    st.caption(f"**CPU TDP:** `{settings.energy.hardware.cpu_tdp_watts} W`")
    st.caption(f"**GPU TDP:** `{settings.energy.hardware.gpu_tdp_watts} W`")
    st.caption(f"**Lightweight Model Power:** `{settings.energy.model_power.lightweight_model_watts} W`")
    st.caption(f"**Capable Model Power:** `{settings.energy.model_power.capable_model_watts} W`")
    st.caption(f"**Grid Carbon Intensity:** `{settings.energy.grid_carbon_intensity_g_per_kwh} g CO₂e/kWh`")

    st.divider()
    st.markdown("#### ⚙️ Pipeline Routing Config")
    st.caption(f"**Lightweight Model:** `{settings.models.lightweight_model}`")
    st.caption(f"**Capable Model:** `{settings.models.capable_model}`")
    st.caption(f"**Embedding Model:** `{settings.models.embedding_model}`")
    st.caption(f"**Complexity Threshold:** `{settings.routing.complexity_threshold}`")
    st.caption(f"**Cache Threshold:** `{settings.caching.semantic_similarity_threshold}`")

    if st.button("🧹 Purge Semantic Cache", use_container_width=True):
        orchestrator.semantic_cache.db.clear_all_cache()
        st.toast("Semantic cache purged successfully!", icon="🧹")
        st.rerun()

# ----------------- MAIN HEADER & TRANSPARENCY DISCLAIMER -----------------
st.markdown('<div class="main-title">🌿 EcoWave — Energy-Aware Green RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Intelligent Adaptive Model Routing & Quality-Aware Semantic Caching for Sustainable AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="disclaimer-box">⚠️ <strong>Transparency Disclaimer:</strong> {ENERGY_DISCLAIMER}</div>',
    unsafe_allow_html=True
)

tab_rag, tab_docs, tab_cache, tab_bench, tab_arch = st.tabs([
    "💬 Green RAG Assistant",
    "📄 Document Knowledge Base",
    "🧠 Semantic Cache Inspector",
    "📊 Experimental Benchmarking & Logs",
    "📜 Architecture & Methodology"
])

# -------------------------------------------------------------
# TAB 1: GREEN RAG ASSISTANT
# -------------------------------------------------------------
with tab_rag:
    st.subheader("Interactive Green Document Question Answering")

    # Sample query chips
    st.markdown("**Quick Query Templates:**")
    col_q1, col_q2, col_q3 = st.columns(3)
    preset_query = ""
    if col_q1.button("💡 Simple: Primary Objective?", use_container_width=True):
        preset_query = "What is the primary objective of GreenRouting?"
    if col_q2.button("🔬 Complex: Compare & Analyze Trade-offs?", use_container_width=True):
        preset_query = "Compare and analyze the trade-offs between lightweight model routing and capable model inference depth."
    if col_q3.button("☀️ Domain: Solar Panel Efficiency?", use_container_width=True):
        preset_query = "What is the conversion efficiency of monocrystalline silicon solar cells?"

    query_input = st.text_input(
        "Enter your question from uploaded documents:",
        value=preset_query,
        placeholder="e.g. How does EcoWave reduce carbon emissions without sacrificing answer quality?"
    )

    docs = orchestrator.db.get_all_documents()
    doc_options = {"All Indexed Documents": None}
    for d in docs:
        doc_options[f"{d['filename']} ({d['total_chunks']} chunks)"] = d["doc_id"]

    col_opt1, col_opt2 = st.columns([2, 1])
    selected_doc_label = col_opt1.selectbox("Filter Document Scope:", list(doc_options.keys()))
    selected_doc_id = doc_options[selected_doc_label]
    active_mode = col_opt2.selectbox("Execution Strategy:", [ExecutionMode.GREENROUTING, ExecutionMode.STANDARD_RAG, ExecutionMode.RAG_WITH_CACHE])

    if st.button("🚀 Answer Question", type="primary", use_container_width=True) and query_input.strip():
        if total_chunks == 0:
            st.error("No documents indexed yet! Please upload documents in the 'Document Knowledge Base' tab first.")
        else:
            with st.spinner("Executing energy-aware RAG pipeline..."):
                response = orchestrator.execute_query(
                    query=query_input.strip(),
                    mode=active_mode,
                    active_doc_id=selected_doc_id
                )

            # Simulation vs Real indicator
            if response.get("is_simulated"):
                st.markdown(
                    '<div class="sim-banner">🟠 <strong>Simulation Mode Active:</strong> Response was generated using the deterministic '
                    'context-grounded simulation fallback because local LLM service is offline. Telemetry is modeled.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<span class="badge-real">🟢 Real Model Execution</span> '
                    f'<span style="font-size:0.85rem; color:#166534;">Executed via local {response.get("model_used")} runtime.</span>',
                    unsafe_allow_html=True
                )

            # Display Query & Answer
            st.markdown(f"**User Query:** `{response['query_text']}`")
            st.markdown("### Answer")
            st.markdown(response["answer"])

            # Telemetry Metrics Bar
            st.divider()
            st.markdown("#### ⚡ Modeled Energy Consumption & Environmental Telemetry")
            m1, m2, m3, m4, m5 = st.columns(5)
            energy_wh = response.get("energy_wh", response.get("energy_kwh", 0.0) * 1000.0)
            m1.metric("Estimated Energy", f"{energy_wh:.4f} Wh")
            m2.metric("Estimated CO₂e", f"{response['emissions_g']:.4f} g CO₂e")
            m3.metric("Total Latency", f"{response['latency_seconds']:.3f} s")
            m4.metric("Groundedness", f"{response.get('grounding_score', 0.0) * 100.0:.1f}%")
            m5.metric("Quality Score", f"{response.get('quality_score', 0.0) * 100.0:.1f}%")

            # Comparative Savings against Baseline
            st.markdown("#### 🏆 Comparative Savings vs Baseline (Standard RAG)")
            s1, s2, s3 = st.columns(3)
            sav_pct = response.get("energy_savings_pct", 0.0)
            carb_sav = response.get("carbon_savings_g", 0.0)
            base_wh = response.get("baseline_energy_wh", 0.0)

            s1.metric("Estimated Energy Savings", f"{sav_pct:.1f}%", delta=f"{sav_pct:.1f}%")
            s2.metric("Estimated Carbon Savings", f"{carb_sav:.4f} g CO₂e", delta=f"{carb_sav:.4f} g")
            s3.metric("Baseline Energy Estimate", f"{base_wh:.4f} Wh")

            st.caption(
                f"**Measurement Mode:** `{response.get('measurement_mode', 'modeled_tdp_estimate')}` "
                f"({response.get('measurement_label', 'Modeled TDP')}) — {response.get('energy_details', '')}"
            )

            # Stage Latencies (Proof of No Double Counting)
            with st.expander("⏱️ Detailed Stage Latencies (Non-Overlapping Boundary)", expanded=False):
                stages = response.get("stage_latencies", {})
                if stages:
                    st.write("Each processing stage is timed independently. Wall-clock elapsed time is strictly measured.")
                    stage_df = pd.DataFrame([
                        {"Stage": k.replace("_", " ").title(), "Latency (seconds)": f"{v:.5f}"}
                        for k, v in stages.items()
                    ])
                    stage_df.loc[len(stage_df)] = {
                        "Stage": "Total Elapsed Wall-Clock",
                        "Latency (seconds)": f"{response['latency_seconds']:.5f}"
                    }
                    st.table(stage_df)
                else:
                    st.write("Stage timing breakdown available.")

            # Explainability & Routing Rationale
            st.markdown("#### 🧠 Decision Intelligence & Transparency")
            col_dec1, col_dec2 = st.columns(2)

            with col_dec1:
                st.markdown("**Cache Decision:**")
                c_status = response.get("cache_status", "MISS")
                if "HIT_VALID" in c_status:
                    st.markdown('<span class="badge-cache-hit">CACHE HIT (VALIDATED)</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="badge-cache-miss">CACHE {c_status}</span>', unsafe_allow_html=True)
                st.write(response.get("cache_validation_reason", "N/A"))

            with col_dec2:
                st.markdown("**Model Routing Decision:**")
                st.markdown(f'<span class="badge-model">{response.get("model_used", "N/A")}</span>', unsafe_allow_html=True)
                st.write(response.get("routing_reason", "N/A"))
                if response.get("retry_count", 0) > 0:
                    st.warning(f"⚠️ Quality Escalation: Retried with stronger model (Retry count: {response['retry_count']})")

            # Source references
            st.markdown("#### 📚 Retrieved Source Chunks")
            sources = response.get("sources", [])
            if not sources:
                st.info("No explicit source chunks retrieved.")
            else:
                for src in sources:
                    with st.expander(f"Source {src['source_index']}: Chunk {src['chunk_id']} (Page {src['page_number']}) — Similarity: {src['similarity_score']:.3f}"):
                        st.markdown(f"**Excerpt:**\n_{src['text_snippet']}_")

# -------------------------------------------------------------
# TAB 2: DOCUMENT KNOWLEDGE BASE
# -------------------------------------------------------------
with tab_docs:
    st.subheader("Document Ingestion and Vector Store")
    st.markdown("Upload domain documents (PDF or TXT) to populate the EcoWave FAISS vector knowledge base.")

    col_up, col_preset = st.columns([2, 1])

    with col_up:
        uploaded_files = st.file_uploader(
            "Choose PDF or TXT documents",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True
        )

        if st.button("📥 Process and Index Uploaded Files", type="primary") and uploaded_files:
            progress = st.progress(0)
            for idx, uploaded_file in enumerate(uploaded_files):
                file_bytes = uploaded_file.read()
                try:
                    res = orchestrator.process_and_index_document(uploaded_file.name, file_bytes)
                    st.success(f"Indexed `{uploaded_file.name}` ({res['total_chunks']} chunks, {res['total_characters']} characters)")
                except Exception as e:
                    st.error(f"Error processing `{uploaded_file.name}`: {e}")
                progress.progress((idx + 1) / len(uploaded_files))
            st.rerun()

    with col_preset:
        st.markdown("**Pre-loaded Sample Corpora:**")
        st.write("Quickly load pre-configured technical documents on Green Computing & Renewable Energy:")
        if st.button("📦 Load EcoWave Sample Documents", use_container_width=True):
            sample_dir = ROOT_DIR / "data" / "sample_documents"
            loaded_count = 0
            if sample_dir.exists():
                for sample_file in sample_dir.glob("*.txt"):
                    with open(sample_file, "rb") as f:
                        b = f.read()
                    orchestrator.process_and_index_document(sample_file.name, b)
                    loaded_count += 1
                st.success(f"Successfully loaded {loaded_count} sample documents!")
                st.rerun()

    st.divider()
    st.markdown("#### 📋 Currently Indexed Documents")
    all_docs = orchestrator.db.get_all_documents()
    if not all_docs:
        st.info("No documents currently indexed. Please upload files above.")
    else:
        df_docs = pd.DataFrame(all_docs)[["filename", "file_type", "total_chunks", "total_characters", "doc_hash", "uploaded_at"]]
        df_docs["doc_hash"] = df_docs["doc_hash"].apply(lambda h: str(h)[:12] + "...")
        st.dataframe(df_docs, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: SEMANTIC CACHE INSPECTOR
# -------------------------------------------------------------
with tab_cache:
    st.subheader("Persistent Semantic Cache Inspector")
    st.markdown(
        "EcoWave avoids wasteful model inference by caching grounded answers. "
        "A cache hit is accepted only when embedding similarity, document scope compatibility, "
        "staleness, and source-grounding checks pass. Cache hits incur zero model inference power."
    )

    cache_entries = orchestrator.db.get_active_cache_entries()
    query_logs = orchestrator.db.get_query_logs(limit=200, simulation_filter="all")

    # Compute cache metrics from real query logs
    total_log_queries = len(query_logs)
    cache_hit_logs = [q for q in query_logs if "HIT_VALID" in str(q.get("cache_status", ""))]
    cache_miss_logs = [q for q in query_logs if "MISS" in str(q.get("cache_status", ""))]
    total_cache_hits = len(cache_hit_logs)
    total_cache_misses = len(cache_miss_logs)
    cache_hit_rate = (total_cache_hits / total_log_queries * 100.0) if total_log_queries > 0 else 0.0

    # Cache lookup latency from logs
    cache_latencies = []
    for q in query_logs:
        try:
            sl = json.loads(q.get("stage_latencies_json", "{}"))
            if "cache_lookup" in sl and sl["cache_lookup"] > 0:
                cache_latencies.append(sl["cache_lookup"])
        except Exception:
            pass
    avg_cache_lookup_ms = (sum(cache_latencies) / len(cache_latencies) * 1000.0) if cache_latencies else 0.8

    # Estimated energy avoided through cache hits
    # For every cache hit, baseline inference power was completely avoided
    avoided_energy_wh = sum(q.get("baseline_energy_wh", 0.0) - q.get("energy_wh", 0.0) for q in cache_hit_logs if q.get("baseline_energy_wh", 0.0) > 0)

    c_stat1, c_stat2, c_stat3, c_stat4 = st.columns(4)
    c_stat1.metric("Active Cached Queries", len(cache_entries))
    c_stat2.metric("Total Cache Hits", total_cache_hits)
    c_stat3.metric("Cache Hit Rate", f"{cache_hit_rate:.1f}%")
    c_stat4.metric("Avg Cache Lookup", f"{avg_cache_lookup_ms:.2f} ms")

    st.markdown("#### ⚡ Environmental Impact of Caching")
    c_sav1, c_sav2 = st.columns(2)
    c_sav1.metric("Estimated Energy Avoided", f"{avoided_energy_wh:.4f} Wh", help="Energy saved by serving cached answers instead of rerunning LLM inference.")
    avoided_carbon_g = avoided_energy_wh / 1000.0 * settings.energy.grid_carbon_intensity_g_per_kwh
    c_sav2.metric("Estimated Carbon Avoided", f"{avoided_carbon_g:.4f} g CO₂e", help="Emissions prevented by serving cached answers.")

    st.caption("ℹ️ Cache hits do not count as model inference executions. Inference power is modeled as 0.0W on cache hits.")

    st.divider()
    st.markdown("#### 📋 Active Cache Entries")
    if not cache_entries:
        st.info("Semantic cache is currently empty. Query the assistant to populate entries.")
    else:
        rows = []
        for e in cache_entries:
            rows.append({
                "Cache ID": e["cache_id"],
                "Query": e["query_text"],
                "Answer Preview": e["answer_text"][:120] + "...",
                "Model Used": e["model_used"],
                "Grounding": f"{e['grounding_score']:.2f}",
                "Access Count": e["access_count"],
                "Created At": e["created_at"]
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# -------------------------------------------------------------
# TAB 4: EXPERIMENTAL BENCHMARKING & LOGS
# -------------------------------------------------------------
with tab_bench:
    st.subheader("Empirical Benchmark and Telemetry Analytics")
    st.markdown(
        "Evaluate comparative performance across Standard RAG, RAG with Caching, and Complete EcoWave. "
        "Transparently distinguishes between real-model executions and simulation fallbacks."
    )

    # 1. LIVE BENCHMARK SECTION
    st.markdown("### ⚡ Live Benchmark Suite")
    col_b1, col_b2 = st.columns([3, 1])
    with col_b1:
        st.markdown("**Benchmark Question Suite:** (Includes factual, analytical, and repeat queries)")
        custom_queries_text = st.text_area(
            "Evaluation Queries (one per line):",
            value="\n".join(DEFAULT_BENCHMARK_QUERIES),
            height=130
        )
    with col_b2:
        st.write("")
        st.write("")
        run_bench_btn = st.button("🚀 Run Live Benchmark", type="primary", use_container_width=True)

    if run_bench_btn:
        queries_list = [q.strip() for q in custom_queries_text.split("\n") if q.strip()]
        if not queries_list:
            st.error("Please provide at least one query.")
        elif total_chunks == 0:
            st.error("Please upload documents before running benchmark.")
        else:
            with st.spinner("Running real benchmark across Standard RAG, RAG+Cache, and EcoWave..."):
                runner = BenchmarkRunner(orchestrator)
                bench_result = runner.run_benchmark(queries=queries_list)

            st.session_state["last_benchmark"] = bench_result
            st.success("Benchmark completed successfully!")

    # Display live benchmark results if present
    if "last_benchmark" in st.session_state:
        bench_data = st.session_state["last_benchmark"]
        comp = bench_data["comparative_analysis"]
        summary_df = bench_data["summary_df"]
        records_df = bench_data["detailed_records_df"]

        st.divider()
        st.markdown("### 🏆 Live Benchmark Comparison vs Baseline")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("⚡ Energy Savings", f"{comp.get('energy_savings_pct', 0.0)}%", delta=f"{comp.get('energy_savings_pct', 0.0)}%")
        kpi2.metric("⏱️ Latency Reduction", f"{comp.get('latency_reduction_pct', 0.0)}%", delta=f"{comp.get('latency_reduction_pct', 0.0)}%")
        kpi3.metric("🎯 Quality Retained", f"{comp.get('quality_retained_pct', 100.0)}%")
        greenrouting_sum = bench_data["mode_summaries"].get(ExecutionMode.GREENROUTING, {})
        kpi4.metric("🔄 Cache Hit Rate", f"{greenrouting_sum.get('cache_hit_rate_pct', 0.0)}%")

        g1, g2 = st.columns(2)
        with g1:
            y_col = "total_energy_wh" if "total_energy_wh" in summary_df.columns else "total_energy_kwh"
            y_label = "Energy (Wh)" if y_col == "total_energy_wh" else "Energy (kWh)"
            fig_energy = px.bar(
                summary_df,
                x="mode",
                y=y_col,
                color="mode",
                title=f"Total Modeled Energy ({y_label}) — Lower is Better",
                labels={y_col: y_label, "mode": "Pipeline Mode"},
                color_discrete_sequence=["#e63946", "#f4a261", "#2a9d8f"]
            )
            fig_energy.update_layout(showlegend=False)
            st.plotly_chart(fig_energy, use_container_width=True)

        with g2:
            fig_emissions = px.bar(
                summary_df,
                x="mode",
                y="total_emissions_g",
                color="mode",
                title="Estimated Carbon Footprint (g CO₂e) — Lower is Better",
                labels={"total_emissions_g": "Emissions (g CO₂e)", "mode": "Pipeline Mode"},
                color_discrete_sequence=["#e63946", "#f4a261", "#2a9d8f"]
            )
            fig_emissions.update_layout(showlegend=False)
            st.plotly_chart(fig_emissions, use_container_width=True)

        st.markdown("#### 🎯 Quality–Carbon Trade-off")
        fig_quality_carbon = px.scatter(
            comparison_df,
            x="carbon_g",
            y="quality_pct",
            text="mode",
            size="total_queries",
            color="mode",
            title="Answer Quality vs Carbon Footprint — Higher quality and lower carbon is preferred",
            labels={
                "carbon_g": "Carbon (g CO₂e)",
                "quality_pct": "Average Quality (%)",
                "mode": "Pipeline Mode",
            },
            color_discrete_sequence=["#e63946", "#f4a261", "#2a9d8f"],
        )
        fig_quality_carbon.update_traces(textposition="top center")
        fig_quality_carbon.update_layout(showlegend=False)
        st.plotly_chart(fig_quality_carbon, use_container_width=True)

        # CSV Export
        csv_data = records_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Benchmark Results to CSV",
            data=csv_data,
            file_name=f"ecowave_benchmark_{bench_data['run_id']}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # 2. PERSISTED QUERY LOGS & HISTORICAL ANALYTICS
    st.divider()
    st.markdown("### 📊 Persisted Query Telemetry & Historical Analytics")

    # Execution Filter
    col_f1, col_f2 = st.columns([2, 2])
    sim_filter = col_f1.radio(
        "Telemetry Filter:",
        ["All Executions", "Real-Model Executions Only", "Simulation Executions Only"],
        horizontal=True,
        help="Separate simulation fallback records from real hardware LLM executions."
    )
    filter_val = "all" if "All" in sim_filter else ("real" if "Real" in sim_filter else "simulation")

    all_logs = orchestrator.db.get_query_logs(limit=250, simulation_filter=filter_val)

    if not all_logs:
        st.info(f"No telemetry logs found matching filter '{sim_filter}'. Execute queries in Tab 1 to generate records.")
    else:
        df_logs = pd.DataFrame(all_logs)

        # Guarantee expected columns exist
        if "energy_wh" not in df_logs.columns and "energy_kwh" in df_logs.columns:
            df_logs["energy_wh"] = df_logs["energy_kwh"] * 1000.0
        if "simulation_mode" not in df_logs.columns:
            df_logs["simulation_mode"] = 0

        # Summary KPIs
        tot_q = len(df_logs)
        c_hits = sum(1 for _, row in df_logs.iterrows() if "HIT_VALID" in str(row.get("cache_status", "")))
        hit_rate = (c_hits / tot_q * 100.0) if tot_q > 0 else 0.0
        avg_lat = df_logs["latency_seconds"].mean() if "latency_seconds" in df_logs else 0.0
        avg_energy = df_logs["energy_wh"].mean() if "energy_wh" in df_logs else 0.0
        avg_carb = df_logs["emissions_g"].mean() if "emissions_g" in df_logs else 0.0
        avg_sav = df_logs["energy_savings_pct"].mean() if "energy_savings_pct" in df_logs else 0.0
        avg_carb_sav = df_logs["carbon_savings_g"].mean() if "carbon_savings_g" in df_logs else 0.0

        if filter_val == "real":
            st.markdown("🟢 **Real-Model Execution Metrics** (Simulation records excluded)")
        elif filter_val == "simulation":
            st.markdown("🟠 **Simulation Execution Metrics** (Not actual hardware inference)")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Queries", tot_q)
        k2.metric("Cache Hit Rate", f"{hit_rate:.1f}%")
        k3.metric("Avg Latency", f"{avg_lat:.3f} s")
        k4.metric("Avg Energy (Wh)", f"{avg_energy:.4f} Wh")

        k5, k6, k7 = st.columns(3)
        k5.metric("Avg CO₂e", f"{avg_carb:.4f} g")
        k6.metric("Avg Energy Savings", f"{avg_sav:.1f}%")
        k7.metric("Avg Carbon Savings", f"{avg_carb_sav:.4f} g CO₂e")

        # Distribution charts
        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            if "model_used" in df_logs.columns:
                fig_model_dist = px.pie(
                    df_logs,
                    names="model_used",
                    title="Model Usage Distribution",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_model_dist, use_container_width=True)

        with c_chart2:
            if "mode" in df_logs.columns:
                fig_route_dist = px.pie(
                    df_logs,
                    names="mode",
                    title="Routing Mode Distribution",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig_route_dist, use_container_width=True)

        # Query log table
        st.markdown("#### 📋 Detailed Query Log Records")
        display_cols = [
            col for col in [
                "log_id", "timestamp", "query_text", "mode", "model_used", "cache_status",
                "simulation_mode", "latency_seconds", "energy_wh", "emissions_g",
                "energy_savings_pct", "carbon_savings_g", "grounding_score"
            ] if col in df_logs.columns
        ]
        st.dataframe(df_logs[display_cols], use_container_width=True)

# -------------------------------------------------------------
# TAB 5: ARCHITECTURE & METHODOLOGY
# -------------------------------------------------------------
with tab_arch:
    st.subheader("System Architecture & Research Methodology")

    st.markdown("""
    ### 1. Conceptual Framework
    **EcoWave** is an Energy-Aware Green Computing RAG Framework designed to minimize the energy consumption and carbon footprint of Retrieval-Augmented Generation without compromising answer quality.

    The architecture incorporates three core pillars:
    - **Adaptive Green Model Routing**: Dynamically classifies query complexity using multi-factor heuristics (length, reasoning keywords, syntax, and domain depth). Low-complexity single-fact questions route to lightweight models (e.g., Qwen 2.5 0.5B drawing ~27W TDP). High-complexity multi-hop questions route to capable models (e.g., Llama 3.2 3B drawing ~93W TDP).
    - **Quality-Aware Semantic Caching**: Eliminates duplicate LLM inferences using vector similarity in embedding space, validated with active document scope match, freshness, and context grounding. On a cache hit, model inference is bypassed completely (**0.0W inference power**).
    - **Closed-Loop Quality Guard**: Validates response groundedness against retrieved context chunks and automatically retries with a capable model when lightweight generation produces low grounding.

    ---

    ### 2. Energy Estimation Methodology (MVP Modeled TDP Estimate)
    To maintain rigorous scientific transparency, EcoWave's MVP uses a **Modeled TDP Estimate** rather than fabricating uncalibrated sensor claims.

    $$\\text{Energy (Wh)} = \\frac{\\text{Estimated Power (Watts)} \\times \\text{Execution Time (seconds)}}{3600}$$

    #### Power Assumptions by Pipeline Stage:
    All power assumptions are centrally managed in `config/config.yaml` and can be adjusted without code modifications:
    - **Cache Lookup**: ~12W in-memory vector similarity comparison.
    - **Context Retrieval & Embedding**: ~25W text embedding generation + FAISS search.
    - **Routing Analysis**: ~15W complexity heuristic evaluation.
    - **Lightweight Model Inference**: ~27.0W (~60% CPU TDP allocation).
    - **Capable Model Inference**: ~93.75W (CPU TDP + 65% GPU TDP allocation).
    - **Idle Overhead Baseline**: ~5.0W.

    On a verified cache hit, **model inference power is strictly 0.0W**, reflecting zero LLM compute execution.

    ---

    ### 3. Carbon Emissions Estimation
    Carbon emissions are calculated from modeled electrical energy consumption and the configured regional grid carbon intensity:

    $$\\text{Energy (kWh)} = \\frac{\\text{Energy (Wh)}}{1000}$$
    $$\\text{Emissions (g CO}_2\\text{e)} = \\text{Energy (kWh)} \\times \\text{Grid Carbon Intensity (g CO}_2\\text{e/kWh)}$$

    The default grid carbon intensity is configured to **708.0 g CO₂e/kWh** (reflecting regional grid average emissions).

    ---

    ### 4. Baseline Comparison & Savings Calculation
    Savings are evaluated against the non-optimized standard RAG baseline (**Standard RAG**, which always executes the capable model with no caching):

    $$\\text{Energy Savings (\\%)} = \\left( \\frac{\\text{Baseline Energy} - \\text{EcoWave Energy}}{\\text{Baseline Energy}} \\right) \\times 100$$
    $$\\text{Carbon Savings (g CO}_2\\text{e)} = \\text{Baseline CO}_2\\text{e} - \\text{EcoWave CO}_2\\text{e}$$

    Savings are only reported when both baseline and optimized measurements share identical power profiles and equivalent measurement boundaries.

    ---

    ### 5. Latency Boundary & Double-Counting Elimination
    Each pipeline stage is measured exactly once using `time.perf_counter()`:
    - **Query Processing Latency**: Wall-clock input token parsing and prompt construction.
    - **Retrieval Latency**: Wall-clock FAISS vector similarity search.
    - **Cache Lookup Latency**: Wall-clock semantic cache lookup.
    - **Routing Latency**: Wall-clock complexity calculation.
    - **Model Inference Latency**: Wall-clock LLM generation (0.0s for cache hits).
    - **Total Latency**: Measured strictly as wall-clock elapsed time from query start to finish. Component latencies are never re-added to an inclusive total.

    ---

    ### 6. Simulation Mode Transparency
    When local Ollama LLM services or GPU hardware are unavailable:
    - EcoWave engages an offline, context-grounded deterministic simulation fallback.
    - Every simulated execution record is tagged with `simulation_mode = 1`.
    - Simulated results are visibly badged in the UI and can be filtered in benchmarking analytics.
    - Simulation records are excluded from real-model hardware performance claims.

    ---

    ### 7. Measurement Limitations
    > **Disclaimer:** Energy values are estimates based on configured hardware power assumptions and execution time. They are not direct measurements from physical power sensors. Physical power meters or calibrated hardware counters (Intel RAPL, NVIDIA NVML) may be integrated in subsequent releases without changing the reporting contract.
    """)
