"""
GreenRouting: An Energy-Aware Retrieval-Augmented Generation Framework
with Adaptive Model Routing and Quality-Aware Caching.
Streamlit Main Dashboard Application.
"""
import os
import sys
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

# Page Configuration
st.set_page_config(
    page_title="GreenRouting | Energy-Aware RAG",
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
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 14px 18px;
        border-left: 4px solid #2d6a4f;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
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
    st.markdown("### 🌿 GreenRouting Control Center")
    st.markdown("**Energy-Aware RAG Framework**")
    st.divider()

    # System Diagnostics
    st.markdown("#### 🖥️ System Health")
    ollama_conn = orchestrator.llm_client.is_connected
    if ollama_conn:
        st.success(f"🟢 Ollama: Connected ({len(orchestrator.llm_client.available_models)} models)")
    else:
        st.warning("🟠 Ollama: Offline (Simulated Deterministic RAG Mode Active)")

    total_docs = len(orchestrator.db.get_all_documents())
    total_chunks = orchestrator.vector_store.get_total_chunks()
    active_cache_count = len(orchestrator.db.get_active_cache_entries())

    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Indexed Docs", total_docs)
    col_s2.metric("Total Chunks", total_chunks)
    st.metric("Active Cache Entries", active_cache_count)

    st.divider()
    st.markdown("#### ⚙️ Configuration")
    exec_mode = st.selectbox(
        "Default Mode",
        [ExecutionMode.GREENROUTING, ExecutionMode.STANDARD_RAG, ExecutionMode.RAG_WITH_CACHE],
        help="GreenRouting applies adaptive routing & quality caching; Standard RAG always uses capable model."
    )

    st.caption(f"**Lightweight Model:** `{settings.models.lightweight_model}`")
    st.caption(f"**Capable Model:** `{settings.models.capable_model}`")
    st.caption(f"**Embedding Model:** `{settings.models.embedding_model}`")
    st.caption(f"**Routing Complexity Threshold:** `{settings.routing.complexity_threshold}`")
    st.caption(f"**Cache Similarity Threshold:** `{settings.caching.semantic_similarity_threshold}`")

    if st.button("🧹 Purge Semantic Cache", use_container_width=True):
        orchestrator.semantic_cache.db.clear_all_cache()
        st.toast("Semantic cache purged successfully!", icon="🧹")
        st.rerun()

# ----------------- MAIN HEADER -----------------
st.markdown('<div class="main-title">🌿 GreenRouting</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">An Energy-Aware Retrieval-Augmented Generation Framework with Adaptive Model Routing and Quality-Aware Caching</div>',
    unsafe_allow_html=True
)

tab_rag, tab_docs, tab_cache, tab_bench, tab_arch = st.tabs([
    "💬 Green RAG Assistant",
    "📄 Document Knowledge Base",
    "🧠 Semantic Cache Inspector",
    "📊 Experimental Benchmarking",
    "📜 Architecture & Methodology"
])

# -------------------------------------------------------------
# TAB 1: GREEN RAG ASSISTANT
# -------------------------------------------------------------
with tab_rag:
    st.subheader("Interactive Document Question Answering")

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
        placeholder="e.g. How does GreenRouting reduce carbon emissions without sacrificing quality?"
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

            # Display Answer
            st.markdown("### Answer")
            st.markdown(response["answer"])

            # Telemetry Metrics Bar
            st.divider()
            st.markdown("#### ⚡ Green Telemetry & Environmental Impact")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Energy Consumed", f"{response['energy_kwh'] * 1e6:.2f} µWh" if response['energy_kwh'] < 1e-3 else f"{response['energy_kwh']:.6f} kWh")
            m2.metric("Carbon Emissions", f"{response['emissions_g']:.4f} g CO₂eq")
            m3.metric("Latency", f"{response['latency_seconds']:.3f} s")
            m4.metric("Groundedness", f"{response.get('grounding_score', 0.0) * 100.0:.1f}%")
            m5.metric("Quality Score", f"{response.get('quality_score', 0.0) * 100.0:.1f}%")

            st.caption(f"**Measurement Method:** `{response.get('measurement_label', 'Modeled')}` — {response.get('energy_details', '')}")

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
            st.markdown("#### 📚 Source Citations & References")
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
    st.markdown("Upload domain documents (PDF or TXT) to populate the GreenRouting FAISS vector knowledge base.")

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
                    if res.get("duplicate"):
                        st.info(
                            f"`{uploaded_file.name}` is already indexed as "
                            f"`{res['filename']}`; duplicate upload skipped."
                        )
                    else:
                        st.success(
                            f"Indexed `{uploaded_file.name}` "
                            f"({res['total_chunks']} chunks, {res['total_characters']} characters)"
                        )
                except Exception as e:
                    st.error(f"Error processing `{uploaded_file.name}`: {e}")
                progress.progress((idx + 1) / len(uploaded_files))
            st.rerun()

    with col_preset:
        st.markdown("**Pre-loaded Sample Corpora:**")
        st.write("Quickly load pre-configured technical documents on Green Computing & Renewable Energy:")
        if st.button("📦 Load GreenRouting Sample Documents", use_container_width=True):
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
        df_docs["doc_hash"] = df_docs["doc_hash"].apply(lambda h: h[:12] + "...")
        st.dataframe(df_docs, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: SEMANTIC CACHE INSPECTOR
# -------------------------------------------------------------
with tab_cache:
    st.subheader("Persistent Semantic Cache Inspector")
    st.markdown(
        "GreenRouting avoids wasteful inference by caching grounded answers. "
        "A cache hit is only accepted when vector similarity, active document compatibility, "
        "staleness, and source-grounding checks pass."
    )

    cache_entries = orchestrator.db.get_active_cache_entries()

    c_stat1, c_stat2, c_stat3 = st.columns(3)
    c_stat1.metric("Active Cached Queries", len(cache_entries))
    total_hits = sum(e.get("access_count", 1) - 1 for e in cache_entries) if cache_entries else 0
    c_stat2.metric("Total Inferences Saved", total_hits)
    c_stat3.metric("Similarity Threshold", f"{settings.caching.semantic_similarity_threshold}")

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
# TAB 4: EXPERIMENTAL BENCHMARKING
# -------------------------------------------------------------
with tab_bench:
    st.subheader("Empirical Benchmark and Ablation Study")
    st.markdown(
        "Run real comparative benchmarks across the three paradigms to measure empirical savings in "
        "**Energy (kWh)**, **Carbon Emissions (g CO₂eq)**, and **Latency (s)** without sacrificing answer quality."
    )

    col_b1, col_b2 = st.columns([3, 1])
    with col_b1:
        st.markdown("**Benchmark Question Suite:** (Includes factual, analytical, and repeat queries)")
        custom_queries_text = st.text_area(
            "Evaluation Queries (one per line):",
            value="\n".join(DEFAULT_BENCHMARK_QUERIES),
            height=140
        )
    with col_b2:
        st.write("")
        st.write("")
        run_bench_btn = st.button("⚡ Run Live Benchmark", type="primary", use_container_width=True)

    if run_bench_btn:
        queries_list = [q.strip() for q in custom_queries_text.split("\n") if q.strip()]
        if not queries_list:
            st.error("Please provide at least one query.")
        elif total_chunks == 0:
            st.error("Please upload documents before running benchmark.")
        else:
            with st.spinner("Running real benchmark across Standard RAG, RAG+Cache, and GreenRouting..."):
                runner = BenchmarkRunner(orchestrator)
                bench_result = runner.run_benchmark(queries=queries_list)

            st.session_state["last_benchmark"] = bench_result
            st.success("Benchmark completed successfully!")

    # Display benchmark results if present in session state or database
    if "last_benchmark" in st.session_state:
        bench_data = st.session_state["last_benchmark"]
        comp = bench_data["comparative_analysis"]
        summary_df = bench_data["summary_df"]
        records_df = bench_data["detailed_records_df"]

        st.divider()
        st.markdown("### 🏆 GreenRouting Comparative Performance vs Baseline")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("⚡ Energy Savings", f"{comp.get('energy_savings_pct', 0.0)}%", delta=f"{comp.get('energy_savings_pct', 0.0)}%")
        kpi2.metric("⏱️ Latency Reduction", f"{comp.get('latency_reduction_pct', 0.0)}%", delta=f"{comp.get('latency_reduction_pct', 0.0)}%")
        kpi3.metric("🎯 Quality Retained", f"{comp.get('quality_retained_pct', 100.0)}%")
        greenrouting_sum = bench_data["mode_summaries"].get(ExecutionMode.GREENROUTING, {})
        kpi4.metric("🔄 Cache Hit Rate", f"{greenrouting_sum.get('cache_hit_rate_pct', 0.0)}%")

        # Visualizations with Plotly
        st.markdown("#### 📈 Comparative Analytics")
        g1, g2 = st.columns(2)

        with g1:
            fig_energy = px.bar(
                summary_df,
                x="mode",
                y="total_energy_kwh",
                color="mode",
                title="Total Energy Consumption (kWh) — Lower is Better",
                labels={"total_energy_kwh": "Energy (kWh)", "mode": "Pipeline Mode"},
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
                title="Estimated Carbon Footprint (g CO₂eq) — Lower is Better",
                labels={"total_emissions_g": "Emissions (g CO₂eq)", "mode": "Pipeline Mode"},
                color_discrete_sequence=["#e63946", "#f4a261", "#2a9d8f"]
            )
            fig_emissions.update_layout(showlegend=False)
            st.plotly_chart(fig_emissions, use_container_width=True)

        g3, g4 = st.columns(2)
        with g3:
            fig_lat = px.bar(
                summary_df,
                x="mode",
                y="avg_latency_sec",
                color="mode",
                title="Average Response Latency (seconds) — Lower is Better",
                labels={"avg_latency_sec": "Latency (s)", "mode": "Pipeline Mode"},
                color_discrete_sequence=["#e63946", "#f4a261", "#2a9d8f"]
            )
            fig_lat.update_layout(showlegend=False)
            st.plotly_chart(fig_lat, use_container_width=True)

        with g4:
            fig_qual = px.bar(
                summary_df,
                x="mode",
                y="avg_grounding_score",
                color="mode",
                title="Answer Grounding Quality Score (0 - 1.0) — Maintained",
                labels={"avg_grounding_score": "Grounding Score", "mode": "Pipeline Mode"},
                color_discrete_sequence=["#457b9d", "#457b9d", "#2a9d8f"]
            )
            fig_qual.update_layout(showlegend=False)
            st.plotly_chart(fig_qual, use_container_width=True)

        st.markdown("#### 📄 Detailed Query Execution Records")
        st.dataframe(records_df, use_container_width=True)

        # CSV Export
        csv_data = records_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Benchmark Results to CSV",
            data=csv_data,
            file_name=f"greenrouting_benchmark_{bench_data['run_id']}.csv",
            mime="text/csv",
            use_container_width=True
        )

# -------------------------------------------------------------
# TAB 5: ARCHITECTURE & METHODOLOGY
# -------------------------------------------------------------
with tab_arch:
    st.subheader("System Architecture & Research Methodology")

    st.markdown("""
    ### 1. Conceptual Framework
    **GreenRouting** tackles the computational and environmental costs of Retrieval-Augmented Generation (RAG) by applying:
    - **Adaptive Green Model Routing**: Queries are dynamically categorized by computational complexity. Simple, single-fact lookups are routed to an ultra-lightweight language model (e.g. Qwen 2.5 0.5B or Llama 3.2 1B) drawing ~25–35W TDP. Multi-hop and comparative queries route to capable models (e.g. Llama 3.2 3B or Mistral 7B) drawing ~75–95W TDP.
    - **Quality-Aware Semantic Caching**: Eliminates duplicate LLM inferences using cosine similarity in embedding space, fortified with multi-factor validation (document scope match, source support, freshness, and grounding thresholds).
    - **Closed-Loop Quality Guard**: If a lightweight model produces an uncertain response or low context grounding, GreenRouting automatically escalates the query to the capable model with retry tracking.

    ### 2. Mathematical Formulations
    #### Query Complexity Score ($C \in [0, 1]$):
    $$C = w_{\\text{len}} f_{\\text{len}} + w_{\\text{reason}} f_{\\text{reason}} + w_{\\text{syntax}} f_{\\text{syntax}} + w_{\\text{domain}} f_{\\text{domain}}$$
    Where $C < \\theta_{\\text{route}} \\implies \\text{Lightweight Model}$, else $\\text{Capable Model}$.

    #### Energy and Carbon Estimation:
    $$E_{\\text{kWh}} = \\frac{P_{\\text{TDP}} \\times t_{\\text{latency}}}{3600 \\times 1000}$$
    $$\\text{Emissions (kg CO}_2\\text{eq)} = E_{\\text{kWh}} \\times \\text{Emission Factor}_{\\text{grid}}$$

    ### 3. Verification Protocol
    - **Ground Truth Grounding**: Evaluates vocabulary overlap between answer claims and retrieved context chunks.
    - **Context Compatibility**: Validates whether the cached answer matches the hash and ID of currently active documents.
    """)
