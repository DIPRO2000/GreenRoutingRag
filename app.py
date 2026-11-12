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
from greenrouting.benchmark import BenchmarkRunner, DEFAULT_BENCHMARK_QUERIES, build_comparison_frame
from greenrouting.parameter_optimizer import run_parameter_optimization
from greenrouting.cache_validator import CacheValidationStatus
from greenrouting.query_intent import DOCUMENT_SUMMARY, detect_query_intent
from greenrouting.document_selection import build_document_options
from greenrouting.document_processor import MIN_PDF_PAGE_TEXT_CHARS

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

    if orchestrator.semantic_cache.storage.backend_name == "Redis":
        st.success("Redis quality cache: Connected")
    else:
        st.info("Redis unavailable; semantic cache is using SQLite fallback")

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
        orchestrator.semantic_cache.storage.clear_all_cache()
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
    doc_options = build_document_options(docs)

    col_opt1, col_opt2 = st.columns([2, 1])
    selected_doc_label = col_opt1.selectbox("Filter Document Scope:", list(doc_options.keys()))
    selected_doc_id = doc_options[selected_doc_label]
    active_mode = col_opt2.selectbox("Execution Strategy:", [ExecutionMode.GREENROUTING, ExecutionMode.STANDARD_RAG, ExecutionMode.RAG_WITH_CACHE])

    if st.button("🚀 Answer Question", type="primary", use_container_width=True) and query_input.strip():
        if total_chunks == 0:
            st.error("No documents indexed yet! Please upload documents in the 'Document Knowledge Base' tab first.")
        else:
            query_intent = detect_query_intent(query_input.strip())
            response = None
            if query_intent == DOCUMENT_SUMMARY:
                if selected_doc_id is None:
                    st.warning("Select one document in the Document Scope dropdown before asking for its summary.")
                else:
                    try:
                        with st.spinner("Summarizing all sections of the selected document..."):
                            response = orchestrator.summarize_document(
                                selected_doc_id, query_text=query_input.strip(), mode=active_mode
                            )
                    except Exception as exc:
                        st.error(f"Could not summarize this document: {exc}")
            else:
                with st.spinner("Executing energy-aware RAG pipeline..."):
                    response = orchestrator.execute_query(
                        query=query_input.strip(),
                        mode=active_mode,
                        active_doc_id=selected_doc_id
                    )

            if response is not None:
                # Display Answer
                st.markdown("### Document Summary" if response.get("is_document_summary") else "### Answer")
                if response.get("is_simulated"):
                    st.warning("This answer was generated by the offline simulation fallback, not by the configured Ollama model.")
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

                if response.get("is_document_summary"):
                    generation_details = response.get("generation_details", [])
                    with st.expander(f"Summary model calls and energy estimates ({len(generation_details)} calls)"):
                        st.dataframe(
                            pd.DataFrame([
                                {
                                    "Stage": item.get("stage", "generation"),
                                    "Model": item["model_name"],
                                    "Latency (s)": item["latency_seconds"],
                                    "Modeled Power (W)": item["power_watts"],
                                    "Energy (kWh)": item["energy_kwh"],
                                    "Estimated CO2e (g)": item["energy_kwh"] * settings.energy.emission_factor_kg_per_kwh * 1000.0,
                                }
                                for item in generation_details
                            ]),
                            use_container_width=True,
                            hide_index=True,
                        )

                # Explainability & Routing Rationale
                st.markdown("#### 🧠 Decision Intelligence & Transparency")
                col_dec1, col_dec2, col_dec3 = st.columns(3)

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

                with col_dec3:
                    st.markdown("**Retrieval Decision:**")
                    st.write(
                        f"Used {response.get('retrieved_chunk_count', 0)} of "
                        f"top-{response.get('retrieval_top_k', 0)} chunks."
                    )
                    st.write(response.get("retrieval_reason", "N/A"))

                # Source references
                st.markdown("#### 📚 Source Citations & References")
                sources = response.get("sources", [])
                if response.get("is_document_summary"):
                    st.info(
                        f"All {response['retrieved_chunk_count']} document chunks were included. "
                        f"Source pages: {', '.join(map(str, response.get('summary_pages', [])))}."
                    )
                    with st.expander(f"Source passages used ({len(sources)} chunks)"):
                        st.dataframe(
                            pd.DataFrame([
                                {
                                    "Chunk": source["chunk_id"],
                                    "Page": source["page_number"],
                                    "Text excerpt": source["text_snippet"],
                                }
                                for source in sources
                            ]),
                            use_container_width=True,
                            hide_index=True,
                        )
                elif not sources:
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
                    if res.get("low_text_pages"):
                        pages = ", ".join(map(str, res["low_text_pages"]))
                        st.warning(
                            f"Limited or no selectable text was found on PDF page(s) {pages} "
                            f"(fewer than {MIN_PDF_PAGE_TEXT_CHARS} characters per page). "
                            "These pages may be scanned or image-heavy, so answers and summaries may miss their content. "
                            "OCR is not available."
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

        delete_options = {
            f"{doc['filename']} ({doc['total_chunks']} chunks)": doc["doc_id"]
            for doc in all_docs
        }
        selected_delete_label = st.selectbox("Select a document to remove:", list(delete_options.keys()))
        if st.button("🗑️ Delete selected document", type="secondary"):
            deletion = orchestrator.delete_document(delete_options[selected_delete_label])
            if deletion.get("deleted"):
                st.success(
                    f"Removed {deletion['filename']} ({deletion['removed_chunk_count']} chunks). "
                    f"Rebuilt index with {deletion['remaining_chunks']} remaining chunks."
                )
                st.rerun()
            else:
                st.warning("That document is no longer indexed. Refreshing the list.")
                st.rerun()

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
    prebuilt_tab, custom_tab = st.tabs(["Prebuilt Benchmark", "Custom Document Benchmark"])
    with prebuilt_tab:
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
                simulated_count = sum(
                    bool(record.get("is_simulated", False))
                    for record in bench_result["detailed_records_df"].to_dict("records")
                )
                if simulated_count:
                    st.warning(
                        f"{simulated_count} benchmark query result(s) used offline simulation. "
                        "Treat these as functional checks, not real model measurements."
                    )

        st.divider()
        st.markdown("### Quality-Constrained Threshold Tuning (Optuna)")
        st.caption(
            "Runs GreenRouting repeatedly with different routing and cache thresholds, then recommends "
            "the lowest-energy settings that meet the quality floor. This can take several minutes. "
            "Recommendations are experimental and do not change the app settings."
        )
        tune_col1, tune_col2, tune_col3 = st.columns([1, 1, 2])
        with tune_col1:
            optimizer_trials = st.number_input("Optuna trials", min_value=2, max_value=12, value=4, step=1)
        with tune_col2:
            quality_floor = st.number_input(
                "Minimum quality score",
                min_value=0.0,
                max_value=1.0,
                value=float(settings.routing.min_acceptable_quality_score),
                step=0.05,
                format="%.2f",
            )
        with tune_col3:
            tune_btn = st.button("Find lower-energy settings", use_container_width=True)

        if tune_btn:
            tuning_queries = [q.strip() for q in custom_queries_text.split("\n") if q.strip()]
            if not tuning_queries:
                st.error("Enter at least one evaluation question before tuning.")
            elif total_chunks == 0:
                st.error("Upload or load a document before tuning settings.")
            else:
                try:
                    with st.spinner(
                        f"Running {int(optimizer_trials)} trials over {len(tuning_queries)} questions. "
                        "This sends repeated requests to your configured model..."
                    ):
                        tuning_result = run_parameter_optimization(
                            orchestrator,
                            queries=tuning_queries,
                            n_trials=int(optimizer_trials),
                            min_quality_score=float(quality_floor),
                        )
                    st.session_state["last_parameter_optimization"] = tuning_result
                except (RuntimeError, ValueError) as exc:
                    st.error(str(exc))

        if "last_parameter_optimization" in st.session_state:
            tuning_result = st.session_state["last_parameter_optimization"]
            if tuning_result["feasible"]:
                best = tuning_result["recommended"]
                st.success(
                    f"Recommended trial {best['trial']}: quality {best['avg_quality_score']:.2f}, "
                    f"energy {best['total_energy_kwh']:.8f} kWh."
                )
                st.write({
                    "complexity_threshold": best["complexity_threshold"],
                    "cache_similarity_threshold": best["cache_similarity_threshold"],
                    "quality_floor": tuning_result["min_quality_score"],
                })
            else:
                best_quality = tuning_result["best_quality_trial"]
                st.warning(
                    "No tested settings met the quality floor. "
                    f"The highest measured score was {best_quality['avg_quality_score']:.2f}; "
                    "lower the floor only if that reflects your evaluation requirements."
                )
            if any(row["contains_simulated_results"] for row in tuning_result["trials"]):
                st.warning("At least one trial used simulated model output; do not present its energy result as a live measurement.")
            st.caption(tuning_result["quality_metric_note"])
            st.dataframe(pd.DataFrame(tuning_result["trials"]), use_container_width=True, hide_index=True)

        # Restore the latest persisted run after a Streamlit restart.
        if "last_benchmark" not in st.session_state:
            saved_runs = orchestrator.db.get_all_benchmark_results()
            if saved_runs:
                latest_run_id = saved_runs[0]["run_id"]
                mode_summaries = {}
                for saved in saved_runs:
                    if saved["run_id"] != latest_run_id:
                        continue
                    summary = dict(saved)
                    summary["total_emissions_g"] = summary["total_emissions_kg"] * 1000.0
                    summary["cache_hit_rate_pct"] = (
                        round(summary["cache_hits"] / summary["total_queries"] * 100.0, 1)
                        if summary["total_queries"] else 0.0
                    )
                    mode_summaries[summary["mode"]] = summary
                baseline = mode_summaries.get(ExecutionMode.STANDARD_RAG, {})
                green = mode_summaries.get(ExecutionMode.GREENROUTING, {})
                comparative = {}
                if baseline and green:
                    comparative = {
                        "energy_savings_pct": round((baseline["total_energy_kwh"] - green["total_energy_kwh"]) / baseline["total_energy_kwh"] * 100.0, 2) if baseline["total_energy_kwh"] else 0.0,
                        "latency_reduction_pct": round((baseline["avg_latency_sec"] - green["avg_latency_sec"]) / baseline["avg_latency_sec"] * 100.0, 2) if baseline["avg_latency_sec"] else 0.0,
                        "quality_retained_pct": round(green["avg_quality_score"] / baseline["avg_quality_score"] * 100.0, 2) if baseline["avg_quality_score"] else 100.0,
                    }
                st.session_state["last_benchmark"] = {
                    "run_id": latest_run_id,
                    "mode_summaries": mode_summaries,
                    "comparative_analysis": comparative,
                    "summary_df": pd.DataFrame(list(mode_summaries.values())),
                    "detailed_records_df": pd.DataFrame(),
                }

        # Display benchmark results if present in session state or database
        if "last_benchmark" in st.session_state:
            bench_data = st.session_state["last_benchmark"]
            comp = bench_data["comparative_analysis"]
            summary_df = bench_data["summary_df"]
            comparison_df = build_comparison_frame(summary_df)
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
                    "carbon_g": "Carbon (g CO₂eq)",
                    "quality_pct": "Average Quality (%)",
                    "mode": "Pipeline Mode",
                },
                color_discrete_sequence=["#e63946", "#f4a261", "#2a9d8f"],
            )
            fig_quality_carbon.update_traces(textposition="top center")
            fig_quality_carbon.update_layout(showlegend=False)
            st.plotly_chart(fig_quality_carbon, use_container_width=True)

            st.markdown("#### 📋 Quality, Cost, and Savings Summary")
            comparison_columns = [
                "mode",
                "quality_pct",
                "grounding_pct",
                "avg_latency_sec",
                "cache_hit_rate_pct",
                "total_energy_kwh",
                "carbon_g",
                "energy_savings_vs_standard_pct",
                "carbon_savings_vs_standard_pct",
            ]
            st.dataframe(
                comparison_df[comparison_columns].rename(
                    columns={
                        "mode": "Mode",
                        "quality_pct": "Quality (%)",
                        "grounding_pct": "Grounding (%)",
                        "avg_latency_sec": "Avg Latency (s)",
                        "cache_hit_rate_pct": "Cache Hits (%)",
                        "total_energy_kwh": "Energy (kWh)",
                        "carbon_g": "Carbon (g CO₂eq)",
                        "energy_savings_vs_standard_pct": "Energy Savings vs Standard (%)",
                        "carbon_savings_vs_standard_pct": "Carbon Savings vs Standard (%)",
                    }
                ),
                use_container_width=True,
            )
            st.caption(
                "Interpretation: the strongest result lowers energy/carbon while keeping answer quality and grounding comparable. "
                "Negative savings are preserved as observed measurements rather than hidden."
            )

            st.markdown("#### 📄 Detailed Query Execution Records")
            if records_df.empty:
                st.info("Per-query details are available immediately after a run; saved history contains mode summaries.")
            else:
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


    with custom_tab:
        st.subheader("Custom Document Benchmark")
        st.markdown(
            "Compare Standard RAG, RAG with Cache, and GreenRouting on one document using your own question and a paraphrase. "
            "Both prompts run in order during one comparison."
        )
        st.info(
            "The first prompt normally misses in the cache modes. The paraphrase may hit only if the first answer passes "
            "quality checks and the paraphrase is sufficiently similar. Standard RAG never uses cache. "
            "Document-wide summary requests bypass semantic caching."
        )
        source_kind = st.radio(
            "Choose document source",
            ["Select indexed document", "Upload and index a document"],
            horizontal=True,
            key="custom_benchmark_source_kind",
        )
        selected_doc_id = None
        selected_doc_label = None
        uploaded_custom_doc = None
        indexed_documents = []
        if source_kind == "Select indexed document":
            indexed_documents = orchestrator.db.get_all_documents()
            if not indexed_documents:
                st.warning("No indexed documents yet. Upload one in Document Knowledge Base or choose the upload option here.")
            else:
                document_options = {
                    f"{doc['filename']} ({doc['total_chunks']} chunks, ID {doc['doc_id']})": doc["doc_id"]
                    for doc in indexed_documents
                }
                selected_doc_label = st.selectbox(
                    "Document to benchmark",
                    list(document_options),
                    key="custom_benchmark_document",
                )
                selected_doc_id = document_options[selected_doc_label]
        else:
            uploaded_custom_doc = st.file_uploader(
                "Choose a PDF, TXT, or Markdown file",
                type=["pdf", "txt", "md"],
                key="custom_benchmark_upload",
            )
            st.caption("Uploaded files are added to Document Knowledge Base and remain indexed until you delete them there.")

        custom_question = st.text_input(
            "Question",
            placeholder="e.g. What is the main objective described in the document?",
            key="custom_benchmark_question",
        )
        custom_paraphrase = st.text_input(
            "Paraphrase for cache test",
            placeholder="Ask the same thing using different wording",
            key="custom_benchmark_paraphrase",
        )
        st.caption("Use a factual or analytical question. A whole-document summary follows a separate path and bypasses semantic cache.")
        run_custom_bench = st.button(
            "Run Custom Comparison",
            type="primary",
            use_container_width=True,
            disabled=(source_kind == "Select indexed document" and not indexed_documents),
            key="run_custom_benchmark",
        )

        if run_custom_bench:
            if not custom_question.strip() or not custom_paraphrase.strip():
                st.error("Enter both a question and its paraphrase.")
            elif custom_question.strip().casefold() == custom_paraphrase.strip().casefold():
                st.error("The paraphrase must use different wording from the original question.")
            elif source_kind == "Upload and index a document" and uploaded_custom_doc is None:
                st.error("Choose a document to upload.")
            else:
                try:
                    if source_kind == "Upload and index a document":
                        indexed = orchestrator.process_and_index_document(
                            uploaded_custom_doc.name,
                            uploaded_custom_doc.getvalue(),
                        )
                        selected_doc_id = indexed["doc_id"]
                        selected_doc_label = (
                            f"{indexed['filename']} ({indexed['total_chunks']} chunks, ID {indexed['doc_id']})"
                        )
                        if indexed.get("duplicate"):
                            st.info("This file was already indexed; reusing its existing document entry.")
                        else:
                            st.success(f"Indexed {indexed['filename']} for this comparison.")
                    if not orchestrator.db.get_document_chunks(selected_doc_id):
                        st.error("The selected document has no indexed text chunks to benchmark.")
                    else:
                        prompts = [custom_question.strip(), custom_paraphrase.strip()]
                        with st.spinner("Running both prompts through all three strategies. This may take a few minutes..."):
                            custom_result = BenchmarkRunner(orchestrator).run_benchmark(
                                queries=prompts,
                                active_doc_id=selected_doc_id,
                                include_answers=True,
                            )
                        st.session_state["last_custom_benchmark"] = {
                            "result": custom_result,
                            "document_label": selected_doc_label,
                            "queries": prompts,
                        }
                        st.success("Custom comparison completed. Your existing semantic cache was restored afterward.")
                except (RuntimeError, ValueError) as exc:
                    st.error(f"Could not run the custom comparison: {exc}")

        if "last_custom_benchmark" in st.session_state:
            custom_saved = st.session_state["last_custom_benchmark"]
            custom_result = custom_saved["result"]
            st.markdown(f"#### Results for {custom_saved['document_label']}")
            custom_records = custom_result["detailed_records_df"]
            summary_columns = [
                "mode", "query_index", "query", "cache_status", "cache_validation_reason",
                "model_used", "latency_seconds", "quality_score", "grounding_score",
                "energy_kwh", "emissions_g", "is_simulated",
            ]
            st.dataframe(
                custom_records[summary_columns].rename(columns={
                    "mode": "Strategy", "query_index": "Prompt #", "query": "Prompt",
                    "cache_status": "Cache", "cache_validation_reason": "Cache explanation",
                    "model_used": "Model", "latency_seconds": "Latency (s)",
                    "quality_score": "Quality", "grounding_score": "Grounding",
                    "energy_kwh": "Energy (kWh)", "emissions_g": "Carbon (g CO2e)",
                    "is_simulated": "Simulated",
                }),
                use_container_width=True,
                hide_index=True,
            )
            for record in custom_records.to_dict("records"):
                with st.expander(f"{record['mode']} ? Prompt {record['query_index']}: {record['query']}"):
                    st.markdown("**Answer**")
                    st.write(record.get("answer") or "No answer text was returned.")
                    st.caption(
                        f"Cache: {record.get('cache_status') or 'N/A'} ? "
                        f"{record.get('cache_validation_reason') or 'No cache explanation returned.'}"
                    )
            if custom_records["is_simulated"].any():
                st.warning("Some outputs used offline simulation. Treat these as functional checks, not live model measurements.")
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
