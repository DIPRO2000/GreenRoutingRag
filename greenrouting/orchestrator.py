"""
GreenRouting / EcoWave Orchestrator.
Unifies document indexing, retrieval, adaptive routing, quality-aware caching,
energy tracking, and persistent telemetry logging into a single cohesive pipeline.
Eliminates latency double-counting and enforces transparent Modeled TDP energy accounting.
"""
import time
import logging
import uuid
from typing import Dict, Any, List, Optional
from greenrouting.config import Settings
from greenrouting.database import DatabaseManager
from greenrouting.document_processor import DocumentProcessor
from greenrouting.vector_store import EmbeddingService, FaissVectorStore
from greenrouting.llm_interface import OllamaClient
from greenrouting.rag_pipeline import RagPipeline, RAG_SYSTEM_PROMPT
from greenrouting.adaptive_router import AdaptiveGreenRouter
from greenrouting.semantic_cache import SemanticCache
from greenrouting.cache_validator import CacheValidationStatus
from greenrouting.energy_tracker import EnergyTracker
from greenrouting.evaluator import RagEvaluator
from greenrouting.retrieval_policy import choose_top_k

logger = logging.getLogger(__name__)


class ExecutionMode:
    STANDARD_RAG = "STANDARD_RAG"          # Capable model only, no cache
    RAG_WITH_CACHE = "RAG_WITH_CACHE"      # Capable model with standard semantic cache
    GREENROUTING = "GREENROUTING"          # Quality-aware cache + adaptive green routing + retry


class GreenRoutingOrchestrator:
    """Master orchestrator for the GreenRouting / EcoWave energy-aware RAG framework."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.db = DatabaseManager(settings.app.db_path)
        self.embedding_service = EmbeddingService(settings.models.embedding_model)
        self.vector_store = FaissVectorStore(settings.app.faiss_index_dir, self.embedding_service)
        self.llm_client = OllamaClient(
            base_url=settings.models.ollama_base_url,
            enable_simulation_fallback=settings.models.enable_simulation_fallback
        )
        self.doc_processor = DocumentProcessor(
            chunk_size=settings.rag.chunk_size,
            chunk_overlap=settings.rag.chunk_overlap
        )
        self.semantic_cache = SemanticCache(
            db_manager=self.db,
            embedding_service=self.embedding_service,
            caching_config=settings.caching
        )
        self.router = AdaptiveGreenRouter(
            llm_client=self.llm_client,
            model_config=settings.models,
            routing_config=settings.routing
        )
        self.energy_tracker = EnergyTracker(settings.energy)
        self.evaluator = RagEvaluator(self.embedding_service)
        self.rag_pipeline = RagPipeline(
            vector_store=self.vector_store,
            llm_client=self.llm_client,
            rag_config=settings.rag,
            model_config=settings.models
        )

    def process_and_index_document(self, filename: str, file_bytes: bytes) -> Dict[str, Any]:
        """Process an uploaded document, chunk it, embed it, and store metadata in SQLite & FAISS."""
        doc_meta, chunks = self.doc_processor.process_document(filename, file_bytes)

        existing = self.db.get_document_by_hash(doc_meta["doc_hash"])
        if existing:
            logger.info(
                "Skipping duplicate document '%s'; already indexed as '%s'.",
                filename,
                existing["filename"],
            )
            return {
                "doc_id": existing["doc_id"],
                "filename": existing["filename"],
                "total_chunks": existing["total_chunks"],
                "total_characters": existing["total_characters"],
                "duplicate": True,
            }

        # Save to SQLite
        self.db.save_document(
            doc_id=doc_meta["doc_id"],
            filename=doc_meta["filename"],
            doc_hash=doc_meta["doc_hash"],
            file_type=doc_meta["file_type"],
            total_chunks=doc_meta["total_chunks"],
            total_characters=doc_meta["total_characters"]
        )
        self.db.save_chunks(chunks)

        # Add to Vector Store
        self.vector_store.add_chunks(chunks)

        return {
            "doc_id": doc_meta["doc_id"],
            "filename": doc_meta["filename"],
            "total_chunks": doc_meta["total_chunks"],
            "total_characters": doc_meta["total_characters"],
            "duplicate": False,
        }

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Delete one document and rebuild the vector index from remaining DB chunks."""
        document = next((doc for doc in self.db.get_all_documents() if doc["doc_id"] == doc_id), None)
        if document is None:
            return {
                "deleted": False,
                "doc_id": doc_id,
                "filename": None,
                "removed_chunk_count": 0,
                "invalidated_cache_count": 0,
                "remaining_documents": len(self.db.get_all_documents()),
                "remaining_chunks": self.vector_store.get_total_chunks(),
            }

        previous_chunks = self.db.get_all_chunks()
        try:
            deletion = self.db.delete_document(doc_id)
            remaining_chunks = self.db.get_all_chunks()
            self.vector_store.rebuild(remaining_chunks)
        except Exception:
            # Restore the index from the pre-delete snapshot when rebuilding fails.
            self.vector_store.rebuild(previous_chunks)
            raise

        return {
            "deleted": True,
            "doc_id": doc_id,
            "filename": document["filename"],
            "removed_chunk_count": deletion["removed_chunk_count"],
            "invalidated_cache_count": deletion["invalidated_cache_count"],
            "remaining_documents": len(self.db.get_all_documents()),
            "remaining_chunks": self.vector_store.get_total_chunks(),
        }

    def execute_query(
        self,
        query: str,
        mode: str = ExecutionMode.GREENROUTING,
        active_doc_id: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute query answering under the specified mode:
        1. Context Retrieval from vector store with stage timing.
        2. Semantic Cache lookup with stage timing (bypasses LLM on hit).
        3. Adaptive Model Routing or Capable Model generation with stage timing.
        4. Non-double-counted wall-clock latency measurement via time.perf_counter().
        5. Transparent Modeled TDP Energy & Carbon estimation.
        6. Baseline comparison (energy/carbon savings).
        7. Single persistent query log record generation including query_text.
        """
        overall_start = time.perf_counter()
        query_id = f"q_{uuid.uuid4().hex[:12]}"

        active_doc_ids = [active_doc_id] if active_doc_id else [d["doc_id"] for d in self.db.get_all_documents()]
        all_chunks = self.db.get_all_chunks()
        available_chunks = sum(
            1 for chunk in all_chunks
            if active_doc_id is None or chunk.get("doc_id") == active_doc_id
        )

        # Retrieval policy decision
        t_prep_start = time.perf_counter()
        if top_k is not None:
            k = max(1, int(top_k)) if available_chunks else 0
            retrieval_complexity_score = 0.0
            retrieval_reason = f"Explicit top_k override requested: {k}."
        else:
            retrieval_decision = choose_top_k(
                query=query,
                available_chunks=available_chunks,
                default_top_k=self.settings.rag.top_k_retrieval,
            )
            k = retrieval_decision.top_k
            retrieval_complexity_score = retrieval_decision.complexity_score
            retrieval_reason = retrieval_decision.reason
        prep_latency = time.perf_counter() - t_prep_start

        # 1. Retrieve Context Chunks
        t_retrieval_start = time.perf_counter()
        search_results = self.vector_store.search(query, top_k=k, doc_id_filter=active_doc_id)
        retrieval_latency = time.perf_counter() - t_retrieval_start
        retrieved_chunks = [item[0] for item in search_results]
        similarity_scores = [item[1] for item in search_results]

        if not retrieved_chunks:
            early_elapsed = time.perf_counter() - overall_start
            return {
                "query_id": query_id,
                "query_text": query,
                "answer": "No relevant documents found. Please upload and index documents first.",
                "mode": mode,
                "model_used": "None",
                "cache_status": CacheValidationStatus.MISS,
                "cache_validation_reason": "No documents indexed.",
                "complexity_score": 0.0,
                "routing_reason": "N/A",
                "retry_count": 0,
                "retrieved_chunks": [],
                "sources": [],
                "latency_seconds": round(early_elapsed, 4),
                "stage_latencies": {
                    "query_processing": round(prep_latency, 5),
                    "retrieval": round(retrieval_latency, 5),
                    "cache_lookup": 0.0,
                    "routing": 0.0,
                    "model_inference": 0.0,
                },
                "energy_wh": 0.0,
                "energy_kwh": 0.0,
                "emissions_g": 0.0,
                "emissions_kg": 0.0,
                "grounding_score": 0.0,
                "quality_score": 0.0,
                "is_simulated": False,
                "measurement_label": "N/A",
                "measurement_mode": self.energy_tracker.measurement_mode,
                "baseline_energy_wh": 0.0,
                "baseline_emissions_g": 0.0,
                "energy_savings_pct": 0.0,
                "carbon_savings_g": 0.0,
                "decision_trace": self._build_decision_trace(
                    cache_status=CacheValidationStatus.MISS,
                    cache_reason="No documents indexed.",
                    retrieval_top_k=k,
                    retrieved_chunk_count=0,
                    retrieval_reason=retrieval_reason,
                    model_used="None",
                    complexity_score=0.0,
                    routing_reason="N/A",
                    retry_count=0,
                    energy_kwh=0.0,
                    emissions_g=0.0,
                    latency_seconds=early_elapsed,
                    measurement_label="N/A",
                )
            }

        prompt = self.rag_pipeline.build_prompt(query, retrieved_chunks)

        # 2. Semantic Cache Lookup
        t_cache_start = time.perf_counter()
        cache_status = CacheValidationStatus.MISS
        cache_reason = "Caching not enabled for this mode."
        cached_answer = None

        if mode in [ExecutionMode.RAG_WITH_CACHE, ExecutionMode.GREENROUTING]:
            eval_doc_ids = active_doc_ids if mode == ExecutionMode.GREENROUTING else None
            eval_chunks = retrieved_chunks if mode == ExecutionMode.GREENROUTING else None

            cache_result = self.semantic_cache.lookup(
                query=query,
                current_doc_ids=eval_doc_ids,
                retrieved_chunks=eval_chunks
            )
            cache_status = cache_result["status"]
            cache_reason = cache_result["validation_reason"]

            if cache_status == CacheValidationStatus.HIT_VALID:
                cached_answer = cache_result["cached_answer"]

        cache_lookup_latency = time.perf_counter() - t_cache_start

        # --- PATH A: Cache Hit Valid (Bypass LLM Inference entirely) ---
        if cached_answer is not None:
            # Wall-clock elapsed time
            total_elapsed = time.perf_counter() - overall_start

            # Clear, non-overlapping stage breakdown
            stage_latencies = {
                "query_processing": max(0.0001, total_elapsed - retrieval_latency - cache_lookup_latency),
                "retrieval": retrieval_latency,
                "cache_lookup": cache_lookup_latency,
                "routing": 0.0,
                "model_inference": 0.0,  # Zero inference on cache hit!
            }

            # Energy and Carbon Accounting using Modeled TDP
            energy_telemetry = self.energy_tracker.track_stages(
                stage_latencies=stage_latencies,
                model_used="Cached Response (Zero LLM Inference)",
                is_cache_hit=True,
                retry_count=0
            )

            # Quality Evaluation
            eval_metrics = self.evaluator.evaluate(query, cached_answer, retrieved_chunks)

            # Baseline Calculation (Standard RAG baseline)
            baseline = self.energy_tracker.calculate_baseline(
                retrieval_latency=retrieval_latency,
                baseline_inference_latency=0.8,  # Standard capable model baseline assumption
                baseline_model=self.settings.models.capable_model
            )
            savings = self.energy_tracker.calculate_savings(
                baseline_energy_wh=baseline["baseline_energy_wh"],
                optimized_energy_wh=energy_telemetry["energy_wh"],
                baseline_emissions_g=baseline["baseline_emissions_g"],
                optimized_emissions_g=energy_telemetry["emissions_g"],
                is_simulated=False
            )

            response_data = {
                "query_id": query_id,
                "query_text": query,
                "answer": cached_answer,
                "mode": mode,
                "model_used": "Cached Response (Zero LLM Inference)",
                "cache_status": cache_status,
                "cache_validation_reason": cache_reason,
                "complexity_score": 0.0,
                "routing_reason": "Bypassed LLM inference via verified semantic cache match.",
                "retry_count": 0,
                "retrieved_chunks": retrieved_chunks,
                "sources": self._format_sources(retrieved_chunks, similarity_scores),
                "latency_seconds": round(total_elapsed, 4),
                "stage_latencies": {k_s: round(v_s, 5) for k_s, v_s in stage_latencies.items()},
                "energy_wh": energy_telemetry["energy_wh"],
                "energy_kwh": energy_telemetry["energy_kwh"],
                "emissions_g": energy_telemetry["emissions_g"],
                "emissions_kg": energy_telemetry["emissions_kg"],
                "grounding_score": eval_metrics["groundedness"],
                "quality_score": eval_metrics["composite_score"],
                "is_simulated": False,
                "measurement_label": energy_telemetry["measurement_label"],
                "measurement_mode": energy_telemetry["measurement_mode"],
                "energy_details": energy_telemetry["details"],
                "baseline_energy_wh": baseline["baseline_energy_wh"],
                "baseline_emissions_g": baseline["baseline_emissions_g"],
                "energy_savings_pct": savings["energy_savings_pct"],
                "carbon_savings_g": savings["carbon_savings_g"],
                "savings_details": savings,
            }
            response_data["decision_trace"] = self._build_decision_trace(
                cache_status=cache_status,
                cache_reason=cache_reason,
                retrieval_top_k=k,
                retrieved_chunk_count=len(retrieved_chunks),
                retrieval_reason=retrieval_reason,
                model_used=response_data["model_used"],
                complexity_score=0.0,
                routing_reason=response_data["routing_reason"],
                retry_count=0,
                energy_kwh=response_data["energy_kwh"],
                emissions_g=response_data["emissions_g"],
                latency_seconds=response_data["latency_seconds"],
                measurement_label=response_data["measurement_label"],
            )

            # Persist single complete log record
            self._log_execution(response_data)
            return response_data

        # --- PATH B: Cache Miss (Model Generation Required) ---
        complexity_score = 0.0
        routing_reason = "Capable model used by default."
        retry_count = 0
        routing_latency = 0.0

        if mode == ExecutionMode.GREENROUTING:
            # Measure routing decision + inference
            t_gen_start = time.perf_counter()
            router_result = self.router.route_and_execute(
                query=query,
                prompt=prompt,
                context_chunks=retrieved_chunks,
                system_prompt=RAG_SYSTEM_PROMPT
            )
            final_answer = router_result["answer"]
            model_used = router_result["final_model"]
            complexity_score = router_result["complexity_score"]
            routing_reason = router_result["routing_reason"]
            retry_count = router_result["retry_count"]
            inference_latency = router_result.get("latency_seconds", 0.0)
            routing_latency = max(0.0001, (time.perf_counter() - t_gen_start) - inference_latency)
            is_simulated = bool(router_result.get("is_simulated", False))
        else:
            # Standard RAG: Always use capable model
            model_used = self.settings.models.capable_model
            t_gen_start = time.perf_counter()
            gen_result = self.llm_client.generate(
                prompt=prompt,
                model_name=model_used,
                system_prompt=RAG_SYSTEM_PROMPT
            )
            final_answer = gen_result.get("answer", "")
            inference_latency = gen_result.get("latency_seconds", time.perf_counter() - t_gen_start)
            routing_latency = 0.0
            is_simulated = bool(gen_result.get("is_simulated", False))

        # Check simulation status if forced or disconnected
        if not self.llm_client.is_connected or getattr(self.settings.models, "force_simulation_mode", False):
            is_simulated = True

        # Total elapsed wall-clock time (strictly measured, no double counting!)
        total_elapsed = time.perf_counter() - overall_start

        # Stage latencies breakdown
        other_latency = max(0.0001, total_elapsed - (retrieval_latency + cache_lookup_latency + routing_latency + inference_latency))
        stage_latencies = {
            "query_processing": other_latency,
            "retrieval": retrieval_latency,
            "cache_lookup": cache_lookup_latency,
            "routing": routing_latency,
            "model_inference": inference_latency,
        }

        # Energy & Emissions tracking using Modeled TDP Estimate
        energy_telemetry = self.energy_tracker.track_stages(
            stage_latencies=stage_latencies,
            model_used=model_used,
            is_cache_hit=False,
            retry_count=retry_count
        )

        # Quality Evaluation
        eval_metrics = self.evaluator.evaluate(query, final_answer, retrieved_chunks)
        grounding_score = eval_metrics["groundedness"]
        quality_score = eval_metrics["composite_score"]

        # Baseline Calculation & Savings
        if mode == ExecutionMode.STANDARD_RAG:
            baseline_energy_wh = energy_telemetry["energy_wh"]
            baseline_emissions_g = energy_telemetry["emissions_g"]
            savings = {
                "energy_savings_wh": 0.0,
                "energy_savings_pct": 0.0,
                "carbon_savings_g": 0.0,
                "is_comparable": True,
                "is_simulated": is_simulated,
                "note": "Standard RAG represents the baseline path."
            }
        else:
            # Baseline represents non-optimized Standard RAG (capable model, no cache)
            baseline_inf_lat = inference_latency if model_used == self.settings.models.capable_model else inference_latency * 1.8
            baseline = self.energy_tracker.calculate_baseline(
                retrieval_latency=retrieval_latency,
                baseline_inference_latency=baseline_inf_lat,
                baseline_model=self.settings.models.capable_model
            )
            baseline_energy_wh = baseline["baseline_energy_wh"]
            baseline_emissions_g = baseline["baseline_emissions_g"]
            savings = self.energy_tracker.calculate_savings(
                baseline_energy_wh=baseline_energy_wh,
                optimized_energy_wh=energy_telemetry["energy_wh"],
                baseline_emissions_g=baseline_emissions_g,
                optimized_emissions_g=energy_telemetry["emissions_g"],
                is_simulated=is_simulated
            )

        # Store in Semantic Cache if caching is enabled
        if mode in [ExecutionMode.RAG_WITH_CACHE, ExecutionMode.GREENROUTING] and final_answer:
            self.semantic_cache.store(
                query=query,
                answer=final_answer,
                source_doc_ids=active_doc_ids,
                source_chunks=retrieved_chunks,
                model_used=model_used,
                grounding_score=grounding_score
            )

        response_data = {
            "query_id": query_id,
            "query_text": query,
            "answer": final_answer,
            "mode": mode,
            "model_used": model_used,
            "cache_status": cache_status,
            "cache_validation_reason": cache_reason,
            "complexity_score": complexity_score,
            "routing_reason": routing_reason,
            "retry_count": retry_count,
            "retrieved_chunks": retrieved_chunks,
            "sources": self._format_sources(retrieved_chunks, similarity_scores),
            "latency_seconds": round(total_elapsed, 4),
            "stage_latencies": {k_s: round(v_s, 5) for k_s, v_s in stage_latencies.items()},
            "energy_wh": energy_telemetry["energy_wh"],
            "energy_kwh": energy_telemetry["energy_kwh"],
            "emissions_g": energy_telemetry["emissions_g"],
            "emissions_kg": energy_telemetry["emissions_kg"],
            "grounding_score": grounding_score,
            "quality_score": quality_score,
            "is_simulated": is_simulated,
            "measurement_label": energy_telemetry["measurement_label"],
            "measurement_mode": energy_telemetry["measurement_mode"],
            "energy_details": energy_telemetry["details"],
            "baseline_energy_wh": baseline_energy_wh,
            "baseline_emissions_g": baseline_emissions_g,
            "energy_savings_pct": savings["energy_savings_pct"],
            "carbon_savings_g": savings["carbon_savings_g"],
            "savings_details": savings,
        }
        response_data["decision_trace"] = self._build_decision_trace(
            cache_status=cache_status,
            cache_reason=cache_reason,
            retrieval_top_k=k,
            retrieved_chunk_count=len(retrieved_chunks),
            retrieval_reason=retrieval_reason,
            model_used=model_used,
            complexity_score=complexity_score,
            routing_reason=routing_reason,
            retry_count=retry_count,
            energy_kwh=energy_telemetry["energy_kwh"],
            emissions_g=energy_telemetry["emissions_g"],
            latency_seconds=total_elapsed,
            measurement_label=energy_telemetry["measurement_label"],
        )

        # Persist single complete log record
        self._log_execution(response_data)
        return response_data

    def _build_decision_trace(
        self,
        *,
        cache_status: str,
        cache_reason: str,
        retrieval_top_k: int,
        retrieved_chunk_count: int,
        retrieval_reason: str,
        model_used: str,
        complexity_score: float,
        routing_reason: str,
        retry_count: int,
        energy_kwh: float,
        emissions_g: float,
        latency_seconds: float,
        measurement_label: str,
    ) -> Dict[str, Any]:
        """Build the stable, display-ready explanation contract."""
        return {
            "cache": {
                "status": str(cache_status),
                "reason": cache_reason,
            },
            "retrieval": {
                "top_k": retrieval_top_k,
                "chunk_count": retrieved_chunk_count,
                "reason": retrieval_reason,
            },
            "routing": {
                "model": model_used,
                "complexity_score": complexity_score,
                "reason": routing_reason,
                "retry_count": retry_count,
            },
            "impact": {
                "energy_kwh": energy_kwh,
                "emissions_g": emissions_g,
                "latency_seconds": latency_seconds,
                "measurement_label": measurement_label,
            },
        }

    def _format_sources(self, chunks: List[Dict[str, Any]], scores: List[float]) -> List[Dict[str, Any]]:
        sources = []
        for i, (chunk, score) in enumerate(zip(chunks, scores), start=1):
            sources.append({
                "source_index": i,
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "page_number": chunk.get("page_number", 1),
                "similarity_score": round(score, 4),
                "text_snippet": chunk.get("text_content", "")[:250] + "..." if len(chunk.get("text_content", "")) > 250 else chunk.get("text_content", "")
            })
        return sources

    def _log_execution(self, data: Dict[str, Any]):
        """Persist a single execution record to SQLite database."""
        try:
            self.db.log_query({
                "query_id": data.get("query_id"),
                "query_text": data.get("query_text", ""),
                "answer_text": data.get("answer", "")[:1000],
                "mode": data.get("mode", ""),
                "model_used": data.get("model_used", ""),
                "cache_status": str(data.get("cache_status", "")),
                "cache_validation_reason": data.get("cache_validation_reason", ""),
                "complexity_score": data.get("complexity_score", 0.0),
                "routing_reason": data.get("routing_reason", ""),
                "retry_count": data.get("retry_count", 0),
                "grounding_score": data.get("grounding_score", 0.0),
                "quality_score": data.get("quality_score", 0.0),
                "latency_seconds": data.get("latency_seconds", 0.0),
                "stage_latencies": data.get("stage_latencies", {}),
                "energy_wh": data.get("energy_wh", 0.0),
                "energy_kwh": data.get("energy_kwh", 0.0),
                "emissions_g": data.get("emissions_g", 0.0),
                "emissions_kg": data.get("emissions_kg", 0.0),
                "energy_source": data.get("measurement_mode", "modeled_tdp_estimate"),
                "simulation_mode": 1 if data.get("is_simulated") else 0,
                "source_chunks": data.get("sources", []),
                "baseline_energy_wh": data.get("baseline_energy_wh", 0.0),
                "baseline_emissions_g": data.get("baseline_emissions_g", 0.0),
                "energy_savings_pct": data.get("energy_savings_pct", 0.0),
                "carbon_savings_g": data.get("carbon_savings_g", 0.0),
            })
        except Exception as e:
            logger.warning("Could not log query execution: %s", e)
