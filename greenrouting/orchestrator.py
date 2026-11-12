"""
GreenRouting Orchestrator.
Unifies document indexing, retrieval, adaptive routing, quality-aware caching,
energy tracking, and persistent telemetry logging into a single cohesive pipeline.
"""
import time
import logging
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
from greenrouting.document_summarizer import DocumentSummarizer

logger = logging.getLogger(__name__)


class ExecutionMode:
    STANDARD_RAG = "STANDARD_RAG"          # Capable model only, no cache
    RAG_WITH_CACHE = "RAG_WITH_CACHE"      # Capable model with standard semantic cache
    GREENROUTING = "GREENROUTING"                  # Quality-aware cache + adaptive green routing + retry


class GreenRoutingOrchestrator:
    """Master orchestrator for the GreenRouting energy-aware RAG framework."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.db = DatabaseManager(settings.app.db_path)
        self.embedding_service = EmbeddingService(settings.models.embedding_model)
        self.vector_store = FaissVectorStore(settings.app.faiss_index_dir, self.embedding_service)
        self.llm_client = OllamaClient(
            base_url=settings.models.ollama_base_url,
            enable_simulation_fallback=settings.models.enable_simulation_fallback
        )
        self.document_summarizer = DocumentSummarizer(
            llm_client=self.llm_client,
            model_name=settings.models.capable_model,
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
                "page_count": doc_meta.get("page_count"),
                "low_text_pages": doc_meta.get("low_text_pages", []),
                "duplicate": True,
            }

        try:
            self.db.save_document(
                doc_id=doc_meta["doc_id"],
                filename=doc_meta["filename"],
                doc_hash=doc_meta["doc_hash"],
                file_type=doc_meta["file_type"],
                total_chunks=doc_meta["total_chunks"],
                total_characters=doc_meta["total_characters"]
            )
            self.db.save_chunks(chunks)
            self.vector_store.add_chunks(chunks)
        except Exception as exc:
            try:
                self.db.delete_document(doc_meta["doc_id"])
                self.vector_store.rebuild(self.db.get_all_chunks())
            except Exception as rollback_error:
                logger.exception("Failed to roll back document '%s' after indexing failed.", filename)
                raise RuntimeError(
                    f"Indexing '{filename}' failed and its rollback also failed: {rollback_error}"
                ) from exc
            raise

        return {
            "doc_id": doc_meta["doc_id"],
            "filename": doc_meta["filename"],
            "total_chunks": doc_meta["total_chunks"],
            "total_characters": doc_meta["total_characters"],
            "page_count": doc_meta.get("page_count"),
            "low_text_pages": doc_meta.get("low_text_pages", []),
            "duplicate": False,
        }

    def summarize_document(
        self,
        doc_id: str,
        query_text: str = "Summarize this document.",
        mode: str = ExecutionMode.GREENROUTING,
    ) -> Dict[str, Any]:
        """Summarize every stored chunk belonging to the selected document."""
        chunks = self.db.get_document_chunks(doc_id)
        summary_result = self.document_summarizer.summarize(chunks)
        answer = summary_result["summary"]
        energy = self.energy_tracker.track_operation(
            operation_name="document_wide_summary",
            latency_seconds=summary_result["latency_seconds"],
            generation_runs=summary_result["generation_runs"],
        )
        quality = self.evaluator.evaluate(query_text, answer, chunks)
        model_used = summary_result["model_used"]
        sources = [
            {
                "source_index": index,
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "page_number": chunk.get("page_number", 1),
                "text_snippet": (
                    chunk.get("text_content", "")[:250] + "..."
                    if len(chunk.get("text_content", "")) > 250
                    else chunk.get("text_content", "")
                ),
            }
            for index, chunk in enumerate(chunks, start=1)
        ]
        summary_pages = sorted({source["page_number"] for source in sources if source["page_number"] is not None})
        routing_reason = (
            "Whole-document summaries use the configured capable model for each section and combine pass. "
            "Cache reuse and adaptive routing are not applied to this summary path."
        )
        response = {
            "answer": answer,
            "query_text": query_text,
            "mode": mode,
            "model_used": model_used,
            "is_simulated": summary_result["is_simulated"],
            "is_document_summary": True,
            "summary_call_count": summary_result["call_count"],
            "cache_status": "BYPASSED",
            "cache_validation_reason": "Document-wide summaries are generated from all chunks and are not cached.",
            "complexity_score": 0.0,
            "routing_reason": routing_reason,
            "retry_count": 0,
            "retrieval_top_k": len(chunks),
            "retrieved_chunk_count": len(chunks),
            "retrieval_reason": f"Processed all {len(chunks)} indexed chunks in bounded batches and combined their summaries.",
            "retrieved_chunks": chunks,
            "sources": sources,
            "summary_pages": summary_pages,
            "latency_seconds": energy["latency_seconds"],
            "energy_kwh": energy["energy_kwh"],
            "emissions_kg": energy["emissions_kg"],
            "emissions_g": energy["emissions_g"],
            "grounding_score": quality["groundedness"],
            "quality_score": quality["composite_score"],
            "measurement_label": energy["measurement_label"],
            "energy_details": energy["details"],
            "generation_details": energy.get("generation_details", []),
        }
        response["decision_trace"] = self._build_decision_trace(
            cache_status=response["cache_status"],
            cache_reason=response["cache_validation_reason"],
            retrieval_top_k=len(chunks),
            retrieved_chunk_count=len(chunks),
            retrieval_reason=response["retrieval_reason"],
            model_used=model_used,
            complexity_score=0.0,
            routing_reason=routing_reason,
            retry_count=0,
            energy_kwh=energy["energy_kwh"],
            emissions_g=energy["emissions_g"],
            latency_seconds=energy["latency_seconds"],
            measurement_label=energy["measurement_label"],
        )
        self._log_execution(response)
        return response

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
        remaining_chunks = [chunk for chunk in previous_chunks if chunk.get("doc_id") != doc_id]
        # Build the new index before committing the document deletion. If embedding
        # or index construction fails, the document remains available in SQLite.
        self.vector_store.rebuild(remaining_chunks)
        try:
            deletion = self.db.delete_document(doc_id)
            self.semantic_cache.storage.invalidate_document(doc_id)
        except Exception:
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
        1. Context Retrieval from vector store.
        2. Semantic Cache lookup (if mode supports caching).
        3. Adaptive Model Routing or Capable Model generation.
        4. Energy & Emissions tracking.
        5. Quality & Grounding evaluation.
        6. Persistent query logging in SQLite.
        """
        start_time = time.perf_counter()
        active_doc_ids = [active_doc_id] if active_doc_id else [d["doc_id"] for d in self.db.get_all_documents()]

        all_chunks = self.db.get_all_chunks()
        available_chunks = sum(
            1 for chunk in all_chunks
            if active_doc_id is None or chunk.get("doc_id") == active_doc_id
        )
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

        # 1. Retrieve Context Chunks
        search_results = self.vector_store.search(query, top_k=k, doc_id_filter=active_doc_id)
        retrieved_chunks = [item[0] for item in search_results]
        similarity_scores = [item[1] for item in search_results]

        if not retrieved_chunks:
            return {
                "answer": "No relevant documents found. Please upload and index documents first.",
                "mode": mode,
                "model_used": "None",
                "is_simulated": False,
                "cache_status": CacheValidationStatus.MISS,
                "cache_validation_reason": "No documents indexed.",
                "complexity_score": 0.0,
                "routing_reason": "N/A",
                "retry_count": 0,
                "retrieval_top_k": k,
                "retrieved_chunk_count": 0,
                "retrieval_reason": retrieval_reason,
                "retrieval_complexity_score": retrieval_complexity_score,
                "retrieved_chunks": [],
                "sources": [],
                "latency_seconds": 0.001,
                "energy_kwh": 0.0,
                "emissions_g": 0.0,
                "grounding_score": 0.0,
                "measurement_label": "N/A",
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
                    latency_seconds=0.001,
                    measurement_label="N/A",
                )
            }

        prompt = self.rag_pipeline.build_prompt(query, retrieved_chunks)

        # 2. Semantic Cache Lookup
        cache_status = CacheValidationStatus.MISS
        cache_reason = "Caching not enabled for this mode."
        cached_answer = None

        if mode in [ExecutionMode.RAG_WITH_CACHE, ExecutionMode.GREENROUTING]:
            cache_result = self.semantic_cache.lookup(
                query=query,
                current_doc_ids=active_doc_ids,
                retrieved_chunks=retrieved_chunks
            )
            cache_status = cache_result["status"]
            cache_reason = cache_result["validation_reason"]

            if cache_status == CacheValidationStatus.HIT_VALID:
                cached_answer = cache_result["cached_answer"]

        # If cache hit valid, bypass LLM entirely
        if cached_answer is not None:
            elapsed = time.perf_counter() - start_time
            energy_telemetry = self.energy_tracker.track_operation(
                operation_name="cache_hit_retrieval",
                latency_seconds=elapsed,
                is_cache_hit=True
            )
            eval_metrics = self.evaluator.evaluate(query, cached_answer, retrieved_chunks)

            response_data = {
                "answer": cached_answer,
                "query_text": query,
                "mode": mode,
                "model_used": "Cached Response (Zero LLM Inference)",
                "is_simulated": False,
                "cache_status": cache_status,
                "cache_validation_reason": cache_reason,
                "complexity_score": 0.0,
                "routing_reason": "Bypassed LLM inference via verified semantic cache match.",
                "retry_count": 0,
                "retrieval_top_k": k,
                "retrieved_chunk_count": len(retrieved_chunks),
                "retrieval_reason": retrieval_reason,
                "retrieval_complexity_score": retrieval_complexity_score,
                "retrieved_chunks": retrieved_chunks,
                "sources": self._format_sources(retrieved_chunks, similarity_scores),
                "latency_seconds": elapsed,
                "energy_kwh": energy_telemetry["energy_kwh"],
                "emissions_kg": energy_telemetry["emissions_kg"],
                "emissions_g": energy_telemetry["emissions_g"],
                "grounding_score": eval_metrics["groundedness"],
                "quality_score": eval_metrics["composite_score"],
                "measurement_label": energy_telemetry["measurement_label"],
                "energy_details": energy_telemetry["details"],
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
            self._log_execution(response_data)
            return response_data

        # 3. LLM Generation
        complexity_score = 0.0
        routing_reason = "Capable model used by default."
        retry_count = 0

        if mode == ExecutionMode.GREENROUTING:
            # Adaptive Green Router
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
            gen_latency = router_result["latency_seconds"]
            generation_runs = router_result.get("generation_runs", [])
            is_simulated = bool(router_result.get("is_simulated", False))
        else:
            # Standard RAG: Always use capable model
            model_used = self.settings.models.capable_model
            gen_result = self.llm_client.generate(
                prompt=prompt,
                model_name=model_used,
                system_prompt=RAG_SYSTEM_PROMPT
            )
            final_answer = gen_result.get("answer", "")
            is_simulated = bool(gen_result.get("is_simulated", False))
            gen_latency = gen_result.get("latency_seconds", 0.0)
            generation_runs = [{
                "model_name": model_used,
                "latency_seconds": gen_latency,
                "eval_count": gen_result.get("eval_count", 0)
            }]
            retry_count = 0

        # Total end-to-end query latency
        total_elapsed = time.perf_counter() - start_time


        # 4. Energy and Carbon Tracking
        energy_telemetry = self.energy_tracker.track_operation(
            operation_name=f"query_execution_{mode}",
            latency_seconds=gen_latency,
            model_name=model_used,
            is_cache_hit=False,
            retry_count=retry_count,
            generation_runs=generation_runs or None
        )

        # 5. Quality Evaluation
        eval_metrics = self.evaluator.evaluate(query, final_answer, retrieved_chunks)
        grounding_score = eval_metrics["groundedness"]
        quality_score = eval_metrics["composite_score"]

        # 6. Store in Semantic Cache if caching is enabled
        if mode in [ExecutionMode.RAG_WITH_CACHE, ExecutionMode.GREENROUTING] and final_answer:
            self.semantic_cache.store(
                query=query,
                answer=final_answer,
                source_doc_ids=sorted({
                    chunk.get("doc_id") for chunk in retrieved_chunks
                    if chunk.get("doc_id")
                }),
                source_chunks=retrieved_chunks,
                model_used=model_used,
                grounding_score=grounding_score
            )

        response_data = {
            "answer": final_answer,
            "query_text": query,
            "mode": mode,
            "model_used": model_used,
            "is_simulated": is_simulated,
            "cache_status": cache_status,
            "cache_validation_reason": cache_reason,
            "complexity_score": complexity_score,
            "routing_reason": routing_reason,
            "retry_count": retry_count,
            "retrieval_top_k": k,
            "retrieved_chunk_count": len(retrieved_chunks),
            "retrieval_reason": retrieval_reason,
            "retrieval_complexity_score": retrieval_complexity_score,
            "retrieved_chunks": retrieved_chunks,
            "sources": self._format_sources(retrieved_chunks, similarity_scores),
            "latency_seconds": round(total_elapsed, 4),
            "energy_kwh": energy_telemetry["energy_kwh"],
            "emissions_kg": energy_telemetry["emissions_kg"],
            "emissions_g": energy_telemetry["emissions_g"],
            "grounding_score": grounding_score,
            "quality_score": quality_score,
            "measurement_label": energy_telemetry["measurement_label"],
            "energy_details": energy_telemetry["details"],
            "generation_details": energy_telemetry.get("generation_details", []),
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
        try:
            self.db.log_query({
                "query_text": data.get("query_text", ""),
                "answer_text": data.get("answer", "")[:500],
                "mode": data.get("mode", ""),
                "model_used": data.get("model_used", ""),
                "cache_status": data.get("cache_status", ""),
                "cache_validation_reason": data.get("cache_validation_reason", ""),
                "complexity_score": data.get("complexity_score", 0.0),
                "routing_reason": data.get("routing_reason", ""),
                "retry_count": data.get("retry_count", 0),
                "grounding_score": data.get("grounding_score", 0.0),
                "latency_seconds": data.get("latency_seconds", 0.0),
                "energy_kwh": data.get("energy_kwh", 0.0),
                "emissions_kg": data.get("emissions_kg", 0.0),
                "energy_source": data.get("measurement_label", "MODELED")
            })
        except Exception as e:
            logger.warning("Could not log query execution: %s", e)
