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
        k = top_k or self.settings.rag.top_k_retrieval
        active_doc_ids = [active_doc_id] if active_doc_id else [d["doc_id"] for d in self.db.get_all_documents()]

        # 1. Retrieve Context Chunks
        search_results = self.vector_store.search(query, top_k=k, doc_id_filter=active_doc_id)
        retrieved_chunks = [item[0] for item in search_results]
        similarity_scores = [item[1] for item in search_results]

        if not retrieved_chunks:
            return {
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
                "latency_seconds": 0.001,
                "energy_kwh": 0.0,
                "emissions_g": 0.0,
                "grounding_score": 0.0,
                "measurement_label": "N/A"
            }

        prompt = self.rag_pipeline.build_prompt(query, retrieved_chunks)

        # 2. Semantic Cache Lookup
        cache_status = CacheValidationStatus.MISS
        cache_reason = "Caching not enabled for this mode."
        cached_answer = None

        if mode in [ExecutionMode.RAG_WITH_CACHE, ExecutionMode.GREENROUTING]:
            # For standard RAG with cache, pass no doc_ids to simulate naive caching
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
                "mode": mode,
                "model_used": "Cached Response (Zero LLM Inference)",
                "cache_status": cache_status,
                "cache_validation_reason": cache_reason,
                "complexity_score": 0.0,
                "routing_reason": "Bypassed LLM inference via verified semantic cache match.",
                "retry_count": 0,
                "retrieved_chunks": retrieved_chunks,
                "sources": self._format_sources(retrieved_chunks, similarity_scores),
                "latency_seconds": elapsed,
                "energy_kwh": energy_telemetry["energy_kwh"],
                "emissions_kg": energy_telemetry["emissions_kg"],
                "emissions_g": energy_telemetry["emissions_g"],
                "grounding_score": eval_metrics["groundedness"],
                "quality_score": eval_metrics["composite_score"],
                "measurement_label": energy_telemetry["measurement_label"],
                "energy_details": energy_telemetry["details"]
            }
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
        else:
            # Standard RAG: Always use capable model
            model_used = self.settings.models.capable_model
            gen_result = self.llm_client.generate(
                prompt=prompt,
                model_name=model_used,
                system_prompt=RAG_SYSTEM_PROMPT
            )
            final_answer = gen_result.get("answer", "")
            gen_latency = gen_result.get("latency_seconds", 0.0)

        total_elapsed = (time.perf_counter() - start_time) + gen_latency

        # 4. Energy and Carbon Tracking
        energy_telemetry = self.energy_tracker.track_operation(
            operation_name=f"query_execution_{mode}",
            latency_seconds=total_elapsed,
            model_name=model_used,
            is_cache_hit=False,
            retry_count=retry_count
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
                source_doc_ids=active_doc_ids,
                source_chunks=retrieved_chunks,
                model_used=model_used,
                grounding_score=grounding_score
            )

        response_data = {
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
            "energy_kwh": energy_telemetry["energy_kwh"],
            "emissions_kg": energy_telemetry["emissions_kg"],
            "emissions_g": energy_telemetry["emissions_g"],
            "grounding_score": grounding_score,
            "quality_score": quality_score,
            "measurement_label": energy_telemetry["measurement_label"],
            "energy_details": energy_telemetry["details"]
        }

        self._log_execution(response_data)
        return response_data

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
