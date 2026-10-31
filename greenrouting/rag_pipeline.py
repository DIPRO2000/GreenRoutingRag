"""
Basic RAG Pipeline for GreenRouting.
Retrieves context chunks from FAISS and synthesizes grounded answers with source citations.
"""
import logging
from typing import List, Dict, Any, Optional
from greenrouting.vector_store import FaissVectorStore
from greenrouting.llm_interface import OllamaClient
from greenrouting.config import RagConfig, ModelConfig

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are GreenRouting Assistant, an energy-efficient and accurate document question-answering assistant.
Use ONLY the provided Context chunks to answer the Question.
If the answer cannot be found in the context, explicitly state: "The provided context does not contain enough information to answer this question."
Do not speculate or extrapolate beyond what is clearly stated in the context."""


class RagPipeline:
    """Standard context-retrieval and question-answering pipeline."""

    def __init__(self, vector_store: FaissVectorStore, llm_client: OllamaClient,
                 rag_config: RagConfig, model_config: ModelConfig):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.rag_config = rag_config
        self.model_config = model_config

    def build_prompt(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Construct a structured prompt incorporating retrieved chunks with source markers."""
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            chunk_text = chunk.get("text_content", "").strip()
            page = chunk.get("page_number", 1)
            chunk_id = chunk.get("chunk_id", f"c{i}")
            context_blocks.append(f"[Source {i} | Chunk: {chunk_id} | Page: {page}]:\n{chunk_text}")

        context_str = "\n\n".join(context_blocks)
        prompt = f"Context:\n{context_str}\n\nQuestion:\n{query}\n\nAnswer:"
        return prompt

    def answer_query(self, query: str, model_name: Optional[str] = None,
                     doc_id_filter: Optional[str] = None, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute standard RAG answering:
        1. Retrieve top-k chunks from vector store.
        2. Format context prompt.
        3. Call LLM for grounded answer.
        4. Return response with detailed source chunk citations.
        """
        k = top_k or self.rag_config.top_k_retrieval
        selected_model = model_name or self.model_config.capable_model

        # 1. Retrieve chunks
        search_results = self.vector_store.search(query, top_k=k, doc_id_filter=doc_id_filter)
        
        if not search_results:
            return {
                "answer": "No relevant context found in uploaded documents. Please ensure documents are uploaded and indexed.",
                "retrieved_chunks": [],
                "sources": [],
                "model_used": selected_model,
                "is_simulated": False,
                "latency_seconds": 0.01,
                "prompt_eval_count": 0,
                "eval_count": 0
            }

        retrieved_chunks = [item[0] for item in search_results]
        similarity_scores = [item[1] for item in search_results]

        # 2. Build structured prompt
        prompt = self.build_prompt(query, retrieved_chunks)

        # 3. Call LLM
        llm_response = self.llm_client.generate(
            prompt=prompt,
            model_name=selected_model,
            system_prompt=RAG_SYSTEM_PROMPT
        )

        # 4. Prepare source references for citation display
        sources = []
        for i, (chunk, score) in enumerate(zip(retrieved_chunks, similarity_scores), start=1):
            sources.append({
                "source_index": i,
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "page_number": chunk.get("page_number", 1),
                "similarity_score": round(score, 4),
                "text_snippet": chunk.get("text_content", "")[:200] + "..." if len(chunk.get("text_content", "")) > 200 else chunk.get("text_content", "")
            })

        return {
            "answer": llm_response.get("answer", ""),
            "retrieved_chunks": retrieved_chunks,
            "sources": sources,
            "model_used": llm_response.get("model_used", selected_model),
            "is_simulated": llm_response.get("is_simulated", False),
            "latency_seconds": llm_response.get("latency_seconds", 0.0),
            "eval_count": llm_response.get("eval_count", 0),
            "prompt_eval_count": llm_response.get("prompt_eval_count", 0)
        }
