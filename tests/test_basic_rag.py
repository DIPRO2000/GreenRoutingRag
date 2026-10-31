"""
Unit tests for Basic RAG Pipeline in GreenRouting.
"""
import pytest
from pathlib import Path
from greenrouting.config import RagConfig, ModelConfig
from greenrouting.document_processor import DocumentProcessor
from greenrouting.vector_store import EmbeddingService, FaissVectorStore
from greenrouting.llm_interface import OllamaClient
from greenrouting.rag_pipeline import RagPipeline


def test_basic_rag_flow(tmp_path):
    index_dir = tmp_path / "rag_faiss"
    embedding_service = EmbeddingService()
    vector_store = FaissVectorStore(index_dir=index_dir, embedding_service=embedding_service)
    llm_client = OllamaClient(enable_simulation_fallback=True)

    rag_config = RagConfig(top_k_retrieval=2)
    model_config = ModelConfig(capable_model="llama3.2:3b")

    pipeline = RagPipeline(
        vector_store=vector_store,
        llm_client=llm_client,
        rag_config=rag_config,
        model_config=model_config
    )

    processor = DocumentProcessor(chunk_size=200, chunk_overlap=30)
    sample_text = (
        "Green AI focuses on developing energy-efficient artificial intelligence algorithms. "
        "GreenRouting achieves energy efficiency by applying adaptive routing and quality-aware caching. "
        "This substantially reduces kilowatt-hour consumption during inference."
    )

    _, chunks = processor.process_document("green_ai.txt", sample_text.encode("utf-8"), doc_id="doc_green")
    vector_store.add_chunks(chunks)

    # Ask question
    response = pipeline.answer_query("How does GreenRouting achieve energy efficiency?")

    assert "answer" in response
    assert len(response["retrieved_chunks"]) > 0
    assert len(response["sources"]) > 0
    assert response["sources"][0]["chunk_id"].startswith("doc_green")
    assert response["latency_seconds"] > 0.0
