"""
Unit tests for VectorStore and EmbeddingService in GreenRouting.
"""
import pytest
import shutil
from pathlib import Path
from greenrouting.vector_store import EmbeddingService, FaissVectorStore


@pytest.fixture
def temp_index_dir(tmp_path):
    index_dir = tmp_path / "faiss_test"
    yield index_dir
    if index_dir.exists():
        shutil.rmtree(index_dir, ignore_errors=True)


def test_embedding_and_faiss_lifecycle(temp_index_dir):
    embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
    vector_store = FaissVectorStore(index_dir=temp_index_dir, embedding_service=embedding_service)

    chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "chunk_index": 0,
            "text_content": "Carbon emissions from large language models are a growing environmental concern.",
            "page_number": 1,
            "character_count": 78
        },
        {
            "chunk_id": "c2",
            "doc_id": "d1",
            "chunk_index": 1,
            "text_content": "Solar and wind energy are renewable power resources with minimal carbon footprint.",
            "page_number": 1,
            "character_count": 82
        }
    ]

    added = vector_store.add_chunks(chunks)
    assert added == 2
    assert vector_store.get_total_chunks() == 2

    # Query search
    results = vector_store.search("What are the environmental impacts of LLMs?", top_k=1)
    assert len(results) == 1
    top_chunk, score = results[0]
    assert top_chunk["chunk_id"] == "c1"
    assert -1.0 <= score <= 1.0

    # Persistence verification: reload vector store
    reloaded_store = FaissVectorStore(index_dir=temp_index_dir, embedding_service=embedding_service)
    assert reloaded_store.get_total_chunks() == 2
    reloaded_results = reloaded_store.search("renewable energy solar", top_k=1)
    assert len(reloaded_results) == 1
    assert reloaded_results[0][0]["chunk_id"] == "c2"
