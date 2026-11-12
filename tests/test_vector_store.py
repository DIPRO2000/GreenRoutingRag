"""
Unit tests for VectorStore and EmbeddingService in GreenRouting.
"""
import pytest
import shutil
import os
import subprocess
import sys
import numpy as np
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


def test_fallback_embeddings_are_stable_across_python_processes():
    script = (
        "import json; from greenrouting.vector_store import DeterministicFallbackEmbedder; "
        "print(json.dumps(DeterministicFallbackEmbedder().encode(['green energy']).tolist()))"
    )
    outputs = []
    for seed in ("1", "98765"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        outputs.append(subprocess.check_output([sys.executable, "-c", script], env=env, text=True))

    assert outputs[0] == outputs[1]


def test_faiss_scoped_search_expands_until_it_finds_requested_count(temp_index_dir):
    store = FaissVectorStore(temp_index_dir, EmbeddingService())
    chunks = [
        {"chunk_id": f"c{i}", "doc_id": "wanted" if i == 5 else "other", "text_content": f"text {i}"}
        for i in range(10)
    ]
    store.add_chunks(chunks)

    class RankedIndex:
        ntotal = len(chunks)

        def __init__(self):
            self.requested = []

        def search(self, _query, k):
            self.requested.append(k)
            indices = np.arange(k, dtype=np.int64).reshape(1, -1)
            scores = np.linspace(1.0, 0.1, k, dtype=np.float32).reshape(1, -1)
            return scores, indices

    index = RankedIndex()
    store.use_faiss = True
    store.faiss_index = index

    results = store.search("query", top_k=1, doc_id_filter="wanted")

    assert len(results) == 1
    assert results[0][0]["chunk_id"] == "c5"
    assert index.requested[-1] >= 6
