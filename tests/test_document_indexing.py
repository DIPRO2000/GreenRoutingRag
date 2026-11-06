"""Tests for document indexing and duplicate protection."""

from greenrouting.config import load_settings
from greenrouting.orchestrator import GreenRoutingOrchestrator


def test_duplicate_document_upload_is_skipped(tmp_path):
    """The same document bytes must not be indexed twice."""
    settings = load_settings()
    settings.app.db_path = tmp_path / "documents.db"
    settings.app.faiss_index_dir = tmp_path / "faiss"
    orchestrator = GreenRoutingOrchestrator(settings)
    content = b"GreenRouting stores document chunks for grounded retrieval."

    first = orchestrator.process_and_index_document("first.txt", content)
    second = orchestrator.process_and_index_document("renamed-copy.txt", content)

    assert first["duplicate"] is False
    assert second["duplicate"] is True
    assert second["doc_id"] == first["doc_id"]
    assert len(orchestrator.db.get_all_documents()) == 1
    assert orchestrator.vector_store.get_total_chunks() == first["total_chunks"]
