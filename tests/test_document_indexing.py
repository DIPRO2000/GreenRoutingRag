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


def _orchestrator_with_documents(tmp_path):
    settings = load_settings()
    settings.app.db_path = tmp_path / "documents.db"
    settings.app.faiss_index_dir = tmp_path / "faiss"
    orchestrator = GreenRoutingOrchestrator(settings)
    first = orchestrator.process_and_index_document(
        "first.txt", b"First document contains solar energy facts."
    )
    second = orchestrator.process_and_index_document(
        "second.txt", b"Second document contains wind energy facts."
    )
    return orchestrator, first, second


def test_delete_document_rebuilds_index_without_deleted_chunks(tmp_path):
    orchestrator, first, second = _orchestrator_with_documents(tmp_path)

    result = orchestrator.delete_document(first["doc_id"])

    assert result["deleted"] is True
    assert result["filename"] == "first.txt"
    assert result["removed_chunk_count"] == first["total_chunks"]
    assert [doc["doc_id"] for doc in orchestrator.db.get_all_documents()] == [second["doc_id"]]
    assert orchestrator.vector_store.get_total_chunks() == second["total_chunks"]
    assert all(chunk["doc_id"] == second["doc_id"] for chunk in orchestrator.vector_store.chunks)


def test_delete_last_document_leaves_empty_index(tmp_path):
    orchestrator, first, _ = _orchestrator_with_documents(tmp_path)

    second_id = [doc["doc_id"] for doc in orchestrator.db.get_all_documents() if doc["doc_id"] != first["doc_id"]][0]
    orchestrator.delete_document(first["doc_id"])
    result = orchestrator.delete_document(second_id)

    assert result["deleted"] is True
    assert orchestrator.db.get_all_documents() == []
    assert orchestrator.vector_store.get_total_chunks() == 0
    assert orchestrator.vector_store.search("anything") == []


def test_delete_unknown_document_does_not_change_index(tmp_path):
    orchestrator, _, _ = _orchestrator_with_documents(tmp_path)
    before_docs = orchestrator.db.get_all_documents()
    before_chunks = orchestrator.vector_store.get_total_chunks()

    result = orchestrator.delete_document("missing-doc-id")

    assert result["deleted"] is False
    assert orchestrator.db.get_all_documents() == before_docs
    assert orchestrator.vector_store.get_total_chunks() == before_chunks


def test_delete_keeps_database_document_when_index_rebuild_fails(tmp_path, monkeypatch):
    orchestrator, first, second = _orchestrator_with_documents(tmp_path)
    original_chunks = orchestrator.db.get_all_chunks()

    def fail_rebuild(_chunks):
        raise RuntimeError("test rebuild failure")

    monkeypatch.setattr(orchestrator.vector_store, "rebuild", fail_rebuild)
    try:
        orchestrator.delete_document(first["doc_id"])
    except RuntimeError:
        pass
    else:
        raise AssertionError("delete should report the failed index rebuild")

    assert {doc["doc_id"] for doc in orchestrator.db.get_all_documents()} == {
        first["doc_id"], second["doc_id"]
    }
    assert orchestrator.db.get_all_chunks() == original_chunks
