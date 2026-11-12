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


def test_index_failure_rolls_back_document_rows_and_restores_existing_index(tmp_path, monkeypatch):
    orchestrator, first, second = _orchestrator_with_documents(tmp_path)
    original_chunks = orchestrator.db.get_all_chunks()
    original_add_chunks = orchestrator.vector_store.add_chunks
    add_calls = 0

    def fail_new_document_once(chunks):
        nonlocal add_calls
        add_calls += 1
        if add_calls == 1:
            raise RuntimeError("test embedding index failure")
        return original_add_chunks(chunks)

    monkeypatch.setattr(orchestrator.vector_store, "add_chunks", fail_new_document_once)

    try:
        orchestrator.process_and_index_document("broken.txt", b"This upload cannot be indexed.")
    except RuntimeError as exc:
        assert "test embedding index failure" in str(exc)
    else:
        raise AssertionError("index failure should be reported")

    assert {doc["doc_id"] for doc in orchestrator.db.get_all_documents()} == {
        first["doc_id"],
        second["doc_id"],
    }
    assert orchestrator.db.get_all_chunks() == original_chunks
    assert orchestrator.vector_store.chunks == original_chunks


def test_document_selector_labels_distinguish_same_filename_and_chunk_count():
    from greenrouting.document_selection import build_document_options

    options = build_document_options([
        {"doc_id": "doc-a", "filename": "report.pdf", "total_chunks": 4},
        {"doc_id": "doc-b", "filename": "report.pdf", "total_chunks": 4},
    ])

    assert len(options) == 3  # Includes the all-documents choice.
    assert {value for value in options.values() if value is not None} == {"doc-a", "doc-b"}


def test_document_upload_result_includes_pdf_extraction_warnings(tmp_path, monkeypatch):
    settings = load_settings()
    settings.app.db_path = tmp_path / "extraction-warning.db"
    settings.app.faiss_index_dir = tmp_path / "extraction-warning-faiss"
    orchestrator = GreenRoutingOrchestrator(settings)
    original_process = orchestrator.doc_processor.process_document

    def process_with_extraction_warning(filename, file_bytes):
        metadata, chunks = original_process("source.txt", file_bytes)
        metadata["filename"] = filename
        metadata["file_type"] = "PDF"
        metadata.update(page_count=3, low_text_pages=[2, 3])
        return metadata, chunks

    monkeypatch.setattr(orchestrator.doc_processor, "process_document", process_with_extraction_warning)

    result = orchestrator.process_and_index_document("mixed.pdf", b"Readable document text.")

    assert result["page_count"] == 3
    assert result["low_text_pages"] == [2, 3]


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


def test_document_chunk_loader_returns_every_chunk_in_document_order(tmp_path):
    settings = load_settings()
    settings.app.db_path = tmp_path / "summary-source.db"
    settings.app.faiss_index_dir = tmp_path / "summary-faiss"
    settings.rag.chunk_size = 100
    settings.rag.chunk_overlap = 10
    orchestrator = GreenRoutingOrchestrator(settings)
    text = " ".join(
        f"Section {index} explains a distinct detail of the document for later summary."
        for index in range(12)
    )
    indexed = orchestrator.process_and_index_document("many-sections.txt", text.encode())
    orchestrator.process_and_index_document("other.txt", b"Another separate document.")

    chunks = orchestrator.db.get_document_chunks(indexed["doc_id"])

    assert len(chunks) == indexed["total_chunks"]
    assert len(chunks) > 1
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert {chunk["doc_id"] for chunk in chunks} == {indexed["doc_id"]}
    assert orchestrator.db.get_document_chunks("missing-document") == []
