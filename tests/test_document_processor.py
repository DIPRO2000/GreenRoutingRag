"""
Unit tests for DocumentProcessor in GreenRouting.
"""
import pytest
from greenrouting.document_processor import DocumentProcessor, DocumentProcessorError


def test_txt_processing_and_chunking():
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    sample_text = (
        "GreenRouting is an energy-aware retrieval-augmented generation framework. "
        "It dynamically routes user questions between lightweight and capable models. "
        "Furthermore, GreenRouting implements quality-aware semantic caching to minimize unnecessary inference."
    )
    raw_bytes = sample_text.encode("utf-8")
    doc_meta, chunks = processor.process_document("test_guide.txt", raw_bytes, doc_id="doc_test_1")

    assert doc_meta["doc_id"] == "doc_test_1"
    assert doc_meta["file_type"] == "TXT"
    assert doc_meta["total_characters"] > 0
    assert len(chunks) >= 2
    assert chunks[0]["doc_id"] == "doc_test_1"
    assert "chunk_id" in chunks[0]
    assert chunks[0]["page_number"] == 1


def test_invalid_empty_document():
    processor = DocumentProcessor(chunk_size=200, chunk_overlap=30)
    with pytest.raises(DocumentProcessorError):
        processor.process_document("empty.txt", b"")


def test_invalid_extension():
    processor = DocumentProcessor(chunk_size=200, chunk_overlap=30)
    with pytest.raises(DocumentProcessorError) as exc_info:
        processor.process_document("bad_file.exe", b"binary content")
    assert "Unsupported file extension" in str(exc_info.value)


def test_chunk_overlap_validation():
    with pytest.raises(ValueError):
        DocumentProcessor(chunk_size=100, chunk_overlap=120)


def test_pdf_processing_reports_low_and_empty_text_pages(monkeypatch):
    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakePdfReader:
        def __init__(self, _stream):
            self.pages = [
                FakePage("Readable page with enough text. " + "detail " * 12),
                FakePage("Only a short heading."),
                FakePage(""),
                FakePage("x" * 80),
            ]

    monkeypatch.setattr("greenrouting.document_processor.PdfReader", FakePdfReader)
    processor = DocumentProcessor()

    metadata, chunks = processor.process_document("mixed.pdf", b"fake pdf", doc_id="mixed-pdf")

    assert metadata["page_count"] == 4
    assert metadata["low_text_pages"] == [2, 3]
    assert {chunk["page_number"] for chunk in chunks} == {1, 2, 4}


def test_fully_scanned_pdf_error_explains_ocr_is_unavailable(monkeypatch):
    class FakePage:
        def extract_text(self):
            return ""

    class FakePdfReader:
        def __init__(self, _stream):
            self.pages = [FakePage(), FakePage()]

    monkeypatch.setattr("greenrouting.document_processor.PdfReader", FakePdfReader)

    with pytest.raises(DocumentProcessorError, match="OCR is not available"):
        DocumentProcessor().process_document("scanned.pdf", b"fake pdf")
