"""
Document Processor for GreenRouting.
Extracts, cleans, validates, and chunks text from PDF and TXT files.
"""
import hashlib
import uuid
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from pypdf import PdfReader

MIN_PDF_PAGE_TEXT_CHARS = 80


class DocumentProcessorError(Exception):
    """Custom exception for document processing errors."""
    pass


class DocumentProcessor:
    """Handles parsing, validation, chunking, and metadata generation for documents."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def compute_sha256(content_bytes: bytes) -> str:
        """Compute SHA-256 hash of document bytes for context verification."""
        return hashlib.sha256(content_bytes).hexdigest()

    def extract_text_from_txt(self, file_path_or_bytes: Any) -> Tuple[str, List[Tuple[int, str]]]:
        """Extract text from a TXT file or raw bytes."""
        if isinstance(file_path_or_bytes, (str, Path)):
            with open(file_path_or_bytes, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        elif isinstance(file_path_or_bytes, bytes):
            text = file_path_or_bytes.decode("utf-8", errors="replace")
        else:
            raise DocumentProcessorError("Unsupported input type for TXT extraction")

        clean_text = self._clean_text(text)
        if not clean_text.strip():
            raise DocumentProcessorError("TXT document contains no readable text.")

        # Single page representation for TXT
        return clean_text, [(1, clean_text)]

    def extract_text_from_pdf(self, file_path_or_bytes: Any) -> Tuple[str, List[Tuple[int, str]]]:
        """Extract text from a PDF file with page tracking."""
        full_text, pages_content, _, _ = self._extract_pdf_text_and_page_stats(file_path_or_bytes)
        return full_text, pages_content

    def _extract_pdf_text_and_page_stats(
        self,
        file_path_or_bytes: Any,
    ) -> Tuple[str, List[Tuple[int, str]], int, List[int]]:
        """Extract PDF text and identify pages with little or no selectable text."""
        try:
            if isinstance(file_path_or_bytes, (str, Path)):
                reader = PdfReader(str(file_path_or_bytes))
            else:
                import io
                reader = PdfReader(io.BytesIO(file_path_or_bytes))

            if len(reader.pages) == 0:
                raise DocumentProcessorError("PDF contains 0 pages.")

            pages_content: List[Tuple[int, str]] = []
            full_text_list: List[str] = []
            low_text_pages: List[int] = []

            for idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                cleaned = self._clean_text(page_text)
                if len(cleaned) < MIN_PDF_PAGE_TEXT_CHARS:
                    low_text_pages.append(idx)
                if cleaned:
                    pages_content.append((idx, cleaned))
                    full_text_list.append(cleaned)

            full_text = "\n\n".join(full_text_list)
            if not full_text.strip():
                raise DocumentProcessorError(
                    "PDF text extraction found no readable text. The file may be scanned or image-only; OCR is not available."
                )

            return full_text, pages_content, len(reader.pages), low_text_pages
        except Exception as e:
            if isinstance(e, DocumentProcessorError):
                raise
            raise DocumentProcessorError(f"Failed to extract text from PDF: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """Standardize whitespace and remove non-printable characters."""
        # Replace Windows / Mac line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Replace multiple spaces / tabs with single space
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def chunk_text(self, pages_content: List[Tuple[int, str]], doc_id: str) -> List[Dict[str, Any]]:
        """
        Split text across pages into overlapping chunks with metadata.
        Uses recursive boundary splitting (paragraphs, sentences, words).
        """
        chunks: List[Dict[str, Any]] = []
        chunk_idx = 0

        step = self.chunk_size - self.chunk_overlap

        for page_num, content in pages_content:
            text_len = len(content)
            if text_len == 0:
                continue

            # If page text is shorter than chunk_size, create one chunk
            if text_len <= self.chunk_size:
                chunks.append({
                    "chunk_id": f"{doc_id}_c{chunk_idx}",
                    "doc_id": doc_id,
                    "chunk_index": chunk_idx,
                    "text_content": content,
                    "page_number": page_num,
                    "character_count": len(content)
                })
                chunk_idx += 1
                continue

            # Sliding window with boundary awareness
            start = 0
            while start < text_len:
                end = min(start + self.chunk_size, text_len)
                segment = content[start:end]

                # Try to adjust boundary to the nearest space or punctuation if not at text end
                if end < text_len:
                    last_break = max(segment.rfind("\n"), segment.rfind(". "), segment.rfind(" "))
                    if last_break > self.chunk_size // 2:
                        end = start + last_break + 1
                        segment = content[start:end]

                cleaned_chunk = segment.strip()
                if cleaned_chunk:
                    chunks.append({
                        "chunk_id": f"{doc_id}_c{chunk_idx}",
                        "doc_id": doc_id,
                        "chunk_index": chunk_idx,
                        "text_content": cleaned_chunk,
                        "page_number": page_num,
                        "character_count": len(cleaned_chunk)
                    })
                    chunk_idx += 1

                start += step

        return chunks

    def process_document(self, file_name: str, file_bytes: bytes, doc_id: str = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Complete processing pipeline: extract, validate, chunk, and return document & chunk dicts.
        """
        if not file_bytes:
            raise DocumentProcessorError("Empty document bytes received.")

        ext = Path(file_name).suffix.lower()
        if ext == ".pdf":
            file_type = "PDF"
            full_text, pages, page_count, low_text_pages = self._extract_pdf_text_and_page_stats(file_bytes)
        elif ext in [".txt", ".md"]:
            file_type = "TXT"
            full_text, pages = self.extract_text_from_txt(file_bytes)
            page_count = None
            low_text_pages = []
        else:
            raise DocumentProcessorError(f"Unsupported file extension: '{ext}'. GreenRouting supports .pdf and .txt files.")

        doc_hash = self.compute_sha256(file_bytes)
        if not doc_id:
            doc_id = str(uuid.uuid4())[:12]

        chunks = self.chunk_text(pages, doc_id)
        if not chunks:
            raise DocumentProcessorError("No valid chunks could be generated from document.")

        doc_meta = {
            "doc_id": doc_id,
            "filename": file_name,
            "doc_hash": doc_hash,
            "file_type": file_type,
            "total_chunks": len(chunks),
            "total_characters": len(full_text),
            "page_count": page_count,
            "low_text_pages": low_text_pages,
        }

        return doc_meta, chunks
