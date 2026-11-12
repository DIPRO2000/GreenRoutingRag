"""Lightweight intent detection for document-wide summary requests."""
import re


DOCUMENT_SUMMARY = "DOCUMENT_SUMMARY"
DOCUMENT_QUESTION = "DOCUMENT_QUESTION"

_SUMMARY_PATTERNS = (
    r"\bsummary\b",
    r"\bsummari[sz](?:e|ing|ation)?\b",
    r"\boverview\b",
    r"\b(?:main|key)\s+(?:ideas?|points?|themes?)\b",
    r"\bkey\s+takeaways\b",
    r"\bwhat\s+is\s+(?:this|the|that)\s+(?:document|file|pdf)\s+about\b",
    r"\bwhat\s+does\s+(?:this|the|that)\s+(?:document|file|pdf)\s+cover\b",
    r"\b(?:tell|explain)\s+me\s+what\s+(?:this|the|that)\s+(?:document|file|pdf)\s+is\s+about\b",
)


def detect_query_intent(query: str) -> str:
    """Classify common whole-document summary requests; default to normal Q&A."""
    if not isinstance(query, str) or not query.strip():
        return DOCUMENT_QUESTION

    normalized_query = " ".join(query.lower().split())
    if any(re.search(pattern, normalized_query) for pattern in _SUMMARY_PATTERNS):
        return DOCUMENT_SUMMARY
    return DOCUMENT_QUESTION
