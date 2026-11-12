"""Tests for distinguishing document summaries from ordinary document questions."""

try:
    from greenrouting.query_intent import DOCUMENT_QUESTION, DOCUMENT_SUMMARY, detect_query_intent
except ImportError:
    DOCUMENT_QUESTION = DOCUMENT_SUMMARY = None
    detect_query_intent = None


def test_detects_common_document_summary_requests():
    assert callable(detect_query_intent), "query intent classifier has not been implemented"
    requests = [
        "Give me a summary of this document",
        "Summarize this PDF",
        "What is this document about?",
        "Give me an overview of the file",
        "List the key points in this document",
    ]

    assert all(detect_query_intent(request) == DOCUMENT_SUMMARY for request in requests)


def test_keeps_specific_document_questions_on_question_path():
    assert callable(detect_query_intent), "query intent classifier has not been implemented"
    questions = [
        "What are the main skills listed in this CV?",
        "How many years of experience does the document mention?",
        "What is the document's conclusion about neural networks?",
        "Compare the two energy sources described in the report.",
    ]

    assert all(detect_query_intent(question) == DOCUMENT_QUESTION for question in questions)
