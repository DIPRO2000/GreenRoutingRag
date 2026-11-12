"""Tests for chunked, document-wide summarization."""

import pytest
from greenrouting.config import load_settings
from greenrouting.orchestrator import GreenRoutingOrchestrator
from greenrouting.rag_pipeline import RAG_SYSTEM_PROMPT

try:
    from greenrouting.document_summarizer import DocumentSummarizer
except ImportError:
    DocumentSummarizer = None


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        is_final = "combine the section summaries" in kwargs["prompt"].lower()
        call_num = len(self.calls)
        return {
            "answer": "Complete document summary." if is_final else f"Section summary {call_num}.",
            "model_used": kwargs["model_name"],
            "is_simulated": call_num == 2,
            "latency_seconds": 0.1,
            "eval_count": 12,
        }


def test_summarizer_reads_all_chunks_in_batches_then_combines_them():
    assert DocumentSummarizer is not None, "document summarizer has not been implemented"
    llm = FakeLLM()
    summarizer = DocumentSummarizer(llm, model_name="llama3.2:3b", max_batch_chars=100)
    chunks = [
        {"chunk_id": f"chunk-{i}", "page_number": i + 1, "text_content": f"Text for source chunk {i}. " + ("detail " * 8)}
        for i in range(4)
    ]

    result = summarizer.summarize(chunks)

    map_prompts = [call["prompt"] for call in llm.calls if "summarize this section" in call["prompt"].lower()]
    assert len(map_prompts) == 4
    assert all(f"chunk-{i}" in " ".join(map_prompts) for i in range(4))
    assert result["summary"] == "Complete document summary."
    assert result["call_count"] == len(llm.calls) == 5
    assert result["is_simulated"] is True
    assert result["latency_seconds"] == pytest.approx(0.5)


def test_summarizer_rejects_missing_or_blank_document_content():
    assert DocumentSummarizer is not None, "document summarizer has not been implemented"
    summarizer = DocumentSummarizer(FakeLLM(), model_name="llama3.2:3b")

    with pytest.raises(ValueError, match="No readable text"):
        summarizer.summarize([])

    with pytest.raises(ValueError, match="No readable text"):
        summarizer.summarize([{"chunk_id": "empty", "text_content": "  "}])


def test_summarizer_handles_many_batches_with_recursive_combine_passes():
    assert DocumentSummarizer is not None, "document summarizer has not been implemented"

    class RecursiveLLM:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["system_prompt"] == DocumentSummarizer._COMBINE_SYSTEM_PROMPT:
                answer = "sum"
            else:
                answer = "section"
            return {"answer": answer, "latency_seconds": 0.01, "is_simulated": False}

    llm = RecursiveLLM()
    summarizer = DocumentSummarizer(llm, model_name="llama3.2:3b", max_batch_chars=25)
    chunks = [
        {
            "chunk_id": f"part-{index}",
            "page_number": index + 1,
            "text_content": f"Beginning middle ending content marker {index} and supporting detail.",
        }
        for index in range(20)
    ]

    result = summarizer.summarize(chunks)

    section_prompts = [
        call["prompt"] for call in llm.calls
        if call["system_prompt"] == DocumentSummarizer._SECTION_SYSTEM_PROMPT
    ]
    combine_prompts = [
        call["prompt"] for call in llm.calls
        if call["system_prompt"] == DocumentSummarizer._COMBINE_SYSTEM_PROMPT
    ]
    assert len(section_prompts) > len(chunks)
    assert all(f"part-{index}" in " ".join(section_prompts) for index in range(len(chunks)))
    assert len(combine_prompts) > 1
    assert result["summary"] == "sum"
    assert result["call_count"] == len(llm.calls)


def test_orchestrator_summarizes_chunks_from_the_selected_document(tmp_path):
    settings = load_settings()
    settings.app.db_path = tmp_path / "summary.db"
    settings.app.faiss_index_dir = tmp_path / "faiss"
    settings.rag.chunk_size = 120
    settings.rag.chunk_overlap = 10
    orchestrator = GreenRoutingOrchestrator(settings)
    text = " ".join(
        f"Section {index} contains a unique fact about document summarization."
        for index in range(12)
    )
    document = orchestrator.process_and_index_document("summary.txt", text.encode())
    source_chunks = orchestrator.db.get_document_chunks(document["doc_id"])
    for index, chunk in enumerate(source_chunks):
        chunk["page_number"] = index // 3 + 1
    orchestrator.db.get_document_chunks = lambda requested_doc_id: (
        source_chunks if requested_doc_id == document["doc_id"] else []
    )
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        if kwargs["system_prompt"] == DocumentSummarizer._COMBINE_SYSTEM_PROMPT:
            answer = "Combined document summary."
        else:
            answer = "Section summary."
        return {"answer": answer, "model_used": kwargs["model_name"], "is_simulated": False, "latency_seconds": 0.1}

    orchestrator.llm_client.generate = fake_generate

    result = orchestrator.summarize_document(document["doc_id"])

    assert result["answer"] == "Combined document summary."
    assert len(calls) == 2
    assert all(chunk["chunk_id"] in calls[0]["prompt"] for chunk in source_chunks)
    assert result["model_used"] == settings.models.capable_model
    assert result["summary_call_count"] == 2
    assert result["is_document_summary"] is True
    assert result["cache_status"] == "BYPASSED"
    assert result["retrieved_chunk_count"] == len(source_chunks)
    assert result["energy_kwh"] > 0
    assert result["emissions_g"] > 0
    assert {source["chunk_id"] for source in result["sources"]} == {
        chunk["chunk_id"] for chunk in source_chunks
    }
    assert result["summary_pages"] == sorted({chunk["page_number"] for chunk in source_chunks})
    assert {detail["stage"] for detail in result["generation_details"]} == {"section", "combine"}

    call_count_before_question = len(calls)
    question_result = orchestrator.execute_query(
        "What detail is described in section 0?",
        mode="STANDARD_RAG",
        active_doc_id=document["doc_id"],
    )
    assert question_result.get("is_document_summary") is not True
    assert question_result["retrieved_chunk_count"] <= question_result["retrieval_top_k"]
    assert len(calls) == call_count_before_question + 1
    assert calls[-1]["system_prompt"] == RAG_SYSTEM_PROMPT
