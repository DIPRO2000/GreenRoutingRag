"""Hierarchical summaries for documents that exceed a single prompt's context."""
from typing import Any, Dict, List


class DocumentSummarizer:
    """Summarize source chunks in bounded batches, then combine the summaries."""

    _SECTION_SYSTEM_PROMPT = (
        "Summarize only the provided document text. Preserve its central ideas and important facts. "
        "Do not add outside information. Keep this section summary concise."
    )
    _COMBINE_SYSTEM_PROMPT = (
        "Combine the provided section summaries into one coherent document summary. "
        "Preserve key ideas and important facts, remove repetition, and do not add outside information."
    )

    def __init__(self, llm_client: Any, model_name: str, max_batch_chars: int = 3500):
        if max_batch_chars < 1:
            raise ValueError("max_batch_chars must be positive")
        self.llm_client = llm_client
        self.model_name = model_name
        self.max_batch_chars = max_batch_chars

    def summarize(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return one summary after processing all readable source chunks."""
        source_records = [
            {
                "chunk_id": chunk.get("chunk_id", f"chunk-{index + 1}"),
                "page_number": chunk.get("page_number"),
                "text_content": str(chunk.get("text_content", "")).strip(),
            }
            for index, chunk in enumerate(chunks)
            if str(chunk.get("text_content", "")).strip()
        ]
        if not source_records:
            raise ValueError("No readable text was found in the selected document.")

        generation_runs: List[Dict[str, Any]] = []
        simulated = False
        last_model_used = self.model_name

        def generate_groups(records: List[Dict[str, Any]], stage: str) -> List[Dict[str, Any]]:
            nonlocal simulated, last_model_used
            summaries = []
            for group_index, group in enumerate(self._make_batches(records), start=1):
                context_blocks = []
                for record in group:
                    page = record.get("page_number")
                    page_label = f"; page {page}" if page is not None else ""
                    context_blocks.append(
                        f"[Chunk {record['chunk_id']}{page_label}]\n{record['text_content']}"
                    )
                context = "\n\n".join(context_blocks)
                if stage == "section":
                    question = "Summarize this section in a few concise sentences."
                    system_prompt = self._SECTION_SYSTEM_PROMPT
                else:
                    question = "Combine the section summaries into one concise document summary."
                    system_prompt = self._COMBINE_SYSTEM_PROMPT
                prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                result = self.llm_client.generate(
                    prompt=prompt,
                    model_name=self.model_name,
                    system_prompt=system_prompt,
                )
                answer = str(result.get("answer", "")).strip()
                if not answer:
                    raise RuntimeError(f"The model returned an empty {stage} summary.")
                run_latency = float(result.get("latency_seconds", 0.0))
                generation_runs.append({
                    "model_name": self.model_name,
                    "stage": stage,
                    "latency_seconds": run_latency,
                    "eval_count": result.get("eval_count", 0),
                })
                simulated = simulated or bool(result.get("is_simulated", False))
                last_model_used = result.get("model_used", self.model_name)
                summaries.append({
                    "chunk_id": f"{stage}-{group_index}",
                    "page_number": None,
                    "text_content": answer,
                })
            return summaries

        partial_summaries = generate_groups(source_records, "section")
        # Repeatedly combine batches so even a very long document fits small contexts.
        for _ in range(8):
            final_summaries = generate_groups(partial_summaries, "combine")
            if len(final_summaries) == 1:
                latency = sum(run["latency_seconds"] for run in generation_runs)
                return {
                    "summary": final_summaries[0]["text_content"],
                    "model_used": last_model_used,
                    "is_simulated": simulated,
                    "latency_seconds": latency,
                    "generation_runs": generation_runs,
                    "call_count": len(generation_runs),
                }
            if len(final_summaries) >= len(partial_summaries):
                raise RuntimeError("The model did not shorten intermediate summaries enough to combine them.")
            partial_summaries = final_summaries

        raise RuntimeError("The document is too long to combine its summaries within the configured number of passes.")

    def _make_batches(self, records: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Keep each prompt's source text within the configured character budget."""
        batches: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_chars = 0

        for record in records:
            text = record["text_content"]
            pieces = []
            while len(text) > self.max_batch_chars:
                end = self.max_batch_chars
                boundary = text.rfind(" ", 0, end)
                if boundary > 0:
                    end = boundary
                pieces.append(text[:end])
                text = text[end:].lstrip()
            if text:
                pieces.append(text)

            for part_index, piece in enumerate(pieces, start=1):
                if current and current_chars + len(piece) > self.max_batch_chars:
                    batches.append(current)
                    current = []
                    current_chars = 0
                current.append({
                    **record,
                    "chunk_id": f"{record['chunk_id']}-part-{part_index}" if len(pieces) > 1 else record["chunk_id"],
                    "text_content": piece,
                })
                current_chars += len(piece)

        if current:
            batches.append(current)
        return batches
