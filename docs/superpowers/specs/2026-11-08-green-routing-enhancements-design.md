# GreenRouting RAG Enhancements Design

**Date:** 2026-11-08  
**Target demonstration:** 2026-11-13  
**Status:** Design proposal for approval before implementation

## Intent

GreenRouting already indexes uploaded documents, retrieves source chunks, routes between Ollama models, validates semantic-cache hits, and records energy/carbon telemetry. This design adds four visible capabilities that make the sustainability claim easier to demonstrate:

1. A document can be removed safely, followed by a consistent vector-index rebuild.
2. Retrieval can use fewer or more chunks (`top_k`) according to query difficulty and document evidence.
3. The application explains its cache, retrieval, and routing decisions in plain language.
4. A benchmark chart compares quality, energy, carbon, latency, and cache savings across execution modes.

The features are intentionally incremental. They must work with the current base branch and must not require unfinished Workstreams 2 or 3 to be delivered first.

## Goals

- Keep SQLite metadata and the FAISS/NumPy vector index consistent after document deletion.
- Reduce prompt size and retrieval work for simple questions without hiding relevant evidence.
- Make every green decision understandable to a teacher or demo audience.
- Produce a repeatable visual comparison between `STANDARD_RAG`, `RAG_WITH_CACHE`, and `GREENROUTING`.
- Preserve current PDF/TXT/Markdown upload behavior, duplicate-upload protection, source citations, and local Ollama execution.
- Keep all four capabilities testable without depending on live network services.

## Non-goals

- Do not replace the existing semantic-cache implementation or change its similarity threshold in this slice.
- Do not redesign `AdaptiveGreenRouter` or introduce learned routing before its owner delivers Workstream 2/3 changes.
- Do not claim that modeled energy values are laboratory-grade measurements; the UI must label them as modeled or sensor-based.
- Do not add cloud hosting, external LLM APIs, live electricity-price APIs, OCR, or a new vector database for this milestone.

## Constraints and decisions

- Python 3.11 environment (`greenrouting-py311`) remains the supported development environment.
- Ollama remains local and continues to provide the lightweight/capable models.
- Existing SQLite, FAISS/NumPy, pandas, Plotly, and Streamlit dependencies are reused.
- New behavior should be added through small methods or a focused helper module rather than a broad rewrite.
- Tests use temporary databases/index directories and deterministic fake services where possible.
- A deletion action is destructive from the user’s perspective, so the UI must require an explicit button action and report exactly what was removed.
- Rebuilds are derived from SQLite’s remaining chunks; the database is the source of truth.

## Existing extension points

- `greenrouting/database.py` already stores documents, chunks, cache entries, query logs, and benchmark summaries; it also has `delete_document`.
- `greenrouting/vector_store.py` already has `add_chunks`, `search(top_k=...)`, `clear`, persistence, and chunk metadata.
- `greenrouting/orchestrator.py` owns indexing and query execution and already accepts an optional `top_k`.
- `greenrouting/adaptive_router.py` already returns a `routing_reason` and retry information.
- `app.py` already has document, cache, assistant, and benchmark tabs and displays cache/routing explanations.
- `benchmark.py` already returns per-mode summaries and detailed records suitable for additional Plotly charts.

## Chosen approach

Use four small vertical slices, implemented in this order:

1. **Deletion and rebuild foundation** — make index consistency reliable before changing retrieval behavior.
2. **Adaptive retrieval policy** — compute a bounded `top_k`, pass it through the existing orchestrator, and expose the decision in telemetry.
3. **Explainability contract and UI** — standardize the plain-language decision fields and show them alongside the existing metrics.
4. **Quality–carbon benchmark visualization** — add charts and a small table that make trade-offs measurable rather than anecdotal.

Each slice has its own tests and can be demonstrated independently. The first and third slices can be completed without any changes to friends’ cache/router internals. The second slice only consumes the existing `top_k` interface. The fourth slice consumes benchmark output and does not alter model selection.

## Feature 1: Document deletion and index rebuilding

### User behavior

In the Document Knowledge Base tab, each indexed document is listed with a safe, explicit delete control. After confirmation, the app removes that document’s SQLite metadata and chunks, invalidates cache entries whose source documents include that document, rebuilds the vector index from all remaining SQLite chunks, and refreshes the displayed counts.

### Proposed interfaces

- `DatabaseManager.delete_document(doc_id: str) -> None` remains the database operation; it must also remove or invalidate dependent cache entries in the same logical operation.
- Add a vector-store rebuild operation such as `FaissVectorStore.rebuild(chunks: List[Dict[str, Any]]) -> int`, which clears persisted vectors and re-embeds exactly the supplied chunks.
- Add `GreenRoutingOrchestrator.delete_document(doc_id: str) -> Dict[str, Any]`, returning the deleted filename, removed chunk count, invalidated cache count, and remaining document/chunk counts.

### Safety rules

- Unknown document IDs return a non-success result and do not rebuild.
- The last document may be deleted; the resulting index is empty and future queries show the existing “upload documents” message.
- Rebuilding must not duplicate surviving chunks.
- A failed rebuild must be logged and surfaced instead of silently claiming success.

## Feature 2: Energy-aware retrieval with adaptive `top_k`

### User behavior

Simple factual questions retrieve a small context (for example, two chunks). Analytical or multi-part questions retrieve more context (up to the configured maximum). The answer page shows the chosen `top_k`, the reason, and the number of chunks actually used.

### Proposed policy

Create a pure helper, for example `greenrouting/retrieval_policy.py`:

`choose_top_k(query: str, available_chunks: int, default_top_k: int, min_top_k: int = 2, max_top_k: int = 5) -> RetrievalDecision`

`RetrievalDecision` contains `top_k`, `complexity_score`, and `reason`. The first implementation is deterministic and local: use query length, question-word count, and comparison/analysis markers. Clamp the result to available chunks and configured bounds. This is a transparent heuristic, not machine learning.

### Integration

- `execute_query` uses the policy only when the caller does not explicitly provide `top_k`.
- An explicit `top_k` remains an override for tests and advanced users.
- The chosen value is passed to the existing `vector_store.search` call.
- The response adds `retrieval_top_k`, `retrieved_chunk_count`, and `retrieval_reason`.
- Cache lookup and answer generation use the same retrieved chunk set; no cache algorithm change is required.

### Guardrails

- Never return `top_k < 1` when documents exist.
- Never request more chunks than are available.
- If the heuristic cannot classify a query, use the configured default and say so.
- Quality evaluation and source citations remain enabled for every value of `top_k`.

## Feature 3: Explainable routing and cache messages

### User behavior

The assistant page presents a short “Why this decision?” section containing:

- cache result: hit, miss, or invalidated, plus the validation reason;
- retrieval result: selected `top_k` and why;
- model result: lightweight/capable model, complexity score, and whether quality escalation occurred;
- environmental result: energy, carbon, latency, and measurement method.

### Proposed response contract

Keep existing fields for compatibility and add a nested, display-ready object:

```python
"decision_trace": {
    "cache": {"status": str, "reason": str},
    "retrieval": {"top_k": int, "chunk_count": int, "reason": str},
    "routing": {"model": str, "complexity_score": float, "reason": str, "retry_count": int},
    "impact": {"energy_kwh": float, "emissions_g": float, "latency_seconds": float, "measurement_label": str}
}
```

The nested object is additive; existing callers can continue reading `routing_reason`, `cache_validation_reason`, and telemetry fields. For cache hits, the routing explanation explicitly says that LLM inference was bypassed. For no-document responses, the trace explains that no retrieval was possible.

## Feature 4: Quality–carbon benchmark chart

### User behavior

After a benchmark run, the Experimental Benchmarking tab shows:

- grouped bars for total energy and total carbon by mode;
- a quality-versus-carbon scatter plot, with one point per mode;
- a compact comparison table containing average quality, groundedness, latency, cache-hit rate, and energy/carbon savings relative to `STANDARD_RAG`.

### Data contract

Extend the benchmark result only with derived fields or UI data frames. Do not alter the meaning of existing measurements. The chart must handle zero/near-zero baselines without division errors and must display negative savings honestly when a run is slower or more energy-intensive.

### Interpretation text

The UI should state that a desirable result is lower carbon/energy with comparable quality, not simply the fastest answer. If `GREENROUTING` has negative savings in a short noisy run, show the result and label it as an observed run rather than forcing it to zero.

## Phase sequence and dependency map

| Phase | Deliverable | Depends on | Evidence of completion |
|---|---|---|---|
| 0 | Design and baseline snapshot | Current branch | Existing non-benchmark tests pass; current counts recorded |
| 1 | Delete one document and rebuild index | None | Unit/integration tests prove no orphan chunks and correct search counts |
| 2 | Adaptive `top_k` retrieval | Phase 1 optional, current query flow | Deterministic policy tests and query response fields pass |
| 3 | Decision trace and UI explanations | Phase 2 for retrieval line | AppTest sees trace values for cache hit, cache miss, and no-doc response |
| 4 | Benchmark comparison charts | Existing benchmark output; Phase 3 optional | AppTest/plot construction succeeds and CSV export remains valid |
| 5 | Demo hardening and handoff | Phases 1–4 | PDF/TXT/MD, multi-file, duplicate, deletion, repeat-query, and benchmark demo checklist complete |

For the November 13 demonstration, Phases 1–4 are the target. Phase 5 is rehearsal and documentation, not a new architectural feature.

## Verification strategy

Every phase follows test-driven development:

1. Add a failing unit test for the new behavior.
2. Implement the smallest change that passes it.
3. Run focused tests, then the non-benchmark suite.
4. Run Streamlit `AppTest` for the visible behavior.
5. Record any benchmark instability separately; do not hide negative observed savings.

Required scenarios:

- delete a middle document, delete the last document, and request an unknown document;
- rebuild after duplicate-protected uploads;
- short factual, long analytical, and empty/ambiguous queries;
- explicit `top_k` override;
- cache hit, cache miss, quality retry, and no-indexed-document response;
- benchmark with one query, repeated queries, and a zero/near-zero energy fixture.

## Risks and fallback decisions

- **HDD latency:** rebuilding embeddings may take time. Show a progress/spinner and keep the operation explicit; correctness is more important than instant deletion.
- **Embedding service unavailable:** use the project’s existing offline/simulation behavior; never delete database data unless the rebuild result is known.
- **Model/router changes arriving later:** keep the new decision trace additive and map missing fields to clear defaults.
- **Noisy energy measurements:** preserve raw values and explain that local runs are modeled/sensor-dependent.
- **Deadline pressure:** if time becomes short, ship deletion/rebuild, decision trace, and one benchmark chart first; adaptive retrieval can fall back to the configured fixed `top_k` while retaining its interface and tests.

## Success criteria

The teacher can upload multiple files, ask a question, see why GreenRouting selected its retrieval/model/cache path, delete one file and observe that it no longer contributes sources, and run a benchmark that visibly compares quality against energy/carbon. The implementation remains local, reproducible, and honest about measured versus modeled environmental impact.
