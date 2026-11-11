# GreenRouting RAG Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement safe document deletion/rebuilding, adaptive retrieval, explainable decisions, and quality–carbon benchmark visuals in the existing GreenRouting app.

**Architecture:** Keep SQLite as the source of truth for documents and chunks. Rebuild the persisted FAISS/NumPy index from surviving SQLite chunks after deletion. Add adaptive retrieval as a pure deterministic policy consumed by the existing orchestrator `top_k` path. Extend response data additively with retrieval and decision-trace fields, then render those fields and existing benchmark summaries in Streamlit.

**Tech Stack:** Python 3.11, pytest, SQLite, FAISS/NumPy, pandas, Plotly, Streamlit AppTest, existing Ollama/embedding services.

**Spec:** `docs/superpowers/specs/2026-11-08-green-routing-enhancements-design.md`

## Global Constraints

- Use the current `feature/base-rag` branch and existing local architecture; do not create a new vector database or cloud dependency.
- Preserve duplicate-upload protection, current PDF/TXT/Markdown upload behavior, source citations, and existing response fields.
- Keep adaptive retrieval deterministic, bounded, and overridable by an explicit `top_k` argument.
- Preserve raw benchmark values, including negative observed savings; never clamp results to make a chart look better.
- Follow TDD: each production behavior starts with a failing test and is verified with focused tests plus the non-benchmark suite.
- Do not commit or push during this implementation unless the user explicitly asks.

## Review Focus

- Deleting the last document must leave a valid empty index and a usable app.
- Re-uploading a duplicate after deletion must be accepted exactly once.
- A one-chunk corpus must never receive `top_k=0` or an out-of-range request.
- An explicit `top_k` override must take precedence over the adaptive policy.
- Cache-hit, cache-miss, retry, and no-document responses must all contain a complete decision trace.
- Benchmark charts must render with one query and with zero/near-zero baseline energy.

---

### Task 1: Document deletion and vector-index rebuilding

**Files:**
- Modify: `greenrouting/database.py`
- Modify: `greenrouting/vector_store.py`
- Modify: `greenrouting/orchestrator.py`
- Modify: `app.py`
- Test: `tests/test_document_indexing.py`

**Interfaces:**
- `DatabaseManager.get_all_chunks() -> List[Dict[str, Any]]` returns all persisted chunks in stable document/chunk order.
- `FaissVectorStore.rebuild(chunks: List[Dict[str, Any]]) -> int` clears and re-embeds exactly the supplied chunks.
- `GreenRoutingOrchestrator.delete_document(doc_id: str) -> Dict[str, Any]` returns `deleted`, filename, removed chunk count, invalidated cache count, and remaining counts.

- [ ] **Step 1: Write failing deletion tests**
  - Index two documents, delete one, assert the response identifies the deleted file, SQLite has one document, and the vector store contains only the surviving chunks.
  - Delete the final document and assert zero documents/chunks plus an empty search result.
  - Pass an unknown ID and assert `deleted` is false and existing data is unchanged.

- [ ] **Step 2: Run the focused tests to verify RED**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_document_indexing.py -k deletion --basetemp .pytest-tmp-task1`

  Expected: FAIL because the orchestrator deletion/rebuild interface is not implemented.

- [ ] **Step 3: Implement the database and vector-store interfaces**
  - Add `get_all_chunks` to `DatabaseManager`.
  - Update document deletion to remove dependent cache entries whose `source_doc_ids` include the deleted ID and return deletion counts without breaking existing callers.
  - Add `FaissVectorStore.rebuild` that uses `clear`, then `add_chunks` when chunks remain.

- [ ] **Step 4: Implement `GreenRoutingOrchestrator.delete_document`**
  - Read the existing document and chunks.
  - Delete the database rows, rebuild from remaining chunks, and return a structured result.
  - If rebuilding fails, surface the failure and attempt to restore the previous vector index from the pre-delete chunk snapshot.

- [ ] **Step 5: Add the Streamlit delete control**
  - Show a delete button per indexed document.
  - Call the orchestrator method only after an explicit click, show removed counts, and rerun the page.

- [ ] **Step 6: Run focused tests and the non-benchmark suite**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_document_indexing.py --basetemp .pytest-tmp-task1-green`

  Expected: PASS for the deletion tests.

  Run: `python -m pytest -q -p no:cacheprovider -k 'not benchmark' --basetemp .pytest-tmp-task1-suite`

  Expected: existing non-benchmark tests pass.

### Task 2: Deterministic adaptive `top_k` retrieval

**Files:**
- Create: `greenrouting/retrieval_policy.py`
- Modify: `greenrouting/orchestrator.py`
- Modify: `app.py`
- Create: `tests/test_retrieval_policy.py`
- Modify: `tests/test_basic_rag.py` or add an orchestrator-focused test file if fixtures require it.

**Interfaces:**
- `RetrievalDecision` dataclass with `top_k: int`, `complexity_score: float`, and `reason: str`.
- `choose_top_k(query: str, available_chunks: int, default_top_k: int, min_top_k: int = 2, max_top_k: int = 5) -> RetrievalDecision`.

- [ ] **Step 1: Write failing policy tests**
  - A short factual query chooses the minimum bounded value.
  - A comparison/analysis query chooses a larger value.
  - One available chunk clamps to one.
  - Empty/ambiguous text falls back to the configured default within bounds.

- [ ] **Step 2: Run the focused policy tests to verify RED**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_retrieval_policy.py --basetemp .pytest-tmp-task2`

  Expected: FAIL because the policy module does not exist.

- [ ] **Step 3: Implement the pure retrieval policy**
  - Use only local deterministic signals: token count, query length, and comparison/analysis markers.
  - Clamp to available chunks and `[min_top_k, max_top_k]` while allowing a one-chunk corpus.
  - Return a plain-language reason suitable for the UI.

- [ ] **Step 4: Integrate the policy into `execute_query`**
  - When `top_k` is explicitly provided, preserve it after safe positive clamping.
  - Otherwise call `choose_top_k`, pass the result to `vector_store.search`, and add `retrieval_top_k`, `retrieved_chunk_count`, `retrieval_reason`, and `retrieval_complexity_score` to every response path.

- [ ] **Step 5: Add integration tests**
  - Assert a normal query response includes the retrieval fields and the returned chunk count does not exceed the selected `top_k`.
  - Assert explicit `top_k=1` is honored.

- [ ] **Step 6: Run focused tests and the non-benchmark suite**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_retrieval_policy.py tests/test_basic_rag.py --basetemp .pytest-tmp-task2-green`

  Expected: PASS.

  Run: `python -m pytest -q -p no:cacheprovider -k 'not benchmark' --basetemp .pytest-tmp-task2-suite`

  Expected: existing non-benchmark tests pass.

### Task 3: Complete decision trace and explainable UI

**Files:**
- Modify: `greenrouting/orchestrator.py`
- Modify: `app.py`
- Create: `tests/test_decision_trace.py`

**Interfaces:**
- Every `execute_query` response contains `decision_trace` with `cache`, `retrieval`, `routing`, and `impact` sections.
- Existing flat fields (`cache_validation_reason`, `routing_reason`, `energy_kwh`, etc.) remain available for compatibility.

- [ ] **Step 1: Write failing decision-trace tests**
  - A no-document response contains all four trace sections with safe values.
  - A normal generated response maps cache, retrieval, routing, retry, and impact fields into the trace.
  - A cache-hit response states that LLM inference was bypassed.

- [ ] **Step 2: Run the focused tests to verify RED**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_decision_trace.py --basetemp .pytest-tmp-task3`

  Expected: FAIL because `decision_trace` is not present on all response paths.

- [ ] **Step 3: Add the additive response contract**
  - Build the trace in one orchestrator helper so generated, cache-hit, and no-document paths use the same shape.
  - Use the retrieval fields from Task 2 and current router/cache explanations.

- [ ] **Step 4: Render the trace in Streamlit**
  - Add a compact Retrieval Decision panel beside the existing cache/model panels.
  - Display selected `top_k`, actual chunks, cache reason, model reason, retry warning, and measurement method in plain language.

- [ ] **Step 5: Add an AppTest smoke test or response-level UI fixture**
  - Load the app with a temporary settings/database fixture and assert the new labels render without exceptions.

- [ ] **Step 6: Run focused tests and the non-benchmark suite**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_decision_trace.py --basetemp .pytest-tmp-task3-green`

  Expected: PASS.

  Run: `python -m pytest -q -p no:cacheprovider -k 'not benchmark' --basetemp .pytest-tmp-task3-suite`

  Expected: existing non-benchmark tests pass.

### Task 4: Quality–carbon benchmark visualization

**Files:**
- Modify: `app.py`
- Create: `tests/test_benchmark_visuals.py`

**Interfaces:**
- The existing `BenchmarkRunner.run_benchmark` result remains the source of truth.
- Add a small pure helper only if needed to normalize summary data for charts; preserve all raw summary columns.

- [ ] **Step 1: Write failing benchmark-visual tests**
  - A normal summary frame produces energy/carbon bar data and quality-versus-carbon data for every mode.
  - A zero/near-zero baseline does not raise a division error and preserves the observed values.

- [ ] **Step 2: Run the focused tests to verify RED**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_benchmark_visuals.py --basetemp .pytest-tmp-task4`

  Expected: FAIL because the chart-data helper is not present.

- [ ] **Step 3: Implement the chart-data helper or directly test the existing summary contract**
  - Prefer a pure helper returning a copy of the summary frame with display columns such as carbon grams and quality percentage.
  - Do not clamp negative savings or mutate the benchmark’s raw values.

- [ ] **Step 4: Add Streamlit charts and comparison table**
  - Keep the existing energy chart.
  - Add carbon-emissions bars, a quality-versus-carbon scatter, and a table with quality, groundedness, latency, cache hit rate, energy, and carbon.
  - Add interpretation text explaining that lower impact with comparable quality is the target.

- [ ] **Step 5: Run focused tests, non-benchmark tests, and an AppTest smoke check**

  Run: `python -m pytest -q -p no:cacheprovider tests/test_benchmark_visuals.py --basetemp .pytest-tmp-task4-green`

  Expected: PASS.

  Run: `python -m pytest -q -p no:cacheprovider -k 'not benchmark' --basetemp .pytest-tmp-task4-suite`

  Expected: existing non-benchmark tests pass.

### Task 5: Final demo hardening

**Files:**
- Modify: `README.md` or `SETUP.md` only if commands or screenshots need updating.
- Test: existing suite plus manual/AppTest checklist.

- [ ] **Step 1: Run the full suite and record the known benchmark instability separately**

  Run: `python -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp-final`

  Expected: report exact pass/fail counts. Do not hide or rewrite a negative observed energy-savings result.

- [ ] **Step 2: Exercise the demo checklist**
  - Upload PDF, TXT, and Markdown.
  - Upload two files in one action.
  - Repeat a query for a cache hit.
  - Ask a longer analytical query and observe adaptive retrieval.
  - Delete one document and confirm its sources disappear.
  - Delete the final document and confirm the empty state.
  - Reload and run the benchmark charts.

- [ ] **Step 3: Update setup/demo notes if needed**
  - Document the delete/rebuild behavior, adaptive retrieval explanation, and benchmark interpretation.

