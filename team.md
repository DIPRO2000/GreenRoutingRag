# GreenRouting RAG — Team Work Plan

> **Team size:** 6 members  
> **Work structure:** 4 parallel workstreams  
> **Submission:** 13 November 2026  
> **Starting point:** GreenRouting base already copied into this repository

This plan uses four workstreams, not four-person groups. Two workstreams have two members each and two have one member each, so all six members have clear ownership. Replace `Member 1`–`Member 6` with actual names before sharing.

## What Already Exists in the GreenRouting Base

The copied base already contains working-looking modules for:

- Streamlit application and five dashboard tabs (`app.py`)
- PDF, TXT, and Markdown document processing (`greenrouting/document_processor.py`)
- Sentence Transformer embeddings and FAISS/NumPy vector search (`greenrouting/vector_store.py`)
- Ollama client plus an offline simulation fallback (`greenrouting/llm_interface.py`)
- Basic RAG prompt and source display (`greenrouting/rag_pipeline.py`)
- Semantic cache and multi-layer cache validation (`greenrouting/semantic_cache.py`, `cache_validator.py`)
- Rule-based complexity scoring and adaptive model routing (`complexity_analyzer.py`, `adaptive_router.py`)
- Heuristic quality/grounding checks (`quality_guard.py`, `evaluator.py`)
- SQLite persistence for documents, cache, logs, and benchmarks (`database.py`)
- Energy/carbon estimate module (`energy_tracker.py`)
- Standard RAG, RAG-with-cache, and GreenRouting benchmark modes (`benchmark.py`)
- Automated tests under `tests/`

This means our work is now mainly **stabilisation, correction, integration, evaluation, and project-specific extension**, not building every module from zero.

## Important Base Risks to Assign Immediately

These are part of Workstream 1 and Workstream 3, not optional cleanup:

1. Recreate the Python virtual environment; the copied environment was intentionally excluded and the original environment pointed to another computer.
2. Verify real Ollama mode before using benchmark numbers in the report.
3. Label current TDP energy values as estimates unless actual sensor sampling is implemented.
4. Fix the orchestrator latency calculation, which currently adds generation latency twice.
5. Add `query_text` to the response before writing query logs.
6. Make the offline embedding fallback use a stable hash, so persisted indexes still work after restarting Python.
7. Obtain the original author's permission and keep `CREDITS.md` up to date; the inherited base had no licence file.

## Project Goal

Deliver a local document chatbot that:

1. answers questions from uploaded documents;
2. reuses a validated answer to a very similar question;
3. chooses a cache, small model, or stronger model;
4. records clearly labelled energy and CO2e estimates; and
5. compares GreenRouting RAG with a standard local RAG baseline.

## First Rule: Work on the Existing File Boundaries

Each workstream owns the files listed below. Do not create a second parallel implementation of a module already present in `greenrouting/`. Members may edit another workstream's file only after agreeing on the interface with its owner.

Use one GitHub repository with one branch per workstream:

```text
feature/base-rag
feature/cache-router
feature/energy-dashboard
docs/benchmark-report
```

## Shared Result Contract

The current orchestrator already returns most of this data. Workstreams should preserve the names below or update all consumers together.

| Field | Meaning |
|---|---|
| `query_text` | Original user question; required for logs and benchmark records. |
| `answer` | Final answer shown to the user. |
| `sources` | Source document, page, chunk, and similarity information. |
| `mode` | `STANDARD_RAG`, `RAG_WITH_CACHE`, or `GREENROUTING`. |
| `route` | `cache`, `small_model`, or `strong_model`; add if missing. |
| `cache_status` | `MISS`, `HIT_VALID`, or a rejection reason. |
| `cache_score` | Similarity score used for the cache decision. |
| `model_used` | Ollama model name, cached response, or simulation label. |
| `complexity_score` | Rule-based query complexity score. |
| `routing_reason` | Plain-English explanation of the route. |
| `retry_count` | Number of quality-triggered escalations. |
| `latency_seconds` | End-to-end response time, measured once. |
| `prompt_tokens` / `output_tokens` | Ollama counts or clearly labelled estimates. |
| `energy_kwh` | Measured or modelled energy, with method label. |
| `emissions_g` | Estimated grams of CO2e. |
| `measurement_label` | `Modeled TDP Estimate`, `Cache Reuse`, or `N/A`. |
| `grounding_score` / `quality_score` | Heuristic evaluation scores, not proof of factual correctness. |

## Workstream 1 — Base Stabilisation, Documents, and RAG

| Item | Assignment |
|---|---|
| Members | Member 1 and Member 2 |
| Branch | `feature/base-rag` |
| Main responsibility | Make the inherited base RAG reliable on our machines. |
| Owned files | `greenrouting/document_processor.py`, `vector_store.py`, `rag_pipeline.py`, `llm_interface.py`, `config.py`, related tests |

### Member 1: Environment and Base RAG

- Create a new Python 3.10/3.11 virtual environment for this repository.
- Install `requirements.txt` and run the existing tests.
- Verify Ollama connectivity and confirm the exact small/strong model tags.
- Test the current end-to-end flow using the included sample documents.
- Fix only startup, import, model-availability, and obvious RAG failures.

### Member 2: Document and Vector Store Reliability

- Verify PDF, TXT, and Markdown ingestion with page/source metadata.
- Add duplicate-file protection using the existing SHA-256 document hash.
- Test persistence after closing and reopening the app.
- Replace the process-random `hash()` fallback embedding with a stable method, or document that the real Sentence Transformer is required.
- Add DOCX support only if the basic PDF/TXT flow is already stable.

### Deliverable

From a fresh environment, a member can upload the sample PDF/TXT files, ask a question, receive a source-grounded answer, restart the app, and still retrieve the indexed documents.

### Dependencies

- Can start immediately and is the first live blocker for every other team.
- Must publish the working startup commands and current response fields by **2 November**.
- Workstream 2 may use mock RAG responses while this workstream is fixing the base.

## Workstream 2 — Semantic Cache and Green Router

| Item | Assignment |
|---|---|
| Members | Member 3 and Member 4 |
| Branch | `feature/cache-router` |
| Main responsibility | Make the inherited cache and routing behaviour safe, explainable, and testable. |
| Owned files | `greenrouting/semantic_cache.py`, `cache_validator.py`, `complexity_analyzer.py`, `adaptive_router.py`, `quality_guard.py`, related tests |

### Member 3: Quality-Aware Semantic Cache

- Test the existing cosine-similarity threshold using identical, rephrased, and unrelated questions.
- Verify document-scope checks, stale-entry checks, source-support checks, and grounding thresholds.
- Confirm that a cache hit bypasses Ollama completely.
- Add cache bypass and cache-clear behaviour to the existing UI if needed.
- Make the cache result expose `cache_score` and a plain-English validation reason.
- Do not add Redis for the first submission; SQLite plus the current local/vector approach is enough for one local user.

### Member 4: Rule-Based Green Router

- Keep the current rule-based complexity analyser as the MVP router.
- Verify that simple questions go to the lightweight model and analytical questions go to the capable model.
- Verify the quality guard's escalation path when the lightweight answer is weak.
- Add the explicit `route` field if the current response only exposes `model_used`.
- Ensure an unavailable Ollama model fails quickly and shows a clear setup error instead of waiting unnecessarily.
- Call this **rule-based adaptive routing**, not trained machine-learning routing.

### Deliverable

A controlled test can demonstrate all three paths:

```text
validated cache hit → no LLM call
simple cache miss    → lightweight Ollama model
complex/weak answer  → capable model or quality escalation
```

### Dependencies

- Can start immediately with fake LLM responses and the current modules.
- Needs Workstream 1's confirmed Ollama model names for live testing.
- Must publish the final cache/router result fields to Workstream 3 by **4 November**.

## Workstream 3 — Energy Accounting and Streamlit Dashboard

| Item | Assignment |
|---|---|
| Members | Member 5 |
| Branch | `feature/energy-dashboard` |
| Main responsibility | Make energy, carbon, routing, and savings visible without overstating accuracy. |
| Owned files | `greenrouting/energy_tracker.py`, `greenrouting/orchestrator.py`, `app.py`, `config/config.yaml`, dashboard-related tests |

### Tasks

- Fix the orchestrator's double-counted latency before calculating energy.
- Add the missing `query_text` to persisted logs.
- Decide and document one measurement mode for the MVP: actual sensor sampling if genuinely implemented, or `Modeled TDP Estimate` using configured laptop power assumptions.
- Do not label a TDP calculation as a direct sensor reading.
- Ensure simulation mode is visibly labelled and excluded from final real-model claims.
- Keep the current Streamlit tabs and improve only the parts needed for the demo: Green RAG Assistant, Document Knowledge Base, Semantic Cache Inspector, Experimental Benchmarking, and Architecture & Methodology.
- Show route, model, cache result, latency, energy, CO2e, quality/grounding score, and source chunks.

### Deliverable

A live query produces one complete result/log row and the dashboard correctly summarises cache hits, model usage, latency, modelled energy, and estimated carbon savings.

### Dependencies

- Can build dashboard layout with existing sample data immediately.
- Needs Workstream 2's final result fields by **4 November**.
- Needs a stable live end-to-end pipeline by **8 November** for final screenshots and measurements.

## Workstream 4 — Tests, Benchmark, Report, and Presentation

| Item | Assignment |
|---|---|
| Members | Member 6 |
| Branch | `docs/benchmark-report` |
| Main responsibility | Produce evidence that the inherited base and our extensions work. |
| Owned files | `tests/`, `docs/benchmark.md`, `docs/results.csv`, report/presentation material |

### Tasks

- Prepare 3–5 related demonstration documents; avoid mixing unrelated topics.
- Create 25–40 questions containing direct facts, analytical questions, and rephrased duplicates.
- Record expected source pages or answer points for each question.
- Run three comparisons using the same documents and question set:
  - **Standard RAG:** capable model every time, cache disabled.
  - **RAG with cache:** capable model with ordinary cache behaviour.
  - **GreenRouting RAG:** validated cache plus adaptive routing and quality escalation.
- Record answer usefulness, source correctness, cache hits, selected model, latency, energy, CO2e, and retries.
- Clearly state whether each result came from real Ollama or simulation mode.
- Add regression tests for every base bug fixed by Workstreams 1–3.
- Write the teacher demo script and explain why this project is different from a general chatbot.

### Deliverable

A reproducible benchmark table, charts, screenshots, test results, limitations, and a 3–5 minute presentation.

### Dependencies

- Can begin immediately with question design, expected answers, and report structure.
- Needs integrated RAG, cache/router, and metrics by **9 November** to run the final benchmark.
- Must report unsafe cache hits, missing sources, negative energy savings, or simulation-mode results honestly.

## Dependency Map

```text
Existing GreenRouting base
          ↓
Workstream 1 stabilises environment, RAG, documents, and persistence
          ↓
Workstream 2 validates cache and router against the stable RAG interface
          ↓
Workstream 3 connects corrected metrics and dashboard to live responses
          ↓
Workstream 4 runs the benchmark and prepares the evidence

In parallel:
- Workstream 2 can use mock LLM responses before Workstream 1 is finished.
- Workstream 3 can use sample CSV/SQLite rows before live integration.
- Workstream 4 can prepare documents, questions, tests, and report sections immediately.
```

## Schedule and Integration Checkpoints

| Date | What must be ready | Responsible |
|---|---|---|
| 31 October | New environment plan, model choice, document set, permission/credits confirmation, shared result contract | Everyone |
| 1–2 November | Fresh environment works; base RAG answers from sample documents; existing tests attempted | Workstream 1 |
| 1–4 November | Cache tests, router tests, quality escalation tests, mock dashboard | Workstreams 2 and 3 |
| 3 November | First live integration: base RAG result passes through cache/router | Workstreams 1 + 2 |
| 4–6 November | Latency/logging/energy corrections and live Streamlit query flow | Workstreams 1 + 2 + 3 |
| 7 November | Multi-document upload and rephrased-query cache demo | Workstreams 1 + 2 |
| 8 November | MVP feature freeze; one complete demo works on a clean machine | Everyone |
| 9–10 November | Standard/RAG-cache/GreenRouting benchmark and bug fixes | Workstream 4 + affected owners |
| 11 November | Dashboard screenshots, charts, report and presentation draft | Workstreams 3 + 4 |
| 12 November | Full rehearsal, regression test, credits/licence check, submission packaging | Everyone |
| 13 November | Submit and demonstrate | Everyone |

## Definition of Ready Before Integration

| Module | Ready only when it can show this |
|---|---|
| Base RAG | A known question receives an answer supported by an uploaded document and source. |
| Document store | The same index can be reopened after restarting the app. |
| Cache | A rephrased duplicate hits; an unrelated question misses; context mismatch is rejected. |
| Router | The route changes for controlled simple and complex questions. |
| Energy tracker | Each value says whether it is direct, modelled, or simulated. |
| Dashboard | One live result appears with route, model, latency, energy, CO2e, and sources. |
| Benchmark | The same questions can be run against all three modes and exported. |

## Scope Rules for the 13 November MVP

### Must finish

- Use the inherited base with attribution/permission.
- Recreate the environment and verify the app on at least one clean machine.
- Local PDF/TXT/MD RAG with multiple documents.
- Existing FAISS/NumPy retrieval and semantic cache, after validation.
- Rule-based cache/small-model/strong-model routing.
- Corrected latency and clearly labelled energy/CO2e estimates.
- Streamlit dashboard and three-mode benchmark.
- Regression tests and presentation evidence.

### Only if the MVP is stable

- DOCX ingestion.
- Actual NVML power sampling.
- Better source-support checks.
- Cache duplicate detection and document deletion/reindexing.

### Later semester

- Redis and multi-user storage.
- Scikit-learn trained routing model.
- Electricity Maps live carbon API.
- RAGAS automated evaluation.
- OCR for scanned PDFs.
- Optuna optimisation.
- Authentication and deployment.

If a new feature does not improve the 13 November live demo or its evidence, place it in future work instead of risking the base system.

## Team Communication Rules

1. Post a short update every evening: completed, next task, blocker, and branch/file changed.
2. Tell the team before changing the shared result contract.
3. Do not commit private uploaded documents, API keys, `.env` files, databases, or generated indexes.
4. Use mock data when a dependency is not ready so parallel work continues.
5. Merge small, tested changes; do not wait until the final day.
6. Every benchmark number must record real Ollama, modelled, or simulation mode.
7. Do not claim inherited base code as newly written by the whole group.

## Suggested Daily Update Format

```text
Name / Workstream:
Completed today:
Working on next:
Blocked by / needs from another team:
Branch or file changed:
Test or demo evidence:
```
