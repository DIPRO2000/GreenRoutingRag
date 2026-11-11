# GreenRouting RAG

## An Energy-Aware Document Question Answering System

**GreenRouting RAG** is a local document chatbot that tries to answer questions with less unnecessary AI computation. A user uploads documents, asks a question in plain language, and receives an answer based on those documents. The system also shows the estimated energy and carbon impact of answering the question.

The main idea is simple: do not use the biggest AI model for every question. First reuse a safe previous answer when a very similar question has already been asked. If a new answer is needed, use a smaller model for simple questions and a stronger model only when necessary.

> **Submission target:** 13 November 2026

> **New team member?** Start with the command-by-command [setup manual](SETUP.md).

## Problem We Are Solving

Normal RAG (Retrieval-Augmented Generation) systems generally run document search and an AI model for every question. In a college, office, or company, people often ask the same policy questions repeatedly in different words. This repeats computation, increases response time, and uses electricity unnecessarily.

GreenRouting RAG makes the decision process visible and measurable. It aims to reduce **avoidable** local computation while keeping answers useful.

## What the Prototype Will Do by 13 November

The submission version will be a working local web application with the following flow:

```text
Upload PDFs / TXT / Markdown files
            ↓
Ask a question in normal language
            ↓
Check for a safe, similar past question
   ↓ cache hit                 ↓ cache miss
Reuse old answer          Search uploaded documents
   ↓                            ↓
Save energy          Select small or stronger local model
            ↓
Answer with source information
            ↓
Show time, energy estimate, CO2e estimate, and savings
```

### Planned MVP Features

| Feature | What it means in simple words |
|---|---|
| Document upload | Upload text-based PDF, TXT, or Markdown files to create a knowledge base. DOCX support is planned later. |
| Document chatbot | Ask questions and receive answers based on the uploaded documents. |
| Semantic cache | If a similar question was answered earlier, reuse that answer instead of making the AI work again. |
| Green routing | Decide between cache, a small AI model, and a stronger AI model. |
| Local AI | Use Ollama models on the laptop rather than requiring a paid cloud AI API. |
| Energy and carbon log | Record estimated energy use and estimated CO2e for each question. |
| Green dashboard | Show total queries, cache hits, latency, energy, CO2e, and estimated savings against a standard baseline. |
| Benchmark | Test 25–40 prepared questions and compare normal RAG with GreenRouting RAG. |

## What We Will Demonstrate

1. Upload a set of documents, such as college policies, HR rules, or green-computing reports.
2. Ask a question such as: `What is the minimum attendance requirement?`
3. The system retrieves relevant passages and generates an answer.
4. Ask the same thing in another way: `How much attendance do I need?`
5. The system identifies a similar earlier question, reuses the stored answer, and displays: **Cache hit — AI call avoided**.
6. Open the dashboard to compare GreenRouting RAG with a standard setup that always uses the stronger model.

## Why This Is Different From ChatGPT

ChatGPT can analyse uploaded documents and will often be faster or produce stronger general answers. GreenRouting RAG is not intended to replace it. Our project focuses on a different engineering question:

> How can an organisation answer repeated questions from its own documents while reducing unnecessary local AI computation and showing the energy/carbon trade-off?

| ChatGPT | GreenRouting RAG |
|---|---|
| General cloud AI assistant | Purpose-built local document assistant |
| Its internal routing and energy use are not controlled by this project | Routing decision and local measurement are visible in the dashboard |
| Good for broad, one-off questions | Designed for a known document collection and repeated questions |
| File handling depends on product/account settings | Documents can remain on the local machine when only local models are used |

We will **not** claim that a laptop is always greener than a cloud AI service. Our valid claim is narrower and testable: **GreenRouting RAG reduces estimated energy and CO2e compared with our own standard local baseline that uses the stronger model for every question.**

## Technology Stack

| Tool / technology | Why we need it |
|---|---|
| Python | Main programming language; connects all parts of the project. |
| Streamlit | Creates the simple web interface and dashboard. |
| Ollama | Runs AI language models locally on the laptop. |
| Small and medium local models | Lets the router choose a lower- or higher-compute answer path. Examples: a 2–3B model and a 7B quantised model. |
| Sentence Transformers | Converts questions into numerical meaning vectors so similar questions can be recognised. |
| FAISS | Fast local search for document chunks and cached questions/answers. |
| LangChain (light use) | Helps organise document loading, splitting, retrieval, and prompts. |
| Energy tracker | Models energy and CO2e from configured CPU/GPU power assumptions and generation time. |
| pandas | Stores and analyses query logs and benchmark results. |
| Matplotlib or Plotly | Creates charts for the dashboard/report. |
| PyPDF / docx reader | Extracts text from PDFs and Word documents. |

### File Formats for the First Version

| Supported now | Later enhancement |
|---|---|
| Text-based PDF | Scanned/image-only PDFs using OCR |
| TXT | Images and handwritten notes |
| DOCX and scanned PDFs | Excel/CSV data analysis mode |

## Hardware and Software Requirements

The application can run without a powerful gaming GPU, but the experience becomes slower. An SSD is recommended; an HDD works but increases model-loading and first-time document-indexing time.

| Requirement | Minimum workable setup | Recommended for a smooth demo |
|---|---|---|
| Operating system | 64-bit Windows 10/11 or Linux | Windows 11 or Ubuntu/Linux |
| CPU | Modern 4-core processor | Modern 6-core processor or better |
| RAM | 8 GB (small models, may be slow) | 16 GB or more |
| GPU | Not required; CPU mode works slowly | NVIDIA GPU with 6 GB+ VRAM; 8 GB+ is better |
| Storage | 20 GB free | 40 GB free on SSD |
| Internet | Needed initially to install packages/models | Needed initially; app can work locally afterwards |
| Python | Python 3.10 or 3.11 | Python 3.11 |
| Local model runner | Ollama | Ollama with GPU support |

### Model Plan

- **Small model:** 2–3 billion parameters, quantised; used for simple queries.
- **Stronger local model:** around 7 billion parameters, quantised; used for difficult queries when the GPU has sufficient memory.
- **CPU-only fallback:** use only the small model and clearly label it as a slower demo mode.

## Step-by-Step Roadmap

### Phase 1 — Project Setup and Basic RAG

**Goal:** A user uploads documents and gets an answer with a source.

1. Create a clean project structure and Python environment.
2. Install Ollama, download a small local model, and verify it runs.
3. Add PDF/TXT/Markdown text extraction.
4. Split documents into small searchable chunks and store them in FAISS.
5. Build the Streamlit upload page and chat page.
6. Return answers with the most relevant document source/chunk.

### Phase 2 — Semantic Cache

**Goal:** Avoid repeating equivalent questions.

1. Embed every new user question with Sentence Transformers.
2. Store the question, answer, source, and timestamp in a local FAISS cache.
3. Check similarity before calling the RAG pipeline.
4. Display cache hit/miss and similarity score.
5. Use a conservative threshold so unrelated questions are not given a cached answer.

### Phase 3 — Green Routing and Measurement

**Goal:** Make a transparent choice that saves computation.

1. Define three paths: cache, small model, stronger model.
2. Route based on cache confidence, retrieved-document confidence, and simple question rules.
3. Record response time, chosen path, prompt/output length, and model name.
4. Estimate energy/CO2e using configured CPU/GPU power assumptions; label all results as modeled estimates.

### Phase 4 — Dashboard and Benchmark

**Goal:** Produce evidence for the presentation and report.

1. Add a dashboard for query count, cache-hit rate, model usage, time, energy, CO2e, and savings.
2. Prepare 25–40 test questions, including repeated/rephrased questions.
3. Run each set against two modes:
   - **Standard baseline:** always use the stronger model and never reuse a cached answer.
   - **GreenRouting:** cache first, then route to a suitable model.
4. Create charts and a small results table for the report.
5. Prepare a 3–5 minute demonstration script and screenshots.

## Submission Schedule

| Date range | Main output |
|---|---|
| 29 October – 2 November | Local model setup, document upload, basic RAG chatbot |
| 3 – 6 November | Semantic cache and persistent query logs |
| 7 – 9 November | Green routing and energy/carbon recording |
| 10 – 11 November | Dashboard and benchmark runs |
| 12 November | Bug fixes, screenshots, report material, demo rehearsal |
| 13 November | Submit working MVP, benchmark results, and presentation |

## Features for the Later Semester

These are valuable extensions but are not required for the first submission:

- Live electricity-grid carbon intensity through Electricity Maps or a regional source.
- Learned routing model instead of rule-based routing.
- Automated answer-quality evaluation using RAGAS.
- Redis or a database for multiple users and larger caches.
- Cache expiry, eviction policy, and user feedback on cached answers.
- OCR for scanned PDFs.
- Authentication, role-based document access, and cloud/on-premise deployment.
- More detailed lifecycle and hardware-carbon accounting.

## Limitations and Honest Claims

| Limitation | How we will handle it |
|---|---|
| CO2e is an estimate, not an exact physical measurement | Show the formula/assumptions and label the dashboard values as estimated. |
| Short AI queries are difficult to measure perfectly | Compare many queries and report totals/averages, not exaggerated single-query conclusions. |
| A semantic cache can return a wrong old answer | Use a strict similarity threshold, show cache status, and allow the user to bypass cache. |
| Local smaller models may give weaker answers | Route difficult questions to the stronger model and compare answer quality in the benchmark. |
| Laptop/CPU-only inference can be slow | Use small quantised models and keep the demonstration document set manageable. |
| Text extraction from scanned PDFs is poor | Support text-based PDFs in version 1; add OCR later. |
| Local RAG only knows uploaded documents | Clearly say it is a document assistant, not an internet search engine. |
| HDD storage can slow startup/index building | Load model/index once and recommend an SSD for best performance. |

## Open-Source Repositories Already Downloaded

### 1. GreenRAG (`konstvr/GreenRAG`)

**What it provides:** a notebook-based RAG example with local Ollama models and energy/CO2 monitoring.

**Licence:** MIT. It can legally be reused and modified provided its copyright and licence notice are retained.

**How it can help us:**

- Review its energy-monitoring approach as background; this project currently uses its own modeled estimator.
- Reuse or adapt small, clearly attributed parts only if needed.
- Use it as a reference while we build a cleaner Streamlit application.

**What we will build ourselves:** the document upload UI, semantic cache integration, green router, dashboard, benchmark, and project documentation. We must not submit the notebook unchanged.

### 2. Semantic Prompt Cache (`renswickd/semantic-prompt-cache`)

**What it provides:** a modular semantic-cache/RAG example using Sentence Transformers, FAISS, LangChain, and Streamlit.

**Licence status:** no `LICENSE` file was found in the downloaded repository. A public repository without a licence should be treated as reference material only unless the author gives written permission or adds a licence.

**How it can help us safely:**

- Study its project structure and cache flow.
- Use the ideas to design our own cache module.
- Test its concepts, but write our implementation independently.

**Important quality note:** cache similarity scoring and thresholds must be tested carefully before use. A cache should never be accepted only because an example repository has it.

## Academic Integrity and Attribution

1. Create and submit a new repository for GreenRouting RAG.
2. Keep a `CREDITS.md` or acknowledgement section naming both repositories and their links.
3. Retain the MIT licence/copyright notice if any GreenRAG code is copied or modified.
4. Do not copy code from the unlicensed semantic-cache repository into the submitted project without permission.
5. Explain clearly in the report what the team implemented: routing policy, dashboard, measurement logic, benchmark, and evaluation.

## Suggested Teacher Pitch

> GreenRouting RAG is a local, energy-aware document chatbot. It answers questions from uploaded documents, but avoids unnecessary AI processing by safely reusing similar past answers and selecting an appropriate local model. It makes the energy and estimated carbon impact visible and compares this approach with a normal RAG system that always uses a stronger model.

## Definition of Success for the 13 November Submission

The project is successful if the team can demonstrate all of the following reliably:

- Upload a text-based document.
- Ask a document-grounded question and show its source.
- Ask a rephrased question and demonstrate a safe cache hit.
- Show a routing decision and the selected model/path.
- Show dashboard values for energy/CO2e estimates and baseline savings.
- Present a small benchmark proving the trade-off between answer quality, response time, and estimated environmental impact.
