# Day22 - LLMOps: Prompt Versioning, RAG Evaluation, and Guardrails

## Submission Information

Student: Thai Thi Yen Nhi  
Student ID: 2A202600783  
GitHub Repository: https://github.com/Lemin9802/Day22-Lab_2A202600783_Thai-Thi-Yen-Nhi  
LangSmith Project: https://smith.langchain.com/o/e641930e-16b4-43e2-b2ca-6d49399af522/projects/p/884f2bd9-18f3-4def-94bf-41e33c3aaef4

This repository contains the completed Day22 Track 2 LLMOps Prompt Versioning lab. The lab implements a PDF-based RAG pipeline, LangSmith tracing, Prompt Hub versioning, deterministic A/B routing, RAGAS evaluation, and Guardrails custom validators.

---

## Completed Tasks

## Task 1 - LangSmith RAG Pipeline

Implemented in:

- `src/01_langsmith_rag_pipeline.py`
- `src/utils/data_loader.py`
- `src/utils/llm_factory.py`
- `src/qa_pairs.py`

Main features:

- Loads the PDF knowledge base from `data/new_sdlc_vibe_coding.pdf`.
- Splits the PDF into text chunks.
- Builds a FAISS vectorstore.
- Uses hybrid retrieval with FAISS + BM25.
- Adds section-anchor retrieval for ambiguous topics such as AI agent components, harness, architecture, maintenance, and orchestrator skills.
- Builds a RAG chain with LangChain Expression Language.
- Uses `@traceable` to send RAG runs to LangSmith.
- Runs 50 PDF-grounded questions.

Evidence:

- `evidence/01_langsmith_traces.png`
- `evidence/01_langsmith_project_overview.png`

LangSmith result:

- Project: `day22-lab`
- More than 50 successful traces
- Error rate: 0%

---

## Task 2 - Prompt Hub and A/B Routing

Implemented in:

- `src/02_prompt_hub_ab_routing.py`

Main features:

- Creates two semantically different prompt versions.
- Pushes both prompts to LangSmith Prompt Hub.
- Pulls both prompts back from Prompt Hub before use.
- Routes requests deterministically with `md5(request_id) % 2`.
- Logs every request ID, selected version, question, and answer.

Prompt versions:

- V1: `day22-sdlc-vibe-coding-v1-concise`
- V2: `day22-sdlc-vibe-coding-v2-structured`

Prompt behavior:

- V1: concise factual RAG answer.
- V2: structured engineering RAG answer.

Routing result:

- V1 count: 29
- V2 count: 21

Evidence:

- `evidence/02_prompt_hub.png`
- `evidence/02_ab_routing_log.txt`

---

## Task 3 - RAGAS Evaluation

Implemented in:

- `src/03_ragas_evaluation.py`

Main features:

- Evaluates all 50 QA pairs through both prompt versions.
- Total evaluated samples: 100.
- Uses `SingleTurnSample` with `user_input`, `response`, `retrieved_contexts`, and `reference`.
- Evaluates faithfulness, answer relevancy, context recall, and context precision.
- Saves JSON, CSV, and chart evidence.

Full RAGAS result:

| Metric | Overall | V1 | V2 |
|---|---:|---:|---:|
| Faithfulness | 0.9717 | 0.9543 | 0.9940 |
| Answer relevancy | 0.8182 | 0.8472 | 0.7891 |
| Context recall | 0.9617 | 0.9580 | 0.9654 |
| Context precision | 0.8889 | 0.8333 | 0.9167 |

V1 vs V2 analysis:

V1 achieved higher answer relevancy because it was designed to be concise and direct. Its responses stayed closer to the exact user question and avoided extra explanation unless needed.

V2 achieved higher faithfulness, context recall, and context precision because it used a structured engineering format. It often used more retrieved context and remained strongly grounded in the source material.

Both prompt versions passed the faithfulness target by a wide margin.

Evidence:

- `evidence/03_ragas_scores.png`
- `evidence/03_ragas_report.json`
- `evidence/03_ragas_rows.csv`
- `data/ragas_report.json`
- `evidence/README.md`

---

## Task 4 - Guardrails Validators

Implemented in:

- `src/04_guardrails_validator.py`

Main features:

### Custom PII Validator

Detects and redacts:

- Email addresses
- Phone numbers
- SSN-like numbers
- Credit-card-like numbers

Uses:

- `PIIDetector(on_fail=OnFailAction.FIX)`

Evidence:

- `evidence/04_pii_demo_log.txt`

### Custom JSON Formatter Validator

Repairs:

- Markdown fenced JSON
- Single-quoted Python-style dictionaries
- Trailing commas
- Invalid JSON through fallback JSON

Uses:

- `JSONFormatter(on_fail=OnFailAction.FIX)`

Evidence:

- `evidence/04_json_demo_log.txt`

---

## Repository Structure

Main files and folders:

- `data/new_sdlc_vibe_coding.pdf`  
  PDF knowledge base used as the single source of truth.

- `data/ragas_report.json`  
  Full RAGAS report saved in the data directory.

- `evidence/README.md`  
  Evidence summary, V1 vs V2 analysis, and run_all verification.

- `evidence/01_langsmith_traces.png`  
  LangSmith trace list screenshot.

- `evidence/01_langsmith_project_overview.png`  
  LangSmith project overview screenshot.

- `evidence/02_prompt_hub.png`  
  Prompt Hub screenshot showing both prompt versions.

- `evidence/02_ab_routing_log.txt`  
  A/B routing log showing deterministic routing results.

- `evidence/03_ragas_scores.png`  
  RAGAS score chart.

- `evidence/03_ragas_report.json`  
  Full RAGAS evidence report.

- `evidence/03_ragas_rows.csv`  
  Row-level RAGAS results.

- `evidence/04_pii_demo_log.txt`  
  Guardrails PII validator demo log.

- `evidence/04_json_demo_log.txt`  
  Guardrails JSON formatter demo log.

- `src/01_langsmith_rag_pipeline.py`  
  Task 1 RAG pipeline with LangSmith tracing.

- `src/02_prompt_hub_ab_routing.py`  
  Task 2 Prompt Hub push/pull and A/B routing.

- `src/03_ragas_evaluation.py`  
  Task 3 RAGAS evaluation.

- `src/04_guardrails_validator.py`  
  Task 4 Guardrails custom validators.

- `src/qa_pairs.py`  
  50 PDF-grounded QA pairs.

- `src/run_all.py`  
  Unified entrypoint for running lab steps.

- `src/utils/data_loader.py`  
  PDF loading, text splitting, FAISS, hybrid retriever, and section-anchor retriever.

- `src/utils/llm_factory.py`  
  LLM and embedding provider factory.

---

## Knowledge Base

The original text knowledge base was replaced with a PDF-only knowledge base:

- `data/new_sdlc_vibe_coding.pdf`

The old `data/knowledge_base.txt` was intentionally removed because this submission uses the PDF as the single source of truth.

---

## Environment

The implementation was run with:

- Python 3.12
- Ollama local provider
- LLM model: `qwen2.5:7b`
- Embedding model: `nomic-embed-text`
- LangSmith project: `day22-lab`

Required environment variables are documented in `.env.example`.

The `.env` file is intentionally excluded from Git and was not committed.

---

## Installation

Create and activate a virtual environment, then install dependencies:

    pip install -r requirements.txt

For local Ollama execution, make sure the required models are available:

    ollama pull qwen2.5:7b
    ollama pull nomic-embed-text

---

## How to Run Each Task

Run individual task files:

    python src/01_langsmith_rag_pipeline.py
    python src/02_prompt_hub_ab_routing.py
    python src/03_ragas_evaluation.py
    python src/04_guardrails_validator.py

Task 3 full RAGAS evaluation can take around one hour with local Ollama because it evaluates 100 samples.

For a quick Task 3 smoke test:

    python src/03_ragas_evaluation.py --limit 1
    python src/03_ragas_evaluation.py --limit 2 --no-eval

---

## run_all.py Entrypoint

The repository includes a unified entrypoint:

- `src/run_all.py`

Supported commands:

    python src/run_all.py --help
    python src/run_all.py --step 1
    python src/run_all.py --step 2
    python src/run_all.py --step 3
    python src/run_all.py --step 4
    python src/run_all.py --step all

For quick RAGAS smoke tests:

    python src/run_all.py --step 3 --ragas-limit 1
    python src/run_all.py --step 3 --ragas-limit 2 --ragas-no-eval

The following commands were tested successfully:

- `python src/run_all.py --help`
- `python src/run_all.py --step 4`
- `python src/run_all.py --step 3 --ragas-limit 1 --ragas-no-eval`

---

## Evidence Index

| Task | Evidence File | Description |
|---|---|---|
| Task 1 | `evidence/01_langsmith_traces.png` | LangSmith trace list showing RAG runs |
| Task 1 | `evidence/01_langsmith_project_overview.png` | Project overview showing trace count and error rate |
| Task 2 | `evidence/02_prompt_hub.png` | Prompt Hub screenshot showing V1 and V2 prompts |
| Task 2 | `evidence/02_ab_routing_log.txt` | Deterministic A/B routing log |
| Task 3 | `evidence/03_ragas_scores.png` | RAGAS score chart |
| Task 3 | `evidence/03_ragas_report.json` | Full RAGAS report |
| Task 3 | `evidence/03_ragas_rows.csv` | Row-level RAGAS results |
| Task 4 | `evidence/04_pii_demo_log.txt` | PII validator demo log |
| Task 4 | `evidence/04_json_demo_log.txt` | JSON formatter demo log |
| Bonus | `evidence/README.md` | Summary, V1/V2 analysis, and run_all verification |

---

## Code Quality Notes

The code is organized into reusable functions with docstrings across:

- PDF loading and retrieval
- RAG pipeline setup
- Prompt Hub push/pull
- Deterministic A/B routing
- RAGAS evaluation
- Guardrails validators
- Unified `run_all.py` execution

The implementation includes error handling and fallbacks, including:

- PDF file validation
- Empty PDF text validation
- Hybrid retrieval and section-anchor retrieval
- JSON repair and fallback JSON output
- Guardrails `on_fail=OnFailAction.FIX`
- RAGAS compatibility handling for evaluation calls

---

## Final Submission Links

GitHub repository:

- https://github.com/Lemin9802/Day22-Lab_2A202600783_Thai-Thi-Yen-Nhi

LangSmith project:

- https://smith.langchain.com/o/e641930e-16b4-43e2-b2ca-6d49399af522/projects/p/884f2bd9-18f3-4def-94bf-41e33c3aaef4
