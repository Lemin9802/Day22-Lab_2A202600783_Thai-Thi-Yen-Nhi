# Day22 Evidence Summary

## Task 1 - LangSmith RAG Pipeline

The RAG pipeline uses the PDF knowledge base `data/new_sdlc_vibe_coding.pdf`. The pipeline loads the PDF, splits it into chunks, builds a FAISS vectorstore, and uses a hybrid retriever with BM25 plus section-anchor retrieval for ambiguous topics such as AI agent components, harness, architecture, maintenance, and orchestrator skills.

Evidence files:

- `01_langsmith_traces.png`
- `01_langsmith_project_overview.png`

The LangSmith project shows more than 50 successful traces for the `rag-query` run, with no errors.

## Task 2 - Prompt Hub and A/B Routing

Two prompt versions were pushed to LangSmith Prompt Hub and then pulled back for use in the RAG pipeline.

- V1: `day22-sdlc-vibe-coding-v1-concise`
- V2: `day22-sdlc-vibe-coding-v2-structured`

The A/B router uses deterministic routing with `md5(request_id) % 2`.

Routing result:

- V1 count: 29
- V2 count: 21

Evidence files:

- `02_prompt_hub.png`
- `02_ab_routing_log.txt`

## Task 3 - RAGAS Evaluation

RAGAS was run on 50 QA pairs across both prompt versions, for a total of 100 evaluated samples.

Overall metrics:

- Faithfulness: 0.9717
- Answer relevancy: 0.8182
- Context recall: 0.9617
- Context precision: 0.8889

Metrics by prompt version:

| Metric | V1 | V2 |
|---|---:|---:|
| Faithfulness | 0.9543 | 0.9940 |
| Answer relevancy | 0.8472 | 0.7891 |
| Context recall | 0.9580 | 0.9654 |
| Context precision | 0.8333 | 0.9167 |

### V1 vs V2 Analysis

V1 produced higher answer relevancy because it was designed to be concise and direct. Its answers stayed closer to the exact user question and avoided extra explanation unless needed. This made V1 better for short factual questions.

V2 produced higher faithfulness, context recall, and context precision because it used a more structured engineering format. By organizing answers around definition, importance, and SDLC implication, V2 often used the retrieved context more completely and stayed strongly grounded in the source material. However, this extra structure sometimes made V2 less directly aligned with short reference answers, which explains its lower answer relevancy score.

Overall, V2 is stronger for grounded engineering explanations, while V1 is stronger for concise factual QA. Both prompt versions passed the faithfulness target by a wide margin.

Evidence files:

- `03_ragas_scores.png`
- `03_ragas_report.json`
- `03_ragas_rows.csv`

## Task 4 - Guardrails Validators

The Guardrails task implements two custom validators.

The PII validator detects and redacts:

- Email addresses
- Phone numbers
- SSN-like numbers
- Credit-card-like numbers

The JSON formatter repairs:

- Markdown fenced JSON
- Single-quoted Python-style dictionaries
- Trailing commas
- Invalid JSON through fallback JSON

Evidence files:

- `04_pii_demo_log.txt`
- `04_json_demo_log.txt`

## Code Quality Notes

The code is organized into reusable functions with docstrings across the RAG pipeline, prompt routing, RAGAS evaluation, data loading, and Guardrails validators. The implementation includes error handling and fallbacks, including PDF loading validation, section-anchor retrieval for ambiguous queries, JSON repair fallback, Guardrails `on_fail=FIX`, and compatibility handling for RAGAS evaluation.

## Running All Steps

The project supports running each step through `src/run_all.py` without editing source files.

Examples:

- `python src/run_all.py --step 1`
- `python src/run_all.py --step 2`
- `python src/run_all.py --step 3`
- `python src/run_all.py --step 4`
- `python src/run_all.py --step all`

For quick smoke tests of the long RAGAS step:

- `python src/run_all.py --step 3 --ragas-limit 1`
- `python src/run_all.py --step 3 --ragas-limit 2 --ragas-no-eval`

## Submission Links

GitHub repository:

- `https://github.com/Lemin9802/Day22-Lab_2A202600783_Thai-Thi-Yen-Nhi`

LangSmith project:

- `https://smith.langchain.com/o/e641930e-16b4-43e2-b2ca-6d49399af522/projects/p/884f2bd9-18f3-4def-94bf-41e33c3aaef4`

## run_all.py Verification

The project includes a single entrypoint at `src/run_all.py` for running each lab step without editing source files.

The following commands were tested successfully:

- `python src/run_all.py --help`
- `python src/run_all.py --step 4`
- `python src/run_all.py --step 3 --ragas-limit 1 --ragas-no-eval`

Verification result:

- `--help` displayed the available CLI options.
- `--step 4` successfully ran the Guardrails validator demo and regenerated the Task 4 evidence logs.
- `--step 3 --ragas-limit 1 --ragas-no-eval` successfully loaded the PDF knowledge base, built the retriever, generated V1 and V2 RAG samples, and skipped the long RAGAS metric evaluation as expected.

This confirms that the lab steps can be executed through `run_all.py` without modifying source files. The full `--step all` command is also supported, but Step 3 full RAGAS evaluation is intentionally long because it evaluates 100 samples with local Ollama.
