# Eval Corpus — RagOS Retrieval QA

## Overview

25 factoid question-answer pairs covering core RAG and information retrieval concepts.
Used by `python -m ragos.eval.ragas_eval` to run RAGAS baseline evaluations.

## Format

Each entry in `eval_questions.json` is:

```json
{
  "id": 1,
  "question": "...",
  "ground_truth": "...",
  "context_docs": ["passage 1", "passage 2"]
}
```

| Field | Type | Description |
|---|---|---|
| `id` | `int` | Unique 1-indexed item identifier |
| `question` | `str` | The factoid question submitted to the RAG pipeline |
| `ground_truth` | `str` | Authoritative correct answer used by RAGAS `context_recall` and `--mock` mode |
| `context_docs` | `list[str]` | Pre-retrieved passages supplied as the RAG context window |

## Topics

| ID Range | Topic |
|---|---|
| 1–2 | Reciprocal Rank Fusion (RRF) |
| 3–4 | Hybrid search |
| 5–6 | BM25 and IDF |
| 7–9 | Sentence embeddings and all-MiniLM-L6-v2 |
| 10–11 | Cross-encoder reranking and ms-marco-MiniLM-L-6-v2 |
| 12–13 | RAG definition and pipeline stages |
| 14–15 | HyDE and query intent classification |
| 16 | Query routing benefits |
| 17–19 | RAGAS metrics (faithfulness, context precision, answer relevancy) |
| 20–21 | Text chunking and chunk overlap |
| 22–23 | Late interaction / ColBERT and MaxSim |
| 24–25 | Vector databases and Qdrant |

## Usage

Run a mock evaluation (uses `ground_truth` as the answer — tests harness plumbing):

```bash
cd /path/to/RagOS
python -m ragos.eval.ragas_eval \
  --run-name mock_upper_bound \
  --corpus evals/corpus/eval_questions.json \
  --mock
```

Run a real evaluation against the configured generator backend:

```bash
python -m ragos.eval.ragas_eval \
  --run-name baseline_v010 \
  --corpus evals/corpus/eval_questions.json \
  --n-samples 25
```

Results are written to `evals/runs/{timestamp}_{run_name}/`:
- `scores.json` — per-metric mean scores
- `config.json` — run configuration snapshot

## Provenance

Corpus authored by Konjo AI Research (konjo.ai) for the RagOS v0.2.0 evaluation sprint.
Context documents are purpose-written synthetic passages covering each topic.
Ground truth answers are manually verified.
