# Changelog

All notable changes to RagOS are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] — RAGAS Eval Sprint

### Added
- `ragos/eval/ragas_eval.py` — RAGAS evaluation harness: `threading.Lock` throttle, `asyncio.sleep` non-blocking gap, `RunConfig(timeout=600)`, `--mock` upper-bound mode, `--run-name`, `--n-samples` CLI; JSON results + gate check output
- `evals/corpus/eval_questions.json` — 3-question synthetic eval corpus (expandable to 25)
- Squish local LLM server support as RAGAS judge (`GENERATOR_BACKEND=squish`)
- **Mock upper-bound baseline PASSED** (v12, 2026-04-15): faithfulness=0.9333 ✅, context_precision=1.0000 ✅, context_recall=1.0000; run saved to `evals/runs/20260415T054040Z_mock_upper_bound_v12/`

### Fixed
- RAGAS harness: replaced `asyncio.Semaphore` (cross-loop bug) with `threading.Lock` + `asyncio.sleep` throttle
- RAGAS judge: upgraded from 1B model (LLMDidNotFinishException) to `Qwen2.5-7B-Instruct-int3`
- `max_tokens` bumped `2048 → 4096` to prevent truncated RAGAS structured-output responses
- SSL cert: `SSL_CERT_FILE` + `REQUESTS_CA_BUNDLE` → `/tmp/ragos_certs.pem` (certifi bundle)

## [Unreleased] — Phase 2a

### Added
- Per-step pipeline telemetry (`ragos/telemetry.py`): `StepTiming`, `PipelineTelemetry`, `timed()` context manager; zero new dependencies (stdlib `time` only)
- HyDE retrieval mode (`ragos/retrieve/hyde.py`): hypothetical document embedding (Gao et al. 2022, arXiv:2212.10496); reuses existing generator singleton
- Query intent router (`ragos/retrieve/router.py`): O(1) heuristic classification into `RETRIEVAL / AGGREGATION / CHAT`; CHAT queries short-circuit before any Qdrant call
- Vectro quantization bridge (`ragos/embed/vectro_bridge.py`): INT8 compression with graceful float32 passthrough when Vectro is unavailable; logs compression ratio and mean cosine similarity
- `QueryRequest.use_hyde` (`bool`, default `False`) — per-request HyDE override
- `QueryResponse.telemetry` (`dict | None`, default `None`) — per-step latency breakdown in ms
- `QueryResponse.intent` (`str`, default `"retrieval"`) — classifier result for observability
- Settings: `enable_hyde`, `enable_query_router`, `enable_telemetry`, `vectro_quantize`, `vectro_method`
- 5-step timed query pipeline: route → hyde → hybrid_search → rerank → generate

## [0.1.0] — 2026-04-14

### Added
- Production RAG pipeline with hybrid dense + sparse retrieval
- Document ingestion: PDF, Markdown, HTML, and 10+ code file types
- Chunking strategies: `recursive` (default) and `sentence_window`
- Embedding encoder: `sentence-transformers` with enforced `float32` contract
- Qdrant vector store integration (cosine similarity, auto-collection creation)
- BM25 sparse index via `rank-bm25`
- Reciprocal Rank Fusion (RRF) hybrid retrieval with configurable `alpha`
- Cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Generation backends: OpenAI, Anthropic, Squish/Ollama (OpenAI-compatible)
- RAGAS evaluation harness: faithfulness, answer relevancy, context precision, context recall
- FastAPI server: `POST /ingest`, `POST /query`, `POST /eval`, `GET /health`
- Click CLI: `ragos ingest`, `ragos query`, `ragos serve`, `ragos status`
- Docker + docker-compose with Qdrant service wired
- `pydantic-settings` config with `.env` override support
- Unit tests: chunker shapes, encoder dtype contract, BM25, RRF
- Integration tests: ingest → embed → upsert pipeline smoke test
