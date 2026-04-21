# Changelog

All notable changes to KonjoOS are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] — Sprint 6: Semantic Cache

### Added
- `konjoai/cache/semantic_cache.py` — two-level semantic cache: exact dict lookup O(1) + cosine-similarity scan O(n); `OrderedDict` LRU eviction; `threading.Lock` singleton with double-checked locking; `SemanticCacheEntry` dataclass; `_reset_cache()` test helper
- `konjoai/cache/__init__.py` — package with `SemanticCache`, `get_semantic_cache` exports
- `QueryResponse.cache_hit: bool = False` — backward-compatible field; true when response served from cache
- Settings: `cache_enabled: bool = False`, `cache_similarity_threshold: float = 0.95`, `cache_max_size: int = 500`
- `tests/unit/test_semantic_cache.py` — 21 tests covering exact/semantic hit, miss, LRU, invalidate, thread safety, sub-5ms latency gate ✅

### Changed
- `konjoai/retrieve/dense.py` — added `q_vec: np.ndarray | None = None` param; skips re-embed when pre-computed vector is supplied
- `konjoai/retrieve/hybrid.py` — added `q_vec: np.ndarray | None = None` param forwarded to `dense_search`
- `konjoai/api/routes/query.py` — Step 2b: embed → cache lookup → early return on hit; cache store on miss
- `konjoai/api/routes/ingest.py` — cache invalidated after BM25 rebuild to prevent stale hits

### Performance
- Cache hit path: < 5 ms (validated by `test_cache_hit_under_5ms` ✅)
- Zero LLM cost on repeated queries when `CACHE_ENABLED=true`

### Tests
- Suite: **226 passed, 0 failed** (up from 205)



## [Unreleased] — Sprint 12: Self-RAG

### Added
- `konjoai/retrieve/self_rag.py` — Self-RAG pipeline (Asai et al. 2023): `RetrieveDecision`, `RelevanceToken`, `SupportToken`, `UsefulnessToken` (IntEnum), `DocumentCritique`, `SupportScorer`, `SelfRAGPipeline`
- `tests/unit/test_self_rag.py` — 27 tests ✅

### Fixed
- `UsefulnessToken` changed from `str, Enum` to `IntEnum` to fix comparison operators
- Module-level import of `QueryIntent`, `classify_intent` from `konjoai.retrieve.router` (was local-only, caused `NameError` at runtime)
- `@patch("sentence_transformers.CrossEncoder")` in test fixtures to prevent SSL certificate errors during `SupportScorer` initialisation in sandboxed CI

### Tests
- Suite: **329 passed, 0 failed** (up from 302)

---

## [Unreleased] — Sprint 11: CRAG (Corrective RAG)

### Added
- `konjoai/retrieve/crag.py` — Corrective Retrieval-Augmented Generation pipeline: relevance grading, web fallback, knowledge refinement
- `tests/unit/test_crag.py` — tests ✅

### Tests
- Suite: **302 passed, 0 failed** (up from 280)

---

## [Unreleased] — Sprint 10: Adaptive Chunking

### Added
- `konjoai/chunk/adaptive.py` — adaptive chunk sizing based on document structure and content density
- `tests/unit/test_adaptive_chunking.py` — tests ✅

### Tests
- Suite: **280 passed, 0 failed** (up from 255)

---

## [Unreleased] — Sprint 7: Adapter Architecture

### Added
- `konjoai/adapters/base.py` — `BaseAdapter` abstract interface for retrieval backends
- `konjoai/adapters/registry.py` — `AdapterRegistry`: runtime registration and resolution of named adapters
- `konjoai/adapters/__init__.py`
- `tests/unit/test_adapters.py` — tests ✅

### Changed
- Retrieval backends refactored to implement `BaseAdapter` interface; backwards-compatible

### Tests
- Suite: **255 passed, 0 failed** (up from 226)

---

## [Unreleased] — RAGAS Eval Sprint

### Added
- `konjoai/eval/ragas_eval.py` — RAGAS evaluation harness: `threading.Lock` throttle, `asyncio.sleep` non-blocking gap, `RunConfig(timeout=600)`, `--mock` upper-bound mode, `--run-name`, `--n-samples` CLI; JSON results + gate check output
- `evals/corpus/eval_questions.json` — 3-question synthetic eval corpus (expandable to 25)
- Squish local LLM server support as RAGAS judge (`GENERATOR_BACKEND=squish`)
- **Mock upper-bound baseline PASSED** (v12, 2026-04-15): faithfulness=0.9333 ✅, context_precision=1.0000 ✅, context_recall=1.0000; run saved to `evals/runs/20260415T054040Z_mock_upper_bound_v12/`

### Fixed
- RAGAS harness: replaced `asyncio.Semaphore` (cross-loop bug) with `threading.Lock` + `asyncio.sleep` throttle
- RAGAS judge: upgraded from 1B model (LLMDidNotFinishException) to `Qwen2.5-7B-Instruct-int3`
- `max_tokens` bumped `2048 → 4096` to prevent truncated RAGAS structured-output responses
- SSL cert: `SSL_CERT_FILE` + `REQUESTS_CA_BUNDLE` → `/tmp/konjoai_certs.pem` (certifi bundle)

## [Unreleased] — Phase 2a

### Added
- Per-step pipeline telemetry (`konjoai/telemetry.py`): `StepTiming`, `PipelineTelemetry`, `timed()` context manager; zero new dependencies (stdlib `time` only)
- HyDE retrieval mode (`konjoai/retrieve/hyde.py`): hypothetical document embedding (Gao et al. 2022, arXiv:2212.10496); reuses existing generator singleton
- Query intent router (`konjoai/retrieve/router.py`): O(1) heuristic classification into `RETRIEVAL / AGGREGATION / CHAT`; CHAT queries short-circuit before any Qdrant call
- Vectro quantization bridge (`konjoai/embed/vectro_bridge.py`): INT8 compression with graceful float32 passthrough when Vectro is unavailable; logs compression ratio and mean cosine similarity
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
- Click CLI: `konjoai ingest`, `konjoai query`, `konjoai serve`, `konjoai status`
- Docker + docker-compose with Qdrant service wired
- `pydantic-settings` config with `.env` override support
- Unit tests: chunker shapes, encoder dtype contract, BM25, RRF
- Integration tests: ingest → embed → upsert pipeline smoke test
