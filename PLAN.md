# Kyro — Master Plan

> **ቆንጆ** — Beautiful. **根性** — Fighting spirit. **康宙** — Health of the universe. **खोजो** — Search and discover.
> *Make it konjo — build, ship, rest, repeat.*

**Strategic document:** See [`KORE_PLAN.md`](KORE_PLAN.md) for full market analysis, sprint roadmap, and licensing recommendation.

---

## Current State: Sprint 6 Complete (v0.3.0)

- **Tests:** 226 passing, 0 failing
- **Branch:** `main`
- **Stack:** FastAPI + HyDE + ColBERT + hybrid search + RAGAS + Vectro bridge + streaming + semantic cache

---

## Completed Sprints

| Sprint | Version | Focus | Status |
|---|---|---|---|
| 1–5 | v0.2.5 | Foundation: telemetry, routing, HyDE, ColBERT, RAGAS | ✅ |
| 6 | v0.3.0 | Semantic cache: sub-5ms cached responses | ✅ 226 tests |

---

## Active Sprint: Sprint 7 — Adapter Architecture

**Goal:** Sub-5ms cached responses. Eliminate LLM cost for near-duplicate queries (20–40% of prod traffic).

### Implementation Checklist — Sprint 7

| # | File | Change | Status |
|---|---|---|---|
| 1 | `konjoai/adapters/__init__.py` | Package init exporting protocols | ✅ |
| 2 | `konjoai/adapters/base.py` | `VectorStoreAdapter`, `EmbedderAdapter`, `GeneratorAdapter`, `RetrieverAdapter` protocols | ✅ |
| 3 | `tests/unit/test_adapters.py` | Protocol conformance + duck-typing tests | ✅ |

### Sprint 7 Gates

1. All existing 226 tests still pass.
2. `VectorStoreAdapter`, `EmbedderAdapter`, `GeneratorAdapter`, `RetrieverAdapter` protocols defined.
3. Existing `get_store()`, `get_encoder()`, `get_generator()` singletons satisfy the protocols (checked via `isinstance` + `runtime_checkable`).

---

## Phase 1 → Phase 2 Transition: Active Sprint 10 — Adaptive Chunking

### Implementation Checklist — Sprint 10

| # | File | Change | Status |
|---|---|---|---|
| 1 | `konjoai/ingest/adaptive_chunker.py` | `QueryComplexityScorer`, `MultiGranularityChunker`, `AdaptiveRetriever` | ✅ |
| 2 | `konjoai/config.py` | `adaptive_chunking_enabled`, `chunk_sizes_hierarchy` | ✅ |
| 3 | `tests/unit/test_adaptive_chunker.py` | Complexity scoring, multi-granularity, retrieval dispatch | ✅ |

### Implementation Checklist — Sprint 11: CRAG

| # | File | Change | Status |
|---|---|---|---|
| 1 | `konjoai/retrieve/crag.py` | `RelevanceGrade`, `grade_documents()`, `CRAGPipeline` | ✅ |
| 2 | `konjoai/config.py` | `enable_crag`, `crag_relevance_threshold` | ✅ |
| 3 | `konjoai/api/routes/query.py` | Wire CRAG step between hybrid_search and rerank | ✅ |
| 4 | `tests/unit/test_crag.py` | Grade relevant/irrelevant/ambiguous, fallback, threshold | ✅ |

### Implementation Checklist — Sprint 12: Self-RAG

| # | File | Change | Status |
|---|---|---|---|
| 1 | `konjoai/retrieve/self_rag.py` | `SelfRAGTokens`, `SelfRAGCritic`, `SelfRAGPipeline` | ✅ |
| 2 | `konjoai/config.py` | `enable_self_rag`, `self_rag_max_iterations` | ✅ |
| 3 | `konjoai/api/routes/query.py` | Wire Self-RAG as optional post-generate critique | ✅ |
| 4 | `tests/unit/test_self_rag.py` | Retrieve/no-retrieve decision, critique tokens, iteration loop | ✅ |

---

## Sprint Roadmap Summary (Production Release Plan)

| Sprint | Version | Phase | Focus | Gate |
|---|---|---|---|---|
| 1–5 | v0.2.5 | — | Foundation: telemetry, routing, HyDE, ColBERT, RAGAS | ✅ 205 tests |
| 6 | v0.3.0 | — | Semantic cache (sub-5ms cached responses) | ✅ 226 tests |
| **7** | **v0.3.5** | **P1** | **Adapter architecture (swap any backend)** | **✅ Active** |
| 8 | v0.4.0 | P1 | Async pipeline + connection pooling (3× throughput) | ⬜ |
| 9 | v0.5.0 | P1 | Streaming SSE (already exists; harden + OTel hooks) | ⬜ |
| 10 | v0.5.5 | P2 | Adaptive chunking (query-aware granularity) | ✅ |
| 11 | v0.6.0 | P2 | CRAG — retrieval critique + corrective fallback | ✅ |
| 12 | v0.7.0 | P2 | Self-RAG — reflection tokens + critique loop | ✅ |
| 13 | v0.7.5 | P3 | Query decomposition (multi-hop fan-out) | ⬜ |
| 14 | v0.8.0 | P3 | Agentic RAG — ReAct loop | ⬜ |
| 15 | v0.8.5 | P3 | Lightweight GraphRAG (NetworkX + Louvain) | ⬜ |
| 16 | v0.8.7 | P4 | OTel + Prometheus + Grafana | ⬜ |
| 17 | v0.9.0 | P4 | Multi-tenancy + JWT | ⬜ |
| 18 | v0.9.5 | P4 | Auth + rate limiting | ⬜ |
| 19 | v0.9.8 | P5 | Python SDK + MCP server | ⬜ |
| 20 | v1.0.0 | P5 | Helm chart + PyPI + Docs site | ⬜ |

---

## The Seven Konjo Invariants

| # | Invariant | Contract |
|---|---|---|
| K1 | No silent failures | Every component returns or raises. No `except: pass`. |
| K2 | Telemetry on every step | `timed()` wraps all hot-path calls. Latency in every response. |
| K3 | Graceful degradation | Vectro unavailable → float32. RAGAS absent → 501. Cache disabled → transparent fallthrough. |
| K4 | Dtype contracts | Encoder: `float32`. Vectro: `float32` in/out. Qdrant: `float32`. Assert, never assume. |
| K5 | Zero new hard deps | Cache uses `collections.OrderedDict` + `numpy` (already required). |
| K6 | Backward-compatible API | New fields are optional with sensible defaults. |
| K7 | Reproducible evals | Every run → `evals/runs/<timestamp>_<name>/`. Never overwrite. |

---

## Hard Stops

- Tests failing from a previous step.
- Cache returns stale data after ingest (invalidation must be verified in tests).
- Dtype assertion fails at any component boundary.
- NaN/Inf passed to Qdrant.
- Audit log (Sprint 11) contains raw question text (PII leak, OWASP violation).
- p95 query latency regression > 5% from current baseline.

---

*Owner: wesleyscholl / Konjo AI Research*
*See KORE_PLAN.md for full strategic roadmap, market analysis, and licensing recommendation.*
