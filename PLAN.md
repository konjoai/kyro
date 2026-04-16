# Konjo KORE — Master Plan

> **ቆንጆ** — Beautiful. **根性** — Fighting spirit. **康宙** — Health of the universe. **खोजो** — Search and discover.
> *Make it konjo — build, ship, rest, repeat.*

**Strategic document:** See [`KORE_PLAN.md`](KORE_PLAN.md) for full market analysis, sprint roadmap, and licensing recommendation.

---

## Current State: Sprint 5 Complete (v0.2.5)

- **Tests:** 205 passing, 0 failing
- **Branch:** `main`, HEAD `82f893b`
- **Stack:** FastAPI + HyDE + ColBERT + hybrid search + RAGAS + Vectro bridge + streaming

---

## Active Sprint: Sprint 6 — Semantic Cache

**Goal:** Sub-5ms cached responses. Eliminate LLM cost for near-duplicate queries (20–40% of prod traffic).

### Implementation Checklist

| # | File | Change | Status |
|---|---|---|---|
| 1 | `konjoai/config.py` | Add `cache_enabled`, `cache_similarity_threshold`, `cache_max_size` | 🔄 |
| 2 | `konjoai/api/schemas.py` | Add `cache_hit: bool = False` to `QueryResponse` | ⬜ |
| 3 | `konjoai/cache/__init__.py` | New package init | ⬜ |
| 3 | `konjoai/cache/semantic_cache.py` | `SemanticCacheEntry`, `SemanticCache`, `get_semantic_cache()` | ⬜ |
| 4 | `konjoai/retrieve/dense.py` | `q_vec: np.ndarray \| None = None` param | ⬜ |
| 4 | `konjoai/retrieve/hybrid.py` | `q_vec: np.ndarray \| None = None` param, pass-through | ⬜ |
| 5 | `konjoai/api/routes/query.py` | Embed once → cache lookup → full pipeline → cache store | ⬜ |
| 6 | `konjoai/api/routes/ingest.py` | `cache.invalidate()` after `bm25.build()` | ⬜ |
| 7 | `tests/unit/test_semantic_cache.py` | 15+ tests (exact match, semantic match, miss, LRU, invalidate, disabled) | ⬜ |
| 8 | `pytest tests/ -x -q` | All 205+ must pass, 0 regressions | ⬜ |
| 9 | `SESSION.md`, `CHANGELOG.md`, git | Document Sprint 6, commit, push | ⬜ |

### Sprint 6 Gates

1. `pytest tests/ --timeout=120` — 0 failures
2. Cache hit latency < 5 ms (asserted in unit test)
3. Cache miss returns correct results (existing pipeline unaffected)
4. Ingest invalidates cache (stale data protected)
5. `cache_enabled=False` (default) → fully transparent, zero behaviour change
6. `CHANGELOG.md` entry written

---

## Sprint Roadmap Summary

| Sprint | Version | Focus | Gate |
|---|---|---|---|
| 1–5 | v0.2.5 | Foundation: telemetry, routing, HyDE, ColBERT, RAGAS | ✅ 205 tests |
| **6** | **v0.3.0** | **Semantic cache (lightning-fast repeats)** | **⬛ Active** |
| 7 | v0.4.0 | Adapter architecture (swap any backend) | ⬜ |
| 8 | v0.5.0 | Async pipeline + connection pooling (3× throughput) | ⬜ |
| 9 | v0.6.0 | Multi-tenant namespace isolation | ⬜ |
| 10 | v0.7.0 | OpenTelemetry + Prometheus + Grafana | ⬜ |
| 11 | v0.8.0 | Auth + API keys + audit log (enterprise gate) | ⬜ |
| 12 | v1.0.0 | Python SDK + TypeScript client + Helm chart + PyPI | ⬜ |

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
