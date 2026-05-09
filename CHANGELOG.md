# Changelog

All notable changes to KonjoOS are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [unreleased] — K3: Observatory live

### Added
- `demo/observatory.html` — self-contained live dashboard. Dark theme, purple
  accent. Auto-refreshes every 3s by polling `GET /metrics`. Renders:
  - SVG ring gauge for cache hit rate (0–100%).
  - SVG sparkline of the last 60 query latencies.
  - Singleflight coalescing ratio (collapsed / total) with a progress bar.
  - Top-10 most-queried terms leaderboard (stop-words stripped, stems counted).
  - "replay corpus" button that fires the 20-query × 3-tenant seed corpus
    twice through `POST /api/cache/ask` so visitors see the hit-rate climb live.
  - "reset" button wired to `POST /api/cache/reset`.
- `demo/sample_queries.json` — 20 realistic RAG queries across three tenants
  (`acme-support`, `konjo-devrel`, `lumen-research`). Served at
  `GET /api/sample_queries` for the dashboard's replay button.
- `demo/server.py` — `GET /metrics` endpoint backed by real counters from
  `konjoai.cache.SemanticCache.stats()` plus a synchronous singleflight gate
  that mirrors `AsyncSemanticCache.get_or_compute` semantics (concurrent
  callers for the same normalised question collapse onto one LLM synth).
  Metric shape:
  - `cache_hit_rate` (float 0–1) — pulled from `SemanticCache.stats()`.
  - `avg_latency_ms` (float) — mean of `latency_history`.
  - `singleflight_ratio` (float 0–1) — `singleflight_collapsed / total_queries`.
  - `total_queries` (int) — every `ask()` call (hits + misses + collapses).
  - `top_terms` (list of `{term, count}`) — Counter over content words after
    stop-word and length filtering, capped at 10.
  - `latency_history` (list of float, length ≤ 60) — ring buffer of recent ms.
  - Plus convenience fields: `total_hits`, `total_misses`, `cache_size`,
    `singleflight_collapsed`, `calls_avoided`, `time_saved_ms`, `dollars_saved`,
    `history_capacity`, `threshold`, `cache_max_size`.
- `tests/unit/test_demo_metrics.py` — 7 new tests pinning the dashboard
  contract: empty-state shape, field types, hit-rate equals `SemanticCache`
  ground truth, latency history bounded + chronological, leaderboard cap +
  stop-word filter, singleflight ratio under concurrent load, sample-corpus
  shape (20 queries × 3 tenants).

### Changed
- `demo/server.py` — `ask()` now records latency into a 60-slot ring buffer,
  increments stop-word-stripped term counters, and routes misses through the
  new singleflight gate. `reset()` clears the new counters and the inflight
  registry. The handler exposes new GET routes: `/observatory`, `/metrics`,
  `/api/sample_queries`. Static-serving was DRYed into `_serve_static()`.

### Konjo gates
- **K1**: `ask()` waiter path tolerates a leader compute failure — `event.set()`
  fires from `finally`, so waiters never block forever and the inflight slot
  is always cleared.
- **K3**: every metric is a *real* count — `cache_hit_rate` is read straight
  from `SemanticCache.stats()`, `singleflight_collapsed` increments only when
  a thread genuinely waited on a peer's `Event`. No mocks.
- **K6**: purely additive — the existing `/api/cache/*` routes, `stats()`,
  `probe()`, and `seed()` are untouched; 891 tests pass (+7, was 884), 19
  pre-existing failures unchanged.

## [v1.5.0] — Sprint 25: Feedback Collection

### Added
- `konjoai/feedback/` package (stdlib-only, zero new hard deps — K5):
  - `konjoai/feedback/models.py` — `FeedbackEvent` dataclass with OWASP-compliant PII handling: raw question text is NEVER accepted or stored — only `question_hash` (the 16-hex-char SHA-256 prefix). `comment` text is stored as `comment_hash` only. Signal constants: `THUMBS_UP = "thumbs_up"`, `THUMBS_DOWN = "thumbs_down"`, `VALID_SIGNALS` frozenset.
  - `konjoai/feedback/store.py` — `FeedbackStore`: thread-safe bounded `deque` ring buffer (LRU eviction), `record()`, `query()` (filterable by tenant/signal/question_hash, newest-first, limit-capped), `summary()` (total, thumbs_up, thumbs_down, avg_relevance, by_question per-hash breakdown), `clear()`, iterator support. `get_feedback_store()` singleton + `_reset_singleton()` test helper.
  - `konjoai/feedback/__init__.py` — re-exports all public symbols.
- `konjoai/api/routes/feedback.py` — Two endpoints:
  - `POST /feedback` — submit a thumbs-up/down signal with optional `relevance_score` (0.0–1.0), `comment` (stored as hash), `model`, `latency_ms`. Returns HTTP 201 on success, HTTP 404 when disabled (K3), HTTP 422 on validation failure.
  - `GET /feedback/summary` — aggregate statistics: `total`, `thumbs_up`, `thumbs_down`, `avg_relevance`, `by_question` per-hash breakdown. Filterable by `tenant_id`. Returns HTTP 404 when disabled.
- `konjoai/api/app.py` — registers `feedback_route.router`.
- `konjoai/config.py` — two new settings (all K3 opt-in):
  - `feedback_enabled: bool = False` — master switch; when False both endpoints return 404.
  - `feedback_max_events: int = 1000` — ring-buffer capacity (LRU eviction).
- `tests/unit/test_feedback.py` — 55 new tests:
  - `FeedbackEvent`: minimal and full construction, `as_dict()` omits None fields, signal constants, `VALID_SIGNALS`.
  - `FeedbackStore`: construction, record, LRU eviction at max, query newest-first, filter by tenant/signal/question_hash, limit capping, empty store, summary counts/avg_relevance/by_question/tenant_filter, clear, len/iter, thread safety (20 threads × 50 writes), singleton, reset.
  - Config: `feedback_enabled` default False, `feedback_max_events` default 1000.
  - API (disabled): `POST /feedback` → 404, `GET /feedback/summary` → 404, 404 detail mentions `FEEDBACK_ENABLED`.
  - API (enabled): `POST /feedback` → 201 for thumbs_up and thumbs_down, full response contract, with relevance_score, with all optional fields; invalid signal → 422, missing fields → 422, out-of-range score → 422; `GET /feedback/summary` → 200, response contract, reflects submitted feedback, tenant filter.
  - OWASP PII contract: `FeedbackRequest` has no `question` field, `FeedbackEvent` has no `question` field, comment stored as `comment_hash` only, raw comment text absent from stored event.
  - Package exports: all public symbols importable.

### Changed
- `pyproject.toml`, `konjoai/__init__.py`, `helm/kyro/Chart.yaml`, `docs/index.md`, `tests/unit/test_packaging.py`: version bumped `1.4.0 → 1.5.0`; added `test_feedback_importable` to packaging tests.

### Konjo Invariants
- **K1** (no silent failures): `FeedbackStore.record()` wraps writes in `try/except`; errors are emitted as `logger.warning()` and never propagate to the request path.
- **K3** (graceful degradation): `feedback_enabled=False` (the default) makes both endpoints return HTTP 404. The store is only lazily instantiated when feedback is enabled.
- **K5** (zero new hard deps): stdlib only — `collections.deque`, `threading.Lock`. No new packages required.
- **K6** (backward compatible): two new config keys have sensible defaults; no existing config consumer is affected.
- **K7** (multi-tenant safety): `FeedbackEvent.tenant_id` is populated from `request.state.tenant_id` when available; `GET /feedback/summary?tenant_id=<id>` filters by tenant.

### Tests
- Focused: `python -m pytest tests/unit/test_feedback.py -q` → **55 passed in <2s**
- Full regression: `python -m pytest tests/unit/ -q` → **876 passed, 25 skipped** (was 853 — +55 new; pre-existing failures from missing optional packages unchanged)

---

## [v1.4.0] — Sprint 24: Audit Logging

### Added
- `konjoai/audit/` package (pre-existing scaffolding now fully wired):
  - `konjoai/audit/models.py` — `AuditEvent` dataclass with OWASP-compliant PII handling: raw question text, document paths, and user identifiers are NEVER stored — only first-16-hex-chars of SHA-256. `hash_text()` helper. Event type constants: `QUERY`, `INGEST`, `AGENT_QUERY`, `AUTH_FAILURE`, `RATE_LIMITED`.
  - `konjoai/audit/logger.py` — `InMemoryBackend` (bounded `deque` ring buffer, thread-safe), `JsonLinesBackend` (append-only JSONL, thread-safe, parent directory auto-created), `AuditLogger` (enabled/disabled wrapper — K3 no-op path), `get_audit_logger()` singleton with `_reset_singleton()` test helper.
  - `konjoai/audit/__init__.py` — re-exports all public symbols.
- `konjoai/api/routes/audit.py` — Two read-only endpoints:
  - `GET /audit/events` — paginated, filterable (by `tenant_id`, `event_type`, `limit`) event list. Returns HTTP 404 when `audit_enabled=False` (K3: endpoint existence is not an oracle).
  - `GET /audit/stats` — per-event-type counts. Returns HTTP 404 when disabled.
- `konjoai/api/app.py` — registers `audit_route.router`.
- `konjoai/config.py` — four new settings (all K3 opt-in):
  - `audit_enabled: bool = False` — master switch; when False every `log()` call is a pure no-op with zero allocation.
  - `audit_backend: str = "memory"` — `"memory"` (in-process ring buffer) | `"jsonl"` (append-only file).
  - `audit_log_path: str = "logs/audit.jsonl"` — path used when `audit_backend="jsonl"`.
  - `audit_max_memory_events: int = 1000` — ring-buffer capacity for the in-memory backend.
- **Wired into three API routes** (K3: each call is a no-op when `audit_enabled=False`):
  - `POST /query` — logs `AuditEvent(event_type=QUERY, question_hash=hash_text(req.question), intent=..., cache_hit=..., result_count=..., client_ip=...)`.
  - `POST /ingest` — logs `AuditEvent(event_type=INGEST, path_hash=hash_text(req.path), chunks_indexed=..., chunks_deduplicated=...)`.
  - `POST /agent/query` — logs `AuditEvent(event_type=AGENT_QUERY, question_hash=..., result_count=...)`.
- `tests/unit/test_audit.py` — 36 new tests:
  - `hash_text`: determinism, 16-char output, different inputs differ, no plaintext leakage.
  - `AuditEvent`: minimal construction, `as_dict()` omits None fields, includes set fields, event type constants.
  - `InMemoryBackend`: write+query, limit, filter by tenant, filter by event type, LRU eviction, `max_events` guard, stats, thread safety (20 threads × 50 writes).
  - `JsonLinesBackend`: write+query roundtrip, empty file, stats, parent directory auto-creation.
  - `AuditLogger`: disabled no-op, enabled write, write-error safety (K1 — backend errors logged as warnings, never propagated), `query_events` disabled returns `[]`, `stats` disabled returns `{}`, `enabled` property.
  - Singleton: returns same instance, reset clears instance, JSONL backend selected when configured.
  - API (disabled): `GET /audit/events` → 404, `GET /audit/stats` → 404.
  - API (enabled): `GET /audit/events` → 200 with `audit_enabled=True`, `GET /audit/stats` → 200.
  - OWASP PII contract: raw question text absent from serialized event, raw file path absent.
  - Config: `audit_enabled` default `False`, `audit_backend` default `"memory"`, `audit_max_memory_events` default `1000`.

### Changed
- `pyproject.toml`, `konjoai/__init__.py`, `helm/kyro/Chart.yaml`, `docs/index.md`, `tests/unit/test_packaging.py`: version bumped `1.3.0 → 1.4.0`.
- `tests/unit/test_agent_route.py`, `test_query_crag_route.py`, `test_query_self_rag_route.py`, `test_query_decomposition_route.py`, `test_graph_rag.py`, `test_query_route_timeout.py`: `_Settings*` stubs extended with `audit_enabled: bool = False` (K6: backward-compatible).

### Konjo Invariants
- **K1** (no silent failures): `AuditLogger.log()` wraps backend writes in `try/except`; errors are emitted as `logger.warning()` and never propagate to the request path. Question text and document paths are hashed — no raw PII is ever written.
- **K2** (telemetry): the `GET /audit/stats` endpoint surfaces per-event-type counts for operators.
- **K3** (graceful degradation): `audit_enabled=False` (the default) makes every `log()` call a pure no-op (early return, zero allocation). Both API endpoints return HTTP 404 when disabled. Backend write failures degrade to a warning, not a 500.
- **K5** (zero new hard deps): stdlib only — `hashlib`, `json`, `threading`, `collections.deque`, `pathlib`.
- **K6** (backward compatible): four new config keys all have sensible defaults; no existing config consumer is affected. New `_SettingsStub` fields default `False`.
- **K7** (multi-tenant safety): `AuditEvent.tenant_id` is populated from the FastAPI `Request.state.tenant_id` when available; audit entries are queryable by tenant via `GET /audit/events?tenant_id=<id>`.

### Tests
- Focused: `/Library/Developer/CommandLineTools/usr/bin/python3 -m pytest tests/unit/test_audit.py -q` → **36 passed in <10s**
- Full regression: `/Library/Developer/CommandLineTools/usr/bin/python3 -m pytest tests/ -q` → **853 passed, 15 skipped** (was 810 — +43 new tests; 5 pre-existing Python 3.9 compat failures unchanged)

---

## [v1.3.0] — Sprint 23: Async Cache + Singleflight Stampede Protection

### Added
- `konjoai/cache/async_cache.py` — `AsyncSemanticCache(backend, *, singleflight=True, offload_to_thread=True, tenant_provider=...)`. Async wrapper over any sync backend (`SemanticCache`, `RedisSemanticCache`, or anything quacking like the `_SyncBackend` Protocol) exposing `async lookup`, `async store`, `async invalidate`, `async stats`, plus the new singleflight primitive `async get_or_compute(question, q_vec, compute)`.
- **Singleflight stampede protection.** Concurrent identical misses across coroutines (and across replicas of the same pod) collapse to **one** `compute` invocation; every other waiter suspends on an `asyncio.Future` keyed by `(tenant_id, normalised_question)` and reads the populated cache the moment the leader finishes. Eliminates the thundering-herd on hot fresh queries that the Sprint-22 Redis fan-out otherwise made worse.
- **Tenant-scoped in-flight slots.** `_inflight_key(question, tenant)` namespaces the singleflight dict by the Sprint-17 ContextVar tenant so two tenants asking the same string each get their own compute (no cross-tenant leakage and no false collapse).
- **Error propagation that doesn't strand waiters.** `compute` exceptions raise to the leader **and** every follower; the in-flight slot is freed in a `finally` so the next caller can retry without being blocked by a stale future.
- `konjoai/cache/__init__.py`: re-exports `AsyncSemanticCache` and `async_wrap` (`async_cache.wrap` factory).
- New telemetry counters surfaced in `await cache.stats()`: `singleflight_enabled`, `stampedes_collapsed`, `inflight_peak`, `inflight_now` — extends each backend's own stats dict, doesn't replace it.
- `tests/unit/test_async_cache.py` — 12 new tests:
  - Pass-through roundtrip (lookup / store / invalidate)
  - Stats wrapper extends backend stats with the four new counters
  - **Singleflight collapse: 8 concurrent identical misses → exactly 1 compute call**, all 8 receive the same answer, `stampedes_collapsed == 7`
  - Cache hit on the second call bypasses the singleflight path entirely (no false-positive collapse counter increment)
  - **Error propagation: compute exception raises to all 5 waiters, in-flight slot freed**
  - Retry-after-error path works
  - Tenant-keyed in-flight: same question across `acme` / `globex` triggers two distinct computes (two in-flight slots, no collapse)
  - `singleflight=False` → each caller computes independently (the bypass mode)
  - `offload_to_thread=False` keeps everything on the loop (verified via `monkeypatch` on `asyncio.to_thread`)
  - `wrap()` factory parity

### Changed
- `pyproject.toml`, `konjoai/__init__.py`, `helm/kyro/Chart.yaml`, `docs/index.md`, `tests/unit/test_packaging.py`: version bumped `1.2.0 → 1.3.0`.

### Konjo Invariants
- **K1** (no silent failures): leader exceptions propagate to every waiter via `future.set_exception(exc)` and the slot is freed in `finally` so `lookup` retries are unblocked.
- **K3** (graceful degradation): `singleflight=False` makes the wrapper a thin async adapter — useful when callers have already deduped upstream. `offload_to_thread=False` skips the thread hop for callers that prefer pure-loop semantics.
- **K5** (zero new hard deps): pure stdlib `asyncio` — no `aioredis`, no extra synchronisation library.
- **K6** (backward compatible): purely additive. The synchronous `SemanticCache` and `RedisSemanticCache` contracts and call sites are unchanged.
- **K7** (multi-tenant safety): the in-flight key namespaces by tenant so a Redis-fan-out singleflight does not leak responses across tenants under stampede.

### Tests
- Focused: `python3 -m pytest tests/unit/test_async_cache.py -q` → **12 passed in <1s**
- Full regression: `python3 -m pytest tests/unit/ -q` → **810 passed, 15 skipped** (was 798 — +12 tests; 5 pre-existing Python 3.9 compat failures unchanged)

---

## [v1.2.0] — Sprint 22: Distributed Semantic Cache (Redis backend)

### Added
- `konjoai/cache/redis_cache.py`: `RedisSemanticCache` — cross-pod-shared, tenant-namespaced semantic cache backed by Redis. Storage layout per tenant is `<namespace>:<tenant>:entries` (HASH, field=normalised question, value=pickled blob containing question, l2-normalised float32 vector bytes, response, timestamp) and `<namespace>:<tenant>:lru` (ZSET ordered by `time.monotonic()` for eviction). All Redis calls go through a `_safely(op, fn)` wrapper that logs and returns `None` on transport errors so a Redis outage degrades to a cache miss instead of a 500. Tenant prefix is read from the Sprint-17 `_current_tenant_id` ContextVar; the anonymous bucket falls back to `__anonymous__`. Optional per-entry TTL (`cache_redis_ttl_seconds`).
- `konjoai/cache/redis_cache.py::build_redis_cache(...)`: lazy-imports the `redis` package, runs an initial `PING`, and returns the cache or `None` (K3 graceful fallback) when the package is missing or the connection fails. Callers that get `None` fall back to the in-memory backend.
- `konjoai/cache/__init__.py`: re-exports `RedisSemanticCache` and `build_redis_cache` alongside the existing `SemanticCache` / `get_semantic_cache` symbols.
- `konjoai/cache/semantic_cache.py::get_semantic_cache()`: backend selection — when `cache_backend == "redis"` the factory tries `build_redis_cache(...)` first; on `None` it logs a warning and constructs the in-memory `SemanticCache`. The factory still returns `None` when `cache_enabled=False` (zero-overhead off path).
- `konjoai/config.py`: four new settings:
  - `cache_backend: str = "memory"` — `"memory"` or `"redis"`
  - `cache_redis_url: str = "redis://localhost:6379/0"`
  - `cache_redis_namespace: str = "kyro:cache"`
  - `cache_redis_ttl_seconds: int = 0` — `0` disables TTL
- `tests/unit/test_redis_cache.py` — 29 new tests:
  - Construction guardrails (threshold, max_size, ttl_seconds)
  - Exact + cosine roundtrip; below-threshold misses; question normalisation; float32 enforcement
  - LRU eviction order and refresh-on-lookup behaviour
  - Tenant scoping isolation across `acme`/`globex`/anonymous
  - `invalidate()` only drops the active tenant
  - Stats: hit/miss counters and hit_rate
  - TTL: `0` skips `EXPIRE`; `>0` sets it on both hash + zset
  - Graceful degradation: `lookup` returns `None` and `store` does not raise when every Redis op throws
  - `build_redis_cache` returns `None` on missing `redis` package and on `PING` failure
  - Factory dispatch: returns memory backend by default, returns the Redis backend when configured, falls back to memory when `build_redis_cache` returns `None`, and falls back to memory for unknown backends
  - Backend-protocol parity: both backends expose `lookup`/`store`/`invalidate`/`stats` with identical roundtrip semantics

### Changed
- `pyproject.toml`, `konjoai/__init__.py`, `helm/kyro/Chart.yaml`, `docs/index.md`, `tests/unit/test_packaging.py`: version bumped `1.1.0 → 1.2.0`.

### Konjo Invariants
- **K1**: Redis transport errors are logged via `logger.warning` and surfaced as cache misses (no silent failures, no swallowed exceptions in business logic). Pickle-decode errors degrade to a miss with a warning rather than crashing the request.
- **K2**: Hit/miss counters per backend; debug-level breadcrumbs include the active tenant and similarity score.
- **K3**: When the `redis` package is missing, when `PING` fails, when an unknown backend is configured, or when individual Redis ops error, the caller still gets a working cache (in-memory) or a clean miss.
- **K4**: `RedisSemanticCache.store()` asserts `q_vec.dtype == np.float32` at the boundary, identical to the in-memory contract. Vectors are L2-normalised before pickling.
- **K5**: `redis` is **optional** (lazy import). Tests use a 100-line in-process fake; the suite stays runnable without `pip install redis`.
- **K6**: New endpoint surface and config keys are additive; default `cache_backend="memory"` preserves the v1.1.0 behaviour byte-for-byte.
- **K7**: Multi-tenant isolation is enforced via the keyspace, so a Redis instance shared across tenants cannot leak responses between them.

### Tests
- Focused: `python3 -m pytest tests/unit/test_redis_cache.py -q` → **29 passed in ~1s**
- Full regression: `python3 -m pytest tests/unit/ -q` → **798 passed, 15 skipped** (was 769 — +29 tests; 5 pre-existing Python 3.9 compat failures unchanged)

---

## [v1.1.0] — Sprint 21: Streaming Agent (`POST /agent/query/stream`)

### Added
- `konjoai/agent/react.py`: `RAGAgent.run_stream()` — synchronous generator that drives the bounded ReAct loop and yields one event dict per step plus a final `result` event. `RAGAgent.run()` is now a thin consumer of `run_stream()`, preserving its existing `AgentResult` contract.
- `konjoai/api/routes/agent.py`: `POST /agent/query/stream` — Server-Sent Events endpoint that emits one `data:` frame per ReAct step, then a `result` frame (answer + serialized sources/steps), then a `telemetry` frame (populated when `enable_telemetry=True`), then a terminal `[DONE]` sentinel. Bridges the synchronous agent loop to the async stream via `asyncio.Queue` + `asyncio.to_thread`. `request_timeout_seconds` is enforced via `asyncio.wait_for`; breaches return a deterministic 504 with K2 telemetry (`logger.warning`).
- `konjoai/sdk/models.py`: `SDKAgentStreamEvent` — typed frozen dataclass with `type` discriminator and decoded `data` payload.
- `konjoai/sdk/client.py`: `KonjoClient.agent_query_stream()` — iterator over `SDKAgentStreamEvent`. Skips malformed JSON frames and frames missing a `type`, and stops cleanly at `[DONE]`. Wraps `httpx.TimeoutException` as `KyroTimeoutError`.
- `tests/unit/test_agentic.py` (+4 tests): `run_stream` step-then-result sequencing, parser fallback path, empty-question rejection, and a regression that proves `run()` still returns an identical `AgentResult` after the refactor.
- `tests/unit/test_agent_route.py` (+4 tests): SSE frame contract (`step`/`result`/`telemetry`/`[DONE]`), telemetry omission when disabled, 504 on `asyncio.wait_for` timeout, 422 on empty question.
- `tests/unit/test_sdk.py` (+4 tests): typed event yielding, `[DONE]` sentinel termination, malformed/typeless frame skipping, `KyroTimeoutError` mapping.

### Changed
- `pyproject.toml`, `konjoai/__init__.py`, `helm/kyro/Chart.yaml`, `docs/index.md`: version bumped `1.0.0 → 1.1.0`.
- `tests/unit/test_packaging.py`: pinned-version assertion updated to `1.1.0`.

### Konjo Invariants
- **K1** (no silent failures): producer-side exceptions in `run_stream()` are propagated through the SSE bridge; the route surfaces them rather than swallowing them.
- **K2** (telemetry): `agent_stream` step is wrapped in `timed()`; timeout path emits `logger.warning`.
- **K3** (graceful degradation): SDK iterator silently skips malformed/typeless frames so a partial deploy cannot crash a pinned client.
- **K5** (no new hard deps): pure stdlib `asyncio.Queue` + `asyncio.to_thread`; SDK reuses the existing `httpx.Client.stream` codepath.
- **K6** (backward compatible): `RAGAgent.run()`, `POST /agent/query`, and all existing SDK methods are unchanged. New endpoint and new SDK method are purely additive.

### Tests
- Focused: `python3 -m pytest tests/unit/test_agent_route.py tests/unit/test_agentic.py tests/unit/test_sdk.py -q` → **64 passed in 3.25s**
- Full regression: `python3 -m pytest tests/unit/ -q` → **769 passed, 15 skipped** (up from 757 — +12 tests; 5 pre-existing Python 3.9 compat failures unchanged)

---

## [v1.0.0] — Sprint 20: Helm chart + PyPI packaging + Docs site

### Added
- `pyproject.toml` v1.0.0: production classifiers (Development Status :: 5 - Production/Stable, Python 3.11/3.12, Typing :: Typed), six optional extras (`jwt`, `mcp`, `eval`, `observability`, `dev`, `all`), `[project.urls]` (Homepage, Documentation, Repository, Changelog, Bug Tracker), author email, `[tool.hatch.build.targets.wheel]`
- `konjoai/__init__.py`: `__version__ = "1.0.0"`
- `helm/kyro/` — production Helm chart:
  - `Chart.yaml` — `apiVersion: v2`, application type, semver 1.0.0
  - `values.yaml` — `replicaCount=2`, HPA enabled (2–10 pods), ClusterIP service port 8000, full `config.*` env-var map, `secrets.*`, liveness/readiness probes on `/health`, CPU/memory resource limits, pod security context (non-root)
  - `templates/_helpers.tpl` — standard name/label/selector helpers
  - `templates/deployment.yaml` — `apps/v1` Deployment; ConfigMap `envFrom`, secret env injections, security context, checksum annotation for config rolling restarts
  - `templates/service.yaml` — `v1` ClusterIP Service
  - `templates/configmap.yaml` — `v1` ConfigMap, all config keys
  - `templates/hpa.yaml` — `autoscaling/v2` HPA with CPU + memory targets
  - `templates/ingress.yaml` — `networking.k8s.io/v1` Ingress (conditional)
- `.github/workflows/release.yml` — tag-triggered release pipeline: test (3.11 + 3.12) → build sdist/wheel → PyPI (OIDC trusted publishing) + Docker Hub (multi-arch amd64/arm64, `latest` + versioned tag) + Helm OCI push + GitHub Release with artefacts
- `mkdocs.yml` — MkDocs Material theme, versioned docs, social links
- `docs/` — 7-page documentation site: `index.md`, `quickstart.md`, `sdk.md`, `mcp.md`, `api.md`, `configuration.md`, `deployment.md`
- `tests/unit/test_packaging.py` — 52 tests
- `tests/unit/test_helm.py` — 25 tests

### Tests
- Focused run: `python3 -m pytest tests/unit/test_packaging.py tests/unit/test_helm.py -v` → **77 passed in 0.75s**
- Full regression: `python3 -m pytest tests/ --timeout=120` → **764 passed, 15 skipped** (5 pre-existing Python 3.9 compat failures unchanged)

## [v0.9.8] — Sprint 19: Python SDK + MCP Server

### Added
- `konjoai/sdk/` — typed synchronous Python SDK with zero new hard dependencies:
  - `konjoai/sdk/exceptions.py` — `KyroError`, `KyroAuthError`, `KyroRateLimitError` (`retry_after` from `Retry-After` header), `KyroTimeoutError`, `KyroNotFoundError`; clean exception hierarchy for structured error handling
  - `konjoai/sdk/models.py` — stdlib frozen dataclass response models: `SDKQueryResponse`, `SDKIngestResponse`, `SDKHealthResponse`, `SDKAgentQueryResponse`, `SDKAgentStep`, `SDKSourceDoc`, `SDKStreamChunk`
  - `konjoai/sdk/client.py` — `KonjoClient(base_url, *, api_key, jwt_token, timeout)`: wraps `httpx.Client` with `X-API-Key` / `Authorization: Bearer` auth; exposes `query()`, `query_stream()` (SSE iterator), `ingest()`, `health()`, `agent_query()`; context-manager lifecycle
  - `konjoai/sdk/__init__.py` — all public SDK symbols exported
- `konjoai/mcp/` — Model Context Protocol server:
  - `konjoai/mcp/server.py` — `KyroMCPServer`: `TOOLS` constant (4 JSON Schema tool definitions), `list_tools()`, `async dispatch()` (pure Python, no mcp dep required), `from_url()` factory; `run_stdio()` with lazy mcp import (K3/K5)
  - `konjoai/mcp/__init__.py` — `_HAS_MCP` flag, exports
  - `konjoai/mcp/__main__.py` — `python -m konjoai.mcp` click CLI (`--base-url`, `--api-key`, `--jwt-token`, `--timeout`; reads `KYRO_API_KEY` / `KYRO_JWT_TOKEN` env vars)
- MCP tool surface: `kyro_query`, `kyro_ingest`, `kyro_health`, `kyro_agent_query` — each with a well-typed JSON Schema `inputSchema`
- `tests/unit/test_sdk.py` — 46 tests
- `tests/unit/test_mcp.py` — 29 tests

### Tests
- Focused run: `python3 -m pytest tests/unit/test_sdk.py tests/unit/test_mcp.py -v` → **75 passed in 0.40s**
- Full regression: `python3 -m pytest tests/ --timeout=120` → **687 passed, 15 skipped** (5 pre-existing Python 3.9 compat failures unchanged)

---

## [v0.9.5] — Sprint 18: Auth Hardening + Rate Limiting

### Added
- `konjoai/auth/rate_limiter.py` — in-memory sliding-window rate limiter:
  - `_Bucket`: per-(tenant, endpoint) deque of timestamps with per-bucket `threading.Lock`
  - `RateLimiter`: keyed by `(tenant_id, endpoint)`, configurable `max_requests` / `window_seconds`; `enabled=False` is a complete no-op (K3)
  - `RateLimitExceeded`: carries `tenant_id`, `endpoint`, `limit`, `window`
  - `get_rate_limiter()`: module-level singleton reading settings; `_reset_singleton()` for tests
- `konjoai/auth/api_key.py` — static API-key authentication layer:
  - `hash_api_key(plaintext)`: SHA-256 hex digest (stdlib only, K5)
  - `verify_api_key(plaintext, entries)`: timing-safe `hmac.compare_digest` lookup against a `<sha256hex>[:<tenant_id>]` registry
  - `APIKeyResult`: `tenant_id` + `key_hash` resolved on match
- `konjoai/auth/brute_force.py` — IP-based brute-force protection:
  - `_IPRecord`: per-IP sliding-window failure deque + lockout timestamp with `threading.Lock`
  - `BruteForceGuard`: `check_ip` (raises `IPLockedOut` when locked), `record_failure`, `record_success` (clears failures), `is_locked`, `failure_count`, `reset`
  - `IPLockedOut`: carries `ip`, `lockout_seconds`, `retry_after`
  - `get_brute_force_guard()`: module-level singleton; `_reset_singleton()` for tests
- `konjoai/auth/deps.py`:
  - `_resolve_tenant_id(request, credentials)`: core resolution logic; accepts `request=None` for unit tests; evaluates X-API-Key header before Bearer JWT; integrates brute-force guard
  - `get_tenant_id(request, credentials)`: thin FastAPI dep delegating to `_resolve_tenant_id` (non-optional `Request` required by FastAPI DI)
  - `check_rate_limit(request, tenant_id)`: new FastAPI dep; raises 429 on limit breach; no-op when `rate_limiting_enabled=False`
  - `get_brute_force_guard` imported at module level for patchability in tests
- `tests/unit/test_rate_limiter.py` — 30 tests
- `tests/unit/test_api_key_auth.py` — 32 tests
- `tests/unit/test_brute_force.py` — 29 tests

### Changed
- `konjoai/config.py`: added `api_key_auth_enabled=False`, `api_keys=[]`, `rate_limiting_enabled=False`, `rate_limit_requests=60`, `rate_limit_window_seconds=60`, `brute_force_enabled=False`, `brute_force_max_attempts=5`, `brute_force_window_seconds=60`, `brute_force_lockout_seconds=300` (all off by default — K3)
- `konjoai/auth/__init__.py`: updated exports for all Sprint 18 public symbols
- `tests/unit/test_auth.py`: updated `TestGetTenantIdDep` to call `_resolve_tenant_id` (internal helper accepting `request=None`) so existing unit tests compile without a full FastAPI `Request`

### Tests
- Focused run: `python3 -m pytest tests/unit/test_rate_limiter.py tests/unit/test_api_key_auth.py tests/unit/test_brute_force.py -v` → **91 passed in 0.65s**
- Full regression: `python3 -m pytest tests/ --timeout=120` → **607 passed, 15 skipped** (5 pre-existing Python 3.9 compat failures unchanged)

---

## [Unreleased] — Sprint 17: Multi-tenancy + JWT (v0.9.0)

### Added
- `konjoai/auth/` — new package:
  - `tenant.py`: `_current_tenant_id` ContextVar, `get_current_tenant_id()`, `set_current_tenant_id()`, `ANONYMOUS_TENANT` sentinel — zero-overhead tenant propagation through async tasks and `asyncio.to_thread` threads
  - `jwt_auth.py`: `TenantClaims` dataclass, `decode_token()`, `_HAS_JWT` guard (K5: PyJWT optional)
  - `deps.py`: `get_tenant_id` async generator FastAPI dependency — K3 pass-through when disabled, 401 on missing/invalid token, 503 when secret unconfigured, sets ContextVar for downstream store scoping, cleanup in `finally`
  - `__init__.py`: package exports
- `tests/unit/test_auth.py` — 24 tests (+ 9 skipped without PyJWT): `TenantClaims`, `decode_token`, `_HAS_JWT`, ContextVar isolation, `get_tenant_id` dep paths, `QdrantStore` payload injection and filter

### Changed
- `konjoai/config.py`: added `multi_tenancy_enabled=False`, `jwt_secret_key=""`, `jwt_algorithm="HS256"`, `tenant_id_claim="sub"` (K3: off by default)
- `konjoai/store/qdrant.py`: `upsert()` stamps `tenant_id` payload field when ContextVar set; `search()` adds `Filter(must=[FieldCondition(key="tenant_id", ...)])` when ContextVar set (K6: `None` context = no filter = full backward compat)
- `konjoai/api/routes/ingest.py`: `tenant_id: str | None = Depends(get_tenant_id)` injected (sets ContextVar as side-effect; zero ingest logic change)
- `konjoai/api/routes/query.py`: `tenant_id: str | None = Depends(get_tenant_id)` on both `/query` and `/query/stream`
- `requirements.txt`: `# PyJWT>=2.8` documented as optional

### Tests
- Focused run: `python3 -m pytest tests/unit/test_auth.py -v` → **24 passed, 9 skipped in 1.77s**
- Full regression: `python3 -m pytest tests/unit/ -q --tb=short` → **509 passed, 5 pre-existing failures**

---

## [Unreleased] — Sprint 16: OTel + Prometheus Observability Layer (v0.8.7)

### Added
- `konjoai/telemetry.py` — Sprint 16 observability extensions:
  - `_HAS_PROMETHEUS` / `_HAS_OTEL` — import guards (K5: no new hard deps)
  - `KyroMetrics` — Prometheus counters/histograms: `kyro_query_total`, `kyro_query_errors_total`, `kyro_query_latency_ms`, `kyro_cache_hits_total`; no-op when prometheus-client absent or `otel_enabled=False` (K3)
  - `KyroTracer` — thin OTel tracer wrapper; `_noop_span()` fallback when opentelemetry-sdk absent or endpoint unset (K3)
  - `get_metrics()` / `get_tracer()` — module-level singletons (lazy init, reads settings)
  - `record_pipeline_metrics(tel, intent, *, enabled)` — K3-gated push of a completed `PipelineTelemetry` into Prometheus
- `konjoai/api/routes/health.py` — `GET /metrics` Prometheus exposition endpoint; returns 404 when `otel_enabled=False`, 503 when prometheus-client absent (K3)
- `tests/unit/test_telemetry.py` — 26 new Sprint 16 tests (46 passed + 6 skipped when prometheus-client absent)

### Changed
- `konjoai/config.py`: added `otel_enabled: bool = False`, `otel_endpoint: str = ""`, `otel_service_name: str = "kyro"`, `prometheus_port: int = 8001` (K3: off by default)
- `konjoai/api/app.py`: registered `health_route.router`
- `konjoai/api/routes/query.py`: `record_pipeline_metrics(tel, intent.value, enabled=settings.otel_enabled)` call added after pipeline completes; import of `record_pipeline_metrics`
- `requirements.txt`: optional OTel + Prometheus deps documented as comments
- `tests/unit/test_query_crag_route.py`, `test_query_self_rag_route.py`, `test_query_decomposition_route.py`, `test_query_route_timeout.py`, `test_graph_rag.py`: `_SettingsStub` structs updated with `otel_enabled: bool = False`

### Tests
- Focused run: `python3 -m pytest tests/unit/test_telemetry.py -v` → **46 passed, 6 skipped in 0.42s**
- Full regression: `python3 -m pytest tests/unit/ -q --tb=short` → **485 passed, 5 pre-existing failures (Python 3.9 compat)**

---

## [Unreleased] — Sprint 15: Lightweight GraphRAG Community Detection (v0.8.5)

### Added
- `konjoai/retrieve/graph_rag.py` — `GraphRAGRetriever` using NetworkX community detection:
  - `_tokenize()` — stopword-stripped, lowercased token sets for Jaccard similarity
  - `EntityGraph` — Jaccard similarity graph construction (O(n²), n≤20 chunk limit)
  - `CommunityContext` — community label + member content container
  - `GraphRAGResult` — final retrieval result with community summaries
  - `GraphRAGRetriever` — full retriever: build graph → Louvain communities → top-K by relevance
  - `get_graph_rag_retriever()` — singleton factory (feature-flagged off by default)
- `tests/unit/test_graph_rag.py` — 37 tests covering entity graph, community detection, retriever, K3 gate, edge cases

### Changed
- `konjoai/config.py`: added `enable_graph_rag: bool = False`, `graph_rag_max_communities: int = 5`, `graph_rag_similarity_threshold: float = 0.3`
- `konjoai/api/schemas.py`: added `QueryRequest.use_graph_rag: bool = Field(False, ...)`, `QueryResponse.graph_rag_communities: list[str] | None = None`
- `konjoai/api/routes/query.py`: K3 gate (`if settings.enable_graph_rag and req.use_graph_rag`) injected after hybrid retrieval; `X-Use-Graph-Rag` header parsed; `graph_rag_communities` threaded to response
- `requirements.txt`: `networkx>=3.2` added
- `tests/unit/test_query_crag_route.py`, `test_query_self_rag_route.py`, `test_query_decomposition_route.py`, `test_query_route_timeout.py`: `_SettingsStub` updated with 3 new GraphRAG fields

### Tests
- Focused run: `python3 -m pytest tests/unit/test_graph_rag.py -v` → **37 passed in 8.53s**
- Full regression: `python3 -m pytest tests/unit/ -q --tb=short` → **464 passed in 10.16s**

## [Unreleased] — Pre-Sprint-15: Query Route Timeout Parity

### Added
- `asyncio.wait_for` timeout enforcement on `POST /query` and `POST /query/stream`.
- Route-level timeout failure contract: both routes return HTTP `504` with duration detail on overrun.
- `logger.warning(...)` telemetry on both timeout paths (K2 compliance).
- `tests/unit/test_query_route_timeout.py` — 4 tests covering 504 path, happy path, stream 504, detail format.

### Changed
- `konjoai/api/routes/query.py`: inner `_execute()` and `_stream_execute()` closures now wrapped with `asyncio.wait_for(timeout=timeout_seconds)`.
- `tests/unit/test_query_crag_route.py`, `test_query_self_rag_route.py`, `test_query_decomposition_route.py`: `_SettingsStub` updated with `request_timeout_seconds: float = 30.0`.

### Tests
- Focused run: `python3 -m pytest tests/unit/test_query_route_timeout.py -v` → **4 passed**
- Full regression: `python3 -m pytest tests/unit/ -q --tb=short` → **427 passed in 34.57s**
- Commit: `e48ed09` on `main`

## [Unreleased] — Sprint 14: Agentic Route Hardening (Wave 1.1)

### Added
- `POST /agent/query` request-timeout guard using `request_timeout_seconds` from settings.
- Route-level timeout failure contract: returns HTTP `504` with explicit timeout detail.
- `tests/unit/test_agent_route.py::test_agent_query_route_returns_504_on_timeout`

### Changed
- `konjoai/api/routes/agent.py` now wraps bounded ReAct execution with `asyncio.timeout(...)`.

### Tests
- Focused run: `python3 -m pytest tests/unit/test_agentic.py tests/unit/test_agent_route.py -q`
- Result: **6 passed, 0 failed**
- Adjacent route regression run: `python3 -m pytest tests/unit/test_query_decomposition_route.py tests/unit/test_query_crag_route.py tests/unit/test_query_self_rag_route.py -q`
- Result: **9 passed, 0 failed**
- Lint (changed files): `python3 -m ruff check konjoai/api/routes/agent.py tests/unit/test_agent_route.py`
- Result: **All checks passed**

## [Unreleased] — Sprint 14: Agentic RAG Foundation (v0.8.0, Wave 1)

### Added
- `konjoai/agent/react.py` — bounded ReAct-style `RAGAgent` with:
	- JSON action parsing (`retrieve` / `finish`)
	- tool registry abstraction (`ToolRegistry`)
	- max-step guard fallback for deterministic termination
	- step-level Thought/Action/Observation trace (`AgentStep`)
- `konjoai/agent/__init__.py` — agent exports for package-level import
- `konjoai/api/routes/agent.py` — `POST /agent/query` endpoint returning:
	- `answer`, `sources`, `model`, `usage`
	- `steps[]` trace
	- optional telemetry payload (aligned with `enable_telemetry`)
- `tests/unit/test_agentic.py` — unit tests for agent core loop and fallback behavior
- `tests/unit/test_agent_route.py` — route tests for response contract and telemetry on/off

### Changed
- `konjoai/api/app.py` — registered `agent` router so `/agent/query` is available in the main API app

### Tests
- Focused run: `python3 -m pytest tests/unit/test_agentic.py tests/unit/test_agent_route.py -q`
- Result: **5 passed, 0 failed**

## [Unreleased] — Sprint 13: Query Decomposition + Multi-Step Retrieval (v0.7.5)

### Added
- `konjoai/retrieve/decomposition.py`:
	- `QueryDecomposer` (LLM JSON decomposition with deterministic fallback)
	- `ParallelRetriever` (`asyncio.gather` fan-out for sub-query retrieval)
	- `AnswerSynthesizer` (sub-answer synthesis using decomposition hint)
- `QueryRequest.use_decomposition: bool = False` — per-request decomposition opt-in
- `/query` header opt-in support: `use_decomposition`, `use-decomposition`, `x-use-decomposition`
- `QueryResponse` decomposition fields:
	- `decomposition_used`
	- `decomposition_sub_queries`
	- `decomposition_synthesis_hint`
- `tests/unit/test_decomposition.py` — decomposition parser/fallback/retriever/synthesizer unit tests
- `tests/unit/test_query_decomposition_route.py` — `/query` decomposition opt-in route tests (body, header, default-off)

### Changed
- `konjoai/api/routes/query.py`:
	- AGGREGATION intent now supports Sprint 13 decomposition orchestration when explicitly enabled
	- parallel retrieval fan-out over decomposed sub-queries
	- sub-query answer generation + synthesized final answer
	- decomposition telemetry propagation in response payload
- `konjoai/config.py`:
	- added `enable_query_decomposition` and `decomposition_max_sub_queries`
- Route test settings stubs updated for new decomposition settings fields:
	- `tests/unit/test_query_crag_route.py`
	- `tests/unit/test_query_self_rag_route.py`

### Tests
- Focused run: `python3 -m pytest tests/unit/test_decomposition.py tests/unit/test_query_decomposition_route.py tests/unit/test_query_crag_route.py tests/unit/test_query_self_rag_route.py tests/unit/test_router.py -q`
- Result: **65 passed, 0 failed**
- Expanded focused run: `python3 -m pytest tests/unit/test_async_pipeline.py tests/unit/test_decomposition.py tests/unit/test_query_decomposition_route.py tests/unit/test_query_crag_route.py tests/unit/test_query_self_rag_route.py tests/unit/test_router.py -q`
- Result: **76 passed, 0 failed**

## [Unreleased] — Sprint 12: Self-RAG (v0.7.0 refresh)

### Added
- `QueryRequest.use_self_rag: bool = False` — per-request Self-RAG opt-in
- `/query` header opt-in support for Self-RAG: `use_self_rag`, `use-self-rag`, or `x-use-self-rag`
- `QueryResponse` Self-RAG telemetry fields:
	- `self_rag_iteration_scores`
	- `self_rag_total_tokens`
- `tests/unit/test_query_self_rag_route.py` — `/query` Self-RAG opt-in behavior tests (body, header, default-off)

### Changed
- `konjoai/retrieve/self_rag.py` — refreshed Sprint 12 contract implementation:
	- `SelfRAGOrchestrator` bounded iterative loop (max iterations configurable)
	- `SelfRAGCritic` reflection scoring for ISREL / ISSUP / ISUSE
	- refined retrieval callback when `ISSUP < 0.5`
	- compatibility alias retained: `SelfRAGPipeline = SelfRAGOrchestrator`
- `konjoai/api/routes/query.py` — Self-RAG now runs when globally enabled **or** request/header opt-in is set; telemetry carries per-iteration reflection scores and cumulative token usage
- `konjoai/config.py` — `self_rag_max_iterations` default updated to `3` per Sprint 12 contract

### Tests
- Focused run: `python3 -m pytest tests/unit/test_self_rag.py tests/unit/test_query_crag_route.py tests/unit/test_query_self_rag_route.py -q`
- Result: **42 passed, 0 failed**

## [Unreleased] — Sprint 11: CRAG (v0.6.0 refresh)

### Added
- `kyro_production_plan.md` — production rollout plan added to repo root and referenced by planning docs
- `QueryRequest.use_crag: bool = False` — per-request CRAG opt-in
- `/query` header opt-in support: `use_crag`, `use-crag`, or `x-use-crag`
- `QueryResponse` CRAG diagnostics: `crag_scores`, `crag_classification`, `crag_refinement_triggered`
- `tests/unit/test_query_crag_route.py` — `/query` opt-in path tests (body flag, header flag, default-off behavior)

### Changed
- `konjoai/retrieve/crag.py` — replaced legacy CRAG pipeline with `CRAGEvaluator` contract:
	- normalized score classification (`CORRECT > 0.7`, `AMBIGUOUS 0.3–0.7`, `INCORRECT < 0.3`)
	- all-incorrect fallback via `web_fallback()` stub
	- ambiguous refinement path with decomposed sub-queries
- `konjoai/config.py` — replaced `crag_relevance_threshold` with `crag_correct_threshold` and `crag_ambiguous_threshold`
- `konjoai/api/routes/query.py` — CRAG runs when globally enabled or per-request opt-in is set; telemetry includes
	`crag_scores`, `crag_classification`, `crag_refinement_triggered`
- `tests/unit/test_crag.py` — rewritten for Sprint 11 contract gates and synthetic quality checks

### Tests
- Focused run: `python3 -m pytest tests/unit/test_crag.py tests/unit/test_query_crag_route.py -q`
- Result: **11 passed, 0 failed**

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



## [Unreleased] — Sprint 10: Adaptive Chunking (v0.5.5)

### Added

**`konjoai/ingest/chunkers.py`**
- `SemanticSplitter` — embeds sentences with an optional context buffer (`buffer_size`), inserts chunk boundaries where cosine similarity between adjacent sentence embeddings drops below `similarity_threshold`. Implements the Semantic Chunking technique (Kamradt 2023). Produces chunks with `metadata["splitter"] = "semantic"` and `sentence_count`.
- `LateChunker` — encodes *all* sentences in a single batch call (approximating the Jina AI Late Chunking paper, 2024), then finds boundaries post-embedding. Respects `max_chunk_tokens` ceiling. Produces chunks with `metadata["chunker"] = "late"`, `sentence_count`, and `boundary_sim`.
- `_cosine_similarities()` — shared helper for adjacent cosine similarity computation from `(N, dim)` float32 arrays.
- `get_chunker()` updated — now supports `"semantic"` and `"late"` strategies in addition to existing `"recursive"` and `"sentence_window"`. Accepts optional `_encoder` and `similarity_threshold` kwargs.

**`konjoai/retrieve/router.py`**
- `ChunkComplexity(str, Enum)` — SIMPLE / MEDIUM / COMPLEX tiers with docstring explaining counter-intuitive chunk-size ordering.
- `CHUNK_SIZE_MAP: dict[ChunkComplexity, int]` — `{SIMPLE: 256, MEDIUM: 512, COMPLEX: 1024}`.
- `classify_chunk_complexity(query) -> tuple[ChunkComplexity, int]` — maps query via `QueryComplexityScorer` to a complexity tier and its associated chunk size. Lazy-loads the scorer singleton.

**`konjoai/config.py`**
- `chunk_strategy` comment updated to document all four strategies: `"recursive" | "sentence_window" | "semantic" | "late"`.
- `semantic_split_threshold: float = 0.4` — cosine similarity boundary for `SemanticSplitter`.
- `late_chunk_threshold: float = 0.4` — cosine similarity boundary for `LateChunker`.

**`scripts/ablation_chunking.py`** (new)
- CLI ablation harness; runs all four chunking strategies against `evals/corpus/eval_questions.json`.
- Computes offline proxy metrics: `chunk_count`, `avg_chunk_chars`, `std_chunk_chars`, `min/max_chunk_chars`, `within_coherence`, `boundary_sharpness`, `coverage_score`.
- Gate checks: no strategy produces zero/empty chunks; embedding-aware strategies must not be less coherent than recursive by > 0.05.
- Writes `evals/runs/<timestamp>_chunking_ablation/comparison.json`.
- Usage: `python scripts/ablation_chunking.py --quiet` (full metrics) or `--no-encoder` (CI-safe offline mode, skips embedding metrics).

**Tests**
- `tests/unit/test_semantic_splitter.py` — 32 tests: construction validation, empty/single-sentence edge cases, uniform/alternating/block mock encoders, no-empty-chunks invariant, full-content coverage, metadata tags, factory integration.
- `tests/unit/test_late_chunker.py` — 33 tests: construction validation, `max_chunk_tokens` enforcement, `boundary_sim` metadata, semantic vs late tag distinction, factory `chunk_size` → `max_chunk_tokens` wiring.
- `tests/unit/test_router.py` — 28 new tests: `ChunkComplexity` enum values, `CHUNK_SIZE_MAP` ordering, `classify_chunk_complexity` return type, size-map consistency, simple/complex query sizing, empty-query error, monotonicity.

### Tests
- Suite: **423 passed, 0 failed** (up from 329)

---

## [Unreleased] — Sprint 12 (initial scaffold, superseded above)

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

## [Unreleased] — Sprint 10 (initial scaffold, superseded above)

> **Note:** Sprint 10 scaffolding (`konjoai/ingest/adaptive_chunker.py`) was committed prior to this session.  The full Sprint 10 deliverables (SemanticSplitter, LateChunker, QueryComplexityRouter, ablation harness) are documented in the entry above.

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
