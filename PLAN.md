# Kyro — Master Plan

> **ቆንጆ** — Beautiful. **根性** — Fighting spirit. **康宙** — Health of the universe. **खोजो** — Search and discover.
> *Make it konjo — build, ship, rest, repeat.*

**Strategic documents:**
- [`KORE_PLAN.md`](KORE_PLAN.md) — full market analysis, sprint roadmap, licensing recommendation.
- [`kyro_production_plan.md`](kyro_production_plan.md) — production execution plan and operational rollout notes.

---

## Current State: Sprint 25 Complete — v1.5.0 SHIPPED

- **Tests:** 876 passing (+ 25 skipped), pre-existing failures due to missing optional packages
- **Branch:** `claude/advance-all-repos-T5j3q`
- **Stack:** FastAPI + HyDE + ColBERT + hybrid search + RAGAS + Vectro bridge + streaming + distributed semantic cache (Sprint 22 — Redis-backed, tenant-namespaced) + async cache + singleflight stampede protection (Sprint 23 — v1.3.0) + audit logging (Sprint 24 — v1.4.0) + **feedback collection (Sprint 25 — v1.5.0)** + adaptive chunking + CRAG + Self-RAG + Query Decomposition + Agentic RAG + Streaming Agent + GraphRAG + OTel + Prometheus + Multi-tenancy + JWT + Auth hardening + Rate limiting + Python SDK + MCP server + Helm chart + PyPI + Docs site

---

## Completed Sprint: Sprint 25 — Feedback Collection (v1.5.0)

**Goal:** Add a `POST /feedback` endpoint for user relevance feedback (thumbs-up/down + optional continuous relevance score) and a `GET /feedback/summary` aggregation endpoint. Feedback is stored in a thread-safe in-process ring buffer. Feature-flagged off by default (K3). OWASP-compliant: raw question text is NEVER accepted — only `question_hash` (the 16-hex-char SHA-256 prefix emitted by the query audit log). Zero new hard dependencies (stdlib only, K5).

### Implementation Checklist — Sprint 25

| # | File | Change | Status |
|---|---|---|---|
| 1 | `konjoai/feedback/models.py` | `FeedbackEvent` dataclass; `THUMBS_UP`, `THUMBS_DOWN`, `VALID_SIGNALS` constants | ✅ |
| 2 | `konjoai/feedback/store.py` | `FeedbackStore` thread-safe bounded deque; `query()`, `summary()`, `clear()`; `get_feedback_store()` singleton; `_reset_singleton()` test helper | ✅ |
| 3 | `konjoai/feedback/__init__.py` | Re-export all public symbols | ✅ |
| 4 | `konjoai/api/routes/feedback.py` | `POST /feedback` (201) + `GET /feedback/summary` (200); HTTP 404 when disabled (K3); `FeedbackRequest` validates signal enum; comment stored as hash only (OWASP) | ✅ |
| 5 | `konjoai/api/app.py` | Register `feedback_route.router` | ✅ |
| 6 | `konjoai/config.py` | `feedback_enabled: bool = False`, `feedback_max_events: int = 1000` | ✅ |
| 7 | `tests/unit/test_feedback.py` | 55 tests: model, store, config, API disabled/enabled, OWASP contract, package exports | ✅ |
| 8 | `tests/unit/test_packaging.py` | Added `test_feedback_importable` + version bump to 1.5.0 | ✅ |
| 9 | `pyproject.toml` + `konjoai/__init__.py` + `helm/kyro/Chart.yaml` + `docs/index.md` | Version bump 1.4.0 → 1.5.0 | ✅ |

### Sprint 25 Gate Results

1. **K1**: `FeedbackStore.record()` wraps writes in try/except; errors logged as warnings, never propagate to request path. ✅
2. **K3**: `feedback_enabled=False` (default) → `POST /feedback` and `GET /feedback/summary` return HTTP 404. ✅
3. **K5**: stdlib only — `collections.deque`, `threading.Lock`. Zero new hard deps. ✅
4. **K6**: all new config keys have sensible defaults; no existing route or config consumer is affected. ✅
5. **K7**: `FeedbackEvent.tenant_id` populated from request state; `GET /feedback/summary?tenant_id=<id>` filters by tenant. ✅
6. **OWASP**: `FeedbackRequest` has no `question` field — only `question_hash`. Comments stored as SHA-256 hash only. ✅
7. **Tests**: 876 passing (was 853 — +55 new, +1 from test_packaging). Pre-existing failures from missing optional packages unchanged. ✅

---

## Completed Sprint: Sprint 24 — Audit Logging (v1.4.0)

**Goal:** Wire the pre-scaffolded `konjoai/audit/` package into the full API surface — query, ingest, and agent routes — expose a read-only query API (`GET /audit/events`, `GET /audit/stats`), and add four config fields (all off by default, K3). Zero new hard dependencies (stdlib only, K5). OWASP-compliant: raw question text and document paths are NEVER stored — only SHA-256 hashes.

### Implementation Checklist — Sprint 24

| # | File | Change | Status |
|---|---|---|---|
| 1 | `konjoai/api/routes/audit.py` | New `GET /audit/events` + `GET /audit/stats` endpoints; 404 when disabled (K3) | ✅ |
| 2 | `konjoai/api/app.py` | Register `audit_route.router` | ✅ |
| 3 | `konjoai/config.py` | `audit_enabled`, `audit_backend`, `audit_log_path`, `audit_max_memory_events` | ✅ |
| 4 | `konjoai/api/routes/query.py` | Emit `QUERY` `AuditEvent` after pipeline (K3 gated) | ✅ |
| 5 | `konjoai/api/routes/ingest.py` | Emit `INGEST` `AuditEvent` after ingest (K3 gated) | ✅ |
| 6 | `konjoai/api/routes/agent.py` | Emit `AGENT_QUERY` `AuditEvent` after agent run (K3 gated) | ✅ |
| 7 | `tests/unit/test_audit.py` | 36 tests: hash, model, backends, logger, singleton, API, OWASP contract, config | ✅ |
| 8 | `tests/unit/test_agent_route.py` + 5 other route test files | `audit_enabled: bool = False` added to all `_Settings*` stubs (K6) | ✅ |
| 9 | `pyproject.toml` + `konjoai/__init__.py` + `helm/kyro/Chart.yaml` + `docs/index.md` + `tests/unit/test_packaging.py` | Version bump 1.3.0 → 1.4.0 | ✅ |

### Sprint 24 Gate Results

1. **K1**: backend write errors are caught and emitted as `logger.warning()`; never propagate to the request path. ✅
2. **K3**: `audit_enabled=False` (default) → every `log()` is a pure no-op (early return); both API endpoints return 404. ✅
3. **K5**: stdlib only — `hashlib`, `json`, `threading`, `collections.deque`, `pathlib`. Zero new hard deps. ✅
4. **K6**: all four new config keys have sensible defaults; six existing route test stubs extended with `audit_enabled=False`. ✅
5. **K7**: `AuditEvent.tenant_id` populated from request state; `GET /audit/events?tenant_id=<id>` filters by tenant. ✅
6. **OWASP**: raw question text and file paths appear only as 16-hex-char SHA-256 hashes in every serialized event. ✅
7. **Tests**: 853 passing (was 810 — +43 new). 15 skipped, 5 pre-existing Py3.9 compat failures unchanged. ✅

---

## Completed Sprint: Sprint 23 — Async Cache + Singleflight (v1.3.0)

**Goal:** Eliminate the cache-stampede / thundering-herd on hot fresh queries — the failure mode the Sprint-22 Redis fan-out *worsened* by letting replicas race each other into the LLM. Do it without changing the synchronous backend contract.

### Implementation Checklist — Sprint 23

| # | File | Change | Status |
|---|---|---|---|
| 1 | `konjoai/cache/async_cache.py` | `AsyncSemanticCache` async wrapper + `get_or_compute` singleflight primitive; tenant-namespaced in-flight keys; error propagation to all waiters; `wrap()` factory | ✅ |
| 2 | `konjoai/cache/__init__.py` | Re-export `AsyncSemanticCache` and `async_wrap` | ✅ |
| 3 | `tests/unit/test_async_cache.py` | 12 tests: pass-through, stampede collapse (8→1), error propagation (5 waiters), retry after error, tenant scoping, singleflight=False bypass, offload_to_thread=False, wrap() parity | ✅ |
| 4 | `pyproject.toml` + `konjoai/__init__.py` + `helm/kyro/Chart.yaml` + `docs/index.md` + `tests/unit/test_packaging.py` | Version bump 1.2.0 → 1.3.0 | ✅ |
| 5 | `CHANGELOG.md` + `PLAN.md` | Document the new wrapper, the singleflight invariant, the stampede metric | ✅ |

### Sprint 23 Gate Results

1. **K1**: leader exceptions propagate to every waiter; in-flight slot freed in `finally` so retries proceed. ✅
2. **K3**: `singleflight=False` and `offload_to_thread=False` are honoured as a thin-adapter mode. ✅
3. **K5**: pure stdlib `asyncio`. Zero new hard deps. ✅
4. **K6**: synchronous `SemanticCache` / `RedisSemanticCache` contracts unchanged. ✅
5. **K7**: in-flight key namespaces by tenant — concurrent same-question requests across tenants never collapse onto each other. ✅
6. **Tests**: 810 passing (was 798 — +12 new). 15 skipped, 5 pre-existing Py3.9 compat failures unchanged. ✅

---

## Completed Sprint: Sprint 22 — Distributed Semantic Cache (Redis backend) (v1.2.0)

**Goal:** Make the Sprint-6 semantic cache survive pod restarts and shard across the Helm HPA topology (2–10 replicas), without breaking the in-memory contract or adding a hard dependency on Redis.

### Implementation Checklist — Sprint 22

| # | File | Change | Status |
|---|---|---|---|
| 1 | `konjoai/cache/redis_cache.py` | New `RedisSemanticCache` — tenant-namespaced HASH + LRU ZSET, `_safely()` wrapper, optional TTL, pickled entries with l2-normalised float32 bytes | ✅ |
| 2 | `konjoai/cache/redis_cache.py::build_redis_cache(...)` | Lazy `import redis`; PING-and-fallback factory returning `None` on missing package or connection error | ✅ |
| 3 | `konjoai/cache/__init__.py` | Re-export `RedisSemanticCache` and `build_redis_cache` | ✅ |
| 4 | `konjoai/cache/semantic_cache.py::get_semantic_cache()` | Backend dispatch: `memory` | `redis` (with K3 fallback to memory) | ✅ |
| 5 | `konjoai/config.py` | `cache_backend`, `cache_redis_url`, `cache_redis_namespace`, `cache_redis_ttl_seconds` | ✅ |
| 6 | `tests/unit/test_redis_cache.py` | 29 tests across construction, roundtrip, LRU, tenant isolation, invalidate, stats, TTL, graceful degradation, factory dispatch, backend parity | ✅ |
| 7 | `pyproject.toml` + `konjoai/__init__.py` + `helm/kyro/Chart.yaml` + `docs/index.md` + `tests/unit/test_packaging.py` | Version bump 1.1.0 → 1.2.0 | ✅ |
| 8 | `README.md` + `CHANGELOG.md` + `docs/configuration.md` | Document new backend selection + Redis settings | ✅ |

### Sprint 22 Gate Results

1. **K1**: Redis transport errors are logged and surfaced as cache misses; no swallowed business-logic exceptions. ✅
2. **K3**: `cache_backend="redis"` with `redis` not installed or `PING` failing → in-memory fallback, request paths stay green. ✅
3. **K4**: `q_vec` float32 assertion enforced at the `store()` boundary on both backends. ✅
4. **K5**: `redis` is **optional** — full suite runs without it. ✅
5. **K6**: Default `cache_backend="memory"` preserves v1.1.0 behaviour byte-for-byte; new keys are additive. ✅
6. **K7**: Tenant prefix in every Redis key — cross-tenant lookups cannot leak. ✅
7. **Tests**: 798 passing (was 769 — +29 new). 15 skipped, 5 pre-existing Py3.9 compat failures unchanged. ✅

---

## Completed Sprint: Sprint 21 — Streaming Agent (`POST /agent/query/stream`) (v1.1.0)

**Goal:** Bridge the bounded ReAct loop to a Server-Sent Events stream so callers see each Thought/Action/Observation in real time instead of waiting for the full multi-step run to complete.

### Implementation Checklist — Sprint 21

| # | File | Change | Status |
|---|---|---|---|
| 1 | `konjoai/agent/react.py` | `RAGAgent.run_stream()` — sync generator yielding `step` events + final `result` event; `run()` refactored to consume it | ✅ |
| 2 | `konjoai/api/routes/agent.py` | `POST /agent/query/stream` — SSE endpoint, `asyncio.Queue` bridge from `to_thread`, timeout-bounded, telemetry frame, `[DONE]` sentinel | ✅ |
| 3 | `konjoai/sdk/models.py` | `SDKAgentStreamEvent` typed event | ✅ |
| 4 | `konjoai/sdk/client.py` | `KonjoClient.agent_query_stream()` iterator | ✅ |
| 5 | `konjoai/sdk/__init__.py` | Export `SDKAgentStreamEvent` | ✅ |
| 6 | `tests/unit/test_agentic.py` | +4 tests: stream sequencing, parser fallback, empty-question reject, `run()` parity | ✅ |
| 7 | `tests/unit/test_agent_route.py` | +4 tests: SSE frame contract, telemetry-off, 504 timeout, 422 empty-question | ✅ |
| 8 | `tests/unit/test_sdk.py` | +4 tests: typed events, `[DONE]` stop, malformed/typeless skip, timeout mapping | ✅ |
| 9 | `pyproject.toml` + `konjoai/__init__.py` + `helm/kyro/Chart.yaml` + `docs/index.md` + `tests/unit/test_packaging.py` | Version bump 1.0.0 → 1.1.0 | ✅ |
| 10 | `README.md` + `CHANGELOG.md` | Document new endpoint and bumped version | ✅ |

### Sprint 21 Gate Results

1. K1: producer-side exceptions in `run_stream()` propagate through the SSE bridge and surface to the route layer. ✅
2. K2: `agent_stream` step wrapped in `timed()`; timeout path emits `logger.warning`. ✅
3. K3: SDK iterator silently drops malformed/typeless frames; new endpoint is purely additive. ✅
4. K5: pure stdlib bridge (`asyncio.Queue` + `asyncio.to_thread`). Zero new hard deps. ✅
5. K6: `RAGAgent.run()`, `POST /agent/query`, and all existing SDK methods unchanged. ✅
6. **Tests:** 769 passing (was 757 — +12 new). 15 skipped, 5 pre-existing Py3.9 compat failures unchanged. ✅

---

## Active Sprint Delta: Sprint 14 — Agentic RAG Foundation (v0.8.0, Wave 1)

**Goal:** Add a bounded ReAct loop with tool-use traceability and expose it via `/agent/query`.

### Implementation Checklist — Sprint 14 (Wave 1)

| # | File | Change | Status |
|---|---|---|---|
| 1 | `konjoai/agent/react.py` | Add `RAGAgent`, `ToolRegistry`, action parser, bounded ReAct loop | ✅ |
| 2 | `konjoai/agent/__init__.py` | Export `RAGAgent`, `AgentResult`, `AgentStep` | ✅ |
| 3 | `konjoai/api/routes/agent.py` | Add `POST /agent/query` endpoint + telemetry wrapping | ✅ |
| 4 | `konjoai/api/app.py` | Register `agent` router | ✅ |
| 5 | `tests/unit/test_agentic.py` | Add agent core tests (retrieve/finish, parser fallback, max-step guard) | ✅ |
| 6 | `tests/unit/test_agent_route.py` | Add route tests for telemetry on/off and response contract | ✅ |
| 7 | `konjoai/api/routes/agent.py` + `tests/unit/test_agent_route.py` | Enforce `request_timeout_seconds` on `/agent/query`; return HTTP 504 on timeout | ✅ |

### Sprint 14 Gate (Wave 1)

1. Agent loop is bounded (`max_steps`) and never runs unbounded. ✅
2. Tool action trace is returned in API response (`steps[]`). ✅
3. Endpoint preserves K3/K6 behavior (telemetry optional, no breaking change to `/query`). ✅
4. Focused unit tests pass for new agent core and route. ✅
5. Endpoint timeout is enforced and returns deterministic 504 on overrun. ✅

---

## Completed Sprint: Sprint 20 — Helm chart + PyPI packaging + Docs site (v1.0.0)

**Goal:** Ship Kyro as a fully installable, deployable, and documented v1.0.0 release. Produce a production Helm chart, complete `pyproject.toml` packaging with optional extras, a GitHub Actions release workflow (PyPI + Docker Hub + Helm OCI), and an MkDocs documentation site.

### Implementation Checklist — Sprint 20

| # | File | Change | Status |
|---|---|---|---|
| 1 | `pyproject.toml` | Bump to v1.0.0; add `classifiers`, `[project.optional-dependencies]` (jwt, mcp, eval, observability, dev, all), `[project.urls]`, `authors.email`, `[tool.hatch.build]` | ✅ |
| 2 | `konjoai/__init__.py` | `__version__ = "1.0.0"` | ✅ |
| 3 | `helm/kyro/Chart.yaml` | `apiVersion: v2`, `name: kyro`, `version/appVersion: 1.0.0`, `type: application` | ✅ |
| 4 | `helm/kyro/values.yaml` | `replicaCount=2`, `autoscaling.enabled=true`, `config.*` env-var map, `secrets.*`, `livenessProbe/readinessProbe`, `resources`, `ingress` | ✅ |
| 5 | `helm/kyro/templates/_helpers.tpl` | `kyro.name`, `kyro.fullname`, `kyro.chart`, `kyro.labels`, `kyro.selectorLabels`, `kyro.serviceAccountName` | ✅ |
| 6 | `helm/kyro/templates/deployment.yaml` | `apps/v1` Deployment; `envFrom: configMapRef`, secret env injections, liveness/readiness probes, security context | ✅ |
| 7 | `helm/kyro/templates/service.yaml` | `v1` ClusterIP Service, port 8000 | ✅ |
| 8 | `helm/kyro/templates/configmap.yaml` | `v1` ConfigMap — all `config.*` values as env vars | ✅ |
| 9 | `helm/kyro/templates/hpa.yaml` | `autoscaling/v2` HPA — CPU + memory targets, conditional on `autoscaling.enabled` | ✅ |
| 10 | `helm/kyro/templates/ingress.yaml` | `networking.k8s.io/v1` Ingress — conditional on `ingress.enabled` | ✅ |
| 11 | `.github/workflows/release.yml` | Tag-triggered: test → build → PyPI (OIDC trusted publishing) + Docker (multi-arch amd64/arm64) + Helm OCI + GitHub Release | ✅ |
| 12 | `mkdocs.yml` | Material theme, nav: index/quickstart/sdk/mcp/api/configuration/deployment | ✅ |
| 13 | `docs/index.md` | Feature table, version badge | ✅ |
| 14 | `docs/quickstart.md` | Install, Docker Compose, first query, optional features table | ✅ |
| 15 | `docs/sdk.md` | Full KonjoClient API, auth, error handling, response models | ✅ |
| 16 | `docs/mcp.md` | Claude Desktop integration, tool reference, programmatic use | ✅ |
| 17 | `docs/api.md` | All endpoint request/response contracts | ✅ |
| 18 | `docs/configuration.md` | All 40+ env vars with defaults and descriptions | ✅ |
| 19 | `docs/deployment.md` | Docker Compose, Helm install/upgrade, production checklist | ✅ |
| 20 | `tests/unit/test_packaging.py` | 52 tests: version, classifiers, extras, URLs, entry points, imports, mkdocs, docs pages | ✅ |
| 21 | `tests/unit/test_helm.py` | 25 tests: directory structure, Chart.yaml, values.yaml, workflow trigger/jobs | ✅ |

### Sprint 20 Gate Results

1. `konjoai.__version__ == "1.0.0"` and matches `pyproject.toml`. ✅
2. Six optional extras: jwt, mcp, eval, observability, dev, all. ✅
3. Helm chart: 10 files, passes YAML structural validation. ✅
4. Release workflow: tag trigger, 5 jobs (test, build, pypi, docker, helm, github-release). ✅
5. Docs site: 7 pages, MkDocs Material config. ✅
6. **764 passed, 15 skipped** (up from 687 — +77 new tests). ✅

---

## Completed Sprint: Sprint 19 — Python SDK + MCP server (v0.9.8)

**Goal:** Ship a typed synchronous Python SDK (`konjoai.sdk.KonjoClient`) wrapping all Kyro API endpoints, and an MCP (Model Context Protocol) server (`konjoai.mcp.KyroMCPServer`) that exposes Kyro's RAG capabilities as tools consumable by any MCP-compatible agent. Both are production-grade, fully tested, and honour all Seven Konjo Invariants.

**Design:**
- `KonjoClient` wraps `httpx.Client` (already a hard dep via FastAPI/httpx ≥ 0.25) — zero new mandatory deps (K5). Typed response models use stdlib `dataclasses` to avoid transitive pydantic coupling in downstream consumers.
- `KyroMCPServer.dispatch()` routes tool calls and is fully testable without the `mcp` package; the stdio transport (`run_stdio()`) imports `mcp` lazily so the module is always importable (K3/K5 pattern, matching Sprint 16–18 precedent).

### Implementation Checklist — Sprint 19

| # | File | Change | Status |
|---|---|---|---|
| 1 | `konjoai/sdk/exceptions.py` | `KyroError`, `KyroAuthError`, `KyroRateLimitError` (retry_after), `KyroTimeoutError`, `KyroNotFoundError` | ✅ |
| 2 | `konjoai/sdk/models.py` | `SDKQueryResponse`, `SDKIngestResponse`, `SDKHealthResponse`, `SDKAgentQueryResponse`, `SDKAgentStep`, `SDKSourceDoc`, `SDKStreamChunk` — stdlib dataclasses, frozen | ✅ |
| 3 | `konjoai/sdk/client.py` | `KonjoClient`: `query()`, `query_stream()`, `ingest()`, `health()`, `agent_query()`, context-manager lifecycle; `httpx.Client` with X-API-Key / Bearer auth | ✅ |
| 4 | `konjoai/sdk/__init__.py` | Export all public SDK symbols | ✅ |
| 5 | `konjoai/mcp/server.py` | `KyroMCPServer`: `list_tools()`, `async dispatch()`, `from_url()` factory; `TOOLS` constant (4 tools with JSON Schema); `run_stdio()` with lazy mcp import | ✅ |
| 6 | `konjoai/mcp/__init__.py` | `_HAS_MCP` flag, exports | ✅ |
| 7 | `konjoai/mcp/__main__.py` | `python -m konjoai.mcp` CLI via click (--base-url, --api-key, --jwt-token, --timeout) | ✅ |
| 8 | `tests/unit/test_sdk.py` | 46 tests: exception hierarchy, model contracts, construction, query, stream, ingest, health, agent_query, lifecycle | ✅ |
| 9 | `tests/unit/test_mcp.py` | 29 tests: _HAS_MCP, TOOLS schema, server construction, dispatch × 4 tools, error propagation, run_stdio guard | ✅ |

### Sprint 19 Gate Results

1. `KonjoClient` wraps all six API surfaces: query, stream, ingest, health, agent. ✅
2. Streaming SSE correctly handles `data: {"token": ...}` and `data: [DONE]`. ✅
3. All HTTP error codes map to typed exceptions (401→KyroAuthError, 429→KyroRateLimitError with retry_after, 5xx→KyroError). ✅
4. MCP tool dispatch is testable without `mcp` installed (K3). ✅
5. `run_stdio()` raises `RuntimeError` with install hint when `_HAS_MCP=False` (K3). ✅
6. Zero new hard dependencies — httpx already in requirements (K5). ✅
7. **687 passed, 15 skipped** (up from 612 — +75 new tests). ✅

---

## Completed Sprint: Sprint 18 — Auth Hardening + Rate Limiting (v0.9.5)

**Goal:** Harden the auth layer with per-tenant sliding-window rate limiting, an optional static API-key authentication layer (accepted alongside JWT), and IP-based brute-force protection for auth endpoints. All new features are feature-flagged off by default (K3). Zero new mandatory dependencies (K5).

### Implementation Checklist — Sprint 18

| # | File | Change | Status |
|---|---|---|---|
| 1 | `konjoai/auth/rate_limiter.py` | `RateLimiter`: per-(tenant, endpoint) sliding-window rate limiter (deque of timestamps, per-bucket locks); `RateLimitExceeded`; `get_rate_limiter()` singleton | ✅ |
| 2 | `konjoai/auth/api_key.py` | `APIKeyResult`, `hash_api_key()`, `verify_api_key()`: SHA-256 + `hmac.compare_digest` timing-safe check against stored digest registry | ✅ |
| 3 | `konjoai/auth/brute_force.py` | `BruteForceGuard`: per-IP sliding-window failure tracker + lockout; `IPLockedOut`; `get_brute_force_guard()` singleton | ✅ |
| 4 | `konjoai/auth/deps.py` | `get_tenant_id` delegates to `_resolve_tenant_id`; evaluates X-API-Key before Bearer JWT; brute-force check/record integrated; `check_rate_limit` new dependency | ✅ |
| 5 | `konjoai/auth/__init__.py` | Updated exports: `APIKeyResult`, `verify_api_key`, `hash_api_key`, `RateLimiter`, `RateLimitExceeded`, `get_rate_limiter`, `BruteForceGuard`, `IPLockedOut`, `get_brute_force_guard` | ✅ |
| 6 | `konjoai/config.py` | `api_key_auth_enabled=False`, `api_keys=[]`, `rate_limiting_enabled=False`, `rate_limit_requests=60`, `rate_limit_window_seconds=60`, `brute_force_enabled=False`, `brute_force_max_attempts=5`, `brute_force_window_seconds=60`, `brute_force_lockout_seconds=300` | ✅ |
| 7 | `tests/unit/test_rate_limiter.py` | 30 tests: construction, sliding-window eviction, tenant/endpoint isolation, disabled mode, current_count, reset, thread safety, singleton | ✅ |
| 8 | `tests/unit/test_api_key_auth.py` | 32 tests: hash determinism, verify match/no-match/empty, tenant extraction, case-insensitive comparison, dep integration (valid key, invalid, disabled, context var cleanup, API key beats JWT) | ✅ |
| 9 | `tests/unit/test_brute_force.py` | 29 tests: construction, check_ip, record_failure/success, is_locked, failure_count, reset, disabled mode, thread safety, singleton, dep integration (429, failure increment, success clear) | ✅ |
| 10 | `tests/unit/test_auth.py` | Updated 6 `TestGetTenantIdDep` tests to call `_resolve_tenant_id` (internal helper that accepts `request=None`) to preserve backward compat with unit tests while `get_tenant_id` requires a real FastAPI `Request` | ✅ |

### Sprint 18 Gate Results

1. Rate limiting: per-(tenant, endpoint) sliding-window, pure Python, no Redis (K5). ✅
2. API key auth: SHA-256 + `hmac.compare_digest`, timing-safe, optional (K3, K5). ✅
3. Brute-force protection: per-IP failure count + lockout, in-memory, no external dep (K5). ✅
4. K3: all three features off by default — existing API unaffected with zero config change. ✅
5. K6: no breaking changes — all new config fields have sensible defaults. ✅
6. **607 passed, 15 skipped** (up from 509 — +98 new tests). ✅

---

## Completed Sprint: Sprint 17 — Multi-tenancy + JWT (v0.9.0)

**Goal:** Add per-tenant isolation so multiple organisations can share one Kyro deployment. Each tenant's embeddings are scoped by a `tenant_id` payload field in Qdrant; JWT HS256 authentication extracts the tenant from the `sub` claim. Feature-flagged off by default (K3). No breaking API changes (K6). PyJWT is an optional dep (K5).

**Design:** Python `contextvars.ContextVar` propagates the active `tenant_id` through the async task and into `asyncio.to_thread` threads, so `QdrantStore.search()` and `upsert()` automatically scope without any signature changes to `hybrid_search` or `dense_search`.

### Implementation Checklist — Sprint 17

| # | File | Change | Status |
|---|---|---|---|
| 1 | `konjoai/auth/__init__.py` | Package init exporting `TenantClaims`, `decode_token`, `_HAS_JWT`, `ANONYMOUS_TENANT`, `get_current_tenant_id`, `set_current_tenant_id` | ✅ |
| 2 | `konjoai/auth/tenant.py` | `_current_tenant_id` ContextVar, `get_current_tenant_id()`, `set_current_tenant_id()`, `ANONYMOUS_TENANT` sentinel | ✅ |
| 3 | `konjoai/auth/jwt_auth.py` | `TenantClaims` dataclass, `decode_token()`, `_HAS_JWT` guard; raises `RuntimeError` if PyJWT absent, `ValueError` on expired/invalid/missing-claim | ✅ |
| 4 | `konjoai/auth/deps.py` | `get_tenant_id` async generator FastAPI dep; K3 pass-through (returns `None`) when `multi_tenancy_enabled=False`; 401 / 503 on failure; sets ContextVar; cleanup in `finally` | ✅ |
| 5 | `konjoai/config.py` | `multi_tenancy_enabled=False`, `jwt_secret_key=""`, `jwt_algorithm="HS256"`, `tenant_id_claim="sub"` | ✅ |
| 6 | `konjoai/store/qdrant.py` | `upsert()`: adds `tenant_id` to point payload when ContextVar set; `search()`: adds `Filter(must=[FieldCondition(key="tenant_id", ...)])` when ContextVar set | ✅ |
| 7 | `konjoai/api/routes/ingest.py` | `tenant_id: str \| None = Depends(get_tenant_id)` injected (sets ContextVar as side-effect) | ✅ |
| 8 | `konjoai/api/routes/query.py` | `tenant_id: str \| None = Depends(get_tenant_id)` injected on both `/query` and `/query/stream` | ✅ |
| 9 | `requirements.txt` | Optional `# PyJWT>=2.8` documented | ✅ |
| 10 | `tests/unit/test_auth.py` | 24 tests (+ 9 skipped without PyJWT): `TenantClaims`, `decode_token`, `_HAS_JWT`, ContextVar, `get_tenant_id` dep (K3/401/503/valid), `QdrantStore` tenant scoping | ✅ |

### Sprint 17 Gate Results

1. Tenant isolation: `search()` filtered by `tenant_id` FieldCondition; `upsert()` stamps `tenant_id` payload. ✅
2. K3: `multi_tenancy_enabled=False` → `get_tenant_id` yields `None` → no filter → existing behaviour preserved. ✅
3. K6: no breaking changes — all new params/fields default to `None`/`False`. ✅
4. K5: PyJWT documented as optional; absent → `RuntimeError` only when auth actually attempted. ✅
5. `509 passed, 15 skipped` (up from 485 — +24 new). ✅

---

## Completed Sprint: Sprint 16 — OTel + Prometheus Observability (v0.8.7)

**Goal:** Add structured observability using OpenTelemetry (OTel) for distributed tracing and Prometheus for metrics. Feature-flagged off by default (K3). No breaking API changes (K6). No new hard dependencies (K5).

### Implementation Checklist — Sprint 16

| # | File | Change | Status |
|---|---|---|---|
| 1 | `konjoai/telemetry.py` | `KyroMetrics` (Prometheus counters/histograms), `KyroTracer` (OTel span wrapper), `_noop_span()`, `get_metrics()`, `get_tracer()`, `record_pipeline_metrics()`, `_HAS_PROMETHEUS`/`_HAS_OTEL` guards | ✅ |
| 2 | `konjoai/config.py` | `otel_enabled=False`, `otel_endpoint=""`, `otel_service_name="kyro"`, `prometheus_port=8001` | ✅ |
| 3 | `konjoai/api/routes/health.py` | `GET /metrics` Prometheus exposition; 404 when disabled, 503 when dep absent | ✅ |
| 4 | `konjoai/api/app.py` | Register `health_route.router` | ✅ |
| 5 | `konjoai/api/routes/query.py` | `record_pipeline_metrics(tel, intent.value, enabled=settings.otel_enabled)` after pipeline | ✅ |
| 6 | `requirements.txt` | Optional deps documented: `prometheus-client>=0.19`, `opentelemetry-sdk>=1.20`, `opentelemetry-exporter-otlp-proto-grpc>=1.20` | ✅ |
| 7 | `tests/unit/test_telemetry.py` | +26 tests: `_noop_span`, `KyroMetrics` disabled/enabled, `KyroTracer` disabled, singletons, `record_pipeline_metrics`, module flags | ✅ |
| 8 | Existing test stubs (5 files) | Added `otel_enabled: bool = False` to `_SettingsStub` in `test_query_crag_route.py`, `test_query_self_rag_route.py`, `test_query_decomposition_route.py`, `test_query_route_timeout.py`, `test_graph_rag.py` | ✅ |

### Sprint 16 Gate Results

1. All telemetry behind `if settings.otel_enabled` (K3). ✅
2. No breaking changes to existing routes when flag is off (K6). ✅
3. New deps optional or guarded with `_HAS_OTEL`/`_HAS_PROMETHEUS` (K5). ✅
4. All K1–K7 invariants pass on new code. ✅
5. `485 passed, 6 skipped` (up from 464 — +21 new). ✅

---

## Completed Sprint: Sprint 15 — Lightweight GraphRAG (v0.8.5)

**Goal:** Group semantically related retrieved chunks into communities via Louvain-style community detection. Surface one representative per community to the reranker — reducing near-duplicate context while preserving topical diversity. Feature-flagged off by default (K3).

### Implementation Checklist — Sprint 15

| # | File | Change | Status |
|---|---|---|---|
| 1 | `konjoai/retrieve/graph_rag.py` | `EntityGraph` (Jaccard-weighted graph), `GraphRAGRetriever` (community detection via `greedy_modularity_communities`), `CommunityContext`, `GraphRAGResult`, singleton factory | ✅ |
| 2 | `konjoai/config.py` | `enable_graph_rag=False`, `graph_rag_max_communities=5`, `graph_rag_similarity_threshold=0.3` | ✅ |
| 3 | `konjoai/api/schemas.py` | `QueryRequest.use_graph_rag: bool = False`; `QueryResponse.graph_rag_communities: list[str] \| None = None` | ✅ |
| 4 | `konjoai/api/routes/query.py` | Step 3c injection (after hybrid retrieval, before CRAG); `x-use-graph-rag` header support; `graph_rag_enabled` K3 gate | ✅ |
| 5 | `requirements.txt` | `networkx>=3.2` | ✅ |
| 6 | `tests/unit/test_graph_rag.py` | 37 tests covering `_tokenize`, `EntityGraph`, `CommunityContext`, `GraphRAGResult`, `GraphRAGRetriever`, singleton, and K3 route gate (disabled / settings / body / header) | ✅ |
| 7 | Existing test stubs (4 files) | Added `enable_graph_rag`, `graph_rag_max_communities`, `graph_rag_similarity_threshold` to `_SettingsStub` dataclasses | ✅ |

### Sprint 15 Gate Results

1. `greedy_modularity_communities` with Jaccard edges — deterministic, no extra embedding cost. ✅
2. K3: disabled by default; graceful networkx-absent fallback returns raw hybrid results. ✅
3. K6: `use_graph_rag=False` default; `graph_rag_communities=None` default — no breaking change. ✅
4. `464 passed, 0 failed` (up from 427 — +37 tests). ✅

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

## Phase 1 → Phase 2: Sprint 10 Complete — Adaptive Chunking (v0.5.5)

### Implementation Checklist — Sprint 10 (Full Deliverables)

| # | File | Change | Status |
|---|---|---|---|
| 1 | `konjoai/ingest/adaptive_chunker.py` | `QueryComplexityScorer`, `MultiGranularityChunker`, `AdaptiveRetriever` | ✅ |
| 2 | `konjoai/ingest/chunkers.py` | `SemanticSplitter` — cosine boundary detection via buffered sentence embedding | ✅ |
| 3 | `konjoai/ingest/chunkers.py` | `LateChunker` — Jina-style post-embedding split (single-batch full-doc encoding) | ✅ |
| 4 | `konjoai/ingest/chunkers.py` | `get_chunker()` updated to support `"semantic"` and `"late"` strategies | ✅ |
| 5 | `konjoai/retrieve/router.py` | `ChunkComplexity` enum (SIMPLE/MEDIUM/COMPLEX) + `CHUNK_SIZE_MAP` | ✅ |
| 6 | `konjoai/retrieve/router.py` | `classify_chunk_complexity()` — maps query → `(ChunkComplexity, chunk_size)` | ✅ |
| 7 | `konjoai/config.py` | `chunk_strategy` supports `"recursive"\|"sentence_window"\|"semantic"\|"late"` | ✅ |
| 8 | `konjoai/config.py` | `semantic_split_threshold`, `late_chunk_threshold` settings added | ✅ |
| 9 | `scripts/ablation_chunking.py` | Ablation harness — all 4 strategies, proxy metrics, JSON to `evals/runs/` | ✅ |
| 10 | `tests/unit/test_semantic_splitter.py` | 32 tests — construction, splitting, factory | ✅ |
| 11 | `tests/unit/test_late_chunker.py` | 33 tests — construction, splitting, metadata, factory | ✅ |
| 12 | `tests/unit/test_router.py` | 28 new tests — `ChunkComplexity`, `CHUNK_SIZE_MAP`, `classify_chunk_complexity` | ✅ |

**Sprint 10 Gate Results:** 423 passed, 0 failed (up from 329 — +94 tests)

### Implementation Checklist — Sprint 11: CRAG

| # | File | Change | Status |
|---|---|---|---|
| 1 | `konjoai/retrieve/crag.py` | Replace legacy pipeline with `CRAGEvaluator`: normalized scores, `CORRECT/AMBIGUOUS/INCORRECT`, fallback stub, ambiguous refinement | ✅ |
| 2 | `konjoai/config.py` | Add `crag_correct_threshold`, `crag_ambiguous_threshold` | ✅ |
| 3 | `konjoai/api/schemas.py` + `konjoai/api/routes/query.py` | Add `QueryRequest.use_crag`; support `use_crag` header; emit `crag_scores`, `crag_classification`, `crag_refinement_triggered` | ✅ |
| 4 | `tests/unit/test_crag.py` + `tests/unit/test_query_crag_route.py` | Add Sprint 11 contract tests for quality bands, fallback, refinement, and `/query` opt-in behavior | ✅ |

### Implementation Checklist — Sprint 12: Self-RAG

| # | File | Change | Status |
|---|---|---|---|
| 1 | `konjoai/retrieve/self_rag.py` | `SelfRAGOrchestrator` iterative loop + `SelfRAGCritic` + `SelfRAGTokens` + refined retrieval callback contract | ✅ |
| 2 | `konjoai/config.py` | `enable_self_rag`, `self_rag_max_iterations=3` | ✅ |
| 3 | `konjoai/api/schemas.py` + `konjoai/api/routes/query.py` | `QueryRequest.use_self_rag` opt-in + Self-RAG telemetry fields (`self_rag_iteration_scores`, `self_rag_total_tokens`) | ✅ |
| 4 | `tests/unit/test_self_rag.py` + `tests/unit/test_query_self_rag_route.py` | Critique/orchestrator unit tests + `/query` opt-in route tests (body/header/default-off) | ✅ |

### Implementation Checklist — Sprint 13: Query Decomposition + Multi-Step Retrieval

| # | File | Change | Status |
|---|---|---|---|
| 1 | `konjoai/retrieve/decomposition.py` | Added `QueryDecomposer` (LLM JSON output + deterministic fallback), `ParallelRetriever`, and `AnswerSynthesizer` | ✅ |
| 2 | `konjoai/config.py` | Added `enable_query_decomposition` and `decomposition_max_sub_queries` guards | ✅ |
| 3 | `konjoai/api/schemas.py` | Added `QueryRequest.use_decomposition` and decomposition response fields | ✅ |
| 4 | `konjoai/api/routes/query.py` | Added body/header opt-in, parallel fan-out retrieval, sub-answer synthesis, and decomposition telemetry | ✅ |
| 5 | `tests/unit/test_decomposition.py` + `tests/unit/test_query_decomposition_route.py` | Added decomposition unit coverage and `/query` opt-in route behavior tests | ✅ |

---

## Sprint Roadmap Summary (Production Release Plan)

| Sprint | Version | Phase | Focus | Gate |
|---|---|---|---|---|
| 1–5 | v0.2.5 | — | Foundation: telemetry, routing, HyDE, ColBERT, RAGAS | ✅ 205 tests |
| 6 | v0.3.0 | — | Semantic cache (sub-5ms cached responses) | ✅ 226 tests |
| **7** | **v0.3.5** | **P1** | **Adapter architecture (swap any backend)** | **✅ Active** |
| 8 | v0.4.0 | P1 | Async pipeline + connection pooling (3× throughput) | ⬜ |
| 9 | v0.5.0 | P1 | Streaming SSE (already exists; harden + OTel hooks) | ⬜ |
| 10 | v0.5.5 | P2 | Adaptive chunking (SemanticSplitter, LateChunker, ChunkComplexity router, ablation harness) | ✅ 427 tests |
| 11 | v0.6.0 | P2 | CRAG — retrieval critique + corrective fallback | ✅ |
| 12 | v0.7.0 | P2 | Self-RAG — reflection tokens + critique loop | ✅ |
| 13 | v0.7.5 | P3 | Query decomposition (multi-hop fan-out) | ✅ |
| 14 | v0.8.0 | P3 | Agentic RAG — ReAct loop | ✅ |
| 15 | v0.8.5 | P3 | Lightweight GraphRAG (NetworkX + Louvain) | ✅ 464 tests |
| 16 | v0.8.7 | P4 | OTel + Prometheus + Grafana | ✅ 485 tests |
| 17 | v0.9.0 | P4 | Multi-tenancy + JWT | ✅ 509 tests |
| 18 | v0.9.5 | P4 | Auth + rate limiting | ✅ 607 tests |
| 19 | v0.9.8 | P5 | Python SDK + MCP server | ✅ 687 tests |
| 20 | v1.0.0 | P5 | Helm chart + PyPI + Docs site | ✅ 764 tests |
| 21 | v1.1.0 | P5 | Streaming Agent (`POST /agent/query/stream`) | ✅ 769 tests |
| 22 | v1.2.0 | P5 | Distributed Semantic Cache (Redis backend) | ✅ 798 tests |
| 23 | v1.3.0 | P5 | Async Cache + Singleflight stampede protection | ✅ 810 tests |
| 24 | v1.4.0 | P6 | Audit logging (OWASP PII-safe event trail) | ✅ 853 tests |
| 25 | v1.5.0 | P6 | Feedback collection (thumbs-up/down + relevance score) | ✅ 876 tests |

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
