# Konjo KORE — Strategic Roadmap

> **ቆንጆ** — Beautiful. **根性** — Fighting spirit. **康宙** — Health of the universe. **खोजो** — Search and discover.
> *Make it konjo — build, ship, rest, repeat.*

**Version:** v0.3.0+ (KORE era)
**Owner:** Wesley Scholl / Konjo AI Research
**Last Updated:** 2026-07

---

## What Is Konjo KORE?

Konjo KORE is **the pipeline connector** — the data plane that binds together Vectro
(embedding compression), Squish (local LLM inference), Qdrant (vector store), and any
future component into a single, lightning-fast, observable RAG kernel.

**KORE = the connector that connects all the pieces.**

Unlike LangChain or LlamaIndex, KORE is:

| Property | LangChain / LlamaIndex | Konjo KORE |
|---|---|---|
| Design philosophy | Maximize abstraction | Minimize indirection |
| Transitive deps | 200+ | < 20 |
| Built-in telemetry | Callback hell | `timed()` on every step |
| Embedding compression | None | Vectro (4.85× FAISS, SIMD) |
| Local LLM | 3rd-party adapters | Squish (first-class) |
| Eval harness | External | RAGAS wired in |
| Data residency | Requires configuration | Local-first by default |

---

## Market Analysis

### Open Source Community Value

**Who needs this:** ML engineers, researchers, teams building private knowledge bases.

**What they want:**
- A drop-in, configuration-driven RAG backend that does not require reading 40 source files.
- Observable by default (they can profile every step, not just the LLM call).
- Local-first (no API keys required: Squish + Qdrant + local embedding models).
- Not bloated — no 200-dep install that breaks their environment.

**How KORE wins:**
- Apache 2.0 — permissive, safe to adopt in commercial projects.
- One-command demo: `docker compose up && curl -X POST /query`.
- Benchmark-first: every claim has a number behind it (Vectro 4.85×, p95 latency, RAGAS scores).
- Blog-worthy architecture: HyDE + ColBERT + hybrid search in a single, readable pipeline.

**Comparable OSS traction:** Haystack (deepset) — ~18K GitHub stars, built the same niche.
KORE's differentiation: **SIMD-compressed embeddings + local LLM as first-class citizens**, not afterthoughts.

---

### Business Community Value

**Who needs this:** Startups and mid-size companies building document Q&A, internal search,
customer support bots, or code assistants.

**What they want:**
- Production-ready out of the box (auth, streaming, Docker, observability).
- Pluggable backends (swap Qdrant for Pinecone without rewriting the pipeline).
- Cost control: Vectro INT8 compression → 4× fewer vectors stored → 4× lower vector DB bill.
- Eval harness built in (RAGAS in every deploy, not an afterthought sprint before launch).

**KORE's value proposition (dollars):**
- Vectro INT8 compression: a 10M-vector Qdrant collection → 2.5M vectors with ≥0.9999 cosine sim.
  At Pinecone's $0.096/M vectors/month, that is ~$720/month saved on a mid-scale deployment.
- Semantic cache (Sprint 6): repeat queries (FAQ patterns) → sub-1ms response, zero Qdrant calls.
  For a 1000 req/day deployment with 30% repeat rate: 300 req/day × zero LLM cost = real savings.
- Local LLM (Squish): no OpenAI API cost for high-volume deployments.

---

### Enterprise Value

**Who needs this:** Companies with data residency requirements, compliance mandates,
multi-tenant SaaS products, or security-sensitive knowledge bases.

**What they need that open-source alone doesn't deliver:**

| Feature | Current KORE | Enterprise Add-on |
|---|---|---|
| Auth / API Keys | ❌ | ✅ Sprint 11 |
| Multi-tenant namespace isolation | ❌ | ✅ Sprint 9 |
| Audit logging (who queried what, when) | ❌ | ✅ Sprint 11 |
| Role-based access control | ❌ | ✅ Sprint 11 |
| SSO / SAML | ❌ | Future |
| OpenTelemetry + Prometheus | ❌ | ✅ Sprint 10 |
| SLA gates (latency alerts) | ❌ | ✅ Sprint 10 |
| Kubernetes Helm chart | docker-compose only | ✅ Sprint 12 |
| Data residency (fully local stack) | ✅ already (Squish+Vectro+Qdrant) | ✅ already |

**Data residency is KORE's biggest enterprise moat.** Zero external API calls possible:
- Embeddings: local sentence-transformers model
- LLM: Squish (local inference)
- Vector DB: Qdrant (self-hosted)
- Reranker: local cross-encoder
- Result: a completely air-gapped RAG pipeline — invaluable to defense, finance, healthcare.

---

### Licensing Model Recommendation

**Verdict: Yes, KORE is worth licensing. Use the open-core model.**

**Tier 1 — Apache 2.0 (community):**
- Full pipeline: ingest, hybrid search, HyDE, ColBERT, RAGAS, telemetry, Vectro bridge
- Single-tenant, single-namespace
- No auth, no audit log
- Docker Compose deployment
- This gets the GitHub stars and community adoption.

**Tier 2 — Commercial Enterprise License ($15K–50K/year):**
- Multi-tenant namespace isolation (Sprint 9)
- Auth + API keys + RBAC (Sprint 11)
- Audit log (Sprint 11)
- OpenTelemetry + Prometheus + Grafana dashboard (Sprint 10)
- SLA tooling (latency gates, alerting)
- Priority support + private Slack
- This closes $15K–50K/year deals with companies that need compliance.

**Tier 3 — Managed KORE (SaaS, future):**
- Hosted KORE instance (no infra to run)
- $50–500/month per workspace
- Automatic Vectro compression, Squish inference via API
- Converts the portfolio into recurring revenue.

**Comparable:** deepset (Haystack) → ~€8M ARR with this exact model.
**Realistic Year 1 target:** 500 GitHub stars → 5 enterprise inquiries → 2 paying customers
at $20K average = $40K ARR. By Year 2: $200K ARR is achievable with focused sales.

---

## Sprint Roadmap: v0.3.0 → v1.0.0

### ✅ Sprint 1–5 (Complete) — Foundation + Intelligence

- Telemetry (`timed()` on every step)
- Query routing (CHAT/RETRIEVAL/AGGREGATION)
- HyDE (hypothesis document embedding)
- Vectro bridge (INT8 embedding compression, 4.85× FAISS)
- ColBERT MaxSim (late interaction reranking)
- Streaming query response
- RAGAS eval harness (mock + live modes)
- **Result: 205 tests passing**

---

### 🔄 Sprint 6 — Semantic Cache (Lightning-Fast Repeat Queries)

**Goal:** Sub-1ms responses for semantically similar queries. Eliminate LLM cost for FAQ patterns.

**Motivation:** In production, 20–40% of queries are near-duplicates (the same question
rephrased). Every one currently runs the full pipeline: embed → Qdrant → rerank → LLM.
The semantic cache intercepts these with a cosine similarity gate.

**Architecture:**
```
User Question
    │
    ▼
[if cache_enabled]
    ├── Exact string match (O(1) dict) ──────────────────→ cache hit → return
    └── Embed question, cosine similarity scan (O(N·dim))
            ├── sim ≥ threshold ──────────────────────────→ cache hit → return
            └── miss → continue to full pipeline ──────────────────────────────→
                                                                                  │
Full Pipeline (route → hyde → hybrid_search → rerank → generate)                 │
    │                                                                              │
    └── store (question_vec, response) in LRU cache ←──────────────────────────────
```

**Implementation:**
- `konjoai/cache/semantic_cache.py` — `SemanticCache` with `lookup()`, `store()`, `invalidate()`, `stats()`
- LRU eviction via `collections.OrderedDict` — no deps (K5)
- Config: `cache_enabled=False`, `cache_similarity_threshold=0.95`, `cache_max_size=500`
- Wire into query route: embed once, use for both cache and dense retrieval (zero extra cost on miss)
- Wire into ingest route: `cache.invalidate()` on every ingest (stale data protection)
- `QueryResponse.cache_hit: bool = False` (K6: backward-compatible default)

**Gates:**
- `pytest tests/unit/test_semantic_cache.py` — 15+ tests covering lookup, store, LRU eviction, invalidation
- Cache hit latency: < 5 ms wall (measured in test)
- Zero regression in existing 205 tests

---

### Sprint 7 — Abstract Adapter Architecture (The Connector Pattern)

**Goal:** Make KORE genuinely "the connector" — swap any vector store, embedder, or generator
without touching the pipeline code.

**Why this matters:** Currently, KORE is hard-coded to Qdrant + SentenceTransformer.
To support Pinecone or Chroma, someone must fork the codebase. With adapters, they write 50
lines of Python.

**Implementation:**
- `konjoai/adapters/base.py` — abstract ABCs:
  ```python
  class VectorStoreAdapter(ABC):
      def upsert(self, vecs: np.ndarray, ...) -> dict: ...
      def search(self, q_vec: np.ndarray, top_k: int) -> list[SearchResult]: ...

  class EmbedderAdapter(ABC):
      def encode(self, texts: list[str]) -> np.ndarray: ...
      def encode_query(self, text: str) -> np.ndarray: ...

  class GeneratorAdapter(ABC):
      def generate(self, question: str, context: str) -> GenerationResult: ...
  ```
- `konjoai/adapters/qdrant_adapter.py` — wraps existing `QdrantStore`
- `konjoai/adapters/st_adapter.py` — wraps existing `SentenceEncoder`
- Config: `vector_store_type: str = "qdrant"` — extensible registry pattern
- Documentation: "How to add a new vector store in 50 lines"

**Gates:**
- All existing tests pass (adapters are wrappers, not replacements)
- `QdrantAdapter` passes the `VectorStoreAdapter` contract test suite
- README updated with "Supported Backends" section

---

### Sprint 8 — Async Pipeline + Connection Pooling

**Goal:** 3× throughput under concurrent load. Sub-100ms hybrid search.

**Motivation:** FastAPI routes are currently `def` (synchronous). Under concurrent load,
the event loop blocks on embedding calls and Qdrant I/O. Converting to `async def` with
`asyncio.to_thread()` for CPU-bound operations and an async Qdrant client removes this bottleneck.

**Implementation:**
- Convert `query` and `ingest` routes to `async def`
- `asyncio.to_thread()` for CPU-bound: `encoder.encode()`, `reranker.rerank()`
- `QdrantClient(prefer_grpc=True)` already exists — enable async client variant
- Connection pool for Qdrant client (reuse connections across requests)
- Batch embedding queue: accumulate concurrent ingest requests, batch-encode

**Gates:**
- Concurrent load test: 10 simultaneous queries — p95 < 1500 ms
- Zero regression in existing tests (async wrappers are backward-compatible)
- RSS does not grow under sustained load (memory leak test)

---

### Sprint 9 — Multi-tenant Namespace Isolation

**Goal:** Run multiple independent knowledge bases on the same KORE instance.

**Use case:** A SaaS company with 50 customers each needing isolated document search.
Today: deploy 50 separate KORE instances. After Sprint 9: one instance, 50 namespaces.

**Implementation:**
- `X-Konjo-Namespace: {tenant_id}` request header
- Namespace-scoped Qdrant: payload filter `tenant_id == X` OR collection-per-namespace
  (config: `namespace_isolation_mode: "filter" | "collection"`)
- Namespace-scoped BM25 index (dict of sparse indexes, keyed by namespace)
- Namespace-scoped semantic cache (cache entries tagged with namespace)
- Config: `enable_multitenancy: bool = False`, `namespace_isolation_mode: str = "filter"`
- K3: no namespace header → default namespace `"global"` (backward-compatible)

**Gates:**
- Integration test: two namespaces, query namespace A never returns namespace B documents
- Namespace isolation verified with 100% confidence in test suite
- Memory contract: 10 namespaces × 10K vectors each < 2 GB RSS

---

### Sprint 10 — OpenTelemetry + Prometheus Observability

**Goal:** KORE emits production-grade telemetry compatible with any observability stack.

**Motivation:** The current `timed()` system is excellent for development but not for
production monitoring. Enterprise buyers require OpenTelemetry traces (for Datadog, Jaeger,
Honeycomb) and Prometheus metrics (for Grafana dashboards).

**Implementation:**
- `opentelemetry-api` + `opentelemetry-sdk` (optional dep, K5 — no hard dep)
- OTEL span for each pipeline step, replacing `timed()` (backward-compatible: `timed()` stays)
- `/metrics` Prometheus endpoint (fastapi-metrics or manual)
- Metrics to expose:
  - `konjo_query_duration_ms` (histogram, p50/p95/p99)
  - `konjo_cache_hit_total` / `konjo_cache_miss_total`
  - `konjo_ingest_chunks_total`
  - `konjo_pipeline_errors_total`
- Grafana dashboard template: `docs/grafana_dashboard.json`
- K3: OTEL not installed → silent fallback to `timed()`

**Gates:**
- `/metrics` returns valid Prometheus text format
- OTEL traces emitted and visible in Jaeger (local docker smoke test)
- Dashboard template renders without errors in Grafana 10+

---

### Sprint 11 — Auth + Access Control (Enterprise Gate)

**Goal:** API key authentication, rate limiting, and audit logging.

**Implementation:**
- `X-Konjo-Api-Key` header validation
- Key storage: `.env` file or environment variable `KONJO_API_KEYS=key1,key2,...`
  (no database dep for core; enterprise tier adds DB-backed key management)
- Rate limiting: token bucket per key (pure Python, no Redis dep for core)
- Audit log: structured JSON lines to `logs/audit.jsonl`:
  ```json
  {"ts": 1234567890, "key_hash": "sha256:...", "endpoint": "/query", "namespace": "global",
   "question_hash": "sha256:...", "latency_ms": 342, "cache_hit": false}
  ```
  — note: question hash only, never raw question text (OWASP logging rules)
- `KONJO_ADMIN_KEY` env var bootstraps the first admin key
- K3: no `KONJO_API_KEYS` set → auth disabled (open access, backward-compatible)

**Gates:**
- 401 returned when auth enabled and key invalid
- Rate limit triggered after configured threshold (integration test)
- Audit log entries created for every `/query` call
- No raw PII in audit log (hash only)

---

### Sprint 12 — SDK + Distribution

**Goal:** Make KORE the easiest RAG backend to adopt. `pip install`, `docker pull`, done.

**Implementation:**
- Python SDK: `konjo-core-sdk` package on PyPI
  ```python
  from konjo_sdk import KonjoClient
  client = KonjoClient("http://localhost:8000", api_key="...")
  response = client.query("What is the capital of Ethiopia?", stream=True)
  for chunk in response:
      print(chunk.text, end="", flush=True)
  ```
- TypeScript/Node SDK: `@konjo/core-sdk` on npm
  ```typescript
  import { KonjoClient } from "@konjo/core-sdk";
  const client = new KonjoClient({ baseUrl: "...", apiKey: "..." });
  const stream = client.query({ question: "...", stream: true });
  ```
- Helm chart for Kubernetes: `helm/konjo-core/`
- Docker Hub: `konjoai/konjo-core:latest` auto-published on release
- GitHub Actions: release workflow → PyPI + npm + Docker Hub + Helm registry

**Gates:**
- `pip install konjo-core-sdk && python -c "from konjo_sdk import KonjoClient; print('ok')"` works
- `npm install @konjo/core-sdk && node -e "require('@konjo/core-sdk')"` works
- Helm chart deploys cleanly on k3s (CI test)
- `docker compose up` → full stack running in < 60 seconds

---

## Current State: Sprint 5 Complete (v0.2.5)

**Test count:** 205 passing (0 failing)
**Branch:** `main`, HEAD `82f893b`

### Completed Components

```
konjoai/
├── telemetry.py              ✅ StepTiming, PipelineTelemetry, timed()
├── config.py                 ✅ 30 settings keys, pydantic-settings, .env
├── api/
│   ├── app.py                ✅ FastAPI app, CORS, routes mounted
│   ├── schemas.py            ✅ IngestRequest/Response, QueryRequest/Response, HealthResponse
│   └── routes/
│       ├── query.py          ✅ 5-step timed pipeline (route→hyde→hybrid→rerank→generate)
│       ├── ingest.py         ✅ load→chunk→embed→dedup→upsert→BM25→verify
│       ├── eval.py           ✅ RAGAS eval endpoint
│       └── vectro.py         ✅ /vectro/pipeline endpoint
├── embed/
│   ├── encoder.py            ✅ SentenceEncoder, float32 L2-normalised
│   └── vectro_bridge.py      ✅ INT8 bridge, float32 passthrough, compression metrics
├── retrieve/
│   ├── dense.py              ✅ dense_search(query, top_k)
│   ├── sparse.py             ✅ BM25Index, build(), search()
│   ├── hybrid.py             ✅ RRF fusion (alpha=0.7, k=60)
│   ├── hyde.py               ✅ HyDE hypothesis generation + encoding
│   ├── router.py             ✅ QueryIntent (CHAT/RETRIEVAL/AGGREGATION), regex O(1)
│   ├── reranker.py           ✅ CrossEncoderReranker, ms-marco-MiniLM-L-6-v2
│   ├── late_interaction.py   ✅ ColBERT MaxSim (Q×K×D → K scores)
│   └── vectro_retriever.py   ✅ VectroRetriever hybrid adapter
├── store/
│   └── qdrant.py             ✅ QdrantStore, upsert, search, vectro_quantize hook
├── ingest/
│   ├── loaders.py            ✅ .txt, .md, .pdf, .json loaders
│   ├── chunkers.py           ✅ recursive + sentence_window strategies
│   ├── dedup.py              ✅ near-duplicate chunk filtering (cosine threshold)
│   └── rag_bridge.py         ✅ corpus integrity verify
├── generate/
│   └── generator.py          ✅ OpenAI / Anthropic / Squish backends
├── services/
│   └── vectro_pipeline_service.py ✅ Vectro CLI bridge (NF4/PQ/INT8)
└── eval/
    └── ragas_eval.py         ✅ RAGAS harness (--mock, --live-retrieval, --n-samples)
```

### RAGAS Baseline (Sprint 5, mock upper-bound)

| Metric | Score | Gate |
|---|---|---|
| Faithfulness | 0.9333 | ≥ 0.80 ✅ |
| Context Precision | 1.0000 | ≥ 0.75 ✅ |
| Context Recall | 1.0000 | ≥ 0.75 ✅ |

### Vectro Benchmark (Sprint 5, live gate)

| Metric | Value | Gate |
|---|---|---|
| Compression ratio (INT8) | ≥ 4.0× | ✅ |
| Mean cosine similarity | ≥ 0.9999 | ✅ |

---

## The Seven Konjo Invariants

These are non-negotiable architectural properties. Any commit that violates one is a hard stop.

| # | Invariant | In Practice |
|---|---|---|
| K1 | **No silent failures.** | Every component returns a result or raises explicitly. No `except: pass`. |
| K2 | **Telemetry on every step.** | `timed()` context manager wraps all hot-path calls. Latency reported in every response. |
| K3 | **Graceful degradation everywhere.** | Vectro unavailable → float32 passthrough. RAGAS not installed → 501. BM25 not built → dense-only. Cache disabled → transparent fallthrough. |
| K4 | **Dtype contracts at boundaries.** | Encoder output: `float32`. Vectro input/output: `float32`. Qdrant vectors: `float32`. Assert, not assume. |
| K5 | **Zero new hard dependencies for new features.** | Telemetry uses `time.perf_counter()`. Router uses `re`. Cache uses `collections.OrderedDict` + numpy (already required). |
| K6 | **Backward-compatible API evolution.** | New response fields are optional with sensible defaults. Existing API consumers don't break. |
| K7 | **Reproducible evals.** | Every RAGAS run serialized to `evals/runs/<timestamp>_<name>/`. Never overwrite. Seeds logged. |

---

## Success Metrics: KORE v1.0.0

| Metric | Target | Sprint |
|---|---|---|
| Test suite | 0 failures, 300+ tests | All |
| Cached query latency | < 5 ms | S6 |
| Hybrid search latency p95 | < 200 ms | S8 |
| Full pipeline latency p95 | < 1000 ms | S8 |
| Concurrent throughput | 10 req/s sustained | S8 |
| Namespace isolation | 100% verified | S9 |
| Auth: invalid key → 401 | Always | S11 |
| Docker compose → running | < 60 seconds | S12 |
| PyPI install | `pip install konjo-core-sdk` works | S12 |

---

## Hard Stops (do not proceed past these)

- Tests failing from a previous step.
- Dtype boundary assertion fails at any component interface.
- NaN/Inf in vectors passed to Qdrant.
- Cache returns stale data after ingest (invalidation must be verified).
- Multi-tenant isolation test fails (namespace A returns namespace B data).
- Audit log contains raw question text (PII leak, OWASP violation).
- Any new import that increases cold-start time by > 10%.
- A PR that regresses p95 query latency by > 5%.

---

*End of KORE_PLAN.md*
*Owner: wesleyscholl / Konjo AI Research*
*Update this file when architectural contracts change. Never let it drift from the actual implementation.*
