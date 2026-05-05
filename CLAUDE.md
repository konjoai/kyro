# kyro

Production RAG pipeline with hybrid retrieval (BM25 + dense), HyDE, ColBERT reranking, RAGAS evaluation, streaming agent, GraphRAG, distributed semantic cache (Redis), multi-tenancy, OTel/Prometheus, and MCP server.

**v1.2.0** — 798 tests passing. Package name: `konjoai`.

## Stack
Python 3.10+ · FastAPI · Qdrant · sentence-transformers · PyTorch · OpenAI/Anthropic SDK · RAGAS · Redis · Prometheus · OpenTelemetry · Helm · MCP

## Commands
```bash
python -m pytest tests/ -x                   # full test suite
python -m pytest tests/ -x -k "unit"         # unit tests only
uvicorn konjoai.main:app --reload            # dev server
docker compose up                            # full stack (Qdrant + Redis + API)
helm upgrade --install kyro helm/kyro/       # Kubernetes deploy
```

## Critical Constraints
- No `unwrap()` — raise with a clear message or log + re-raise
- No silent failures — `logging.warning` if a fallback path swallows a real error
- Redis is **optional** — `cache_backend="redis"` must fall back to in-memory when `redis` package is absent or `PING` fails (K3 gate)
- `q_vec` must be float32 at the `store()` boundary — assert on both cache backends (K4 gate)
- Prompt injection: system prompt content must never be controllable by request payload
- JWT/tenant prefix in every Redis cache key — cross-tenant lookups must be impossible (K7 gate)
- Never log raw user query content at INFO level — log a hash or truncated prefix in production
- Rate-limit all endpoints by default
- Version bumps touch `pyproject.toml` + `konjoai/__init__.py` + `helm/kyro/Chart.yaml` + `docs/index.md`

## Module Map
| Module | Role |
|--------|------|
| `konjoai/api/` | FastAPI routers: query, agent, health, stream |
| `konjoai/retrieval/` | BM25 + dense hybrid retriever, HyDE, ColBERT reranker |
| `konjoai/cache/` | `SemanticCache` (in-memory) + `RedisSemanticCache` + factory |
| `konjoai/agent/` | Bounded ReAct loop, streaming agent (SSE), tool registry |
| `konjoai/graph/` | GraphRAG: knowledge graph construction + entity retrieval |
| `konjoai/chunking/` | Adaptive chunking, CRAG, self-RAG, query decomposition |
| `konjoai/sdk/` | Python SDK client + `SDKAgentStreamEvent` |
| `konjoai/config.py` | `KyroConfig` — Pydantic settings, backend selection |
| `evals/` | RAGAS evaluation harness |

## Planning Docs
- `PLAN.md` — current sprint state and version history
- `KORE_PLAN.md` — strategic roadmap and market analysis
- `kyro_production_plan.md` — production rollout and operational plan
- `CHANGELOG.md` — all notable changes

## Skills
See `.claude/skills/` — auto-loaded when relevant.
Run `/konjo` to boot a full session (Brief + Discovery + Plan).
