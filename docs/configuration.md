# Configuration

All settings are read from environment variables or a `.env` file in the project root.

## Core

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant instance URL |
| `QDRANT_API_KEY` | — | Qdrant API key (cloud deployments) |
| `QDRANT_COLLECTION` | `konjoai` | Collection name |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `EMBED_DEVICE` | `cpu` | `mps` for Apple Silicon, `cuda` for GPU |
| `EMBED_BATCH_SIZE` | `64` | Embedding batch size |
| `CHUNK_STRATEGY` | `recursive` | `recursive` \| `sentence_window` \| `semantic` \| `late` |
| `CHUNK_SIZE` | `512` | Target chunk size in tokens |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |

## Generation

| Variable | Default | Description |
|---|---|---|
| `GENERATOR_BACKEND` | `openai` | `openai` \| `anthropic` \| `squish` |
| `OPENAI_API_KEY` | — | Required for OpenAI backend |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `ANTHROPIC_API_KEY` | — | Required for Anthropic backend |
| `ANTHROPIC_MODEL` | `claude-3-haiku-20240307` | Anthropic model name |
| `SQUISH_BASE_URL` | `http://localhost:11434/v1` | Local Squish/Ollama endpoint |
| `SQUISH_MODEL` | `qwen3:8b` | Local model name |
| `MAX_TOKENS` | `1024` | Max generation tokens |
| `TEMPERATURE` | `0.1` | Generation temperature |

## Retrieval

| Variable | Default | Description |
|---|---|---|
| `TOP_K_DENSE` | `20` | Dense retrieval candidates |
| `TOP_K_SPARSE` | `20` | Sparse (BM25) retrieval candidates |
| `TOP_K_RERANK` | `5` | Final results after cross-encoder rerank |
| `HYBRID_ALPHA` | `0.7` | Dense weight in RRF fusion |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker model |

## Advanced features (all off by default)

| Variable | Default | Description |
|---|---|---|
| `ENABLE_HYDE` | `false` | HyDE hypothesis embedding |
| `ENABLE_QUERY_ROUTER` | `true` | Skip retrieval for CHAT intent |
| `ENABLE_CRAG` | `false` | Corrective RAG |
| `ENABLE_SELF_RAG` | `false` | Self-RAG reflective loop |
| `ENABLE_QUERY_DECOMPOSITION` | `false` | Multi-hop query fan-out |
| `ENABLE_GRAPH_RAG` | `false` | GraphRAG community retrieval |
| `USE_COLBERT` | `false` | ColBERT MaxSim late-interaction re-scoring |
| `VECTRO_QUANTIZE` | `false` | INT8 quantize via Vectro before upsert |
| `CACHE_ENABLED` | `false` | Semantic response cache |
| `CACHE_BACKEND` | `memory` | `memory` (in-process LRU) or `redis` (cross-pod, tenant-namespaced) — Sprint 22 |
| `CACHE_REDIS_URL` | `redis://localhost:6379/0` | Redis URL for the `redis` backend |
| `CACHE_REDIS_NAMESPACE` | `kyro:cache` | Top-level key prefix in Redis |
| `CACHE_REDIS_TTL_SECONDS` | `0` | Per-entry TTL in seconds; `0` disables expiry |

## Multi-tenancy + Auth (Sprint 17–18)

| Variable | Default | Description |
|---|---|---|
| `MULTI_TENANCY_ENABLED` | `false` | Enable JWT tenant isolation |
| `JWT_SECRET_KEY` | — | HS256 HMAC secret (required when enabled) |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm |
| `TENANT_ID_CLAIM` | `sub` | JWT claim used as tenant_id |
| `API_KEY_AUTH_ENABLED` | `false` | Enable X-API-Key auth |
| `API_KEYS` | `[]` | Comma-separated `sha256hex[:tenant_id]` entries |
| `RATE_LIMITING_ENABLED` | `false` | Per-(tenant, endpoint) sliding-window rate limiting |
| `RATE_LIMIT_REQUESTS` | `60` | Max requests per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Window length in seconds |
| `BRUTE_FORCE_ENABLED` | `false` | Per-IP brute-force lockout |
| `BRUTE_FORCE_MAX_ATTEMPTS` | `5` | Failed attempts before lockout |
| `BRUTE_FORCE_LOCKOUT_SECONDS` | `300` | Lockout duration in seconds |

## Observability (Sprint 16)

| Variable | Default | Description |
|---|---|---|
| `OTEL_ENABLED` | `false` | Enable OTel tracing + Prometheus metrics |
| `OTEL_ENDPOINT` | — | OTLP gRPC endpoint (e.g. `http://localhost:4317`) |
| `OTEL_SERVICE_NAME` | `kyro` | OTel service.name attribute |

## Pipeline

| Variable | Default | Description |
|---|---|---|
| `REQUEST_TIMEOUT_SECONDS` | `30.0` | Per-request timeout ceiling |
| `STREAMING_ENABLED` | `true` | Enable SSE token streaming |
| `ASYNC_ENABLED` | `true` | Async pipeline |
