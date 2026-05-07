# Kyro

**Production RAG pipeline** with hybrid retrieval, reranking, multi-tenancy, and agentic capabilities.

No vendor lock-in — plug in OpenAI, Anthropic, or a local [Squish](https://github.com/squishai/squish) server.

## Features

| Feature | Description |
|---|---|
| Hybrid retrieval | Dense (HNSW) + sparse (BM25) with Reciprocal Rank Fusion |
| HyDE | Hypothesis document embeddings for improved recall |
| CRAG | Corrective RAG with relevance scoring and fallback |
| Self-RAG | Reflective generation with iterative critique loop |
| Query decomposition | Multi-hop fan-out for complex aggregation questions |
| Agentic RAG | Bounded ReAct loop with tool-use trace |
| GraphRAG | Community-based retrieval via NetworkX + Louvain |
| Multi-tenancy | JWT-based tenant isolation with Qdrant payload filtering |
| Auth hardening | API key auth, sliding-window rate limiting, brute-force protection |
| Observability | OTel traces + Prometheus metrics, all K3-gated |
| Audit logging | Immutable event trail with OWASP-compliant hashing, in-memory and JSONL backends |
| Feedback collection | `POST /feedback` thumbs-up/down + relevance score per query; `GET /feedback/summary` aggregates |
| Python SDK | Typed `KonjoClient` for programmatic access |
| MCP server | Expose Kyro tools to any MCP-compatible AI agent |
| AutoRouter | Sprint 25 CRAG-classification → retrieval strategy mapping |

## Version

Current version: **1.5.0** — 878 tests passing.

## License

MIT
