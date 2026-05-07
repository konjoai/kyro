from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Qdrant ────────────────────────────────────────────────────────────────────────────
    qdrant_url: str = Field("http://localhost:6333")
    qdrant_api_key: str | None = None
    qdrant_collection: str = "konjoai"

    # ── Embedding ────────────────────────────────────────────────────────────────────────
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_device: str = "cpu"   # "mps" on Apple Silicon
    embed_batch_size: int = 64

    # ── Chunking ──────────────────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunk_strategy: str = "recursive"  # "recursive" | "sentence_window" | "semantic" | "late"

    # Thresholds for embedding-based chunkers (Sprint 10)
    semantic_split_threshold: float = 0.4    # SemanticSplitter: cosine sim below which a boundary is inserted
    late_chunk_threshold: float = 0.4        # LateChunker: cosine sim below which a boundary is inserted

    # ── Retrieval ─────────────────────────────────────────────────────────────────────────
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_rerank: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    hybrid_alpha: float = 0.7  # dense weight; (1 - alpha) goes to sparse

    # ── Frontier / Experimental ───────────────────────────────────────────────────────────────
    enable_hyde: bool = False            # replace raw query embedding with hypothesis
    enable_query_router: bool = True     # skip retrieval entirely for CHAT queries
    enable_telemetry: bool = True        # attach per-step latency to QueryResponse
    vectro_quantize: bool = False        # quantize embeddings with Vectro before upsert
    vectro_method: str = "int8"          # "int8" | "int4"

    # ── Vectro Retriever ─────────────────────────────────────────────────────────────────────
    use_vectro_retriever: bool = False   # replace konjo-core BM25+dense with VectroRetriever
    vectro_retriever_alpha: float = 0.7  # dense weight for VectroRetriever fusion
    bm25_persist_path: str | None = None # if set, BM25 index is saved/loaded at this path

    # ── Late Interaction (ColBERT-style MaxSim) ─────────────────────────────────────────────────────
    use_colbert: bool = False            # apply MaxSim re-scoring pass after cross-encoder rerank

    # ── Corpus Integrity (RagScanner / Squish) ────────────────────────────────────────────────────
    rag_corpus_dir: str | None = None    # if set, corpus_dir used by /ingest/manifest and /ingest/verify
    rag_auto_verify: bool = False        # if True, auto-verify corpus before each ingest (K3: silent when squish absent)

    # ── Ingest Dedup ──────────────────────────────────────────────────────────────────────────
    dedup_threshold: float | None = None # cosine similarity gate for near-duplicate chunk filtering; None = disabled

    # ── Generation ────────────────────────────────────────────────────────────────────────────
    generator_backend: str = "openai"  # "squish" | "openai" | "anthropic"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-haiku-20240307"
    squish_base_url: str = "http://localhost:11434/v1"
    squish_model: str = "qwen3:8b"
    max_tokens: int = 1024
    temperature: float = 0.1

    # ── Semantic Cache ─────────────────────────────────────────────────────────────────────────
    cache_enabled: bool = False             # off by default; zero behaviour change when False
    cache_similarity_threshold: float = 0.95 # cosine similarity gate for cache hit
    cache_max_size: int = 500               # LRU eviction ceiling
    # Sprint 22 — distributed cache backend selection.
    # "memory"  → in-process LRU (default; single-pod or single-tenant only)
    # "redis"   → Redis-backed cache shared across pods, tenant-namespaced.
    cache_backend: str = "memory"
    cache_redis_url: str = "redis://localhost:6379/0"
    cache_redis_namespace: str = "kyro:cache"
    cache_redis_ttl_seconds: int = 0         # 0 disables TTL; >0 sets per-entry expiry

    # ── RAGAS Eval ───────────────────────────────────────────────────────────────────────────
    ragas_llm: str = "gpt-4o-mini"

    # ── Adaptive Chunking (Sprint 10) ─────────────────────────────────────────────────────────
    adaptive_chunking_enabled: bool = False          # off by default; transparent fallthrough

    # ── Streaming (Sprint 9) ───────────────────────────────────────────────────────────────────
    streaming_enabled: bool = Field(True, description="Enable token streaming via SSE")
    chunk_sizes_hierarchy: list[int] = Field(
        default_factory=lambda: [1024, 512, 128],
        description="Parent → base → child chunk sizes used by MultiGranularityChunker.",
    )

    # ── CRAG — Corrective RAG (Sprint 11) ──────────────────────────────────────────────────
    enable_crag: bool = False                        # off by default; K3 graceful degradation
    crag_correct_threshold: float = 0.7              # score > threshold => CORRECT
    crag_ambiguous_threshold: float = 0.3            # threshold <= score <= correct => AMBIGUOUS

    # ── Self-RAG — Reflective Generation (Sprint 12) ─────────────────────────────────────────────────
    enable_self_rag: bool = False                    # off by default
    self_rag_max_iterations: int = 3                 # max generate → critique cycles

    # ── Query Decomposition — Multi-hop Fan-out (Sprint 13) ───────────────────────────────────────────
    enable_query_decomposition: bool = False         # off by default
    decomposition_max_sub_queries: int = 4           # bounded fan-out guard

    # ── GraphRAG — Community-based Retrieval (Sprint 15) ─────────────────────────────────────────
    enable_graph_rag: bool = False               # off by default; K3 graceful degradation
    graph_rag_max_communities: int = 5           # max communities to surface per query
    graph_rag_similarity_threshold: float = 0.3  # Jaccard edge threshold [0, 1]

    # ── OTel + Prometheus Observability (Sprint 16) ───────────────────────────────────────────────
    otel_enabled: bool = False              # off by default; K3 graceful degradation
    otel_endpoint: str = ""                 # OTLP gRPC endpoint (e.g. "http://localhost:4317")
    otel_service_name: str = "kyro"         # OTel service.name attribute
    prometheus_port: int = 8001             # port for a future standalone Prometheus push server

    # ── Multi-tenancy + JWT (Sprint 17) ──────────────────────────────────────────────────────────
    multi_tenancy_enabled: bool = False  # off by default; K3 graceful degradation
    jwt_secret_key: str = ""             # HS256 HMAC secret (must be non-empty when enabled)
    jwt_algorithm: str = "HS256"         # JWT signing algorithm
    tenant_id_claim: str = "sub"         # JWT claim used as tenant_id

    # ── API Key Auth (Sprint 18) ─────────────────────────────────────────────────────────────────
    api_key_auth_enabled: bool = False   # off by default; K3 graceful degradation
    # Each entry: "<sha256hex>" or "<sha256hex>:<tenant_id>"
    api_keys: list[str] = Field(default_factory=list)

    # ── Rate Limiting (Sprint 18) ─────────────────────────────────────────────────────────────────
    rate_limiting_enabled: bool = False  # off by default; K3 graceful degradation
    rate_limit_requests: int = 60        # max requests per window per (tenant, endpoint)
    rate_limit_window_seconds: int = 60  # sliding window length in seconds

    # ── Brute-Force Protection (Sprint 18) ────────────────────────────────────────────────────
    brute_force_enabled: bool = False    # off by default; K3 graceful degradation
    brute_force_max_attempts: int = 5    # failed auth attempts before lockout
    brute_force_window_seconds: int = 60 # window for counting failures (seconds)
    brute_force_lockout_seconds: int = 300  # lockout duration (seconds)

    # ── Async Pipeline (Sprint 8) ─────────────────────────────────────────────────────────────────
    async_enabled: bool = True               # on by default for async pipeline
    request_timeout_seconds: float = 30.0   # asyncio.timeout ceiling per request
    qdrant_max_connections: int = 10         # httpx connection pool limit for AsyncQdrantClient

    # ── Audit Logging (Sprint 24) ─────────────────────────────────────────────────────────────────
    # K3: off by default — zero cost when disabled.
    # K1: write errors are logged as warnings, never crash the request path.
    # OWASP: raw question text is NEVER stored — only SHA-256([:16]) hashes.
    audit_enabled: bool = False             # master switch; off → every log() is a no-op
    audit_backend: str = "memory"           # "memory" | "jsonl"
    audit_log_path: str = "logs/audit.jsonl"  # only used when audit_backend="jsonl"
    audit_max_memory_events: int = 1000     # ring-buffer capacity for the in-memory backend

    # ── Auto-Route (Sprint 25) ───────────────────────────────────────────────
    auto_route_enabled: bool = False            # off by default; K3 graceful degradation
    auto_route_log_decisions: bool = True       # log AutoRouter decisions when enabled

    # ── Feedback Collection (Sprint 25) ──────────────────────────────────────
    # K3: off by default — POST /feedback and GET /feedback/summary return 404.
    # K5: stdlib only — collections.deque + threading.Lock.
    # OWASP: raw question text is NEVER accepted or stored — only question_hash.
    feedback_enabled: bool = False          # master switch; off → endpoints return 404
    feedback_max_events: int = 1000         # ring-buffer capacity (LRU eviction)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
