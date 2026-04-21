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

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = Field("http://localhost:6333")
    qdrant_api_key: str | None = None
    qdrant_collection: str = "konjoai"

    # ── Embedding ─────────────────────────────────────────────────────────────
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_device: str = "cpu"   # "mps" on Apple Silicon
    embed_batch_size: int = 64

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunk_strategy: str = "recursive"  # "recursive" | "sentence_window"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_rerank: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    hybrid_alpha: float = 0.7  # dense weight; (1 - alpha) goes to sparse

    # ── Frontier / Experimental ──────────────────────────────────────────────
    enable_hyde: bool = False            # replace raw query embedding with hypothesis
    enable_query_router: bool = True     # skip retrieval entirely for CHAT queries
    enable_telemetry: bool = True        # attach per-step latency to QueryResponse
    vectro_quantize: bool = False        # quantize embeddings with Vectro before upsert
    vectro_method: str = "int8"          # "int8" | "int4"

    # ── Vectro Retriever ──────────────────────────────────────────────────────
    use_vectro_retriever: bool = False   # replace konjo-core BM25+dense with VectroRetriever
    vectro_retriever_alpha: float = 0.7  # dense weight for VectroRetriever fusion
    bm25_persist_path: str | None = None # if set, BM25 index is saved/loaded at this path

    # ── Late Interaction (ColBERT-style MaxSim) ───────────────────────────────
    use_colbert: bool = False            # apply MaxSim re-scoring pass after cross-encoder rerank

    # ── Corpus Integrity (RagScanner / Squish) ───────────────────────────────
    rag_corpus_dir: str | None = None    # if set, corpus_dir used by /ingest/manifest and /ingest/verify
    rag_auto_verify: bool = False        # if True, auto-verify corpus before each ingest (K3: silent when squish absent)

    # ── Ingest Dedup ─────────────────────────────────────────────────────────
    dedup_threshold: float | None = None # cosine similarity gate for near-duplicate chunk filtering; None = disabled

    # ── Generation ────────────────────────────────────────────────────────────
    generator_backend: str = "openai"  # "squish" | "openai" | "anthropic"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-haiku-20240307"
    squish_base_url: str = "http://localhost:11434/v1"
    squish_model: str = "qwen3:8b"
    max_tokens: int = 1024
    temperature: float = 0.1

    # ── Semantic Cache ────────────────────────────────────────────────────────
    cache_enabled: bool = False             # off by default; zero behaviour change when False
    cache_similarity_threshold: float = 0.95 # cosine similarity gate for cache hit
    cache_max_size: int = 500               # LRU eviction ceiling

    # ── RAGAS Eval ────────────────────────────────────────────────────────────
    ragas_llm: str = "gpt-4o-mini"

    # ── Adaptive Chunking (Sprint 10) ─────────────────────────────────────────
    adaptive_chunking_enabled: bool = False          # off by default; transparent fallthrough

    # ── Streaming (Sprint 9) ──────────────────────────────────────────────────
    streaming_enabled: bool = Field(True, description="Enable token streaming via SSE")
    chunk_sizes_hierarchy: list[int] = Field(
        default_factory=lambda: [1024, 512, 128],
        description="Parent → base → child chunk sizes used by MultiGranularityChunker.",
    )

    # ── CRAG — Corrective RAG (Sprint 11) ────────────────────────────────────
    enable_crag: bool = False                        # off by default; K3 graceful degradation
    crag_relevance_threshold: float = 0.0            # cross-encoder logit threshold; >0 = RELEVANT
    crag_min_relevant_docs: int = 1                  # fallback triggered below this count

    # ── Self-RAG — Reflective Generation (Sprint 12) ─────────────────────────
    enable_self_rag: bool = False                    # off by default
    self_rag_max_iterations: int = 2                 # max generate → critique cycles

    # ── Async Pipeline (Sprint 8) ─────────────────────────────────────────────
    async_enabled: bool = True               # on by default for async pipeline
    request_timeout_seconds: float = 30.0   # asyncio.timeout ceiling per request
    qdrant_max_connections: int = 10         # httpx connection pool limit for AsyncQdrantClient


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
