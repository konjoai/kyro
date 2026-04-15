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
    qdrant_collection: str = "ragos"

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

    # ── RAGAS Eval ────────────────────────────────────────────────────────────
    ragas_llm: str = "gpt-4o-mini"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
