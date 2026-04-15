from __future__ import annotations

from pydantic import BaseModel, Field


# ── Ingest ──────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    path: str = Field(..., description="File or directory path to ingest.")
    strategy: str = Field("recursive", description="Chunking strategy: 'recursive' or 'sentence_window'.")
    chunk_size: int = Field(512, ge=64, le=4096)
    overlap: int = Field(64, ge=0)


class IngestResponse(BaseModel):
    chunks_indexed: int
    sources_processed: int


# ── Query ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    use_hyde: bool = Field(False, description="Replace the raw query embedding with a Vectro-compatible HyDE hypothesis embedding (Gao et al. 2022).")


class SourceDoc(BaseModel):
    source: str
    content_preview: str  # first 200 chars
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]
    model: str
    usage: dict
    telemetry: dict | None = None       # per-step latency dict; None if telemetry disabled
    intent: str = "retrieval"           # "retrieval" | "aggregation" | "chat"


# ── Eval ─────────────────────────────────────────────────────────────────────

class EvalRequest(BaseModel):
    questions: list[str]
    answers: list[str]
    contexts: list[list[str]]
    ground_truths: list[str] | None = None


class EvalResponse(BaseModel):
    scores: dict[str, float]


# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    vector_count: int
    bm25_built: bool
