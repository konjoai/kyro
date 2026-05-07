from __future__ import annotations

from pydantic import BaseModel, Field


# ── Ingest ───────────────────────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    path: str = Field(..., description="File or directory path to ingest.")
    strategy: str = Field("recursive", description="Chunking strategy: 'recursive' or 'sentence_window'.")
    chunk_size: int = Field(512, ge=64, le=4096)
    overlap: int = Field(64, ge=0)


class IngestResponse(BaseModel):
    chunks_indexed: int
    sources_processed: int
    vectro_metrics: dict | None = None     # K6: None when vectro_quantize=False
    chunks_deduplicated: int = 0           # chunks removed by near-dedup filter; 0 when dedup_threshold=None
    drift_count: int | None = None         # corpus drift count from auto-verify; None when rag_auto_verify=False


class ManifestResponse(BaseModel):
    available: bool
    corpus_dir: str
    file_count: int
    manifest_hash: str
    indexed_at: str


class VerifyResponse(BaseModel):
    available: bool
    ok: bool | None
    total_files: int
    drift_count: int
    drift: list[dict]  # list of {"path": str, "status": str}


# ── Vectro pipeline ─────────────────────────────────────────────────────────────────────────────────────────

class VectroPipelineRequest(BaseModel):
    """Request body for POST /vectro/pipeline."""
    input_jsonl: str = Field(..., description="Path to JSONL file.  Each line: {'id': str, 'vector': [f32]}")
    out_dir: str | None = Field(None, description="Output directory.  Defaults to a tempdir.")
    format: str = Field("nf4", pattern="^(nf4|pq|int8|rq|auto)$",
                        description="Quantization format.  rq/auto are stubs until vectro_lib v5.0.")
    m: int = Field(16, ge=4, le=64, description="HNSW M parameter.")
    ef_construction: int = Field(200, ge=10, description="HNSW construction beam width.")
    ef_search: int = Field(50, ge=10, description="HNSW search beam width.")
    query_file: str | None = Field(None, description="Optional JSONL query file for evaluation.")
    top_k: int = Field(10, ge=1, le=100, description="Nearest neighbours per query.")
    archive: bool = Field(False, description="Archive result to evals/runs/ (K7).")


class VectroPipelineResponse(BaseModel):
    """Response from POST /vectro/pipeline."""
    n_vectors: int
    dims: int
    format: str
    out_dir: str
    index_size_bytes: int
    duration_ms: float
    query_results: list[dict] = Field(default_factory=list)
    binary_path: str = ""


# ── Query ─────────────────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    use_hyde: bool = Field(False, description="Replace the raw query embedding with a Vectro-compatible HyDE hypothesis embedding (Gao et al. 2022).")
    use_crag: bool = Field(False, description="Enable CRAG critique for this request only.")
    use_self_rag: bool = Field(False, description="Enable Self-RAG reflective loop for this request only.")
    use_decomposition: bool = Field(False, description="Enable Sprint 13 query decomposition for this request only.")
    use_graph_rag: bool = Field(False, description="Enable Sprint 15 GraphRAG community-based retrieval for this request only.")
    stream: bool = Field(False, description="Hint only; use POST /query/stream to actually stream tokens.")
    auto_route: bool = Field(False, description="Enable Sprint 25 AutoRouter to map CRAG classification to retrieval strategy for this request only.")


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
    cache_hit: bool = False             # True when response served from semantic cache
    # ── Self-correcting retrieval metadata (Sprints 11–12) ──────────────────────────────────────────────
    crag_confidence: float | None = None    # mean relevance score from CRAG; None when disabled
    crag_fallback: bool | None = None       # True if CRAG triggered corrective fallback
    crag_scores: list[float] | None = None  # per-chunk normalized CRAG scores [0, 1]
    crag_classification: list[str] | None = None  # per-chunk class labels
    crag_refinement_triggered: bool | None = None # True when ambiguous chunks invoked refinement
    self_rag_support: float | None = None   # mean support score from Self-RAG; None when disabled
    self_rag_iterations: int | None = None  # number of Self-RAG generate→critique cycles
    self_rag_iteration_scores: list[dict[str, float]] | None = None  # per-iteration ISREL/ISSUP/ISUSE
    self_rag_total_tokens: int | None = None  # cumulative generated token count across Self-RAG iterations
    decomposition_used: bool | None = None  # True when Sprint 13 decomposition was executed
    decomposition_sub_queries: list[str] | None = None  # generated sub-query list
    decomposition_synthesis_hint: str | None = None  # synthesis hint emitted by QueryDecomposer
    # ── GraphRAG community metadata (Sprint 15) ───────────────────────────────────────────────────────────────────────────
    graph_rag_communities: list[str] | None = None  # community label list; None when GraphRAG disabled
    # ── AutoRouter metadata (Sprint 25) ─────────────────────────────────────────────────────────────────────────────────────
    auto_route_strategy: str | None = None       # RouteStrategy value; None when AutoRouter disabled
    auto_route_rationale: str | None = None      # human-readable rationale from AutoRouter


# ── Eval ─────────────────────────────────────────────────────────────────────────────────────────

class EvalRequest(BaseModel):
    questions: list[str]
    answers: list[str]
    contexts: list[list[str]]
    ground_truths: list[str] | None = None


class EvalResponse(BaseModel):
    scores: dict[str, float]


# ── Health ───────────────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    vector_count: int
    bm25_built: bool
