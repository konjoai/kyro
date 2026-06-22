"""kyro public retrieval HTTP layer.

Thin FastAPI shim around konjoai's existing pipeline so external services can
hit a real RAG endpoint over HTTP. Wires real internals — `hybrid_search`,
`CrossEncoderReranker`, `QdrantStore`, `BM25Index`, `SentenceEncoder`,
`AsyncSemanticCache` — never stubs.

Endpoints:
    POST   /query             run retrieval (question, tenant_id, top_k)
    GET    /retrieval/{id}    fetch a cached retrieval result by ID
    GET    /metrics           live performance stats
    POST   /ingest            index a batch of {text, metadata} documents
    GET    /health            liveness check

Singleflight: identical (tenant_id, question) requests collide on a single
in-flight pipeline invocation via :class:`konjoai.cache.AsyncSemanticCache`.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any, Protocol

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from konjoai.auth.tenant import set_current_tenant_id
from konjoai.cache import AsyncSemanticCache, SemanticCache

logger = logging.getLogger(__name__)


# ── Wire types ────────────────────────────────────────────────────────────────


class IngestItem(BaseModel):
    text: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[IngestItem] = Field(..., min_length=1)
    tenant_id: str | None = Field(None, description="Tenant scope for this batch.")
    source: str | None = Field(None, description="Optional source label applied to every doc.")


class IngestResult(BaseModel):
    indexed: int
    sources: int


class QueryRequestBody(BaseModel):
    question: str = Field(..., min_length=1)
    tenant_id: str | None = Field(None, description="Tenant scope for this query.")
    top_k: int = Field(5, ge=1, le=50)


class ScoredSource(BaseModel):
    source: str
    score: float
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResult(BaseModel):
    id: str
    question: str
    tenant_id: str | None
    results: list[ScoredSource]
    cache_hit: bool
    latency_ms: float


class HealthResult(BaseModel):
    status: str
    version: str
    indexed_documents: int


class MetricsResult(BaseModel):
    queries_total: int
    cache_hits: int
    cache_hit_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    singleflight_collapses: int
    inflight_peak: int
    inflight_now: int
    indexed_documents: int


# ── Pipeline protocol — real impl below; tests inject a stub ──────────────────


class Pipeline(Protocol):
    async def embed_query(self, question: str) -> np.ndarray: ...
    async def retrieve(self, question: str, q_vec: np.ndarray, top_k: int) -> list[ScoredSource]: ...
    async def index(self, request: IngestRequest) -> IngestResult: ...
    def document_count(self) -> int: ...


class KyroPipeline:
    """Default pipeline — calls into the real konjoai modules."""

    async def embed_query(self, question: str) -> np.ndarray:
        from konjoai.embed.encoder import get_encoder

        return await asyncio.to_thread(get_encoder().encode_query, question)

    async def retrieve(self, question: str, q_vec: np.ndarray, top_k: int) -> list[ScoredSource]:
        from konjoai.retrieve.hybrid import hybrid_search
        from konjoai.retrieve.reranker import rerank

        hybrid = await asyncio.to_thread(hybrid_search, question, q_vec=q_vec)
        if not hybrid:
            return []
        ranked = await asyncio.to_thread(rerank, question, hybrid, top_k)
        return [
            ScoredSource(
                source=r.source,
                score=float(r.score),
                content=r.content,
                metadata=r.metadata,
            )
            for r in ranked
        ]

    async def index(self, request: IngestRequest) -> IngestResult:
        from konjoai.embed.encoder import get_encoder
        from konjoai.ingest.chunkers import get_chunker
        from konjoai.ingest.loaders import Document
        from konjoai.retrieve.sparse import get_sparse_index
        from konjoai.store.qdrant import get_store

        chunker = get_chunker("recursive", 512, 64)
        contents: list[str] = []
        sources: list[str] = []
        metadatas: list[dict] = []
        sources_seen: set[str] = set()

        for i, item in enumerate(request.documents):
            label = request.source or item.metadata.get("source") or f"doc-{i}"
            sources_seen.add(label)
            doc = Document(
                content=item.text,
                source=label,
                metadata={k: v for k, v in item.metadata.items() if k != "source"},
            )
            for chunk in chunker.chunk(doc):
                contents.append(chunk.content)
                sources.append(chunk.source)
                metadatas.append(chunk.metadata)

        if not contents:
            raise HTTPException(status_code=422, detail="ingest produced zero chunks")

        encoder = get_encoder()
        embeddings = await asyncio.to_thread(encoder.encode, contents)

        store = get_store()
        await asyncio.to_thread(store.upsert, embeddings, contents, sources, metadatas)

        bm25 = get_sparse_index()
        await asyncio.to_thread(bm25.build, contents, sources, metadatas)

        return IngestResult(indexed=len(contents), sources=len(sources_seen))

    def document_count(self) -> int:
        try:
            from konjoai.store.qdrant import get_store

            return int(get_store().count())
        except Exception as exc:  # noqa: BLE001 — health endpoint is best-effort
            logger.warning("document_count unavailable: %s", exc)
            return -1


# ── Bounded retrieval-by-id store ─────────────────────────────────────────────


class _RetrievalStore:
    """Thread-safe LRU cache mapping retrieval-id → QueryResult."""

    def __init__(self, max_size: int = 1024) -> None:
        self._max = max_size
        self._lock = Lock()
        self._items: OrderedDict[str, QueryResult] = OrderedDict()

    def put(self, result: QueryResult) -> None:
        with self._lock:
            self._items[result.id] = result
            self._items.move_to_end(result.id)
            while len(self._items) > self._max:
                self._items.popitem(last=False)

    def get(self, retrieval_id: str) -> QueryResult | None:
        with self._lock:
            entry = self._items.get(retrieval_id)
            if entry is not None:
                self._items.move_to_end(retrieval_id)
            return entry


# ── Live metrics ──────────────────────────────────────────────────────────────


class _Metrics:
    """In-process counters + bounded latency ring for percentiles."""

    def __init__(self, ring_size: int = 1024) -> None:
        self._lock = Lock()
        self._ring_size = ring_size
        self._latencies_ms: list[float] = []
        self._latencies_idx = 0
        self.queries_total = 0
        self.cache_hits = 0

    def record(self, latency_ms: float, cache_hit: bool) -> None:
        with self._lock:
            self.queries_total += 1
            if cache_hit:
                self.cache_hits += 1
            if len(self._latencies_ms) < self._ring_size:
                self._latencies_ms.append(latency_ms)
            else:
                self._latencies_ms[self._latencies_idx] = latency_ms
                self._latencies_idx = (self._latencies_idx + 1) % self._ring_size

    def snapshot(self) -> tuple[float, float, float]:
        """Return (avg_ms, p50_ms, p95_ms). Zeros when no samples."""
        with self._lock:
            samples = list(self._latencies_ms)
        if not samples:
            return 0.0, 0.0, 0.0
        avg = sum(samples) / len(samples)
        ordered = sorted(samples)
        p50 = ordered[len(ordered) // 2]
        p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
        return avg, p50, p95


# ── App factory ───────────────────────────────────────────────────────────────


def _build_async_cache() -> AsyncSemanticCache:
    """Construct an AsyncSemanticCache with a tenant-aware singleflight key.

    Falls back to a small in-memory SemanticCache when the konjoai semantic
    cache singleton is disabled — singleflight is the *primary* value here,
    not the persistent cache. The default tenant_provider reads
    ``get_current_tenant_id`` from the contextvar that we set on every
    /query request, so concurrent requests from different tenants never
    collide on the same in-flight slot (K7).
    """
    from konjoai.cache import get_semantic_cache

    backend = get_semantic_cache()
    if backend is None:
        backend = SemanticCache(max_size=512, threshold=0.98)
    return AsyncSemanticCache(
        backend,
        singleflight=True,
        offload_to_thread=False,
    )


def create_app(pipeline: Pipeline | None = None) -> FastAPI:
    pipeline_impl: Pipeline = pipeline or KyroPipeline()
    retrieval_store = _RetrievalStore()
    metrics = _Metrics()
    async_cache = _build_async_cache()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        logger.info("kyro api starting")
        yield
        logger.info("kyro api stopping")

    app = FastAPI(
        title="kyro",
        description="Public retrieval API for the kyro RAG pipeline.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.pipeline = pipeline_impl
    app.state.retrieval_store = retrieval_store
    app.state.metrics = metrics
    app.state.async_cache = async_cache

    def get_pipeline(request: Request) -> Pipeline:
        return request.app.state.pipeline

    def get_retrieval_store(request: Request) -> _RetrievalStore:
        return request.app.state.retrieval_store

    def get_metrics(request: Request) -> _Metrics:
        return request.app.state.metrics

    def get_async_cache(request: Request) -> AsyncSemanticCache:
        return request.app.state.async_cache

    # ── Health ───────────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResult, tags=["health"])
    async def health(p: Pipeline = Depends(get_pipeline)) -> HealthResult:
        from konjoai import __version__

        return HealthResult(
            status="ok",
            version=__version__,
            indexed_documents=p.document_count(),
        )

    # ── Query (singleflight-protected) ───────────────────────────────────────

    @app.post("/query", response_model=QueryResult, tags=["query"])
    async def query(
        body: QueryRequestBody,
        request: Request,
        p: Pipeline = Depends(get_pipeline),
        store: _RetrievalStore = Depends(get_retrieval_store),
        m: _Metrics = Depends(get_metrics),
        cache: AsyncSemanticCache = Depends(get_async_cache),
    ) -> QueryResult:
        # Make tenant_id visible to the singleflight key + downstream
        # konjoai modules (Qdrant filter, audit, …) for the duration of
        # this request via konjoai's contextvar.
        tenant_token = set_current_tenant_id(body.tenant_id)

        started = time.perf_counter()
        try:
            q_vec = await p.embed_query(body.question)

            async def _compute() -> list[ScoredSource]:
                return await p.retrieve(body.question, q_vec, body.top_k)

            cached = await cache.lookup(body.question, q_vec)
            if cached is not None:
                cache_hit = True
                sources = cached
            else:
                cache_hit = False
                # Singleflight collapses concurrent (tenant_id, question) misses
                # onto a single _compute invocation.
                sources = await cache.get_or_compute(body.question, q_vec, _compute)
        finally:
            try:
                from konjoai.auth.tenant import _current_tenant_id  # noqa: PLC0415

                _current_tenant_id.reset(tenant_token)
            except (LookupError, ValueError):
                pass

        # Defensive: AsyncSemanticCache stores arbitrary objects; we cache the
        # list[ScoredSource] above so this is a list. Re-trim to the
        # caller-requested top_k (the cached value may be from a larger run).
        sources_typed: list[ScoredSource] = list(sources)[: body.top_k]
        latency_ms = (time.perf_counter() - started) * 1000.0

        result = QueryResult(
            id=str(uuid.uuid4()),
            question=body.question,
            tenant_id=body.tenant_id,
            results=sources_typed,
            cache_hit=cache_hit,
            latency_ms=latency_ms,
        )
        store.put(result)
        m.record(latency_ms, cache_hit)
        return result

    # ── Retrieval-by-id ──────────────────────────────────────────────────────

    @app.get("/retrieval/{retrieval_id}", response_model=QueryResult, tags=["query"])
    async def get_retrieval(
        retrieval_id: str,
        store: _RetrievalStore = Depends(get_retrieval_store),
    ) -> QueryResult:
        entry = store.get(retrieval_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="retrieval id not found")
        return entry

    # ── Ingest ───────────────────────────────────────────────────────────────

    @app.post("/ingest", response_model=IngestResult, tags=["ingest"])
    async def ingest(
        body: IngestRequest,
        p: Pipeline = Depends(get_pipeline),
    ) -> IngestResult:
        tenant_token = set_current_tenant_id(body.tenant_id)
        try:
            return await p.index(body)
        finally:
            try:
                from konjoai.auth.tenant import _current_tenant_id  # noqa: PLC0415

                _current_tenant_id.reset(tenant_token)
            except (LookupError, ValueError):
                pass

    # ── Metrics ──────────────────────────────────────────────────────────────

    @app.get("/metrics", response_model=MetricsResult, tags=["health"])
    async def get_live_metrics(
        p: Pipeline = Depends(get_pipeline),
        m: _Metrics = Depends(get_metrics),
        cache: AsyncSemanticCache = Depends(get_async_cache),
    ) -> MetricsResult:
        avg_ms, p50_ms, p95_ms = m.snapshot()
        cache_stats = await cache.stats()
        hit_rate = m.cache_hits / m.queries_total if m.queries_total else 0.0
        return MetricsResult(
            queries_total=m.queries_total,
            cache_hits=m.cache_hits,
            cache_hit_rate=round(hit_rate, 4),
            avg_latency_ms=round(avg_ms, 3),
            p50_latency_ms=round(p50_ms, 3),
            p95_latency_ms=round(p95_ms, 3),
            singleflight_collapses=int(cache_stats.get("stampedes_collapsed", 0)),
            inflight_peak=int(cache_stats.get("inflight_peak", 0)),
            inflight_now=int(cache_stats.get("inflight_now", 0)),
            indexed_documents=p.document_count(),
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",  # noqa: S104 — container deployment
        port=8000,
        log_level="info",
    )
