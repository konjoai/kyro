"""End-to-end tests for the kyro public retrieval HTTP layer.

Each test uses an in-memory stub Pipeline, an ``httpx.AsyncClient`` over
``ASGITransport``, and a fresh ``create_app`` instance — so the suite runs
without sentence-transformers, qdrant, or any other heavy ML dependency.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from api.main import IngestRequest, IngestResult, ScoredSource, create_app
from konjoai.cache import semantic_cache as _semantic_cache_module


@pytest.fixture(autouse=True)
def _reset_cache_singleton():
    """The konjoai SemanticCache singleton leaks across tests; reset it."""
    _semantic_cache_module._reset_cache()
    yield
    _semantic_cache_module._reset_cache()


@dataclass
class _StubPipeline:
    """Test double — counts calls so singleflight collapse is observable."""

    fixed_results: list[ScoredSource] = field(default_factory=list)
    document_count_value: int = 7
    embed_calls: int = 0
    retrieve_calls: int = 0
    index_calls: int = 0
    retrieve_delay_s: float = 0.0
    embed_dim: int = 16

    async def embed_query(self, question: str) -> np.ndarray:
        self.embed_calls += 1
        # Deterministic per-question vector so cache lookups behave correctly.
        rng = np.random.default_rng(abs(hash(question)) % (2**32))
        v = rng.standard_normal(self.embed_dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-12
        return v.reshape(1, -1).astype(np.float32)

    async def retrieve(self, question: str, q_vec: np.ndarray, top_k: int) -> list[ScoredSource]:
        self.retrieve_calls += 1
        if self.retrieve_delay_s:
            await asyncio.sleep(self.retrieve_delay_s)
        results = self.fixed_results or [
            ScoredSource(
                source=f"doc-{i}",
                score=1.0 - 0.1 * i,
                content=f"Answer about {question} #{i}",
                metadata={"rank": i},
            )
            for i in range(top_k)
        ]
        return results[:top_k]

    async def index(self, request: IngestRequest) -> IngestResult:
        self.index_calls += 1
        return IngestResult(
            indexed=len(request.documents),
            sources=len({d.metadata.get("source") or "default" for d in request.documents}),
        )

    def document_count(self) -> int:
        return self.document_count_value


def _client(pipeline: _StubPipeline) -> AsyncClient:
    app = create_app(pipeline=pipeline)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ── /health ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_reports_version_and_count():
    pipeline = _StubPipeline(document_count_value=42)
    async with _client(pipeline) as c:
        r = await c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["indexed_documents"] == 42
    assert body["version"]


# ── /query ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_query_returns_ranked_sources_with_metadata():
    pipeline = _StubPipeline()
    async with _client(pipeline) as c:
        r = await c.post(
            "/query",
            json={"question": "what is rag?", "tenant_id": "acme", "top_k": 3},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["question"] == "what is rag?"
    assert body["tenant_id"] == "acme"
    assert body["cache_hit"] is False
    assert len(body["results"]) == 3
    assert body["results"][0]["score"] >= body["results"][1]["score"]
    assert "rank" in body["results"][0]["metadata"]
    assert body["latency_ms"] >= 0
    assert pipeline.retrieve_calls == 1


@pytest.mark.asyncio
async def test_query_rejects_empty_question():
    async with _client(_StubPipeline()) as c:
        r = await c.post("/query", json={"question": ""})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_query_rejects_invalid_top_k():
    async with _client(_StubPipeline()) as c:
        r = await c.post("/query", json={"question": "hi", "top_k": 0})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_query_caches_repeat_question_when_cache_enabled(monkeypatch):
    monkeypatch.setenv("CACHE_ENABLED", "true")
    from konjoai.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]

    pipeline = _StubPipeline()
    async with _client(pipeline) as c:
        r1 = await c.post("/query", json={"question": "same q", "tenant_id": "t1", "top_k": 2})
        r2 = await c.post("/query", json={"question": "same q", "tenant_id": "t1", "top_k": 2})
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["cache_hit"] is False
    assert r2.json()["cache_hit"] is True
    # The second request must NOT have invoked the retrieval pipeline.
    assert pipeline.retrieve_calls == 1


# ── /retrieval/{id} ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retrieval_lookup_returns_stored_response():
    async with _client(_StubPipeline()) as c:
        q = await c.post("/query", json={"question": "lookup test", "top_k": 2})
        rid = q.json()["id"]
        r = await c.get(f"/retrieval/{rid}")
    assert r.status_code == 200
    assert r.json()["question"] == "lookup test"
    assert r.json()["id"] == rid


@pytest.mark.asyncio
async def test_retrieval_unknown_id_returns_404():
    async with _client(_StubPipeline()) as c:
        r = await c.get("/retrieval/does-not-exist")
    assert r.status_code == 404


# ── /ingest ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_documents_invokes_pipeline_index():
    pipeline = _StubPipeline()
    async with _client(pipeline) as c:
        r = await c.post(
            "/ingest",
            json={
                "documents": [
                    {"text": "alpha", "metadata": {"source": "a.txt"}},
                    {"text": "beta", "metadata": {"source": "b.txt"}},
                ],
                "tenant_id": "acme",
            },
        )
    assert r.status_code == 200
    assert r.json() == {"indexed": 2, "sources": 2}
    assert pipeline.index_calls == 1


@pytest.mark.asyncio
async def test_ingest_rejects_empty_payload():
    async with _client(_StubPipeline()) as c:
        r = await c.post("/ingest", json={"documents": []})
    assert r.status_code == 422


# ── /metrics ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_metrics_records_query_counts_and_latency():
    pipeline = _StubPipeline()
    async with _client(pipeline) as c:
        for i in range(3):
            await c.post("/query", json={"question": f"q{i}"})
        r = await c.get("/metrics")
    body = r.json()
    assert body["queries_total"] == 3
    assert body["avg_latency_ms"] > 0
    assert body["p95_latency_ms"] >= body["p50_latency_ms"]
    assert body["indexed_documents"] == pipeline.document_count_value


# ── Singleflight (the K2 deliverable) ──────────────────────────────────────


@pytest.mark.asyncio
async def test_singleflight_collapses_concurrent_identical_misses(monkeypatch):
    """8 concurrent requests with the same (tenant_id, question) → 1 compute."""
    monkeypatch.setenv("CACHE_ENABLED", "true")
    from konjoai.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]

    pipeline = _StubPipeline(retrieve_delay_s=0.05)
    async with _client(pipeline) as c:
        coros = [
            c.post(
                "/query",
                json={"question": "stampede", "tenant_id": "acme", "top_k": 2},
            )
            for _ in range(8)
        ]
        responses = await asyncio.gather(*coros)
        m = await c.get("/metrics")

    assert all(r.status_code == 200 for r in responses)
    # Singleflight invariant: only one retrieve actually ran.
    assert pipeline.retrieve_calls == 1
    # Every other waiter must have collapsed onto that single in-flight call.
    metrics = m.json()
    assert metrics["singleflight_collapses"] >= 7


@pytest.mark.asyncio
async def test_singleflight_separates_tenants(monkeypatch):
    """Same question, different tenants → independent compute calls."""
    monkeypatch.setenv("CACHE_ENABLED", "true")
    from konjoai.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]

    pipeline = _StubPipeline(retrieve_delay_s=0.03)
    async with _client(pipeline) as c:
        a = c.post("/query", json={"question": "shared q", "tenant_id": "acme", "top_k": 1})
        b = c.post("/query", json={"question": "shared q", "tenant_id": "globex", "top_k": 1})
        ra, rb = await asyncio.gather(a, b)

    assert ra.status_code == 200
    assert rb.status_code == 200
    # K7: tenant prefix on the singleflight key prevents cross-tenant collapse.
    assert pipeline.retrieve_calls == 2
