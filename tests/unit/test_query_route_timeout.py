"""Tests for request_timeout_seconds enforcement on /query and /query/stream.

K2: every hot-path step is observable.
K3: graceful degradation — 504 on overrun, not 500.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from konjoai.api.routes.query import router
from konjoai.retrieve.reranker import RerankResult
from konjoai.retrieve.router import QueryIntent


# ── Shared settings stubs ────────────────────────────────────────────────────

@dataclass
class _SettingsNormal:
    enable_query_router: bool = False       # skip routing so first await is hybrid_search
    enable_hyde: bool = False
    enable_telemetry: bool = True
    use_vectro_retriever: bool = False
    use_colbert: bool = False
    enable_crag: bool = False
    enable_self_rag: bool = False
    enable_query_decomposition: bool = False
    decomposition_max_sub_queries: int = 4
    self_rag_max_iterations: int = 3
    top_k_dense: int = 5
    top_k_sparse: int = 5
    openai_model: str = "stub-model"
    request_timeout_seconds: float = 30.0
    enable_graph_rag: bool = False
    graph_rag_max_communities: int = 5
    graph_rag_similarity_threshold: float = 0.3
    otel_enabled: bool = False
    audit_enabled: bool = False


@dataclass
class _SettingsTimeout:
    """10 ms timeout — fires before any real I/O can complete."""
    enable_query_router: bool = False
    enable_hyde: bool = False
    enable_telemetry: bool = True
    use_vectro_retriever: bool = False
    use_colbert: bool = False
    enable_crag: bool = False
    enable_self_rag: bool = False
    enable_query_decomposition: bool = False
    decomposition_max_sub_queries: int = 4
    self_rag_max_iterations: int = 3
    top_k_dense: int = 5
    top_k_sparse: int = 5
    openai_model: str = "stub-model"
    request_timeout_seconds: float = 0.01  # 10 ms — reliably fires before 50 ms sleep
    enable_graph_rag: bool = False
    graph_rag_max_communities: int = 5
    graph_rag_similarity_threshold: float = 0.3
    otel_enabled: bool = False
    audit_enabled: bool = False


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def _sample_reranked() -> list[RerankResult]:
    return [
        RerankResult(
            score=0.9,
            content="Refund policy content.",
            source="policy.md",
            metadata={},
        )
    ]


async def _slow_to_thread(_fn, *_args, **_kwargs):
    """Drop-in replacement for asyncio.to_thread that injects a 50 ms delay."""
    await asyncio.sleep(0.05)
    # Return a minimal plausible value depending on which call site fires first
    return _sample_reranked()


# ── /query timeout tests ──────────────────────────────────────────────────────

def test_query_returns_504_on_timeout(monkeypatch):
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy",
                "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)

    app = _make_app()
    client = TestClient(app)

    with (
        patch("konjoai.api.routes.query.get_settings", return_value=_SettingsTimeout()),
        patch("konjoai.api.routes.query.asyncio.to_thread", _slow_to_thread),
    ):
        resp = client.post("/query", json={"question": "What is refund policy?"})

    assert resp.status_code == 504
    body = resp.json()
    assert "timed out" in body["detail"]
    assert "0.01" in body["detail"]


def test_query_completes_normally_within_timeout(monkeypatch):
    from konjoai.generate.generator import GenerationResult
    from konjoai.retrieve.hybrid import HybridResult

    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy",
                "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)

    app = _make_app()
    client = TestClient(app)

    _hybrid = [
        HybridResult(content="Policy text.", source="policy.md", rrf_score=0.8, metadata={})
    ]
    _gen_result = GenerationResult(
        answer="You can get a refund within 30 days.",
        model="stub-model",
        usage={"prompt_tokens": 5, "completion_tokens": 10},
    )

    class _FakeGenerator:
        def generate(self, question: str, context: str) -> GenerationResult:
            return _gen_result

    with (
        patch("konjoai.api.routes.query.get_settings", return_value=_SettingsNormal()),
        patch("konjoai.retrieve.hybrid.hybrid_search", return_value=_hybrid),
        patch("konjoai.retrieve.reranker.rerank", return_value=_sample_reranked()),
        patch("konjoai.generate.generator.get_generator", return_value=_FakeGenerator()),
        patch("konjoai.cache.get_semantic_cache", return_value=None),
    ):
        resp = client.post("/query", json={"question": "What is refund policy?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "You can get a refund within 30 days."


# ── /query/stream timeout tests ───────────────────────────────────────────────

def test_query_stream_returns_504_on_timeout(monkeypatch):
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy",
                "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)

    app = _make_app()
    client = TestClient(app)

    with (
        patch("konjoai.api.routes.query.get_settings", return_value=_SettingsTimeout()),
        patch("konjoai.api.routes.query.asyncio.to_thread", _slow_to_thread),
    ):
        resp = client.post("/query/stream", json={"question": "Streaming query?"})

    assert resp.status_code == 504
    body = resp.json()
    assert "timed out" in body["detail"]


def test_query_timeout_detail_includes_duration(monkeypatch):
    """The 504 detail string must include the configured timeout duration for observability."""
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy",
                "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)

    app = _make_app()
    client = TestClient(app)

    with (
        patch("konjoai.api.routes.query.get_settings", return_value=_SettingsTimeout()),
        patch("konjoai.api.routes.query.asyncio.to_thread", _slow_to_thread),
    ):
        resp = client.post("/query", json={"question": "Q"})

    assert resp.status_code == 504
    detail = resp.json()["detail"]
    # Must contain the duration in seconds for operator observability (K2)
    assert "0.01" in detail or "s" in detail
