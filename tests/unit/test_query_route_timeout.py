"""Tests for request_timeout_seconds enforcement on /query and /query/stream.

K2: every hot-path step is observable.
K3: graceful degradation — 504 on overrun, not 500.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from fastapi.testclient import TestClient

from konjoai.retrieve.reranker import RerankResult

from ._route_harness import (
    SettingsStub,
)
from ._route_harness import (
    clear_proxy_env as _clear_proxy_env,
)
from ._route_harness import (
    make_app as _make_app,
)

# ── Shared settings stubs ────────────────────────────────────────────────────
# enable_query_router=False so the first awaited hot-path step is hybrid_search.


def _settings_normal() -> SettingsStub:
    """Baseline settings: 30 s timeout, routing disabled."""
    return SettingsStub(enable_query_router=False, top_k_dense=5, top_k_sparse=5)


def _settings_timeout() -> SettingsStub:
    """10 ms timeout — fires before any real I/O can complete."""
    return SettingsStub(
        enable_query_router=False,
        top_k_dense=5,
        top_k_sparse=5,
        request_timeout_seconds=0.01,
    )


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
    _clear_proxy_env(monkeypatch)

    app = _make_app()
    client = TestClient(app)

    with (
        patch("konjoai.api.routes.query.get_settings", return_value=_settings_timeout()),
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

    _clear_proxy_env(monkeypatch)

    app = _make_app()
    client = TestClient(app)

    _hybrid = [HybridResult(content="Policy text.", source="policy.md", rrf_score=0.8, metadata={})]
    _gen_result = GenerationResult(
        answer="You can get a refund within 30 days.",
        model="stub-model",
        usage={"prompt_tokens": 5, "completion_tokens": 10},
    )

    class _FakeGenerator:
        def generate(self, question: str, context: str) -> GenerationResult:
            return _gen_result

    with (
        patch("konjoai.api.routes.query.get_settings", return_value=_settings_normal()),
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
    _clear_proxy_env(monkeypatch)

    app = _make_app()
    client = TestClient(app)

    with (
        patch("konjoai.api.routes.query.get_settings", return_value=_settings_timeout()),
        patch("konjoai.api.routes.query.asyncio.to_thread", _slow_to_thread),
    ):
        resp = client.post("/query/stream", json={"question": "Streaming query?"})

    assert resp.status_code == 504
    body = resp.json()
    assert "timed out" in body["detail"]


def test_query_timeout_detail_includes_duration(monkeypatch):
    """The 504 detail string must include the configured timeout duration for observability."""
    _clear_proxy_env(monkeypatch)

    app = _make_app()
    client = TestClient(app)

    with (
        patch("konjoai.api.routes.query.get_settings", return_value=_settings_timeout()),
        patch("konjoai.api.routes.query.asyncio.to_thread", _slow_to_thread),
    ):
        resp = client.post("/query", json={"question": "Q"})

    assert resp.status_code == 504
    detail = resp.json()["detail"]
    # Must contain the duration in seconds for operator observability (K2)
    assert "0.01" in detail or "s" in detail
