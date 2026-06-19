from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from konjoai.retrieve.router import QueryIntent

from ._route_harness import (
    GeneratorStub,
    SettingsStub,
    default_hybrid_results,
    default_reranked_results,
)
from ._route_harness import (
    clear_proxy_env as _clear_proxy_env,
)
from ._route_harness import (
    make_app as _make_app,
)


def _base_patches(self_rag_runner: MagicMock):
    return (
        patch("konjoai.api.routes.query.get_settings", return_value=SettingsStub()),
        patch("konjoai.retrieve.router.classify_intent", return_value=QueryIntent.RETRIEVAL),
        patch("konjoai.retrieve.hybrid.hybrid_search", return_value=default_hybrid_results()),
        patch("konjoai.retrieve.reranker.rerank", return_value=default_reranked_results()),
        patch("konjoai.generate.generator.get_generator", return_value=GeneratorStub()),
        patch("konjoai.cache.get_semantic_cache", return_value=None),
        patch("konjoai.retrieve.self_rag.get_self_rag_pipeline", return_value=self_rag_runner),
    )


def _make_self_rag_result():
    return SimpleNamespace(
        answer="self-rag answer",
        support_score=0.74,
        iterations=2,
        iteration_scores=[
            {"iteration": 1.0, "isrel": 0.42, "issup": 0.33, "isuse": 0.56},
            {"iteration": 2.0, "isrel": 0.86, "issup": 0.74, "isuse": 0.78},
        ],
        total_tokens=37,
        usefulness=SimpleNamespace(name="HIGH"),
    )


def test_query_use_self_rag_body_flag_enables_self_rag(monkeypatch):
    _clear_proxy_env(monkeypatch)

    app = _make_app()
    client = TestClient(app)

    runner = MagicMock()
    runner.run.return_value = _make_self_rag_result()
    patches = _base_patches(runner)

    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
    ):
        resp = client.post(
            "/query",
            json={"question": "what is policy", "use_self_rag": True},
        )

    assert resp.status_code == 200
    body = resp.json()

    assert runner.run.called
    assert body["answer"] == "self-rag answer"
    assert body["self_rag_support"] == 0.74
    assert body["self_rag_iterations"] == 2
    assert body["self_rag_total_tokens"] == 37
    assert body["self_rag_iteration_scores"][0]["issup"] == 0.33
    assert body["telemetry"]["self_rag_total_tokens"] == 37


def test_query_use_self_rag_header_enables_self_rag(monkeypatch):
    _clear_proxy_env(monkeypatch)

    app = _make_app()
    client = TestClient(app)

    runner = MagicMock()
    runner.run.return_value = _make_self_rag_result()
    patches = _base_patches(runner)

    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
    ):
        resp = client.post(
            "/query",
            json={"question": "what is policy"},
            headers={"use_self_rag": "true"},
        )

    assert resp.status_code == 200
    assert runner.run.called


def test_query_skips_self_rag_without_opt_in(monkeypatch):
    _clear_proxy_env(monkeypatch)

    app = _make_app()
    client = TestClient(app)

    runner = MagicMock()
    runner.run.return_value = _make_self_rag_result()
    patches = _base_patches(runner)

    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
    ):
        resp = client.post("/query", json={"question": "what is policy"})

    assert resp.status_code == 200
    assert not runner.run.called
    body = resp.json()
    assert body["self_rag_support"] is None
    assert body["self_rag_iterations"] is None
    assert body["self_rag_iteration_scores"] is None
    assert body["self_rag_total_tokens"] is None
