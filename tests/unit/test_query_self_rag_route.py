from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from konjoai.api.routes.query import router
from konjoai.generate.generator import GenerationResult
from konjoai.retrieve.hybrid import HybridResult
from konjoai.retrieve.reranker import RerankResult
from konjoai.retrieve.router import QueryIntent


@dataclass
class _SettingsStub:
    enable_query_router: bool = True
    enable_hyde: bool = False
    enable_telemetry: bool = True
    use_vectro_retriever: bool = False
    use_colbert: bool = False
    enable_crag: bool = False
    enable_self_rag: bool = False
    enable_query_decomposition: bool = False
    decomposition_max_sub_queries: int = 4
    self_rag_max_iterations: int = 3
    top_k_dense: int = 8
    top_k_sparse: int = 8
    openai_model: str = "stub-model"
    request_timeout_seconds: float = 30.0
    enable_graph_rag: bool = False
    graph_rag_max_communities: int = 5
    graph_rag_similarity_threshold: float = 0.3
    otel_enabled: bool = False
    audit_enabled: bool = False


class _GeneratorStub:
    def generate(self, question: str, context: str) -> GenerationResult:
        _ = (question, context)
        return GenerationResult(
            answer="base answer",
            model="stub-model",
            usage={"prompt_tokens": 10, "completion_tokens": 2},
        )


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def _base_patches(self_rag_runner: MagicMock):
    hybrid_results = [
        HybridResult(content="h1", source="a.txt", rrf_score=0.8, metadata={}),
        HybridResult(content="h2", source="b.txt", rrf_score=0.7, metadata={}),
    ]
    reranked_results = [
        RerankResult(score=0.95, content="h1", source="a.txt", metadata={}),
    ]

    return (
        patch("konjoai.api.routes.query.get_settings", return_value=_SettingsStub()),
        patch("konjoai.retrieve.router.classify_intent", return_value=QueryIntent.RETRIEVAL),
        patch("konjoai.retrieve.hybrid.hybrid_search", return_value=hybrid_results),
        patch("konjoai.retrieve.reranker.rerank", return_value=reranked_results),
        patch("konjoai.generate.generator.get_generator", return_value=_GeneratorStub()),
        patch("konjoai.cache.get_semantic_cache", return_value=None),
        patch("konjoai.retrieve.self_rag.get_self_rag_pipeline", return_value=self_rag_runner),
    )


def _clear_proxy_env(monkeypatch):
    for var in (
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "GRPC_PROXY",
        "grpc_proxy",
    ):
        monkeypatch.delenv(var, raising=False)


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
