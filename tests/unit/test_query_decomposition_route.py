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


def _base_patches(intent: QueryIntent = QueryIntent.AGGREGATION):
    hybrid_results = [
        HybridResult(content="h1", source="a.txt", rrf_score=0.8, metadata={}),
        HybridResult(content="h2", source="b.txt", rrf_score=0.7, metadata={}),
    ]
    reranked_results = [
        RerankResult(score=0.95, content="h1", source="a.txt", metadata={}),
    ]

    return (
        patch("konjoai.api.routes.query.get_settings", return_value=_SettingsStub()),
        patch("konjoai.retrieve.router.classify_intent", return_value=intent),
        patch("konjoai.retrieve.hybrid.hybrid_search", return_value=hybrid_results),
        patch("konjoai.retrieve.reranker.rerank", return_value=reranked_results),
        patch("konjoai.generate.generator.get_generator", return_value=_GeneratorStub()),
        patch("konjoai.cache.get_semantic_cache", return_value=None),
    )


def test_query_use_decomposition_body_flag_enables_sprint13_path(monkeypatch):
    _clear_proxy_env(monkeypatch)

    app = _make_app()
    client = TestClient(app)

    decompose = MagicMock(
        return_value=SimpleNamespace(
            sub_queries=["When was policy updated?", "Who approved the change?"],
            synthesis_hint="Combine chronology and ownership.",
            used_fallback=False,
        )
    )

    patches = _base_patches()
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patch("konjoai.retrieve.decomposition.QueryDecomposer.decompose", decompose),
        patch(
            "konjoai.retrieve.decomposition.AnswerSynthesizer.synthesize",
            return_value="decomposed answer",
        ) as synth,
    ):
        resp = client.post(
            "/query",
            json={"question": "compare policy lifecycle", "use_decomposition": True},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert decompose.called
    assert synth.called
    assert body["answer"] == "decomposed answer"
    assert body["decomposition_used"] is True
    assert len(body["decomposition_sub_queries"]) == 2
    assert body["decomposition_synthesis_hint"] == "Combine chronology and ownership."
    assert body["telemetry"]["decomposition_synthesis_hint"] == "Combine chronology and ownership."


def test_query_use_decomposition_header_enables_sprint13_path(monkeypatch):
    _clear_proxy_env(monkeypatch)

    app = _make_app()
    client = TestClient(app)

    patches = _base_patches()
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patch(
            "konjoai.retrieve.decomposition.QueryDecomposer.decompose",
            return_value=SimpleNamespace(
                sub_queries=["A", "B"],
                synthesis_hint="Use both.",
                used_fallback=False,
            ),
        ) as decompose,
        patch(
            "konjoai.retrieve.decomposition.AnswerSynthesizer.synthesize",
            return_value="decomposed answer",
        ),
    ):
        resp = client.post(
            "/query",
            json={"question": "compare policy lifecycle"},
            headers={"use_decomposition": "true"},
        )

    assert resp.status_code == 200
    assert decompose.called
    assert resp.json()["decomposition_used"] is True


def test_query_skips_decomposition_without_opt_in(monkeypatch):
    _clear_proxy_env(monkeypatch)

    app = _make_app()
    client = TestClient(app)

    patches = _base_patches()
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patch("konjoai.retrieve.decomposition.QueryDecomposer.decompose") as decompose,
        patch("konjoai.retrieve.decomposition.AnswerSynthesizer.synthesize") as synth,
    ):
        resp = client.post("/query", json={"question": "compare policy lifecycle"})

    assert resp.status_code == 200
    body = resp.json()
    assert not decompose.called
    assert not synth.called
    assert body["answer"] == "base answer"
    assert body["decomposition_used"] is None
    assert body["decomposition_sub_queries"] is None
    assert body["decomposition_synthesis_hint"] is None
