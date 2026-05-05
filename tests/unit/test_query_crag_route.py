from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from konjoai.api.routes.query import router
from konjoai.generate.generator import GenerationResult
from konjoai.retrieve.crag import CRAGChunk, CRAGClassification, CRAGResult
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
            answer="stub answer",
            model="stub-model",
            usage={"prompt_tokens": 10, "completion_tokens": 2},
        )


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def _make_crag_result() -> CRAGResult:
    scored = [
        CRAGChunk(
            content="relevant chunk",
            source="a.txt",
            score=0.9,
            metadata={},
            crag_score=0.84,
            classification=CRAGClassification.CORRECT,
        ),
        CRAGChunk(
            content="ambiguous chunk",
            source="b.txt",
            score=0.7,
            metadata={},
            crag_score=0.46,
            classification=CRAGClassification.AMBIGUOUS,
        ),
    ]
    return CRAGResult(
        selected_chunks=[scored[0]],
        scored_chunks=scored,
        fallback_chunks=[],
        crag_scores=[0.84, 0.46],
        crag_classification=["CORRECT", "AMBIGUOUS"],
        refinement_triggered=True,
        fallback_triggered=False,
        mean_selected_score=0.84,
    )


def _base_patches(crag_runner: MagicMock):
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
        patch("konjoai.retrieve.crag.get_crag_pipeline", return_value=crag_runner),
    )


def test_query_use_crag_body_flag_enables_crag(monkeypatch):
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

    app = _make_app()
    client = TestClient(app)

    runner = MagicMock()
    runner.run.return_value = _make_crag_result()
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
        resp = client.post("/query", json={"question": "what is policy", "use_crag": True})

    assert resp.status_code == 200
    body = resp.json()
    assert runner.run.called
    assert body["crag_scores"] == [0.84, 0.46]
    assert body["crag_classification"] == ["CORRECT", "AMBIGUOUS"]
    assert body["crag_refinement_triggered"] is True
    assert body["telemetry"]["crag_scores"] == [0.84, 0.46]
    assert body["telemetry"]["crag_classification"] == ["CORRECT", "AMBIGUOUS"]
    assert body["telemetry"]["crag_refinement_triggered"] is True


def test_query_use_crag_header_enables_crag(monkeypatch):
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

    app = _make_app()
    client = TestClient(app)

    runner = MagicMock()
    runner.run.return_value = _make_crag_result()
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
            headers={"use_crag": "true"},
        )

    assert resp.status_code == 200
    assert runner.run.called


def test_query_skips_crag_without_opt_in(monkeypatch):
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

    app = _make_app()
    client = TestClient(app)

    runner = MagicMock()
    runner.run.return_value = _make_crag_result()
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
    assert body["crag_scores"] is None
    assert body["crag_classification"] is None
    assert body["crag_refinement_triggered"] is None
