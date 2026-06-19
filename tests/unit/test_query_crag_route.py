from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from konjoai.generate.generator import GenerationResult
from konjoai.retrieve.crag import CRAGChunk, CRAGClassification, CRAGResult
from konjoai.retrieve.router import QueryIntent

from ._route_harness import (
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


class _GeneratorStub:
    def generate(self, question: str, context: str) -> GenerationResult:
        _ = (question, context)
        return GenerationResult(
            answer="stub answer",
            model="stub-model",
            usage={"prompt_tokens": 10, "completion_tokens": 2},
        )


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
    return (
        patch("konjoai.api.routes.query.get_settings", return_value=SettingsStub()),
        patch("konjoai.retrieve.router.classify_intent", return_value=QueryIntent.RETRIEVAL),
        patch("konjoai.retrieve.hybrid.hybrid_search", return_value=default_hybrid_results()),
        patch("konjoai.retrieve.reranker.rerank", return_value=default_reranked_results()),
        patch("konjoai.generate.generator.get_generator", return_value=_GeneratorStub()),
        patch("konjoai.cache.get_semantic_cache", return_value=None),
        patch("konjoai.retrieve.crag.get_crag_pipeline", return_value=crag_runner),
    )


def test_query_use_crag_body_flag_enables_crag(monkeypatch):
    _clear_proxy_env(monkeypatch)

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
    _clear_proxy_env(monkeypatch)

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
    _clear_proxy_env(monkeypatch)

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
