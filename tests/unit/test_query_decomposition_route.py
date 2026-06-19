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


def _base_patches(intent: QueryIntent = QueryIntent.AGGREGATION):
    return (
        patch("konjoai.api.routes.query.get_settings", return_value=SettingsStub()),
        patch("konjoai.retrieve.router.classify_intent", return_value=intent),
        patch("konjoai.retrieve.hybrid.hybrid_search", return_value=default_hybrid_results()),
        patch("konjoai.retrieve.reranker.rerank", return_value=default_reranked_results()),
        patch("konjoai.generate.generator.get_generator", return_value=GeneratorStub()),
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
