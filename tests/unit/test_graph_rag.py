"""Tests for the Sprint 15 GraphRAG implementation.

Coverage targets:
- _tokenize helper
- EntityGraph: init, build, detect_communities
- CommunityContext: dataclass contract
- GraphRAGResult: dataclass contract
- GraphRAGRetriever: init, retrieve (empty, single, similar, dissimilar, max_communities)
- get_graph_rag_retriever: singleton, custom params
- K3 gate behaviour in /query route (enabled / disabled)
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from konjoai.retrieve.graph_rag import (
    CommunityContext,
    EntityGraph,
    GraphRAGResult,
    GraphRAGRetriever,
    _tokenize,
    get_graph_rag_retriever,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_chunk(content: str, source: str = "doc.md", rrf_score: float = 0.5):
    """Return a SimpleNamespace that mimics a HybridResult."""
    return SimpleNamespace(content=content, source=source, rrf_score=rrf_score)


# ── _tokenize ─────────────────────────────────────────────────────────────────


def test_tokenize_basic():
    tokens = _tokenize("Hello World foo bar baz")
    assert "hello" in tokens   # 5 chars — included
    assert "world" in tokens
    assert "foo" in tokens
    assert "bar" in tokens
    assert "baz" in tokens


def test_tokenize_filters_short():
    tokens = _tokenize("a bb ccc dddd")
    assert "a" not in tokens
    assert "bb" not in tokens
    assert "ccc" in tokens
    assert "dddd" in tokens


def test_tokenize_returns_frozenset():
    result = _tokenize("the quick brown fox")
    assert isinstance(result, frozenset)


def test_tokenize_empty_string():
    assert _tokenize("") == frozenset()


def test_tokenize_numbers_excluded():
    tokens = _tokenize("value 123 token 99")
    assert "123" not in tokens
    assert "99" not in tokens
    assert "token" in tokens
    assert "value" in tokens


# ── EntityGraph ───────────────────────────────────────────────────────────────


def test_entity_graph_init_valid():
    eg = EntityGraph(similarity_threshold=0.3)
    assert eg.similarity_threshold == 0.3


def test_entity_graph_init_boundary_zero():
    eg = EntityGraph(similarity_threshold=0.0)
    assert eg.similarity_threshold == 0.0


def test_entity_graph_init_boundary_one():
    eg = EntityGraph(similarity_threshold=1.0)
    assert eg.similarity_threshold == 1.0


def test_entity_graph_init_invalid_high():
    with pytest.raises(ValueError, match="similarity_threshold"):
        EntityGraph(similarity_threshold=1.1)


def test_entity_graph_init_invalid_low():
    with pytest.raises(ValueError, match="similarity_threshold"):
        EntityGraph(similarity_threshold=-0.1)


def test_entity_graph_build_creates_nodes():
    pytest.importorskip("networkx")
    eg = EntityGraph(similarity_threshold=0.3)
    contents = ["machine learning models", "deep neural networks", "recipe for cookies"]
    graph = eg.build(contents)
    assert graph.number_of_nodes() == 3


def test_entity_graph_build_similar_chunks_get_edge():
    pytest.importorskip("networkx")
    eg = EntityGraph(similarity_threshold=0.1)
    # These two share several tokens → edge should be created
    contents = [
        "neural network training deep learning model",
        "deep learning neural network model training",
    ]
    graph = eg.build(contents)
    assert graph.number_of_edges() >= 1


def test_entity_graph_build_dissimilar_chunks_no_edge():
    pytest.importorskip("networkx")
    eg = EntityGraph(similarity_threshold=0.8)  # very high threshold
    contents = [
        "quantum physics electron orbit",
        "baking cookies butter flour sugar",
    ]
    graph = eg.build(contents)
    assert graph.number_of_edges() == 0


def test_entity_graph_build_empty_input():
    pytest.importorskip("networkx")
    eg = EntityGraph()
    graph = eg.build([])
    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0


def test_entity_graph_detect_communities_with_edges():
    pytest.importorskip("networkx")
    eg = EntityGraph(similarity_threshold=0.05)
    contents = [
        "machine learning deep neural network training model",
        "neural network deep learning training data model",
        "baking bread flour water yeast",
    ]
    graph = eg.build(contents)
    communities = eg.detect_communities(graph)
    # At least 1 community should exist
    assert len(communities) >= 1
    # All node indices should be covered
    all_nodes = set()
    for comm in communities:
        all_nodes.update(comm)
    assert all_nodes == set(range(3))


def test_entity_graph_detect_communities_no_edges():
    pytest.importorskip("networkx")
    eg = EntityGraph(similarity_threshold=1.0)  # impossible threshold → no edges
    contents = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]
    graph = eg.build(contents)
    assert graph.number_of_edges() == 0
    communities = eg.detect_communities(graph)
    # Each node is its own community
    assert len(communities) == 3
    for comm in communities:
        assert len(comm) == 1


def test_entity_graph_detect_communities_empty_graph():
    pytest.importorskip("networkx")
    import networkx as nx
    eg = EntityGraph()
    empty_graph = nx.Graph()
    communities = eg.detect_communities(empty_graph)
    assert communities == []


# ── CommunityContext ──────────────────────────────────────────────────────────


def test_community_context_dataclass():
    ctx = CommunityContext(
        community_id=0,
        members=["chunk A", "chunk B"],
        sources=["a.md", "b.md"],
        rrf_scores=[0.9, 0.7],
        label="chunk A",
        size=2,
    )
    assert ctx.community_id == 0
    assert ctx.size == 2
    assert len(ctx.members) == 2
    assert ctx.label == "chunk A"


def test_community_context_defaults():
    ctx = CommunityContext(community_id=1)
    assert ctx.members == []
    assert ctx.sources == []
    assert ctx.rrf_scores == []
    assert ctx.label == ""
    assert ctx.size == 0


# ── GraphRAGResult ────────────────────────────────────────────────────────────


def test_graph_rag_result_fields():
    result = GraphRAGResult(
        communities=[],
        community_labels=["label1"],
        representative_chunks=["chunk"],
        n_nodes=3,
        n_edges=2,
        used_fallback=False,
    )
    assert result.n_nodes == 3
    assert result.n_edges == 2
    assert result.used_fallback is False
    assert result.community_labels == ["label1"]


# ── GraphRAGRetriever ─────────────────────────────────────────────────────────


def test_retriever_init_valid():
    r = GraphRAGRetriever(max_communities=3, similarity_threshold=0.4)
    assert r.max_communities == 3
    assert r.entity_graph.similarity_threshold == 0.4


def test_retriever_init_invalid_communities():
    with pytest.raises(ValueError, match="max_communities"):
        GraphRAGRetriever(max_communities=0)


def test_retriever_empty_input():
    pytest.importorskip("networkx")
    r = GraphRAGRetriever()
    result = r.retrieve([])
    assert result.representative_chunks == []
    assert result.communities == []
    assert result.used_fallback is False
    assert result.n_nodes == 0


def test_retriever_single_chunk():
    pytest.importorskip("networkx")
    r = GraphRAGRetriever()
    chunk = _make_chunk("machine learning neural network", rrf_score=0.8)
    result = r.retrieve([chunk])
    assert len(result.representative_chunks) == 1
    assert result.representative_chunks[0] is chunk
    assert len(result.communities) == 1
    assert result.communities[0].size == 1


def test_retriever_similar_chunks_grouped():
    pytest.importorskip("networkx")
    r = GraphRAGRetriever(max_communities=5, similarity_threshold=0.1)
    # These two share many tokens → should end up in the same community
    c1 = _make_chunk("deep learning neural network model training epoch", rrf_score=0.9)
    c2 = _make_chunk("neural network deep learning training epoch model", rrf_score=0.7)
    # This one is dissimilar
    c3 = _make_chunk("fruit salad apple banana orange mango kiwi", rrf_score=0.5)
    result = r.retrieve([c1, c2, c3])
    # c1 and c2 should be in the same community → fewer representatives than chunks
    assert len(result.representative_chunks) <= 3
    # c1 should be the representative of the ML community (highest rrf_score)
    ml_community = next(
        (comm for comm in result.communities if c1.content[:30] in comm.members[0][:30]),
        None,
    )
    assert ml_community is not None
    assert ml_community.rrf_scores[0] >= ml_community.rrf_scores[-1]


def test_retriever_max_communities_respected():
    pytest.importorskip("networkx")
    # With threshold=1.0 every chunk is its own community
    r = GraphRAGRetriever(max_communities=2, similarity_threshold=1.0)
    chunks = [
        _make_chunk(f"unique topic {i} alpha beta gamma delta", rrf_score=float(10 - i))
        for i in range(5)
    ]
    result = r.retrieve(chunks)
    assert len(result.representative_chunks) <= 2
    assert len(result.communities) <= 2


def test_retriever_ranks_communities_by_best_rrf():
    pytest.importorskip("networkx")
    r = GraphRAGRetriever(max_communities=5, similarity_threshold=1.0)
    c_low = _make_chunk("random stuff completely unrelated blah", rrf_score=0.1)
    c_high = _make_chunk("important critical document key term", rrf_score=0.95)
    result = r.retrieve([c_low, c_high])
    # The first community should contain the highest-scoring chunk
    assert result.communities[0].rrf_scores[0] >= result.communities[-1].rrf_scores[0]


def test_retriever_representative_is_highest_rrf_in_community():
    pytest.importorskip("networkx")
    r = GraphRAGRetriever(max_communities=5, similarity_threshold=0.05)
    # Force these three into the same community by sharing many tokens
    c1 = _make_chunk("machine learning deep learning neural network artificial intelligence algorithm", rrf_score=0.9)
    c2 = _make_chunk("machine learning deep learning neural network artificial intelligence training", rrf_score=0.7)
    c3 = _make_chunk("machine learning deep learning neural network artificial intelligence evaluation", rrf_score=0.5)
    result = r.retrieve([c1, c2, c3])
    # The representative from the ML community must be the one with rrf=0.9
    reps_scores = [r.rrf_score for r in result.representative_chunks]
    assert max(reps_scores) == pytest.approx(0.9)


def test_retriever_community_labels_non_empty():
    pytest.importorskip("networkx")
    r = GraphRAGRetriever()
    chunk = _make_chunk("important content about machine learning", rrf_score=0.8)
    result = r.retrieve([chunk])
    assert len(result.community_labels) == 1
    assert len(result.community_labels[0]) > 0


def test_retriever_no_networkx_fallback():
    """When networkx is absent the retriever must return the raw input unchanged."""
    chunk = _make_chunk("some content about retrieval", rrf_score=0.5)
    with patch("konjoai.retrieve.graph_rag._HAS_NETWORKX", False):
        r = GraphRAGRetriever()
        result = r.retrieve([chunk])
    assert result.used_fallback is True
    assert result.representative_chunks == [chunk]
    assert result.communities == []
    assert result.community_labels == []


def test_retriever_graph_failure_falls_back(monkeypatch):
    """If graph construction raises, retriever must fall back gracefully (K1/K2)."""
    pytest.importorskip("networkx")
    chunk = _make_chunk("some content", rrf_score=0.5)

    def _bad_build(contents):
        raise RuntimeError("simulated build failure")

    r = GraphRAGRetriever()
    monkeypatch.setattr(r.entity_graph, "build", _bad_build)
    result = r.retrieve([chunk])
    assert result.used_fallback is True
    assert chunk in result.representative_chunks


# ── get_graph_rag_retriever singleton ─────────────────────────────────────────


def test_get_graph_rag_retriever_returns_instance():
    import konjoai.retrieve.graph_rag as mod
    # Reset singleton so this test is isolated
    mod._graph_rag_retriever = None
    inst = get_graph_rag_retriever(max_communities=3)
    assert isinstance(inst, GraphRAGRetriever)
    assert inst.max_communities == 3


def test_get_graph_rag_retriever_singleton_identity():
    import konjoai.retrieve.graph_rag as mod
    mod._graph_rag_retriever = None
    inst1 = get_graph_rag_retriever()
    inst2 = get_graph_rag_retriever()
    assert inst1 is inst2


# ── K3 gate — /query route integration ───────────────────────────────────────


@dataclass
class _SettingsStubGraphRAG:
    """Minimal settings stub for GraphRAG route tests."""
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
    request_timeout_seconds: float = 30.0
    enable_graph_rag: bool = False
    graph_rag_max_communities: int = 5
    graph_rag_similarity_threshold: float = 0.3
    otel_enabled: bool = False
    audit_enabled: bool = False


@dataclass
class _SettingsStubGraphRAGEnabled:
    """Settings stub with GraphRAG enabled."""
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
    request_timeout_seconds: float = 30.0
    enable_graph_rag: bool = True
    graph_rag_max_communities: int = 3
    graph_rag_similarity_threshold: float = 0.3
    otel_enabled: bool = False
    audit_enabled: bool = False


from fastapi import FastAPI
from fastapi.testclient import TestClient

from konjoai.api.routes.query import router
from konjoai.generate.generator import GenerationResult
from konjoai.retrieve.hybrid import HybridResult
from konjoai.retrieve.reranker import RerankResult


class _GeneratorStub:
    def generate(self, question: str, context: str) -> GenerationResult:
        _ = (question, context)
        return GenerationResult(
            answer="answer",
            model="stub-model",
            usage={},
        )


def _make_hybrid_results():
    return [
        HybridResult(rrf_score=0.9, content="machine learning neural network", source="a.md", metadata={}),
        HybridResult(rrf_score=0.7, content="deep learning training data", source="b.md", metadata={}),
    ]


def _make_reranked():
    return [
        RerankResult(score=0.9, content="machine learning neural network", source="a.md", metadata={})
    ]


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_graph_rag_disabled_by_default():
    """With enable_graph_rag=False, GraphRAG is never invoked."""
    settings_stub = _SettingsStubGraphRAG()
    app = _make_app()
    client = TestClient(app, raise_server_exceptions=True)

    mock_graph_rag = MagicMock()

    with (
        patch("konjoai.api.routes.query.get_settings", return_value=settings_stub),
        patch("konjoai.retrieve.hybrid.hybrid_search", return_value=_make_hybrid_results()),
        patch("konjoai.retrieve.reranker.rerank", return_value=_make_reranked()),
        patch("konjoai.generate.generator.get_generator", return_value=_GeneratorStub()),
        patch("konjoai.cache.get_semantic_cache", return_value=None),
        patch("konjoai.retrieve.graph_rag.get_graph_rag_retriever", mock_graph_rag),
    ):
        resp = client.post("/query", json={"question": "What is ML?"})

    assert resp.status_code == 200
    body = resp.json()
    # GraphRAG disabled → communities field is null
    assert body.get("graph_rag_communities") is None
    mock_graph_rag.assert_not_called()


def test_graph_rag_enabled_via_settings():
    """With enable_graph_rag=True, GraphRAG runs and communities appear in response."""
    pytest.importorskip("networkx")
    settings_stub = _SettingsStubGraphRAGEnabled()
    app = _make_app()
    client = TestClient(app, raise_server_exceptions=True)

    from konjoai.retrieve.graph_rag import GraphRAGResult

    fake_graph_rag_result = GraphRAGResult(
        communities=[],
        community_labels=["label1", "label2"],
        representative_chunks=_make_hybrid_results(),
        n_nodes=2,
        n_edges=0,
        used_fallback=False,
    )
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = fake_graph_rag_result

    with (
        patch("konjoai.api.routes.query.get_settings", return_value=settings_stub),
        patch("konjoai.retrieve.hybrid.hybrid_search", return_value=_make_hybrid_results()),
        patch("konjoai.retrieve.reranker.rerank", return_value=_make_reranked()),
        patch("konjoai.generate.generator.get_generator", return_value=_GeneratorStub()),
        patch("konjoai.cache.get_semantic_cache", return_value=None),
        patch(
            "konjoai.retrieve.graph_rag.get_graph_rag_retriever",
            return_value=mock_retriever,
        ),
    ):
        resp = client.post("/query", json={"question": "What is ML?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["graph_rag_communities"] == ["label1", "label2"]


def test_graph_rag_enabled_via_request_field():
    """use_graph_rag=True in request body activates GraphRAG even when settings flag is off."""
    pytest.importorskip("networkx")
    settings_stub = _SettingsStubGraphRAG()  # enable_graph_rag=False
    app = _make_app()
    client = TestClient(app, raise_server_exceptions=True)

    from konjoai.retrieve.graph_rag import GraphRAGResult

    fake_result = GraphRAGResult(
        communities=[],
        community_labels=["community_zero"],
        representative_chunks=_make_hybrid_results(),
        n_nodes=2,
        n_edges=1,
        used_fallback=False,
    )
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = fake_result

    with (
        patch("konjoai.api.routes.query.get_settings", return_value=settings_stub),
        patch("konjoai.retrieve.hybrid.hybrid_search", return_value=_make_hybrid_results()),
        patch("konjoai.retrieve.reranker.rerank", return_value=_make_reranked()),
        patch("konjoai.generate.generator.get_generator", return_value=_GeneratorStub()),
        patch("konjoai.cache.get_semantic_cache", return_value=None),
        patch(
            "konjoai.retrieve.graph_rag.get_graph_rag_retriever",
            return_value=mock_retriever,
        ),
    ):
        resp = client.post(
            "/query",
            json={"question": "What is ML?", "use_graph_rag": True},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["graph_rag_communities"] == ["community_zero"]


def test_graph_rag_enabled_via_header():
    """x-use-graph-rag header activates GraphRAG even when settings flag is off."""
    pytest.importorskip("networkx")
    settings_stub = _SettingsStubGraphRAG()
    app = _make_app()
    client = TestClient(app, raise_server_exceptions=True)

    from konjoai.retrieve.graph_rag import GraphRAGResult

    fake_result = GraphRAGResult(
        communities=[],
        community_labels=["header_community"],
        representative_chunks=_make_hybrid_results(),
        n_nodes=2,
        n_edges=0,
        used_fallback=False,
    )
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = fake_result

    with (
        patch("konjoai.api.routes.query.get_settings", return_value=settings_stub),
        patch("konjoai.retrieve.hybrid.hybrid_search", return_value=_make_hybrid_results()),
        patch("konjoai.retrieve.reranker.rerank", return_value=_make_reranked()),
        patch("konjoai.generate.generator.get_generator", return_value=_GeneratorStub()),
        patch("konjoai.cache.get_semantic_cache", return_value=None),
        patch(
            "konjoai.retrieve.graph_rag.get_graph_rag_retriever",
            return_value=mock_retriever,
        ),
    ):
        resp = client.post(
            "/query",
            json={"question": "What is ML?"},
            headers={"x-use-graph-rag": "true"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["graph_rag_communities"] == ["header_community"]
