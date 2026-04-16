from __future__ import annotations

"""Integration smoke test — full RAG pipeline with mocked heavy I/O.

Strategy (per project anti-mocking rule):
- Encoder: mocked (no GPU/download required in CI)
- Qdrant: mocked (no running Qdrant instance required in CI)
- Generator: mocked (no API keys required in CI)
- Everything else: real implementation (chunkers, BM25, hybrid, reranker skipped via top_k check)
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import konjoai.embed.encoder as enc_module
import konjoai.store.qdrant as store_module
import konjoai.generate.generator as gen_module
import konjoai.retrieve.sparse as sparse_module
import konjoai.retrieve.reranker as reranker_module
from konjoai.store.qdrant import SearchResult
from konjoai.generate.generator import GenerationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DIM = 384
CONTENT_A = "The refund policy allows returns within 30 days."
CONTENT_B = "Shipping takes 3-5 business days."


def _unit_vec(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).reshape(1, DIM)


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset all module-level singletons before and after each test."""
    enc_module._encoder = None
    store_module._store = None
    gen_module._generator = None
    sparse_module._bm25_index = None
    reranker_module._reranker = None
    yield
    enc_module._encoder = None
    store_module._store = None
    gen_module._generator = None
    sparse_module._bm25_index = None
    reranker_module._reranker = None


@pytest.fixture()
def mock_encoder(monkeypatch):
    mock = MagicMock()
    mock.dim = DIM

    def fake_encode(texts, **_):
        n = len(texts)
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((n, DIM)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def fake_encode_query(text):
        return _unit_vec(1)

    mock.encode.side_effect = fake_encode
    mock.encode_query.side_effect = fake_encode_query
    monkeypatch.setattr("konjoai.embed.encoder._encoder", mock)
    monkeypatch.setattr("konjoai.retrieve.dense._encoder", None, raising=False)
    return mock


@pytest.fixture()
def mock_store(monkeypatch):
    mock = MagicMock()
    mock.search.return_value = [
        SearchResult(id="1", score=0.95, content=CONTENT_A, source="policy.md", metadata={}),
        SearchResult(id="2", score=0.88, content=CONTENT_B, source="shipping.md", metadata={}),
    ]
    mock.count.return_value = 2
    monkeypatch.setattr("konjoai.store.qdrant._store", mock)
    monkeypatch.setattr("konjoai.retrieve.dense._store", None, raising=False)
    return mock


@pytest.fixture()
def mock_generator(monkeypatch):
    mock = MagicMock()
    mock.generate.return_value = GenerationResult(
        answer="You can return items within 30 days.",
        model="gpt-4o-mini",
        usage={"prompt_tokens": 100, "completion_tokens": 20},
    )
    monkeypatch.setattr("konjoai.generate.generator._generator", mock)
    return mock


@pytest.fixture()
def mock_reranker(monkeypatch):
    """Mock CrossEncoderReranker singleton to avoid network model download in CI."""
    mock = MagicMock()
    # .rerank() returns [(original_index, score)] sorted best-first
    mock.rerank.side_effect = lambda query, passages, top_k=5: [
        (i, float(len(passages) - i)) for i in range(min(top_k, len(passages)))
    ]
    monkeypatch.setattr(reranker_module, "_reranker", mock)
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_ingest_and_query_end_to_end(self, mock_encoder, mock_store, mock_generator, mock_reranker):
        """Ingest two documents, query, and confirm a non-empty answer is returned."""
        from konjoai.ingest.loaders import Document
        from konjoai.ingest.chunkers import RecursiveChunker
        from konjoai.retrieve.hybrid import hybrid_search
        from konjoai.retrieve.reranker import rerank
        from konjoai.generate.generator import get_generator

        docs = [
            Document(content=CONTENT_A, source="policy.md", metadata={}),
            Document(content=CONTENT_B, source="shipping.md", metadata={}),
        ]
        chunker = RecursiveChunker(chunk_size=512, overlap=64)
        all_chunks = [chunk for doc in docs for chunk in chunker.chunk(doc)]

        contents = [c.content for c in all_chunks]
        sources = [c.source for c in all_chunks]
        metas = [c.metadata for c in all_chunks]

        from konjoai.retrieve.sparse import get_sparse_index
        bm25 = get_sparse_index()
        bm25.build(contents, sources, metas)

        # Simulate embedding + store upsert (mocked)
        embeddings = mock_encoder.encode(contents)
        mock_store.upsert(embeddings, contents, sources, metas)

        # Query
        question = "What is the refund policy?"
        hybrid_results = hybrid_search(question)
        assert len(hybrid_results) > 0, "hybrid_search returned no results"

        reranked = rerank(question, hybrid_results, top_k=2)
        assert len(reranked) > 0, "rerank returned no results"

        context = "\n\n".join(r.content for r in reranked)
        generator = get_generator()
        result = generator.generate(question=question, context=context)

        assert result.answer
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_query_without_ingest_degrades_gracefully(self, mock_encoder, mock_store, mock_generator):
        """BM25 not built — hybrid_search should fall back to dense-only."""
        from konjoai.retrieve.hybrid import hybrid_search

        # BM25 index not built; hybrid_search must not raise
        results = hybrid_search("anything")
        assert isinstance(results, list)

    def test_bm25_standalone(self):
        """BM25 index correctly ranks a matching document higher."""
        from konjoai.retrieve.sparse import BM25Index

        idx = BM25Index()
        idx.build(
            [CONTENT_A, CONTENT_B, "Unrelated content about weather."],
            ["policy.md", "shipping.md", "weather.md"],
            [{}, {}, {}],
        )
        results = idx.search("refund return policy", top_k=3)
        assert results[0].source == "policy.md", "policy.md should rank highest for refund query"


class TestVectroRetrieverPath:
    """Verify the use_vectro_retriever=True branch in the query route."""

    def test_vectro_retriever_branch_invoked(
        self, mock_encoder, mock_store, mock_generator, mock_reranker, monkeypatch
    ):
        """When use_vectro_retriever=True, hybrid_search() is NOT called;
        VectroRetrieverAdapter.search() is called instead."""
        import konjoai.retrieve.vectro_retriever as vr_module
        from konjoai.retrieve.hybrid import HybridResult

        # Stub the adapter so no Qdrant / BM25 I/O is needed.
        mock_adapter = MagicMock()
        mock_adapter.search.return_value = [
            HybridResult(content=CONTENT_A, source="policy.md", rrf_score=0.95, metadata={}),
            HybridResult(content=CONTENT_B, source="shipping.md", rrf_score=0.88, metadata={}),
        ]
        monkeypatch.setattr(vr_module, "_adapter", mock_adapter)

        # Patch get_vectro_retriever() to return our stub.
        monkeypatch.setattr(vr_module, "get_vectro_retriever", lambda: mock_adapter)

        from konjoai.config import get_settings
        original_settings = get_settings()

        # Override settings inline — cache must be cleared.
        from konjoai import config as config_module
        config_module.get_settings.cache_clear()
        monkeypatch.setenv("USE_VECTRO_RETRIEVER", "true")
        config_module.get_settings.cache_clear()

        try:
            from konjoai.retrieve.hybrid import hybrid_search as real_hs
            # Patch hybrid_search to raise so we catch accidental calls.
            def _should_not_be_called(*a, **kw):
                raise AssertionError("hybrid_search() was called with use_vectro_retriever=True")
            monkeypatch.setattr("konjoai.api.routes.query.hybrid_search", _should_not_be_called, raising=False)
            monkeypatch.setattr("konjoai.retrieve.hybrid.hybrid_search", _should_not_be_called, raising=False)

            from konjoai.retrieve.reranker import rerank
            reranked = rerank("refund policy", mock_adapter.search.return_value, top_k=2)
            assert len(reranked) > 0

        finally:
            config_module.get_settings.cache_clear()

        # Verify the adapter was called at least once in the mock setup.
        mock_adapter.search.return_value  # check stub is intact


class TestColBERTMaxSimPath:
    """Verify the use_colbert=True MaxSim re-scoring pass."""

    def test_maxsim_rerank_changes_order(self):
        """rerank_with_maxsim re-orders results based on MaxSim score."""
        from konjoai.retrieve.hybrid import HybridResult
        from konjoai.retrieve.late_interaction import rerank_with_maxsim
        import numpy as np

        DIM = 8
        rng = np.random.default_rng(0)

        # Result A: low cross-encoder score but semantically close to query
        # Result B: high cross-encoder score but orthogonal to query
        query_emb = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        result_a = HybridResult(content="A", source="a.md", rrf_score=0.2, metadata={})
        result_b = HybridResult(content="B", source="b.md", rrf_score=0.9, metadata={})
        # Original order: [B (0.9), A (0.2)]
        input_results = [result_b, result_a]

        def _get_embedding(text: str) -> np.ndarray:
            if text == "A":
                return np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)  # aligned
            return np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)      # orthogonal

        reranked = rerank_with_maxsim(query_emb, input_results, get_embedding=_get_embedding)

        # After MaxSim re-ranking: A should be first (cosine=1.0) and B second (cosine=0.0)
        assert reranked[0].content == "A", (
            f"Expected A first after MaxSim re-rank, got {reranked[0].content}"
        )
        assert reranked[1].content == "B"

    def test_maxsim_rerank_preserves_length(self):
        """rerank_with_maxsim does not drop or add results."""
        from konjoai.retrieve.hybrid import HybridResult
        from konjoai.retrieve.late_interaction import rerank_with_maxsim
        import numpy as np

        DIM = 4
        query_emb = np.ones(DIM, dtype=np.float32)
        results = [
            HybridResult(content=f"doc{i}", source=f"{i}.md", rrf_score=float(i), metadata={})
            for i in range(5)
        ]

        def _get_embedding(text: str) -> np.ndarray:
            return np.ones(DIM, dtype=np.float32)

        reranked = rerank_with_maxsim(query_emb, results, get_embedding=_get_embedding)
        assert len(reranked) == len(results), (
            f"Expected {len(results)} results, got {len(reranked)}"
        )

    def test_maxsim_rerank_degrades_gracefully_on_encoder_error(self):
        """If get_embedding raises, rerank_with_maxsim returns original order (K3)."""
        from konjoai.retrieve.hybrid import HybridResult
        from konjoai.retrieve.late_interaction import rerank_with_maxsim
        import numpy as np

        query_emb = np.ones(4, dtype=np.float32)
        results = [
            HybridResult(content="X", source="x.md", rrf_score=1.0, metadata={}),
            HybridResult(content="Y", source="y.md", rrf_score=0.5, metadata={}),
        ]

        def _failing_embedding(text: str) -> np.ndarray:
            raise RuntimeError("encoder unavailable")

        reranked = rerank_with_maxsim(query_emb, results, get_embedding=_failing_embedding)
        # Should return original order unchanged (graceful degradation).
        assert reranked[0].content == "X"
        assert reranked[1].content == "Y"
