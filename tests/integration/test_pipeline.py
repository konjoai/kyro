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

import ragos.embed.encoder as enc_module
import ragos.store.qdrant as store_module
import ragos.generate.generator as gen_module
import ragos.retrieve.sparse as sparse_module
from ragos.store.qdrant import SearchResult
from ragos.generate.generator import GenerationResult


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
    yield
    enc_module._encoder = None
    store_module._store = None
    gen_module._generator = None
    sparse_module._bm25_index = None


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
    monkeypatch.setattr("ragos.embed.encoder._encoder", mock)
    monkeypatch.setattr("ragos.retrieve.dense._encoder", None, raising=False)
    return mock


@pytest.fixture()
def mock_store(monkeypatch):
    mock = MagicMock()
    mock.search.return_value = [
        SearchResult(id="1", score=0.95, content=CONTENT_A, source="policy.md", metadata={}),
        SearchResult(id="2", score=0.88, content=CONTENT_B, source="shipping.md", metadata={}),
    ]
    mock.count.return_value = 2
    monkeypatch.setattr("ragos.store.qdrant._store", mock)
    monkeypatch.setattr("ragos.retrieve.dense._store", None, raising=False)
    return mock


@pytest.fixture()
def mock_generator(monkeypatch):
    mock = MagicMock()
    mock.generate.return_value = GenerationResult(
        answer="You can return items within 30 days.",
        model="gpt-4o-mini",
        usage={"prompt_tokens": 100, "completion_tokens": 20},
    )
    monkeypatch.setattr("ragos.generate.generator._generator", mock)
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_ingest_and_query_end_to_end(self, mock_encoder, mock_store, mock_generator):
        """Ingest two documents, query, and confirm a non-empty answer is returned."""
        from ragos.ingest.loaders import Document
        from ragos.ingest.chunkers import RecursiveChunker
        from ragos.retrieve.hybrid import hybrid_search
        from ragos.retrieve.reranker import rerank
        from ragos.generate.generator import get_generator

        docs = [
            Document(content=CONTENT_A, source="policy.md", metadata={}),
            Document(content=CONTENT_B, source="shipping.md", metadata={}),
        ]
        chunker = RecursiveChunker(chunk_size=512, overlap=64)
        all_chunks = [chunk for doc in docs for chunk in chunker.chunk(doc)]

        contents = [c.content for c in all_chunks]
        sources = [c.source for c in all_chunks]
        metas = [c.metadata for c in all_chunks]

        from ragos.retrieve.sparse import get_sparse_index
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
        from ragos.retrieve.hybrid import hybrid_search

        # BM25 index not built; hybrid_search must not raise
        results = hybrid_search("anything")
        assert isinstance(results, list)

    def test_bm25_standalone(self):
        """BM25 index correctly ranks a matching document higher."""
        from ragos.retrieve.sparse import BM25Index

        idx = BM25Index()
        idx.build(
            [CONTENT_A, CONTENT_B, "Unrelated content about weather."],
            ["policy.md", "shipping.md", "weather.md"],
            [{}, {}, {}],
        )
        results = idx.search("refund return policy", top_k=3)
        assert results[0].source == "policy.md", "policy.md should rank highest for refund query"
