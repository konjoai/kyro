"""Unit tests for VectroRetrieverAdapter.

These are pure-unit tests — no real Qdrant, no real encoder, no vectro_py.
The adapter's numpy fallback path is exercised via monkeypatching.
"""

from __future__ import annotations

import numpy as np

from konjoai.retrieve.hybrid import HybridResult

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class _FakeEncoder:
    """Minimal stand-in for konjoai.embed.encoder.get_encoder()."""

    dim = 4

    def encode(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.random((len(texts), self.dim), dtype=np.float64).astype(np.float32)


def _make_corpus(n: int = 5, dim: int = 4) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    rng = np.random.default_rng(0)
    vecs = rng.random((n, dim), dtype=np.float64).astype(np.float32)
    texts = [f"document number {i}" for i in range(n)]
    sources = [f"src_{i}.txt" for i in range(n)]
    ids = [f"id_{i}" for i in range(n)]
    return vecs, texts, sources, ids


# ---------------------------------------------------------------------------
# Shape / dtype contract tests
# ---------------------------------------------------------------------------


def test_hybrid_result_fields():
    """HybridResult must have rrf_score, content, source, metadata."""
    r = HybridResult(rrf_score=0.9, content="hello", source="a.txt", metadata={})
    assert isinstance(r.rrf_score, float)
    assert isinstance(r.content, str)
    assert isinstance(r.source, str)
    assert isinstance(r.metadata, dict)


def test_adapter_search_returns_list(monkeypatch):
    """search() always returns list[HybridResult] (numpy path, no real Qdrant)."""
    from konjoai.retrieve import vectro_retriever as vr

    vecs, texts, sources, ids = _make_corpus()

    # patch corpus load so _ensure_corpus_loaded() doesn't touch Qdrant
    def _patched_load(self):
        self._corpus_vectors = vecs
        self._corpus_texts = texts
        self._corpus_sources = sources
        self._corpus_ids = ids
        self._build_bm25()

    monkeypatch.setattr(vr.VectroRetrieverAdapter, "_load_corpus", _patched_load)

    # patch encoder
    import konjoai.embed.encoder as enc_mod  # type: ignore[import]

    monkeypatch.setattr(enc_mod, "get_encoder", lambda: _FakeEncoder())

    adapter = vr.VectroRetrieverAdapter(alpha=0.7)
    results = adapter.search("document number 0", top_k=3)

    assert isinstance(results, list)
    for r in results:
        assert isinstance(r, HybridResult)
        assert isinstance(r.rrf_score, float)
        assert r.rrf_score >= 0.0


def test_adapter_returns_at_most_top_k(monkeypatch):
    from konjoai.retrieve import vectro_retriever as vr

    vecs, texts, sources, ids = _make_corpus(n=10)

    def _patched_load(self):
        self._corpus_vectors = vecs
        self._corpus_texts = texts
        self._corpus_sources = sources
        self._corpus_ids = ids
        self._build_bm25()

    monkeypatch.setattr(vr.VectroRetrieverAdapter, "_load_corpus", _patched_load)

    import konjoai.embed.encoder as enc_mod

    monkeypatch.setattr(enc_mod, "get_encoder", lambda: _FakeEncoder())

    adapter = vr.VectroRetrieverAdapter(alpha=0.5)
    results = adapter.search("query text", top_k=4)
    assert len(results) <= 4


def test_adapter_empty_corpus_falls_back(monkeypatch):
    """With an empty corpus, search() falls back via the passthrough path."""
    from konjoai.retrieve import vectro_retriever as vr

    # Simulate empty corpus sentinel
    def _patched_load(self):
        self._corpus_vectors = np.empty((0, 4), dtype=np.float32)
        self._corpus_texts = []
        self._corpus_sources = []
        self._corpus_ids = []

    monkeypatch.setattr(vr.VectroRetrieverAdapter, "_load_corpus", _patched_load)

    fallback_called: list[bool] = []

    def _mock_fallback(query, top_k):
        fallback_called.append(True)
        return []

    monkeypatch.setattr(vr.VectroRetrieverAdapter, "_fallback", staticmethod(_mock_fallback))

    adapter = vr.VectroRetrieverAdapter()
    results = adapter.search("hello", top_k=5)
    assert results == []
    assert fallback_called


def test_rebuild_clears_cache(monkeypatch):
    from konjoai.retrieve import vectro_retriever as vr

    vecs, texts, sources, ids = _make_corpus()
    load_count: list[int] = [0]

    def _patched_load(self):
        load_count[0] += 1
        self._corpus_vectors = vecs
        self._corpus_texts = texts
        self._corpus_sources = sources
        self._corpus_ids = ids
        self._build_bm25()

    monkeypatch.setattr(vr.VectroRetrieverAdapter, "_load_corpus", _patched_load)

    import konjoai.embed.encoder as enc_mod

    monkeypatch.setattr(enc_mod, "get_encoder", lambda: _FakeEncoder())

    adapter = vr.VectroRetrieverAdapter()
    adapter.search("doc", top_k=2)
    assert load_count[0] == 1

    adapter.rebuild()
    adapter.search("doc", top_k=2)
    assert load_count[0] == 2


def test_get_vectro_retriever_singleton(monkeypatch):
    """get_vectro_retriever() always returns the same object within a module."""
    from konjoai.retrieve import vectro_retriever as vr

    # Reset module-level singleton for isolation
    vr._adapter = None

    class _FakeSettings:
        vectro_retriever_alpha = 0.7

    import konjoai.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "get_settings", lambda: _FakeSettings())

    a1 = vr.get_vectro_retriever()
    a2 = vr.get_vectro_retriever()
    assert a1 is a2
