"""Unit tests for BM25Index.save() / load() persistence (G6 fix)."""
from __future__ import annotations

import pytest

from konjoai.retrieve.sparse import BM25Index

try:
    import rank_bm25 as _rb  # noqa: F401
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

pytestmark = pytest.mark.skipif(not _HAS_BM25, reason="rank-bm25 not installed")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _built_index(tmp_path) -> tuple[BM25Index, list[str], list[str]]:
    idx = BM25Index()
    contents = ["cats are fluffy", "dogs are loyal", "machine learning rocks"]
    sources = ["doc1.txt", "doc2.txt", "doc3.txt"]
    metas: list[dict] = [{}, {}, {}]
    idx.build(contents, sources, metas)
    return idx, contents, sources


# ---------------------------------------------------------------------------
# Shape / dtype contract tests
# ---------------------------------------------------------------------------

def test_built_flag():
    idx = BM25Index()
    assert not idx.built
    idx.build(["hello world"], ["s1"], [{}])
    assert idx.built


def test_build_then_search():
    idx, contents, _ = _built_index(None)
    results = idx.search("cats", top_k=2)
    assert len(results) <= 2
    assert results[0].content in contents


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

def test_save_creates_file(tmp_path):
    idx, _, _ = _built_index(tmp_path)
    p = tmp_path / "bm25.pkl"
    idx.save(p)
    assert p.exists()
    assert p.stat().st_size > 0


def test_save_before_build_raises(tmp_path):
    idx = BM25Index()
    with pytest.raises(RuntimeError, match="build"):
        idx.save(tmp_path / "bm25.pkl")


def test_load_restores_search(tmp_path):
    idx, contents, _ = _built_index(tmp_path)
    p = tmp_path / "bm25.pkl"
    idx.save(p)

    idx2 = BM25Index()
    assert not idx2.built
    ok = idx2.load(p)
    assert ok
    assert idx2.built

    results = idx2.search("dogs loyal", top_k=3)
    assert len(results) >= 1
    assert results[0].content == "dogs are loyal"


def test_load_missing_file_returns_false(tmp_path):
    idx = BM25Index()
    ok = idx.load(tmp_path / "nonexistent.pkl")
    assert not ok
    assert not idx.built


def test_load_corrupt_file_returns_false(tmp_path):
    p = tmp_path / "bad.pkl"
    p.write_bytes(b"not a pickle")
    idx = BM25Index()
    ok = idx.load(p)
    assert not ok
    assert not idx.built


def test_roundtrip_scores_match(tmp_path):
    """Scores from a restored index must match the original index."""
    idx, _, _ = _built_index(tmp_path)
    p = tmp_path / "bm25.pkl"
    idx.save(p)

    idx2 = BM25Index()
    idx2.load(p)

    q = "machine learning"
    r1 = idx.search(q, top_k=3)
    r2 = idx2.search(q, top_k=3)

    assert [x.content for x in r1] == [x.content for x in r2]
    for a, b in zip(r1, r2):
        assert abs(a.score - b.score) < 1e-6


def test_save_creates_parent_dir(tmp_path):
    idx, _, _ = _built_index(tmp_path)
    nested = tmp_path / "sub" / "dir" / "bm25.pkl"
    idx.save(nested)
    assert nested.exists()
