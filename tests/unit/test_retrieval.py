from __future__ import annotations

import pytest

from konjoai.retrieve.hybrid import reciprocal_rank_fusion
from konjoai.retrieve.sparse import BM25Index
from konjoai.store.qdrant import SearchResult

try:
    import rank_bm25 as _rb  # noqa: F401
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sr(content: str, score: float = 1.0) -> SearchResult:
    return SearchResult(id="x", score=score, content=content, source="s", metadata={})


def _bm25r(content: str, score: float = 1.0):
    from konjoai.retrieve.sparse import BM25Result
    return BM25Result(score=score, content=content, source="s", metadata={})


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_BM25, reason="rank-bm25 not installed")
class TestBM25Index:
    def test_search_before_build_raises(self) -> None:
        idx = BM25Index()
        with pytest.raises(RuntimeError, match="build"):
            idx.search("query", top_k=5)

    def test_built_flag_false_before_build(self) -> None:
        idx = BM25Index()
        assert idx.built is False

    def test_built_flag_true_after_build(self) -> None:
        idx = BM25Index()
        idx.build(["hello world", "foo bar"], ["a", "b"], [{}, {}])
        assert idx.built is True

    def test_search_returns_correct_number(self) -> None:
        idx = BM25Index()
        docs = [f"document about topic {i}" for i in range(10)]
        sources = [f"doc_{i}.txt" for i in range(10)]
        metas = [{} for _ in range(10)]
        idx.build(docs, sources, metas)
        results = idx.search("topic", top_k=3)
        assert len(results) <= 3

    def test_search_result_has_correct_source(self) -> None:
        idx = BM25Index()
        idx.build(["alpha beta"], ["source_a.txt"], [{"key": "val"}])
        results = idx.search("alpha", top_k=1)
        assert len(results) == 1
        assert results[0].source == "source_a.txt"
        assert results[0].content == "alpha beta"


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

class TestRRF:
    def test_empty_inputs_return_empty(self) -> None:
        result = reciprocal_rank_fusion([], [], alpha=0.7)
        assert result == []

    def test_only_dense_results(self) -> None:
        dense = [_sr("doc_a"), _sr("doc_b"), _sr("doc_c")]
        result = reciprocal_rank_fusion(dense, [], alpha=0.7)
        # All three should appear (sparse contributes 0)
        contents = {r.content for r in result}
        assert "doc_a" in contents
        assert "doc_b" in contents
        assert "doc_c" in contents

    def test_only_sparse_results(self) -> None:
        sparse = [_bm25r("doc_x"), _bm25r("doc_y")]
        result = reciprocal_rank_fusion([], sparse, alpha=0.7)
        contents = {r.content for r in result}
        assert "doc_x" in contents
        assert "doc_y" in contents

    def test_alpha_one_ignores_sparse(self) -> None:
        """alpha=1.0 means full weight on dense; sparse scores are 0 and don't change ordering."""
        dense = [_sr("d1"), _sr("d2")]
        sparse = [_bm25r("s1"), _bm25r("s2")]
        result = reciprocal_rank_fusion(dense, sparse, alpha=1.0)
        # d1, d2 should be at the front since they have the full dense RRF score
        # s1, s2 only get their sparse RRF * 0 = 0 contribution
        result_contents = [r.content for r in result]
        assert result_contents.index("d1") < result_contents.index("s1")

    def test_rrf_scores_are_positive(self) -> None:
        dense = [_sr(f"doc{i}") for i in range(5)]
        sparse = [_bm25r(f"doc{i}") for i in range(5)]
        result = reciprocal_rank_fusion(dense, sparse, alpha=0.7)
        for r in result:
            assert r.rrf_score > 0.0

    def test_sorted_descending(self) -> None:
        # doc0 appears first in both lists → highest RRF score
        dense = [_sr("doc0"), _sr("doc1"), _sr("doc2")]
        sparse = [_bm25r("doc0"), _bm25r("doc1"), _bm25r("doc2")]
        result = reciprocal_rank_fusion(dense, sparse, alpha=0.5)
        scores = [r.rrf_score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_deduplication_by_content(self) -> None:
        # Same content in both dense and sparse; should appear only once
        dense = [_sr("shared")]
        sparse = [_bm25r("shared")]
        result = reciprocal_rank_fusion(dense, sparse, alpha=0.5)
        contents = [r.content for r in result]
        assert contents.count("shared") == 1

    def test_rrf_formula_correctness(self) -> None:
        """For the top-ranked document in both lists, verify the exact RRF score."""
        k = 60
        alpha = 0.5
        # doc_a is rank-0 in dense and rank-0 in sparse
        dense = [_sr("doc_a"), _sr("doc_b")]
        sparse = [_bm25r("doc_a"), _bm25r("doc_b")]
        result = reciprocal_rank_fusion(dense, sparse, alpha=alpha, k=k)
        top = next(r for r in result if r.content == "doc_a")
        # Implementation uses enumerate() → rank starts at 0, so top doc score = 1/(k+0) = 1/k
        expected = alpha * (1.0 / k) + (1 - alpha) * (1.0 / k)
        assert abs(top.rrf_score - expected) < 1e-9
