"""Unit tests for Sprint 11 CRAG evaluator.

Covers:
1. CORRECT/AMBIGUOUS/INCORRECT threshold bands.
2. All-incorrect fallback trigger contract.
3. Ambiguous refinement with decomposed sub-queries.
4. Synthetic quality gates:
   - no regression on clean corpus
   - precision gain on noisy corpus
5. Singleton helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from konjoai.retrieve.crag import (
    CRAGClassification,
    CRAGEvaluator,
    _reset_crag,
    get_crag_pipeline,
)

# ── Stub document type ────────────────────────────────────────────────────────


@dataclass
class _Doc:
    content: str
    source: str = "test.txt"
    rrf_score: float = 0.5
    metadata: dict = field(default_factory=dict)

    @property
    def score(self):
        return self.rrf_score


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_docs(n: int, content_prefix: str = "doc") -> list[_Doc]:
    return [_Doc(content=f"{content_prefix} {i}", source=f"src_{i}.txt") for i in range(n)]


class _StubEvaluator(CRAGEvaluator):
    """Deterministic CRAG evaluator for unit tests.

    ``scores`` are consumed sequentially by each internal ``_score_pairs`` call:
    - first call: initial chunk scoring
    - later calls: ambiguous refinement scoring (sub-query pairs)
    """

    def __init__(self, scores: list[float], enable_refine: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._scores = list(scores)
        self.fallback_called = False
        self.enable_refine = enable_refine

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if len(pairs) > len(self._scores):
            raise AssertionError("not enough stub scores for test")
        out = self._scores[: len(pairs)]
        self._scores = self._scores[len(pairs) :]
        return out

    def _reembed_subqueries(self, sub_queries: list[str]) -> None:  # noqa: ARG002
        # No heavy encoder loading in unit tests.
        return

    def _refine_ambiguous(self, query: str, ambiguous_chunks):
        if not self.enable_refine:
            return []
        return super()._refine_ambiguous(query, ambiguous_chunks)

    def web_fallback(self, query: str):  # noqa: ARG002
        self.fallback_called = True
        return []


# ── Classification thresholds ────────────────────────────────────────────────


def test_classification_bands_match_sprint11_contract():
    ev = _StubEvaluator(scores=[0.71, 0.70, 0.30, 0.29])
    result = ev.run("test query", _make_docs(4))

    assert result.crag_classification == [
        CRAGClassification.CORRECT.value,
        CRAGClassification.AMBIGUOUS.value,
        CRAGClassification.AMBIGUOUS.value,
        CRAGClassification.INCORRECT.value,
    ]


def test_all_incorrect_triggers_fallback_stub():
    ev = _StubEvaluator(scores=[0.1, 0.2])
    result = ev.run("q", _make_docs(2))

    assert result.fallback_triggered is True
    assert ev.fallback_called is True
    assert result.selected_chunks == []


def test_ambiguous_chunks_can_be_refined_into_correct():
    docs = _make_docs(2)
    ev = _StubEvaluator(scores=[0.9, 0.5, 0.92], enable_refine=True)
    ev._decompose_query = lambda q: ["subquery"]  # type: ignore[method-assign]

    result = ev.run("complex query", docs)

    assert result.refinement_triggered is True
    assert result.fallback_triggered is False
    assert len(result.selected_chunks) == 2
    assert all(c.classification == CRAGClassification.CORRECT for c in result.selected_chunks)


def test_clean_corpus_has_no_recall_regression():
    """If all chunks are high quality, CRAG should keep all of them."""
    docs = _make_docs(4)
    ev = _StubEvaluator(scores=[0.91, 0.88, 0.94, 0.97])

    result = ev.run("clean corpus query", docs)

    assert len(result.selected_chunks) == len(docs)
    assert result.fallback_triggered is False
    assert result.refinement_triggered is False


def test_noisy_corpus_precision_improves_with_crag_filtering():
    """Synthetic gate: CRAG should improve relevant-chunk precision on noisy sets."""
    docs = [
        _Doc(content="refund policy details", source="relevant_1"),
        _Doc(content="shipping and return timeline", source="relevant_2"),
        _Doc(content="weather report", source="noise_1"),
        _Doc(content="sports scores", source="noise_2"),
        _Doc(content="movie rankings", source="noise_3"),
    ]
    relevant_sources = {"relevant_1", "relevant_2"}

    # Keep the two relevant chunks as CORRECT and classify noisy chunks as INCORRECT/AMBIGUOUS.
    ev = _StubEvaluator(scores=[0.93, 0.89, 0.14, 0.22, 0.41])
    ev._decompose_query = lambda q: [q]  # type: ignore[method-assign]
    result = ev.run("what is our refund policy", docs)

    baseline_precision = len(relevant_sources) / len(docs)
    selected_relevant = sum(1 for d in result.selected_chunks if d.source in relevant_sources)
    crag_precision = selected_relevant / len(result.selected_chunks)

    assert crag_precision > baseline_precision


# ── Module singleton ──────────────────────────────────────────────────────────


def test_get_crag_pipeline_returns_same_instance():
    _reset_crag()
    a = get_crag_pipeline()
    b = get_crag_pipeline()
    assert a is b


def test_reset_crag_creates_new_instance():
    _reset_crag()
    first = get_crag_pipeline()
    _reset_crag()
    second = get_crag_pipeline()
    assert first is not second


def test_reset_crag_cleanup():
    _reset_crag()  # ensure test teardown
