"""Unit tests for konjoai.retrieve.crag — Corrective RAG.

Tests cover:
1. RelevanceGrade enum values.
2. DocumentGrader — uses a deterministic stub cross-encoder; no network call.
3. CRAGPipeline — filtering, fallback trigger, confidence computation.
4. Module singleton get/reset helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from konjoai.retrieve.crag import (
    CRAGPipeline,
    CRAGResult,
    DocumentGrader,
    GradedDocument,
    RelevanceGrade,
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


def _make_grader_with_scores(scores: list[float]) -> DocumentGrader:
    """Return a DocumentGrader whose cross-encoder returns *scores*."""
    grader = DocumentGrader(threshold=0.0)
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array(scores)
    grader._model = mock_model
    return grader


# ── RelevanceGrade ────────────────────────────────────────────────────────────

def test_grade_values():
    assert RelevanceGrade.RELEVANT.value == "relevant"
    assert RelevanceGrade.AMBIGUOUS.value == "ambiguous"
    assert RelevanceGrade.IRRELEVANT.value == "irrelevant"


def test_grade_string_comparison():
    assert RelevanceGrade.RELEVANT == "relevant"


# ── DocumentGrader ────────────────────────────────────────────────────────────

class TestDocumentGrader:
    def test_empty_query_raises(self):
        grader = DocumentGrader()
        with pytest.raises(ValueError, match="non-empty"):
            grader.grade("", [_Doc("some text")])

    def test_empty_docs_returns_empty(self):
        grader = DocumentGrader()
        result = grader.grade("what is X?", [])
        assert result == []

    def test_positive_score_is_relevant(self):
        grader = _make_grader_with_scores([5.0])
        docs = [_Doc("relevant content")]
        graded = grader.grade("question", docs)
        assert graded[0].grade == RelevanceGrade.RELEVANT

    def test_negative_score_is_irrelevant(self):
        grader = _make_grader_with_scores([-5.0])
        docs = [_Doc("off-topic content")]
        graded = grader.grade("question", docs)
        assert graded[0].grade == RelevanceGrade.IRRELEVANT

    def test_zero_score_is_ambiguous(self):
        grader = _make_grader_with_scores([0.0])
        docs = [_Doc("maybe relevant")]
        graded = grader.grade("question", docs)
        assert graded[0].grade == RelevanceGrade.AMBIGUOUS

    def test_grades_match_doc_count(self):
        scores = [3.0, -2.0, 0.5, -3.0]
        grader = _make_grader_with_scores(scores)
        docs = _make_docs(4)
        graded = grader.grade("question", docs)
        assert len(graded) == 4

    def test_relevance_scores_preserved(self):
        grader = _make_grader_with_scores([1.5])
        graded = grader.grade("q", [_Doc("text")])
        assert abs(graded[0].relevance_score - 1.5) < 1e-6

    def test_source_preserved(self):
        grader = _make_grader_with_scores([2.0])
        doc = _Doc("text", source="special.txt")
        graded = grader.grade("q", [doc])
        assert graded[0].source == "special.txt"

    def test_missing_sentence_transformers_raises_on_grade(self):
        grader = DocumentGrader()
        # _model is None and we mock the import to fail
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises((ImportError, TypeError)):
                grader.grade("q", [_Doc("text")])

    def test_custom_threshold(self):
        grader = _make_grader_with_scores([1.0])
        grader._threshold = 2.0       # score 1.0 < 2.0 → AMBIGUOUS
        graded = grader.grade("q", [_Doc("x")])
        assert graded[0].grade == RelevanceGrade.AMBIGUOUS


# ── CRAGPipeline ──────────────────────────────────────────────────────────────

class TestCRAGPipeline:
    def test_all_relevant_no_fallback(self):
        grader = _make_grader_with_scores([5.0, 5.0, 5.0])
        pipeline = CRAGPipeline(grader=grader, min_relevant_docs=1)
        result = pipeline.run("q", _make_docs(3))
        assert not result.needs_fallback
        assert result.discarded_count == 0

    def test_all_irrelevant_triggers_fallback(self):
        grader = _make_grader_with_scores([-5.0, -5.0])
        pipeline = CRAGPipeline(grader=grader, min_relevant_docs=1)
        result = pipeline.run("q", _make_docs(2))
        assert result.needs_fallback
        assert len(result.documents) == 0
        assert result.discarded_count == 2

    def test_partial_irrelevant_removed(self):
        # scores: 3.0 (relevant), -3.0 (irrelevant), 0.0 (ambiguous)
        grader = _make_grader_with_scores([3.0, -3.0, 0.0])
        pipeline = CRAGPipeline(grader=grader, min_relevant_docs=1)
        result = pipeline.run("q", _make_docs(3))
        assert result.discarded_count == 1
        assert len(result.documents) == 2  # relevant + ambiguous kept

    def test_confidence_computed_from_relevant_docs(self):
        grader = _make_grader_with_scores([4.0, 2.0])
        pipeline = CRAGPipeline(grader=grader, min_relevant_docs=1)
        result = pipeline.run("q", _make_docs(2))
        expected = (4.0 + 2.0) / 2
        assert abs(result.overall_confidence - expected) < 1e-5

    def test_empty_docs_no_fallback_when_min_zero(self):
        # If min_relevant_docs=0 and no docs, fallback is not triggered
        grader = _make_grader_with_scores([])
        pipeline = CRAGPipeline(grader=grader, min_relevant_docs=0)
        result = pipeline.run("q", [])
        assert not result.needs_fallback

    def test_returns_crag_result_type(self):
        grader = _make_grader_with_scores([1.0])
        pipeline = CRAGPipeline(grader=grader)
        result = pipeline.run("q", _make_docs(1))
        assert isinstance(result, CRAGResult)

    def test_document_order_preserved(self):
        grader = _make_grader_with_scores([3.0, 2.0, 1.0])
        pipeline = CRAGPipeline(grader=grader, min_relevant_docs=1)
        result = pipeline.run("q", _make_docs(3))
        # All three are RELEVANT; order should match input
        assert result.documents[0].content == "doc 0"

    def test_ambiguous_docs_kept(self):
        grader = _make_grader_with_scores([0.0, 0.0])   # both AMBIGUOUS (at threshold=0)
        pipeline = CRAGPipeline(grader=grader, min_relevant_docs=1)
        result = pipeline.run("q", _make_docs(2))
        # AMBIGUOUS docs are kept; but no RELEVANT → needs_fallback = True
        assert len(result.documents) == 2
        assert result.needs_fallback  # no RELEVANT docs

    def test_zero_confidence_when_no_relevant(self):
        grader = _make_grader_with_scores([-1.0, -2.0])
        pipeline = CRAGPipeline(grader=grader, min_relevant_docs=1)
        result = pipeline.run("q", _make_docs(2))
        assert result.overall_confidence == 0.0


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
    _reset_crag()   # ensure test teardown
