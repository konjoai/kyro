"""Unit tests for konjoai/retrieve/self_rag.py — Sprint 12."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from konjoai.retrieve.self_rag import (
    DocumentCritique,
    RelevanceToken,
    RetrieveDecision,
    SelfRAGPipeline,
    SelfRAGResult,
    SupportScorer,
    SupportToken,
    UsefulnessScorer,
    UsefulnessToken,
    _reset_self_rag,
    decide_retrieve,
    get_self_rag_pipeline,
)

# ── Enum smoke tests ──────────────────────────────────────────────────────────

class TestEnums:
    def test_retrieve_decision_values(self):
        assert RetrieveDecision.YES == "yes"
        assert RetrieveDecision.NO == "no"

    def test_relevance_token_values(self):
        assert RelevanceToken.RELEVANT == "relevant"
        assert RelevanceToken.IRRELEVANT == "irrelevant"

    def test_support_token_values(self):
        assert SupportToken.FULLY_SUPPORTED
        assert SupportToken.PARTIALLY_SUPPORTED
        assert SupportToken.NOT_SUPPORTED

    def test_usefulness_token_int_values(self):
        assert int(UsefulnessToken.VERY_HIGH) == 5
        assert int(UsefulnessToken.HIGH) == 4
        assert int(UsefulnessToken.MEDIUM) == 3
        assert int(UsefulnessToken.LOW) == 2
        assert int(UsefulnessToken.VERY_LOW) == 1


# ── SupportScorer ─────────────────────────────────────────────────────────────

class TestSupportScorerJaccard:
    def setup_method(self):
        self.scorer = SupportScorer()

    def test_identical_strings(self):
        assert self.scorer._jaccard("hello world", "hello world") == pytest.approx(1.0)

    def test_disjoint_strings(self):
        assert self.scorer._jaccard("alpha beta", "gamma delta") == pytest.approx(0.0)

    def test_empty_strings(self):
        assert self.scorer._jaccard("", "") == pytest.approx(0.0)

    def test_partial_overlap(self):
        score = self.scorer._jaccard("cat dog", "cat fish")
        assert 0.0 < score < 1.0


class TestSupportScorerToken:
    def setup_method(self):
        self.scorer = SupportScorer()
        self.scorer._use_fallback = False  # force CE mode for threshold tests

    def test_ce_fully_supported(self):
        assert self.scorer.support_token(2.0) == SupportToken.FULLY_SUPPORTED
        assert self.scorer.support_token(3.5) == SupportToken.FULLY_SUPPORTED

    def test_ce_partially_supported(self):
        assert self.scorer.support_token(-0.5) == SupportToken.PARTIALLY_SUPPORTED
        assert self.scorer.support_token(1.0) == SupportToken.PARTIALLY_SUPPORTED

    def test_ce_not_supported(self):
        assert self.scorer.support_token(-1.0) == SupportToken.NOT_SUPPORTED

    def test_jaccard_fallback_thresholds(self):
        self.scorer._use_fallback = True
        assert self.scorer.support_token(0.25) == SupportToken.FULLY_SUPPORTED
        assert self.scorer.support_token(0.10) == SupportToken.PARTIALLY_SUPPORTED
        assert self.scorer.support_token(0.01) == SupportToken.NOT_SUPPORTED


# ── UsefulnessScorer ──────────────────────────────────────────────────────────

class TestUsefulnessScorer:
    def setup_method(self):
        self.scorer = UsefulnessScorer()

    def test_empty_question_returns_very_low(self):
        token, score = self.scorer.score("", "some answer")
        assert token == UsefulnessToken.VERY_LOW
        assert score == pytest.approx(0.0)

    def test_empty_answer_returns_very_low(self):
        token, score = self.scorer.score("what is python?", "")
        assert token == UsefulnessToken.VERY_LOW
        assert score == pytest.approx(0.0)

    def test_both_empty_returns_very_low(self):
        token, score = self.scorer.score("", "")
        assert token == UsefulnessToken.VERY_LOW
        assert score == pytest.approx(0.0)

    def test_full_overlap_returns_high_usefulness(self):
        # Answer contains all content words from question
        token, score = self.scorer.score(
            "machine learning neural networks",
            "machine learning neural networks deep learning",
        )
        assert int(token) >= int(UsefulnessToken.HIGH)
        assert score > 0.0

    def test_zero_overlap_returns_very_low(self):
        token, _ = self.scorer.score("python programming", "banana orange fruit")
        assert token == UsefulnessToken.VERY_LOW

    def test_stop_words_only_question_uses_all_tokens(self):
        # Question is all stop words — falls back to full token set
        token, score = self.scorer.score("what is the", "what is the answer")
        assert token is not None  # should not crash

    def test_token_int_range(self):
        # Formula: raw = min(int(overlap * 4) + 1, 5) → always in [1, 5]
        for q, a in [
            ("deep learning models", "deep learning"),
            ("retrieval augmented generation", "rag"),
            ("python code syntax", "python code syntax rules"),
        ]:
            token, score = self.scorer.score(q, a)
            assert 1 <= int(token) <= 5


# ── decide_retrieve ───────────────────────────────────────────────────────────

class TestDecideRetrieve:
    def test_chat_intent_returns_no(self):
        with patch("konjoai.retrieve.self_rag.classify_intent") as mock_ci:
            from konjoai.retrieve.router import QueryIntent
            mock_ci.return_value = QueryIntent.CHAT
            result = decide_retrieve("hello there")
        assert result == RetrieveDecision.NO

    def test_non_chat_intent_returns_yes(self):
        with patch("konjoai.retrieve.self_rag.classify_intent") as mock_ci:
            from konjoai.retrieve.router import QueryIntent
            mock_ci.return_value = QueryIntent.RETRIEVAL
            result = decide_retrieve("what is python?")
        assert result == RetrieveDecision.YES

    def test_exception_returns_yes(self):
        with patch("konjoai.retrieve.self_rag.classify_intent", side_effect=RuntimeError("fail")):
            result = decide_retrieve("anything")
        assert result == RetrieveDecision.YES


# ── SelfRAGPipeline ───────────────────────────────────────────────────────────

def _make_doc(content: str, source: str = "test") -> MagicMock:
    doc = MagicMock()
    doc.content = content
    doc.source = source
    return doc


class TestSelfRAGPipelineInit:
    def test_max_iterations_floor_at_one(self):
        pipeline = SelfRAGPipeline(max_iterations=0)
        assert pipeline._max_iterations == 1

    def test_max_iterations_negative_floor(self):
        pipeline = SelfRAGPipeline(max_iterations=-5)
        assert pipeline._max_iterations == 1

    def test_max_iterations_valid(self):
        pipeline = SelfRAGPipeline(max_iterations=3)
        assert pipeline._max_iterations == 3

    def test_custom_scorers_accepted(self):
        s = SupportScorer()
        u = UsefulnessScorer()
        pipeline = SelfRAGPipeline(support_scorer=s, usefulness_scorer=u)
        assert pipeline._support is s
        assert pipeline._usefulness is u


class TestSelfRAGPipelineRun:
    def setup_method(self):
        self.pipeline = SelfRAGPipeline(max_iterations=2)

    def test_empty_question_raises_value_error(self):
        with pytest.raises(ValueError):
            self.pipeline.run("", [], lambda: "answer")

    def test_whitespace_question_raises_value_error(self):
        with pytest.raises(ValueError):
            self.pipeline.run("   ", [], lambda: "answer")

    def test_empty_documents_returns_result(self):
        with patch("konjoai.retrieve.self_rag.decide_retrieve", return_value=RetrieveDecision.YES):
            result = self.pipeline.run(
                "what is python?",
                [],
                lambda: "Python is a programming language.",
            )
        assert isinstance(result, SelfRAGResult)
        assert result.answer == "Python is a programming language."
        assert result.iterations >= 1

    def test_result_has_correct_fields(self):
        # SupportScorer._load_model falls back to jaccard when sentence-transformers absent
        doc = _make_doc("Python is a high-level programming language.")
        with patch("konjoai.retrieve.self_rag.decide_retrieve", return_value=RetrieveDecision.YES):
            result = self.pipeline.run(
                "what is python?",
                [doc],
                lambda: "Python is a programming language.",
            )
        assert isinstance(result.retrieve_decision, RetrieveDecision)
        assert isinstance(result.usefulness, UsefulnessToken)
        assert isinstance(result.support_score, float)
        assert result.iterations >= 1
        assert len(result.document_critiques) == 1

    def test_document_critique_populated(self):
        # SupportScorer._load_model falls back to jaccard when sentence-transformers absent
        doc = _make_doc("Python is interpreted and dynamically typed.")
        with patch("konjoai.retrieve.self_rag.decide_retrieve", return_value=RetrieveDecision.YES):
            result = self.pipeline.run(
                "describe python",
                [doc],
                lambda: "Python is interpreted.",
            )
        critique = result.document_critiques[0]
        assert isinstance(critique, DocumentCritique)
        assert isinstance(critique.relevance, RelevanceToken)
        assert isinstance(critique.support, SupportToken)
        assert isinstance(critique.support_score, float)

    def test_max_iterations_override(self):
        calls = []

        def gen():
            calls.append(1)
            return "answer"

        with patch("konjoai.retrieve.self_rag.decide_retrieve", return_value=RetrieveDecision.YES):
            result = self.pipeline.run("what is python?", [], gen, max_iterations=1)
        assert len(calls) == 1
        assert result.iterations == 1

    def test_no_retrieve_decision(self):
        with patch("konjoai.retrieve.self_rag.decide_retrieve", return_value=RetrieveDecision.NO):
            result = self.pipeline.run(
                "hello",
                [],
                lambda: "Hi there!",
            )
        assert result.retrieve_decision == RetrieveDecision.NO


# ── Singleton ─────────────────────────────────────────────────────────────────

class TestSingleton:
    def setup_method(self):
        _reset_self_rag()

    def teardown_method(self):
        _reset_self_rag()

    def test_get_self_rag_pipeline_returns_instance(self):
        pipeline = get_self_rag_pipeline()
        assert isinstance(pipeline, SelfRAGPipeline)

    def test_get_self_rag_pipeline_same_instance(self):
        p1 = get_self_rag_pipeline()
        p2 = get_self_rag_pipeline()
        assert p1 is p2

    def test_reset_forces_new_instance(self):
        p1 = get_self_rag_pipeline()
        _reset_self_rag()
        p2 = get_self_rag_pipeline()
        assert p1 is not p2
