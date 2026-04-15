"""Unit tests for ragos.retrieve.hyde — generate_hypothesis() and hyde_encode()."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# generate_hypothesis
# ---------------------------------------------------------------------------


class TestGenerateHypothesis:
    def test_returns_string(self) -> None:
        from ragos.retrieve.hyde import generate_hypothesis

        mock_result = MagicMock()
        mock_result.answer = "Paris is the capital of France."
        mock_gen = MagicMock()
        mock_gen.generate.return_value = mock_result

        with patch("ragos.generate.generator.get_generator", return_value=mock_gen):
            result = generate_hypothesis("What is the capital of France?")

        assert isinstance(result, str)

    def test_returns_generator_answer(self) -> None:
        from ragos.retrieve.hyde import generate_hypothesis

        expected = "Paris is the capital of France and it has a rich history."
        mock_result = MagicMock()
        mock_result.answer = expected
        mock_gen = MagicMock()
        mock_gen.generate.return_value = mock_result

        with patch("ragos.generate.generator.get_generator", return_value=mock_gen):
            result = generate_hypothesis("What is the capital of France?")

        assert result == expected

    def test_empty_answer_falls_back_to_question(self) -> None:
        from ragos.retrieve.hyde import generate_hypothesis

        mock_result = MagicMock()
        mock_result.answer = ""
        mock_gen = MagicMock()
        mock_gen.generate.return_value = mock_result

        question = "What is entropy?"
        with patch("ragos.generate.generator.get_generator", return_value=mock_gen):
            result = generate_hypothesis(question)

        assert result == question

    def test_whitespace_answer_falls_back_to_question(self) -> None:
        from ragos.retrieve.hyde import generate_hypothesis

        mock_result = MagicMock()
        mock_result.answer = "   "  # whitespace only — strip() → ""
        mock_gen = MagicMock()
        mock_gen.generate.return_value = mock_result

        question = "What is entropy?"
        with patch("ragos.generate.generator.get_generator", return_value=mock_gen):
            result = generate_hypothesis(question)

        assert result == question

    def test_prompt_contains_question_text(self) -> None:
        from ragos.retrieve.hyde import generate_hypothesis

        mock_result = MagicMock()
        mock_result.answer = "Quantum entanglement is a phenomenon..."
        mock_gen = MagicMock()
        mock_gen.generate.return_value = mock_result

        with patch("ragos.generate.generator.get_generator", return_value=mock_gen):
            generate_hypothesis("What is quantum entanglement?")

        call_args = mock_gen.generate.call_args
        # The question keyword should appear somewhere in the call arguments
        all_args_str = str(call_args)
        assert "quantum entanglement" in all_args_str

    def test_answer_is_stripped(self) -> None:
        from ragos.retrieve.hyde import generate_hypothesis

        mock_result = MagicMock()
        mock_result.answer = "  Some padded answer.  "
        mock_gen = MagicMock()
        mock_gen.generate.return_value = mock_result

        with patch("ragos.generate.generator.get_generator", return_value=mock_gen):
            result = generate_hypothesis("A question?")

        assert result == "Some padded answer."


# ---------------------------------------------------------------------------
# hyde_encode
# ---------------------------------------------------------------------------


class TestHydeEncode:
    def _make_mocks(
        self,
        answer: str = "A hypothetical answer.",
        embedding: np.ndarray | None = None,
    ):
        if embedding is None:
            embedding = np.random.randn(384).astype(np.float32)

        mock_result = MagicMock()
        mock_result.answer = answer
        mock_gen = MagicMock()
        mock_gen.generate.return_value = mock_result

        mock_encoder = MagicMock()
        mock_encoder.encode_query.return_value = embedding

        return mock_gen, mock_encoder

    def test_returns_tuple_of_ndarray_and_str(self) -> None:
        from ragos.retrieve.hyde import hyde_encode

        mock_gen, mock_encoder = self._make_mocks()

        with (
            patch("ragos.generate.generator.get_generator", return_value=mock_gen),
            patch("ragos.retrieve.hyde.get_encoder", return_value=mock_encoder),
        ):
            result = hyde_encode("What is Python?")

        assert isinstance(result, tuple)
        assert len(result) == 2
        embedding, hypothesis = result
        assert isinstance(embedding, np.ndarray)
        assert isinstance(hypothesis, str)

    def test_output_dtype_is_float32(self) -> None:
        from ragos.retrieve.hyde import hyde_encode

        mock_gen, mock_encoder = self._make_mocks(
            embedding=np.ones(384, dtype=np.float32),
        )

        with (
            patch("ragos.generate.generator.get_generator", return_value=mock_gen),
            patch("ragos.retrieve.hyde.get_encoder", return_value=mock_encoder),
        ):
            embedding, _ = hyde_encode("What is Python?")

        assert embedding.dtype == np.float32

    def test_k4_float64_encoder_raises_assertion(self) -> None:
        """K4 contract: encoder returning float64 must raise AssertionError matching 'float32'."""
        from ragos.retrieve.hyde import hyde_encode

        # Encoder returns float64 — violates K4
        mock_gen, mock_encoder = self._make_mocks(
            embedding=np.ones(384, dtype=np.float64),
        )

        with (
            patch("ragos.generate.generator.get_generator", return_value=mock_gen),
            patch("ragos.retrieve.hyde.get_encoder", return_value=mock_encoder),
        ):
            with pytest.raises(AssertionError, match="float32"):
                hyde_encode("What is Python?")

    def test_hypothesis_in_tuple_matches_generator_output(self) -> None:
        from ragos.retrieve.hyde import hyde_encode

        expected_hypothesis = "Python is a high-level programming language."
        mock_gen, mock_encoder = self._make_mocks(answer=expected_hypothesis)

        with (
            patch("ragos.generate.generator.get_generator", return_value=mock_gen),
            patch("ragos.retrieve.hyde.get_encoder", return_value=mock_encoder),
        ):
            _, hypothesis = hyde_encode("What is Python?")

        assert hypothesis == expected_hypothesis

    def test_encoder_receives_hypothesis_not_original_question(self) -> None:
        from ragos.retrieve.hyde import hyde_encode

        hypothesis = "The capital of France is Paris."
        mock_gen, mock_encoder = self._make_mocks(answer=hypothesis)

        with (
            patch("ragos.generate.generator.get_generator", return_value=mock_gen),
            patch("ragos.retrieve.hyde.get_encoder", return_value=mock_encoder),
        ):
            hyde_encode("What is the capital of France?")

        # Encoder should be called with the generated hypothesis
        call_args = str(mock_encoder.encode_query.call_args)
        assert "Paris" in call_args
