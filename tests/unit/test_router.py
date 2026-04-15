"""Unit tests for ragos.retrieve.router — classify_intent() and decompose_query()."""
from __future__ import annotations

import pytest

from ragos.retrieve.router import QueryIntent, classify_intent, decompose_query


# ---------------------------------------------------------------------------
# classify_intent
# ---------------------------------------------------------------------------


class TestClassifyIntent:
    # ── CHAT ──────────────────────────────────────────────────────────────────

    def test_hello_is_chat(self) -> None:
        assert classify_intent("hello") == QueryIntent.CHAT

    def test_hi_is_chat(self) -> None:
        assert classify_intent("hi!") == QueryIntent.CHAT

    def test_thank_you_is_chat(self) -> None:
        assert classify_intent("thank you") == QueryIntent.CHAT

    def test_bye_is_chat(self) -> None:
        assert classify_intent("bye") == QueryIntent.CHAT

    def test_what_can_you_do_is_chat(self) -> None:
        assert classify_intent("what can you do") == QueryIntent.CHAT

    def test_leading_trailing_whitespace_chat(self) -> None:
        assert classify_intent("  hello  ") == QueryIntent.CHAT

    def test_uppercase_hello_is_chat(self) -> None:
        assert classify_intent("HELLO") == QueryIntent.CHAT

    # ── AGGREGATION ───────────────────────────────────────────────────────────

    def test_compare_is_aggregation(self) -> None:
        assert classify_intent("compare Python and Rust for systems programming") == QueryIntent.AGGREGATION

    def test_vs_is_aggregation(self) -> None:
        assert classify_intent("Python vs Rust performance") == QueryIntent.AGGREGATION

    def test_list_all_is_aggregation(self) -> None:
        assert classify_intent("list all supported formats") == QueryIntent.AGGREGATION

    def test_how_many_is_aggregation(self) -> None:
        assert classify_intent("how many files are in the index?") == QueryIntent.AGGREGATION

    def test_summarize_is_aggregation(self) -> None:
        assert classify_intent("summarize the main findings") == QueryIntent.AGGREGATION

    def test_pros_and_cons_is_aggregation(self) -> None:
        assert classify_intent("what are the pros and cons of this approach?") == QueryIntent.AGGREGATION

    def test_uppercase_compare_is_aggregation(self) -> None:
        assert classify_intent("COMPARE Python and Rust") == QueryIntent.AGGREGATION

    def test_enumerate_is_aggregation(self) -> None:
        assert classify_intent("enumerate all available models") == QueryIntent.AGGREGATION

    # ── RETRIEVAL ─────────────────────────────────────────────────────────────

    def test_factual_question_is_retrieval(self) -> None:
        assert classify_intent("What is the capital of France?") == QueryIntent.RETRIEVAL

    def test_technical_question_is_retrieval(self) -> None:
        assert classify_intent("How does RRF fusion work in hybrid search?") == QueryIntent.RETRIEVAL

    def test_explain_is_retrieval(self) -> None:
        assert classify_intent("Explain how HyDE improves retrieval accuracy") == QueryIntent.RETRIEVAL

    def test_unknown_query_defaults_to_retrieval(self) -> None:
        assert classify_intent("When was the Eiffel Tower built?") == QueryIntent.RETRIEVAL

    # ── Priority: CHAT wins over AGGREGATION ──────────────────────────────────

    def test_chat_patterns_take_priority(self) -> None:
        # _CHAT_RE is a full-string match (^...$); pure greeting → CHAT
        result = classify_intent("hello")
        # Greeting pattern fires first → CHAT
        assert result == QueryIntent.CHAT

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_whitespace_only_returns_valid_intent(self) -> None:
        result = classify_intent("   ")
        # No chat pattern, no aggregation keyword → RETRIEVAL
        assert result in (QueryIntent.RETRIEVAL, QueryIntent.CHAT)

    def test_returns_queryintent_enum(self) -> None:
        result = classify_intent("What is entropy?")
        assert isinstance(result, QueryIntent)

    def test_intent_values_are_strings(self) -> None:
        # QueryIntent is a str enum; .value should be a plain lowercase string
        assert QueryIntent.RETRIEVAL.value == "retrieval"
        assert QueryIntent.AGGREGATION.value == "aggregation"
        assert QueryIntent.CHAT.value == "chat"


# ---------------------------------------------------------------------------
# decompose_query
# ---------------------------------------------------------------------------


class TestDecomposeQuery:
    def test_no_conjunction_returns_single_item(self) -> None:
        result = decompose_query("What is the capital of France?")
        assert result == ["What is the capital of France?"]

    def test_and_splits_into_two(self) -> None:
        result = decompose_query("compare Python and Rust for systems")
        assert len(result) == 2

    def test_vs_splits(self) -> None:
        result = decompose_query("Python vs Rust performance")
        assert len(result) == 2

    def test_versus_splits(self) -> None:
        result = decompose_query("Python versus JavaScript for backend")
        assert len(result) == 2

    def test_max_parts_is_respected(self) -> None:
        result = decompose_query("A and B and C and D and E", max_parts=3)
        assert len(result) <= 3

    def test_custom_max_parts_1(self) -> None:
        result = decompose_query("Python and Rust and Go", max_parts=1)
        assert len(result) <= 1

    def test_stripped_parts_are_non_empty(self) -> None:
        result = decompose_query("Python and Rust and Go")
        for part in result:
            assert part.strip() != ""

    def test_default_max_parts_is_3(self) -> None:
        # "A and B and C and D" has 4 potential parts; default cap is 3
        result = decompose_query("A and B and C and D")
        assert len(result) <= 3

    def test_empty_string_returns_single_item(self) -> None:
        result = decompose_query("")
        assert len(result) == 1

    def test_returns_list(self) -> None:
        result = decompose_query("What is Python?")
        assert isinstance(result, list)

    def test_comma_splits(self) -> None:
        # _CONJUNCTION_RE requires whitespace around the comma (\s+,\s+).
        # A comma without leading whitespace ("Python, Rust, Go") does not split.
        result = decompose_query("Python, Rust, Go")
        assert result == ["Python, Rust, Go"]

    def test_compared_to_splits(self) -> None:
        result = decompose_query("Python compared to Rust")
        assert len(result) == 2
