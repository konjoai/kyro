"""Unit tests for konjoai.retrieve.router — classify_intent(), decompose_query(),
classify_chunk_complexity(), and ChunkComplexity."""

from __future__ import annotations

import pytest

from konjoai.retrieve.router import (
    CHUNK_SIZE_MAP,
    ChunkComplexity,
    QueryIntent,
    classify_chunk_complexity,
    classify_intent,
    decompose_query,
)

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


# ---------------------------------------------------------------------------
# ChunkComplexity — enum values
# ---------------------------------------------------------------------------


class TestChunkComplexityEnum:
    def test_values_are_strings(self) -> None:
        assert ChunkComplexity.SIMPLE.value == "simple"
        assert ChunkComplexity.MEDIUM.value == "medium"
        assert ChunkComplexity.COMPLEX.value == "complex"

    def test_chunk_size_map_has_all_tiers(self) -> None:
        assert ChunkComplexity.SIMPLE in CHUNK_SIZE_MAP
        assert ChunkComplexity.MEDIUM in CHUNK_SIZE_MAP
        assert ChunkComplexity.COMPLEX in CHUNK_SIZE_MAP

    def test_chunk_size_order(self) -> None:
        """SIMPLE ≤ MEDIUM ≤ COMPLEX chunk sizes (larger query complexity → larger chunk)."""
        assert CHUNK_SIZE_MAP[ChunkComplexity.SIMPLE] <= CHUNK_SIZE_MAP[ChunkComplexity.MEDIUM]
        assert CHUNK_SIZE_MAP[ChunkComplexity.MEDIUM] <= CHUNK_SIZE_MAP[ChunkComplexity.COMPLEX]

    def test_default_sizes(self) -> None:
        assert CHUNK_SIZE_MAP[ChunkComplexity.SIMPLE] == 256
        assert CHUNK_SIZE_MAP[ChunkComplexity.MEDIUM] == 512
        assert CHUNK_SIZE_MAP[ChunkComplexity.COMPLEX] == 1024


# ---------------------------------------------------------------------------
# classify_chunk_complexity
# ---------------------------------------------------------------------------


class TestClassifyChunkComplexity:
    def test_returns_tuple(self) -> None:
        result = classify_chunk_complexity("What is RRF?")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_chunk_complexity(self) -> None:
        complexity, _ = classify_chunk_complexity("What is Python?")
        assert isinstance(complexity, ChunkComplexity)

    def test_second_element_is_int(self) -> None:
        _, size = classify_chunk_complexity("What is Python?")
        assert isinstance(size, int)

    def test_chunk_size_consistent_with_map(self) -> None:
        complexity, size = classify_chunk_complexity("What is Python?")
        assert CHUNK_SIZE_MAP[complexity] == size

    def test_simple_query_returns_small_chunk_size(self) -> None:
        """Very short queries score low complexity → 256 tokens."""
        _, size = classify_chunk_complexity("hi")
        # "hi" is too short to score as complex; should land in SIMPLE or MEDIUM
        assert size <= CHUNK_SIZE_MAP[ChunkComplexity.MEDIUM]

    def test_complex_query_returns_large_chunk_size(self) -> None:
        """Multi-part comparison queries score high complexity → 1024 tokens."""
        q = (
            "Compare and contrast the pricing strategy of Vendor A versus Vendor B "
            "across five dimensions including support, scalability, and integration"
        )
        complexity, size = classify_chunk_complexity(q)
        assert size >= CHUNK_SIZE_MAP[ChunkComplexity.MEDIUM]

    def test_empty_query_raises(self) -> None:
        """Empty query propagates ValueError from the underlying scorer."""
        with pytest.raises(ValueError):
            classify_chunk_complexity("")

    def test_all_returned_sizes_are_valid(self) -> None:
        queries = [
            "hi",
            "What is the capital of France?",
            "Explain how BM25 ranks documents",
            "Compare all three vendors across five dimensions and list pros and cons",
        ]
        valid_sizes = set(CHUNK_SIZE_MAP.values())
        for q in queries:
            _, size = classify_chunk_complexity(q)
            assert size in valid_sizes, f"Unexpected chunk size {size} for {q!r}"

    def test_complexity_label_monotonicity(self) -> None:
        """More complex queries should not return smaller chunks than simpler ones."""
        _, simple_size = classify_chunk_complexity("hi")
        _, complex_size = classify_chunk_complexity(
            "Compare and contrast A vs B and list all pros and cons across dimensions"
        )
        assert complex_size >= simple_size
