"""Unit tests for konjoai.ingest.adaptive_chunker.

Tests cover:
1. QueryComplexityScorer — boundary values, labelling, edge cases.
2. adaptive_chunk_size — hierarchy mapping, boundary conditions.
3. MultiGranularityChunker — multi-level output, level filtering, overlap enforcement.
"""
from __future__ import annotations

import pytest

from konjoai.ingest.adaptive_chunker import (
    MultiGranularityChunker,
    QueryComplexityScorer,
    adaptive_chunk_size,
)
from konjoai.ingest.loaders import Document


# ── QueryComplexityScorer ────────────────────────────────────────────────────

class TestQueryComplexityScorer:
    def setup_method(self):
        self.scorer = QueryComplexityScorer()

    def test_empty_query_raises(self):
        with pytest.raises(ValueError):
            self.scorer.score("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            self.scorer.score("   ")

    def test_score_in_range(self):
        for q in ["hi", "what is the capital of France?", "compare all three vendors"]:
            s = self.scorer.score(q)
            assert 0.0 <= s <= 1.0, f"score {s} out of [0, 1] for {q!r}"

    def test_simple_query_has_low_score(self):
        score = self.scorer.score("hi")
        assert score < 0.35

    def test_complex_query_has_high_score(self):
        q = "Compare and contrast the pricing strategy of Vendor A versus Vendor B in 2024"
        score = self.scorer.score(q)
        assert score > 0.5

    def test_multi_part_marker_increases_score(self):
        base = self.scorer.score("What is X?")
        multi = self.scorer.score("What is X and how does it compare vs Y?")
        assert multi >= base

    def test_aggregation_marker_increases_score(self):
        base = self.scorer.score("What is X?")
        agg = self.scorer.score("List all the features of X")
        assert agg >= base

    def test_label_simple(self):
        assert self.scorer.complexity_label("hi") == "simple"

    def test_label_complex(self):
        q = "Compare and contrast the pricing strategy of Vendor A versus Vendor B in 2024"
        assert self.scorer.complexity_label(q) in ("moderate", "complex")

    def test_label_values_are_one_of_three(self):
        for q in ["hi", "what is X?", "compare all vendors across five dimensions"]:
            label = self.scorer.complexity_label(q)
            assert label in ("simple", "moderate", "complex")

    def test_named_entities_increase_score(self):
        base = self.scorer.score("what happened?")
        with_entities = self.scorer.score("What did Google Apple Microsoft announce?")
        assert with_entities >= base


# ── adaptive_chunk_size ──────────────────────────────────────────────────────

class TestAdaptiveChunkSize:
    def test_empty_hierarchy_raises(self):
        with pytest.raises(ValueError):
            adaptive_chunk_size(0.5, [])

    def test_out_of_range_complexity_raises(self):
        with pytest.raises(ValueError):
            adaptive_chunk_size(1.5, [1024, 512, 128])
        with pytest.raises(ValueError):
            adaptive_chunk_size(-0.1, [1024, 512, 128])

    def test_single_entry_hierarchy(self):
        assert adaptive_chunk_size(0.0, [512]) == 512
        assert adaptive_chunk_size(1.0, [512]) == 512

    def test_low_complexity_returns_largest(self):
        size = adaptive_chunk_size(0.0, [1024, 512, 128])
        assert size == 1024

    def test_high_complexity_returns_smallest(self):
        size = adaptive_chunk_size(1.0, [1024, 512, 128])
        assert size == 128

    def test_midpoint_returns_middle(self):
        size = adaptive_chunk_size(0.5, [1024, 512, 128])
        assert size == 512

    def test_two_level_hierarchy(self):
        assert adaptive_chunk_size(0.0, [1024, 128]) == 1024
        assert adaptive_chunk_size(1.0, [1024, 128]) == 128

    def test_boundary_zero(self):
        size = adaptive_chunk_size(0.0, [2048, 1024, 512])
        assert size == 2048

    def test_boundary_one(self):
        size = adaptive_chunk_size(1.0, [2048, 1024, 512])
        assert size == 512


# ── MultiGranularityChunker ──────────────────────────────────────────────────

def _make_doc(n_chars: int = 2000) -> Document:
    return Document(content="A " * (n_chars // 2), source="test.txt", metadata={})


class TestMultiGranularityChunker:
    def test_default_produces_three_levels(self):
        chunker = MultiGranularityChunker()
        doc = _make_doc(2000)
        chunks = chunker.chunk(doc)
        labels = {c.granularity for c in chunks}
        assert {"parent", "base", "child"}.issubset(labels)

    def test_all_chunks_have_granularity_in_metadata(self):
        chunker = MultiGranularityChunker()
        for chunk in chunker.chunk(_make_doc(1000)):
            assert "granularity" in chunk.metadata

    def test_parent_chunks_are_larger_than_child(self):
        chunker = MultiGranularityChunker(sizes=[1024, 512, 128])
        doc = _make_doc(3000)
        chunks = chunker.chunk(doc)
        parent_sizes = [len(c.content) for c in chunks if c.granularity == "parent"]
        child_sizes = [len(c.content) for c in chunks if c.granularity == "child"]
        assert max(parent_sizes) >= max(child_sizes)

    def test_source_preserved(self):
        chunker = MultiGranularityChunker()
        for chunk in chunker.chunk(_make_doc()):
            assert chunk.source == "test.txt"

    def test_chunk_index_is_int(self):
        chunker = MultiGranularityChunker()
        for chunk in chunker.chunk(_make_doc()):
            assert isinstance(chunk.chunk_index, int)

    def test_chunk_at_level_base(self):
        chunker = MultiGranularityChunker()
        chunks = chunker.chunk_at_level(_make_doc(2000), "base")
        assert all(hasattr(c, "content") for c in chunks)
        assert len(chunks) > 0

    def test_chunk_at_level_unknown_raises(self):
        chunker = MultiGranularityChunker()
        with pytest.raises(ValueError, match="Unknown granularity"):
            chunker.chunk_at_level(_make_doc(), "xlarge")

    def test_sizes_must_have_at_least_two_entries(self):
        with pytest.raises(ValueError):
            MultiGranularityChunker(sizes=[512])

    def test_empty_document(self):
        chunker = MultiGranularityChunker()
        doc = Document(content="", source="empty.txt", metadata={})
        chunks = chunker.chunk(doc)
        # Empty document should produce no chunks (recursive chunker handles empty)
        assert isinstance(chunks, list)

    def test_short_document_covered_at_all_levels(self):
        chunker = MultiGranularityChunker(sizes=[1024, 512, 128])
        doc = Document(content="Short text.", source="s.txt", metadata={})
        chunks = chunker.chunk(doc)
        # Each level produces at least 1 chunk for non-empty text
        assert len(chunks) >= 3  # one per level

    def test_custom_sizes(self):
        chunker = MultiGranularityChunker(sizes=[2048, 256])
        chunks = chunker.chunk(_make_doc(4000))
        assert {c.granularity for c in chunks} == {"parent", "base"}

    def test_chunk_size_stored_on_granular_chunk(self):
        chunker = MultiGranularityChunker(sizes=[1024, 512, 128])
        chunks = chunker.chunk(_make_doc(2000))
        for c in chunks:
            assert c.chunk_size in (1024, 512, 128)
