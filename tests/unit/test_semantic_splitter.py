"""Unit tests for konjoai.ingest.chunkers.SemanticSplitter.

Test taxonomy:
- Pure unit: all tests use a mock encoder (no sentence-transformers model download).
- No I/O, no process-state mutation.
"""

from __future__ import annotations

import numpy as np
import pytest

from konjoai.ingest.chunkers import SemanticSplitter, get_chunker
from konjoai.ingest.loaders import Document

# ---------------------------------------------------------------------------
# Mock encoder
# ---------------------------------------------------------------------------


def _make_uniform_encoder(n_dims: int = 8):
    """Return an encoder that produces identical unit vectors — no splits occur."""

    def encoder(texts: list[str]) -> np.ndarray:
        arr = np.ones((len(texts), n_dims), dtype=np.float32)
        # L2-normalise
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / norms

    return encoder


def _make_alternating_encoder(n_dims: int = 8):
    """Return an encoder that produces alternating orthogonal vectors.

    Adjacent vectors have cosine similarity 0.0, so every boundary is a split.
    """
    v0 = np.zeros(n_dims, dtype=np.float32)
    v0[0] = 1.0
    v1 = np.zeros(n_dims, dtype=np.float32)
    v1[1] = 1.0

    def encoder(texts: list[str]) -> np.ndarray:
        vecs = [v0 if i % 2 == 0 else v1 for i in range(len(texts))]
        return np.stack(vecs, axis=0)

    return encoder


def _make_block_encoder(n_dims: int = 8, block_size: int = 2):
    """Return an encoder where sentences in the same block share a vector.

    Cosine similarity within a block = 1.0; across blocks = 0.0 (orthogonal).
    Produces exactly ``ceil(N/block_size) - 1`` split points.
    """
    bases = [np.zeros(n_dims, dtype=np.float32) for _ in range(64)]
    for i, b in enumerate(bases):
        b[i % n_dims] = 1.0

    def encoder(texts: list[str]) -> np.ndarray:
        vecs = [bases[i // block_size % len(bases)] for i in range(len(texts))]
        return np.stack(vecs, axis=0)

    return encoder


def _doc(text: str, source: str = "test.md") -> Document:
    return Document(content=text, source=source, metadata={})


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestSemanticSplitterConstruction:
    def test_invalid_threshold_too_high(self):
        with pytest.raises(ValueError, match="similarity_threshold"):
            SemanticSplitter(similarity_threshold=1.5)

    def test_invalid_threshold_negative(self):
        with pytest.raises(ValueError, match="similarity_threshold"):
            SemanticSplitter(similarity_threshold=-0.1)

    def test_zero_threshold_accepted(self):
        # threshold = 0 → nothing splits (all sims ≥ 0 when vectors are non-negative)
        s = SemanticSplitter(similarity_threshold=0.0, _encoder=_make_uniform_encoder())
        assert s.similarity_threshold == 0.0

    def test_one_threshold_accepted(self):
        s = SemanticSplitter(similarity_threshold=1.0, _encoder=_make_uniform_encoder())
        assert s.similarity_threshold == 1.0


# ---------------------------------------------------------------------------
# Chunking — basic correctness
# ---------------------------------------------------------------------------


class TestSemanticSplitterChunk:
    def test_empty_doc_returns_empty(self):
        splitter = SemanticSplitter(_encoder=_make_uniform_encoder())
        assert splitter.chunk(_doc("")) == []

    def test_whitespace_only_returns_empty(self):
        splitter = SemanticSplitter(_encoder=_make_uniform_encoder())
        assert splitter.chunk(_doc("   \n  ")) == []

    def test_single_sentence_returns_one_chunk(self):
        splitter = SemanticSplitter(_encoder=_make_uniform_encoder())
        chunks = splitter.chunk(_doc("This is one sentence."))
        assert len(chunks) == 1
        assert "This is one sentence." in chunks[0].content

    def test_source_propagated(self):
        splitter = SemanticSplitter(_encoder=_make_uniform_encoder())
        chunks = splitter.chunk(_doc("Hello world.", source="myfile.txt"))
        assert all(c.source == "myfile.txt" for c in chunks)

    def test_metadata_propagated(self):
        splitter = SemanticSplitter(_encoder=_make_uniform_encoder())
        doc = Document(content="Sentence one.", source="f.md", metadata={"page": 3})
        chunks = splitter.chunk(doc)
        assert all(c.metadata.get("page") == 3 for c in chunks)

    def test_splitter_tag_in_metadata(self):
        splitter = SemanticSplitter(_encoder=_make_uniform_encoder())
        chunks = splitter.chunk(_doc("Sentence."))
        assert chunks[0].metadata.get("splitter") == "semantic"

    def test_chunk_indices_are_sequential(self):
        splitter = SemanticSplitter(_encoder=_make_uniform_encoder())
        text = "Sentence one. Sentence two. Sentence three."
        chunks = splitter.chunk(_doc(text))
        for i, c in enumerate(chunks):
            assert c.chunk_index == i


# ---------------------------------------------------------------------------
# Splitting behaviour
# ---------------------------------------------------------------------------


class TestSemanticSplitterSplitting:
    def test_uniform_encoder_no_splits(self):
        """All adjacent sims = 1.0 → threshold 0.4 not crossed → single chunk."""
        splitter = SemanticSplitter(similarity_threshold=0.4, _encoder=_make_uniform_encoder())
        text = "First sentence. Second sentence. Third sentence."
        chunks = splitter.chunk(_doc(text))
        assert len(chunks) == 1

    def test_alternating_encoder_splits_every_boundary(self):
        """Adjacent sims = 0.0 → threshold 0.4 always crossed → N-1 split points."""
        splitter = SemanticSplitter(similarity_threshold=0.4, _encoder=_make_alternating_encoder())
        text = "A sentence. Another one. Yet another. And one more."
        chunks = splitter.chunk(_doc(text))
        # 4 sentences → up to 3 splits → up to 4 chunks
        assert len(chunks) >= 3

    def test_block_encoder_splits_at_block_boundaries(self):
        """Block encoder with block_size=2 → split after every 2nd sentence."""
        splitter = SemanticSplitter(similarity_threshold=0.5, _encoder=_make_block_encoder(block_size=2))
        # 6 sentences → 3 blocks → up to 3 chunks
        text = "S1. S2. S3. S4. S5. S6."
        chunks = splitter.chunk(_doc(text))
        assert 2 <= len(chunks) <= 6  # at least some splitting happened

    def test_high_threshold_produces_more_chunks(self):
        splitter_low = SemanticSplitter(similarity_threshold=0.1, _encoder=_make_alternating_encoder())
        splitter_high = SemanticSplitter(similarity_threshold=0.9, _encoder=_make_alternating_encoder())
        text = "A. B. C. D. E. F."
        chunks_low = splitter_low.chunk(_doc(text))
        chunks_high = splitter_high.chunk(_doc(text))
        # Higher threshold → more splits
        assert len(chunks_high) >= len(chunks_low)

    def test_no_empty_chunks(self):
        splitter = SemanticSplitter(similarity_threshold=0.5, _encoder=_make_alternating_encoder())
        text = "Sentence A. Sentence B. Sentence C. Sentence D."
        chunks = splitter.chunk(_doc(text))
        assert all(c.content.strip() for c in chunks), "Empty chunk produced"

    def test_all_content_present(self):
        """Chunks collectively contain all original sentences."""
        splitter = SemanticSplitter(similarity_threshold=0.5, _encoder=_make_alternating_encoder())
        sentences = ["Alpha beta.", "Gamma delta.", "Epsilon zeta.", "Eta theta."]
        text = " ".join(sentences)
        chunks = splitter.chunk(_doc(text))
        for sent in sentences:
            # Each original sentence should be recoverable from the chunks
            assert any(sent.rstrip(".") in c.content for c in chunks), f"Sentence {sent!r} missing from chunks"

    def test_sentence_count_in_metadata(self):
        splitter = SemanticSplitter(similarity_threshold=0.5, _encoder=_make_alternating_encoder())
        text = "A sentence here. Another one there. A third one."
        chunks = splitter.chunk(_doc(text))
        for c in chunks:
            assert "sentence_count" in c.metadata
            assert c.metadata["sentence_count"] >= 1


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


class TestGetChunkerSemantic:
    def test_factory_returns_semantic_splitter(self):
        chunker = get_chunker(
            strategy="semantic",
            similarity_threshold=0.4,
            _encoder=_make_uniform_encoder(),
        )
        assert isinstance(chunker, SemanticSplitter)

    def test_factory_semantic_produces_chunks(self):
        chunker = get_chunker(
            strategy="semantic",
            similarity_threshold=0.4,
            _encoder=_make_uniform_encoder(),
        )
        chunks = chunker.chunk(_doc("One sentence. Two sentences."))
        assert len(chunks) >= 1

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunker(strategy="nonexistent_strategy")
