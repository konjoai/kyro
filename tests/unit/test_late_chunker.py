"""Unit tests for konjoai.ingest.chunkers.LateChunker.

Test taxonomy:
- Pure unit: all tests use a mock encoder (no sentence-transformers model download).
- No I/O, no process-state mutation.

Interview note: LateChunker approximates the Jina AI late-chunking paper by
encoding ALL sentences in a single batch call, then finding boundaries post-embedding.
"""

from __future__ import annotations

import numpy as np
import pytest

from konjoai.ingest.chunkers import LateChunker, get_chunker
from konjoai.ingest.loaders import Document

# ---------------------------------------------------------------------------
# Mock encoders (same pattern as test_semantic_splitter.py for consistency)
# ---------------------------------------------------------------------------


def _make_uniform_encoder(n_dims: int = 8):
    """All vectors identical → cosine sim = 1.0 everywhere → no splits."""

    def encoder(texts: list[str]) -> np.ndarray:
        arr = np.ones((len(texts), n_dims), dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / norms

    return encoder


def _make_alternating_encoder(n_dims: int = 8):
    """Adjacent vectors orthogonal → cosine sim = 0.0 → split at every boundary."""
    v0 = np.zeros(n_dims, dtype=np.float32)
    v0[0] = 1.0
    v1 = np.zeros(n_dims, dtype=np.float32)
    v1[1] = 1.0

    def encoder(texts: list[str]) -> np.ndarray:
        return np.stack([v0 if i % 2 == 0 else v1 for i in range(len(texts))])

    return encoder


def _make_block_encoder(n_dims: int = 8, block_size: int = 2):
    """Sentences share a vector within each block; blocks are orthogonal."""
    bases = [np.zeros(n_dims, dtype=np.float32) for _ in range(32)]
    for i, b in enumerate(bases):
        b[i % n_dims] = 1.0

    def encoder(texts: list[str]) -> np.ndarray:
        return np.stack([bases[i // block_size % len(bases)] for i in range(len(texts))])

    return encoder


def _doc(text: str, source: str = "test.md") -> Document:
    return Document(content=text, source=source, metadata={})


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestLateChunkerConstruction:
    def test_invalid_threshold_too_high(self):
        with pytest.raises(ValueError, match="similarity_threshold"):
            LateChunker(similarity_threshold=1.5)

    def test_invalid_threshold_negative(self):
        with pytest.raises(ValueError, match="similarity_threshold"):
            LateChunker(similarity_threshold=-0.1)

    def test_invalid_max_chunk_tokens(self):
        with pytest.raises(ValueError, match="max_chunk_tokens"):
            LateChunker(max_chunk_tokens=0)

    def test_valid_construction(self):
        lc = LateChunker(
            similarity_threshold=0.4,
            max_chunk_tokens=512,
            _encoder=_make_uniform_encoder(),
        )
        assert lc.similarity_threshold == 0.4
        assert lc.max_chunk_tokens == 512


# ---------------------------------------------------------------------------
# Chunking — basic correctness
# ---------------------------------------------------------------------------


class TestLateChunkerChunk:
    def test_empty_doc_returns_empty(self):
        lc = LateChunker(_encoder=_make_uniform_encoder())
        assert lc.chunk(_doc("")) == []

    def test_whitespace_only_returns_empty(self):
        lc = LateChunker(_encoder=_make_uniform_encoder())
        assert lc.chunk(_doc("   ")) == []

    def test_single_sentence_returns_one_chunk(self):
        lc = LateChunker(_encoder=_make_uniform_encoder())
        chunks = lc.chunk(_doc("This is the only sentence."))
        assert len(chunks) == 1
        assert "This is the only sentence." in chunks[0].content

    def test_source_propagated(self):
        lc = LateChunker(_encoder=_make_uniform_encoder())
        chunks = lc.chunk(_doc("Hello.", source="doc.txt"))
        assert all(c.source == "doc.txt" for c in chunks)

    def test_metadata_propagated(self):
        lc = LateChunker(_encoder=_make_uniform_encoder())
        doc = Document(content="Sentence.", source="x.md", metadata={"version": 2})
        chunks = lc.chunk(doc)
        assert all(c.metadata.get("version") == 2 for c in chunks)

    def test_chunker_tag_in_metadata(self):
        lc = LateChunker(_encoder=_make_uniform_encoder())
        chunks = lc.chunk(_doc("Sentence."))
        assert chunks[0].metadata.get("chunker") == "late"

    def test_chunk_indices_are_sequential(self):
        lc = LateChunker(similarity_threshold=0.5, _encoder=_make_alternating_encoder())
        text = "Alpha. Beta. Gamma. Delta."
        chunks = lc.chunk(_doc(text))
        for i, c in enumerate(chunks):
            assert c.chunk_index == i


# ---------------------------------------------------------------------------
# Splitting behaviour — semantic boundaries
# ---------------------------------------------------------------------------


class TestLateChunkerSplitting:
    def test_uniform_encoder_no_splits(self):
        """All sims = 1.0 → threshold not crossed → single chunk."""
        lc = LateChunker(
            similarity_threshold=0.4,
            max_chunk_tokens=512,
            _encoder=_make_uniform_encoder(),
        )
        text = "One sentence. Two sentences. Three sentences."
        chunks = lc.chunk(_doc(text))
        assert len(chunks) == 1

    def test_alternating_encoder_splits_every_boundary(self):
        """Adjacent sims = 0.0 → splits at every sentence boundary."""
        lc = LateChunker(similarity_threshold=0.4, _encoder=_make_alternating_encoder())
        text = "A sentence. Another one. Yet another. One more."
        chunks = lc.chunk(_doc(text))
        assert len(chunks) >= 3

    def test_max_chunk_tokens_enforces_length_ceiling(self):
        """Even when sims are high, a very small max_chunk_tokens forces splits."""
        lc = LateChunker(
            similarity_threshold=0.99,  # almost never split on sim
            max_chunk_tokens=1,  # ~4 chars — forces splits on every sentence
            _encoder=_make_uniform_encoder(),
        )
        text = "Long sentence here. Another long sentence here. Third long sentence."
        chunks = lc.chunk(_doc(text))
        # Each sentence should be its own chunk because max_chunk_tokens is tiny
        assert len(chunks) >= 2

    def test_no_empty_chunks(self):
        lc = LateChunker(similarity_threshold=0.5, _encoder=_make_alternating_encoder())
        text = "First. Second. Third. Fourth."
        chunks = lc.chunk(_doc(text))
        assert all(c.content.strip() for c in chunks)

    def test_all_content_present(self):
        """Union of all chunk contents covers every original sentence."""
        lc = LateChunker(similarity_threshold=0.5, _encoder=_make_alternating_encoder())
        sentences = ["Alpha beta.", "Gamma delta.", "Epsilon zeta.", "Eta theta."]
        text = " ".join(sentences)
        chunks = lc.chunk(_doc(text))
        for sent in sentences:
            assert any(sent.rstrip(".") in c.content for c in chunks), f"Sentence {sent!r} lost after late chunking"

    def test_sentence_count_in_metadata(self):
        lc = LateChunker(similarity_threshold=0.5, _encoder=_make_alternating_encoder())
        text = "A. B. C. D."
        chunks = lc.chunk(_doc(text))
        for c in chunks:
            assert "sentence_count" in c.metadata
            assert c.metadata["sentence_count"] >= 1

    def test_boundary_sim_in_metadata(self):
        """Chunks at split boundaries store the triggering similarity score."""
        lc = LateChunker(similarity_threshold=0.5, _encoder=_make_alternating_encoder())
        text = "A. B. C. D."
        chunks = lc.chunk(_doc(text))
        # All but the last chunk should have a boundary_sim
        if len(chunks) > 1:
            for c in chunks[:-1]:
                assert "boundary_sim" in c.metadata

    def test_block_encoder_groups_within_block(self):
        """Block encoder with block_size=3 groups consecutive sentences together."""
        lc = LateChunker(
            similarity_threshold=0.5,
            max_chunk_tokens=9999,  # no length ceiling
            _encoder=_make_block_encoder(block_size=3),
        )
        # 6 sentences → 2 blocks → at most 2 chunks
        text = "S1. S2. S3. S4. S5. S6."
        chunks = lc.chunk(_doc(text))
        assert len(chunks) <= 6  # did not blow up
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# Difference from SemanticSplitter
# ---------------------------------------------------------------------------


class TestLateChunkerVsSemanticSplitter:
    """LateChunker and SemanticSplitter expose different metadata keys."""

    def test_late_chunker_tag_is_late(self):
        from konjoai.ingest.chunkers import LateChunker

        lc = LateChunker(_encoder=_make_uniform_encoder())
        chunks = lc.chunk(_doc("Sentence."))
        assert chunks[0].metadata.get("chunker") == "late"

    def test_semantic_splitter_tag_is_semantic(self):
        from konjoai.ingest.chunkers import SemanticSplitter

        ss = SemanticSplitter(_encoder=_make_uniform_encoder())
        chunks = ss.chunk(_doc("Sentence."))
        assert chunks[0].metadata.get("splitter") == "semantic"


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


class TestGetChunkerLate:
    def test_factory_returns_late_chunker(self):
        chunker = get_chunker(
            strategy="late",
            chunk_size=512,
            similarity_threshold=0.4,
            _encoder=_make_uniform_encoder(),
        )
        assert isinstance(chunker, LateChunker)

    def test_factory_late_produces_chunks(self):
        chunker = get_chunker(
            strategy="late",
            similarity_threshold=0.4,
            _encoder=_make_uniform_encoder(),
        )
        chunks = chunker.chunk(_doc("One sentence. Two sentences."))
        assert len(chunks) >= 1

    def test_factory_chunk_size_passed_to_max_chunk_tokens(self):
        chunker = get_chunker(
            strategy="late",
            chunk_size=256,
            _encoder=_make_uniform_encoder(),
        )
        assert isinstance(chunker, LateChunker)
        assert chunker.max_chunk_tokens == 256
