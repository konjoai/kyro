from __future__ import annotations

import pytest
from ragos.ingest.chunkers import RecursiveChunker, SentenceWindowChunker, get_chunker
from ragos.ingest.loaders import Document


def _doc(text: str, source: str = "test.md") -> Document:
    return Document(content=text, source=source, metadata={})


# ---------------------------------------------------------------------------
# RecursiveChunker
# ---------------------------------------------------------------------------

class TestRecursiveChunker:
    def test_raises_if_overlap_gte_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            RecursiveChunker(chunk_size=100, overlap=100)

    def test_single_short_doc_produces_one_chunk(self) -> None:
        chunker = RecursiveChunker(chunk_size=512, overlap=64)
        doc = _doc("Hello, world!")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "Hello, world!"
        assert chunks[0].source == "test.md"
        assert chunks[0].chunk_index == 0

    def test_empty_content_produces_no_chunks(self) -> None:
        chunker = RecursiveChunker(chunk_size=512, overlap=64)
        chunks = chunker.chunk(_doc(""))
        assert chunks == []

    def test_long_doc_produces_multiple_chunks(self) -> None:
        chunker = RecursiveChunker(chunk_size=50, overlap=10)
        long_text = " ".join(f"word{i}" for i in range(50))
        chunks = chunker.chunk(_doc(long_text))
        assert len(chunks) > 1

    def test_chunk_size_respected(self) -> None:
        chunker = RecursiveChunker(chunk_size=50, overlap=5)
        long_text = "a" * 200
        chunks = chunker.chunk(_doc(long_text))
        for chunk in chunks:
            assert len(chunk.content) <= 60  # small slack for split logic

    def test_chunk_indices_are_sequential(self) -> None:
        chunker = RecursiveChunker(chunk_size=30, overlap=5)
        text = "\n\n".join(f"Paragraph {i} with some content." for i in range(10))
        chunks = chunker.chunk(_doc(text))
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_metadata_propagated(self) -> None:
        chunker = RecursiveChunker(chunk_size=512, overlap=64)
        doc = Document(content="hello", source="file.md", metadata={"page": 1})
        chunks = chunker.chunk(doc)
        assert chunks[0].metadata["page"] == 1


# ---------------------------------------------------------------------------
# SentenceWindowChunker
# ---------------------------------------------------------------------------

class TestSentenceWindowChunker:
    def test_single_sentence(self) -> None:
        chunker = SentenceWindowChunker(window_size=3)
        doc = _doc("This is one sentence.")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert "This is one sentence." in chunks[0].content

    def test_multiple_sentences_produce_windows(self) -> None:
        chunker = SentenceWindowChunker(window_size=2)
        doc = _doc("First sentence. Second sentence. Third sentence. Fourth sentence.")
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2  # depends on sentence count vs window_size

    def test_empty_content_produces_no_chunks(self) -> None:
        chunker = SentenceWindowChunker(window_size=3)
        chunks = chunker.chunk(_doc(""))
        assert chunks == []


# ---------------------------------------------------------------------------
# get_chunker factory
# ---------------------------------------------------------------------------

def test_get_chunker_recursive() -> None:
    chunker = get_chunker("recursive", 256, 32)
    assert isinstance(chunker, RecursiveChunker)


def test_get_chunker_sentence_window() -> None:
    chunker = get_chunker("sentence_window", 512, 64)
    assert isinstance(chunker, SentenceWindowChunker)


def test_get_chunker_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        get_chunker("unknown_strategy", 512, 64)
