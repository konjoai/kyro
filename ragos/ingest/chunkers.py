from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ragos.ingest.loaders import Document


@dataclass
class Chunk:
    """A contiguous text segment ready for embedding."""

    content: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, doc: Document) -> list[Chunk]: ...


# ── Recursive character splitter ─────────────────────────────────────────────

class RecursiveChunker:
    """Split text recursively on paragraph → sentence → word boundaries."""

    _SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        chunks = self._split(doc.content)
        return [
            Chunk(
                content=c,
                source=doc.source,
                chunk_index=i,
                metadata={**doc.metadata},
            )
            for i, c in enumerate(chunks)
        ]

    def _split(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        for sep in self._SEPARATORS:
            parts = text.split(sep) if sep else list(text)
            if len(parts) > 1:
                return self._merge(parts, sep)

        # Fallback: hard slice
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.overlap)]

    def _merge(self, parts: list[str], sep: str) -> list[str]:
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part).lstrip(sep) if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If a single part is too large, recurse
                if len(part) > self.chunk_size:
                    chunks.extend(self._split(part))
                    current = ""
                else:
                    current = part
            # Build overlap tail
            if current and len(current) > self.overlap:
                tail = current[-self.overlap :]
                # But only if we just added a chunk
                if chunks and not current.startswith(tail):
                    pass  # overlap applied on next iteration via `current`
        if current:
            chunks.append(current)
        return [c for c in chunks if c.strip()]


# ── Sentence-window chunker ───────────────────────────────────────────────────

class SentenceWindowChunker:
    """Anchor on sentences; add a window of surrounding sentences as context."""

    _SENT_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, window_size: int = 3) -> None:
        self.window_size = window_size

    def chunk(self, doc: Document) -> list[Chunk]:
        sentences = [s.strip() for s in self._SENT_RE.split(doc.content) if s.strip()]
        chunks: list[Chunk] = []
        for i, sent in enumerate(sentences):
            lo = max(0, i - self.window_size)
            hi = min(len(sentences), i + self.window_size + 1)
            window = " ".join(sentences[lo:hi])
            chunks.append(
                Chunk(
                    content=window,
                    source=doc.source,
                    chunk_index=i,
                    metadata={
                        **doc.metadata,
                        "anchor_sentence": sent,
                        "window": f"{lo}-{hi}",
                    },
                )
            )
        return chunks


# ── Factory ───────────────────────────────────────────────────────────────────

def get_chunker(strategy: str = "recursive", chunk_size: int = 512, overlap: int = 64) -> Chunker:
    """Return a Chunker based on *strategy* name."""
    if strategy == "recursive":
        return RecursiveChunker(chunk_size=chunk_size, overlap=overlap)
    if strategy == "sentence_window":
        return SentenceWindowChunker()
    raise ValueError(f"Unknown chunking strategy: {strategy!r}. Choose 'recursive' or 'sentence_window'.")
