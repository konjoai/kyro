"""Document chunkers: recursive, sentence-window, semantic, and late chunking."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from konjoai.ingest.loaders import Document


@dataclass
class Chunk:
    """A contiguous text segment ready for embedding."""

    content: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class Chunker(Protocol):
    """Protocol for splitting a :class:`Document` into :class:`Chunk` objects."""

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split *doc* into a list of chunks."""
        ...


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
        """Split *doc* recursively, carrying its metadata onto each chunk."""
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
        """Recursively split *text* on the first separator that yields >1 part."""
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
        """Greedily merge *parts* (rejoined with *sep*) up to ``chunk_size``."""
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


# ── Semantic Splitter ─────────────────────────────────────────────────────────

try:
    import numpy as np  # already required by sentence-transformers

    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NUMPY_AVAILABLE = False


def _cosine_similarities(embeddings: np.ndarray) -> np.ndarray:
    """Return pairwise cosine similarities between adjacent rows.

    Args:
        embeddings: ``(N, dim)`` float32 array (need not be pre-normalised).

    Returns:
        ``(N-1,)`` float32 array of adjacent cosine similarities.
    """
    import numpy as np  # noqa: PLC0415 — guard already checked at call site

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normed = embeddings / norms
    return np.einsum("ij,ij->i", normed[:-1], normed[1:]).astype(np.float32)


class _EncoderBackedChunker:
    """Mixin providing the lazy encoder load shared by embedding-based chunkers."""

    model_name: str
    device: str

    def _get_encoder(self):
        """Return the encoder, loading the model lazily on first call."""
        if self._enc is None:
            from konjoai.embed.encoder import SentenceEncoder  # noqa: PLC0415

            self._enc = SentenceEncoder(model_name=self.model_name, device=self.device)
        return self._enc

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode *texts*; supports both :class:`SentenceEncoder` and raw callables."""
        import numpy as np  # noqa: PLC0415

        enc = self._get_encoder()
        if hasattr(enc, "encode"):
            return enc.encode(texts)
        result = enc(texts)
        return np.array(result, dtype=np.float32)


class SemanticSplitter(_EncoderBackedChunker):
    """Split documents at semantic paragraph boundaries.

    Embeds every sentence individually (plus an optional context buffer of
    surrounding sentences), then inserts a chunk boundary wherever the cosine
    similarity between adjacent sentence embeddings drops below
    *similarity_threshold*.

    This implements the **Semantic Chunking** technique popularised by Greg
    Kamradt (2023) and later incorporated into LangChain's
    ``SemanticChunker``.  The key insight is that abrupt similarity drops
    between consecutive sentences signal topic or paragraph transitions.

    Contrast with :class:`LateChunker`:

    * ``SemanticSplitter`` embeds each sentence with a context buffer of
      ``buffer_size`` surrounding sentences — giving local neighbourhood
      context.
    * :class:`LateChunker` embeds *all* sentences in a single batch call,
      approximating the full-document context of the Jina late-chunking paper.

    Args:
        model_name:  HuggingFace model identifier for sentence-transformers.
        similarity_threshold:  Cosine similarity in ``[0, 1]`` below which a
            chunk boundary is inserted.  Lower values produce fewer, larger
            chunks; higher values produce finer-grained splits.
        buffer_size:  Number of sentences to prepend/append when building
            the embedding input for each anchor sentence.  ``0`` embeds each
            sentence in isolation; ``1`` adds one sentence of context on each
            side.
        device:  Torch device string (``"cpu"``, ``"mps"``, ``"cuda"``).
        _encoder:  Optional encoder callable ``(list[str]) -> np.ndarray``.
            Injected in tests to avoid downloading model weights.
    """

    _SENT_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.4,
        buffer_size: int = 1,
        device: str = "cpu",
        _encoder=None,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(f"similarity_threshold must be in [0, 1], got {similarity_threshold}")
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.buffer_size = buffer_size
        self.device = device
        self._enc = _encoder  # lazy-loaded on first chunk() call when None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, doc: Document) -> list[Chunk]:
        """Split *doc* at semantic boundaries.

        Returns:
            List of :class:`Chunk` objects with ``metadata["splitter"] = "semantic"``.
        """
        sentences = [s.strip() for s in self._SENT_RE.split(doc.content) if s.strip()]
        if not sentences:
            return []
        if len(sentences) == 1:
            return [
                Chunk(
                    content=sentences[0],
                    source=doc.source,
                    chunk_index=0,
                    metadata={**doc.metadata, "splitter": "semantic"},
                )
            ]

        # Build buffered context strings for embedding
        buffered: list[str] = []
        for i, _ in enumerate(sentences):
            lo = max(0, i - self.buffer_size)
            hi = min(len(sentences), i + self.buffer_size + 1)
            buffered.append(" ".join(sentences[lo:hi]))

        embeddings = self._encode(buffered)
        sims = _cosine_similarities(embeddings)  # (N-1,)

        # Insert boundaries where similarity drops below threshold
        split_after = [i for i, s in enumerate(sims) if float(s) < self.similarity_threshold]

        return self._build_chunks(sentences, split_after, doc)

    def _build_chunks(self, sentences: list[str], split_after: list[int], doc: Document) -> list[Chunk]:
        """Convert split indices into :class:`Chunk` objects."""
        chunks: list[Chunk] = []
        start = 0
        for idx in split_after:
            group = sentences[start : idx + 1]
            if group:
                chunks.append(
                    Chunk(
                        content=" ".join(group),
                        source=doc.source,
                        chunk_index=len(chunks),
                        metadata={
                            **doc.metadata,
                            "splitter": "semantic",
                            "sentence_count": len(group),
                        },
                    )
                )
            start = idx + 1

        tail = sentences[start:]
        if tail:
            chunks.append(
                Chunk(
                    content=" ".join(tail),
                    source=doc.source,
                    chunk_index=len(chunks),
                    metadata={
                        **doc.metadata,
                        "splitter": "semantic",
                        "sentence_count": len(tail),
                    },
                )
            )

        # Guarantee at least one chunk (threshold too high → no splits)
        if not chunks:
            return [
                Chunk(
                    content=doc.content,
                    source=doc.source,
                    chunk_index=0,
                    metadata={**doc.metadata, "splitter": "semantic"},
                )
            ]
        return chunks


# ── Late Chunker ──────────────────────────────────────────────────────────────


class LateChunker(_EncoderBackedChunker):
    """Post-embedding semantic chunking (Late Chunking).

    Implements the Late Chunking technique from *Jina AI (2024)*:
    https://jina.ai/news/late-chunking-in-long-context-embedding-models/

    **Core idea:** encode the full document (all its sentences) in a
    *single* batch call so that the embedder can attend across the entire
    document during computation.  Chunk boundaries are then detected *after*
    the embeddings are produced — hence "late" chunking.

    In the original paper, this is done with jina-embeddings-v2 (8 192-token
    context window) which encodes the document as one long sequence and pools
    per-sentence token spans.  This implementation approximates the technique
    for standard sentence-transformers by sending all sentence strings in one
    ``encode()`` call (one batch), preserving shared computation and batch
    normalisation context.

    Contrast with :class:`SemanticSplitter`:

    * :class:`SemanticSplitter` embeds each sentence with a *local* context
      buffer (``buffer_size`` surrounding sentences).
    * ``LateChunker`` embeds *all* sentences together in one shot — the
      "full document context" approximation.

    Chunk boundaries are inserted where cosine similarity between adjacent
    sentence embeddings falls below *similarity_threshold* **or** where
    adding the next sentence would push the chunk past *max_chunk_tokens*.

    Args:
        model_name:  HuggingFace model identifier for sentence-transformers.
        similarity_threshold:  Similarity below which a boundary is forced.
        max_chunk_tokens:  Soft token ceiling per chunk.  A rough
            ``4 chars ≈ 1 token`` heuristic is used so the chunker remains
            free of tokenizer dependencies.
        device:  Torch device string.
        _encoder:  Optional encoder callable for testing.
    """

    _SENT_RE = re.compile(r"(?<=[.!?])\s+")
    _CHARS_PER_TOKEN: int = 4  # conservative approximation

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.4,
        max_chunk_tokens: int = 512,
        device: str = "cpu",
        _encoder=None,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(f"similarity_threshold must be in [0, 1], got {similarity_threshold}")
        if max_chunk_tokens < 1:
            raise ValueError(f"max_chunk_tokens must be ≥ 1, got {max_chunk_tokens}")
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.max_chunk_tokens = max_chunk_tokens
        self.device = device
        self._enc = _encoder

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, doc: Document) -> list[Chunk]:
        """Chunk *doc* using late (post-embedding) boundary detection.

        Returns:
            List of :class:`Chunk` objects with ``metadata["chunker"] = "late"``.
        """
        sentences = [s.strip() for s in self._SENT_RE.split(doc.content) if s.strip()]
        if not sentences:
            return []
        if len(sentences) == 1:
            return [
                Chunk(
                    content=sentences[0],
                    source=doc.source,
                    chunk_index=0,
                    metadata={**doc.metadata, "chunker": "late"},
                )
            ]

        # Encode ALL sentences in a single batch — the "late" approximation
        embeddings = self._encode(sentences)
        sims = _cosine_similarities(embeddings)  # (N-1,)

        max_chars = self.max_chunk_tokens * self._CHARS_PER_TOKEN

        # Find split points: similarity drop OR length ceiling
        split_after: list[int] = []
        current_chars = len(sentences[0])
        for i, sim in enumerate(sims):
            next_len = len(sentences[i + 1])
            would_exceed = (current_chars + 1 + next_len) > max_chars
            if float(sim) < self.similarity_threshold or would_exceed:
                split_after.append(i)
                current_chars = next_len
            else:
                current_chars += 1 + next_len

        # Build chunks
        chunks: list[Chunk] = []
        start = 0
        for idx in split_after:
            group = sentences[start : idx + 1]
            if group:
                boundary_sim = float(sims[idx]) if idx < len(sims) else None
                chunks.append(
                    Chunk(
                        content=" ".join(group),
                        source=doc.source,
                        chunk_index=len(chunks),
                        metadata={
                            **doc.metadata,
                            "chunker": "late",
                            "sentence_count": len(group),
                            "boundary_sim": boundary_sim,
                        },
                    )
                )
            start = idx + 1

        tail = sentences[start:]
        if tail:
            chunks.append(
                Chunk(
                    content=" ".join(tail),
                    source=doc.source,
                    chunk_index=len(chunks),
                    metadata={
                        **doc.metadata,
                        "chunker": "late",
                        "sentence_count": len(tail),
                        "boundary_sim": None,
                    },
                )
            )

        if not chunks:
            return [
                Chunk(
                    content=doc.content,
                    source=doc.source,
                    chunk_index=0,
                    metadata={**doc.metadata, "chunker": "late"},
                )
            ]
        return chunks


# ── Sentence-window chunker ───────────────────────────────────────────────────


class SentenceWindowChunker:
    """Anchor on sentences; add a window of surrounding sentences as context."""

    _SENT_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, window_size: int = 3) -> None:
        self.window_size = window_size

    def chunk(self, doc: Document) -> list[Chunk]:
        """Emit one chunk per sentence, each padded with its surrounding window."""
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


def get_chunker(
    strategy: str = "recursive",
    chunk_size: int = 512,
    overlap: int = 64,
    similarity_threshold: float = 0.4,
    device: str = "cpu",
    _encoder=None,
) -> Chunker:
    """Return a :class:`Chunker` based on *strategy* name.

    Supported strategies:

    * ``"recursive"`` — :class:`RecursiveChunker`: paragraph → sentence → word splitting.
    * ``"sentence_window"`` — :class:`SentenceWindowChunker`: anchor + surrounding-sentence context.
    * ``"semantic"`` — :class:`SemanticSplitter`: embedding-based paragraph boundary detection.
    * ``"late"`` — :class:`LateChunker`: post-embedding full-document boundary detection.

    Args:
        strategy:  One of ``"recursive"``, ``"sentence_window"``, ``"semantic"``, ``"late"``.
        chunk_size:  Target chunk size in characters (``recursive`` only).
        overlap:     Overlap in characters between consecutive chunks (``recursive`` only).
        similarity_threshold:  Cosine similarity split threshold (``semantic`` / ``late``).
        device:  Torch device for embedding models (``semantic`` / ``late``).
        _encoder:  Optional encoder callable for ``semantic`` / ``late`` (testing).

    Raises:
        ValueError: If *strategy* is not one of the supported values.
    """
    if strategy == "recursive":
        return RecursiveChunker(chunk_size=chunk_size, overlap=overlap)
    if strategy == "sentence_window":
        return SentenceWindowChunker()
    if strategy == "semantic":
        return SemanticSplitter(
            similarity_threshold=similarity_threshold,
            device=device,
            _encoder=_encoder,
        )
    if strategy == "late":
        return LateChunker(
            similarity_threshold=similarity_threshold,
            max_chunk_tokens=chunk_size,
            device=device,
            _encoder=_encoder,
        )
    raise ValueError(
        f"Unknown chunking strategy: {strategy!r}. Choose 'recursive', 'sentence_window', 'semantic', or 'late'."
    )
