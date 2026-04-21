"""Protocol interfaces for all swappable Kyro backends.

Design principles
-----------------
* Each protocol is ``@runtime_checkable`` so ``isinstance()`` works at
  module boundaries — no marker base classes required.
* Protocols carry only the methods used by the pipeline hot path.
  Implementation details (connection strings, model names, etc.) belong
  in concrete classes, not in the protocol.
* K4: All floating-point arrays crossing a protocol boundary must be
  ``float32``.  Asserted by callers, not here.
* K1: Every method either returns a value or raises.  No silent swallowing
  of exceptions.
"""
from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

import numpy as np


# ── VectorStoreAdapter ────────────────────────────────────────────────────────

@runtime_checkable
class VectorStoreAdapter(Protocol):
    """A pluggable vector database / similarity search backend.

    Concrete implementations: Qdrant (default), Chroma, Pinecone, Weaviate.
    """

    def upsert(
        self,
        vectors: list[np.ndarray],
        payloads: list[dict],
        ids: list[str] | None = None,
    ) -> int:
        """Write *vectors* with *payloads* into the store.

        Returns the number of vectors successfully indexed.
        """
        ...

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter: dict | None = None,
    ) -> list[dict]:
        """Return the *top_k* nearest neighbours of *query_vector*.

        Each dict must have keys ``content``, ``source``, ``score``, and
        ``metadata``.
        """
        ...

    def delete_collection(self) -> None:
        """Drop and recreate the collection (used during ingest invalidation)."""
        ...

    def count(self) -> int:
        """Return the total number of vectors currently indexed."""
        ...


# ── EmbedderAdapter ───────────────────────────────────────────────────────────

@runtime_checkable
class EmbedderAdapter(Protocol):
    """A text-to-vector embedding backend.

    Concrete implementations: SentenceTransformers (default), OpenAI
    text-embedding-3-small, Cohere embed-multilingual-v3.
    """

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return a float32 matrix of shape ``(len(texts), dim)``."""
        ...

    def encode_query(self, text: str) -> np.ndarray:
        """Return a float32 vector of shape ``(1, dim)`` for a single query.

        Many models benefit from a query-specific prefix (e.g., E5, BGE).
        Implementations should apply that prefix here and strip it from
        ``encode()`` for document embedding.
        """
        ...

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...


# ── GeneratorAdapter ──────────────────────────────────────────────────────────

@runtime_checkable
class GeneratorAdapter(Protocol):
    """An LLM generation backend.

    Concrete implementations: OpenAI, Anthropic, Squish (local).
    """

    def generate(self, question: str, context: str) -> object:
        """Return a ``GenerationResult``-like object with ``.answer``,
        ``.model``, and ``.usage`` attributes.

        K1: Raise if the API call fails.  Never return an empty string
        silently.
        """
        ...

    def generate_stream(self, question: str, context: str) -> Iterator[str]:
        """Yield response tokens one at a time.

        K1: Raise on the first API error rather than silently stopping
        iteration.
        """
        ...


# ── RetrieverAdapter ──────────────────────────────────────────────────────────

@runtime_checkable
class RetrieverAdapter(Protocol):
    """A hybrid retrieval backend.

    The interface is intentionally minimal: any object that implements
    ``search()`` qualifies.  This allows dense-only, sparse-only, hybrid,
    or graph-augmented retrieval to all be wired into the pipeline
    without if/else branching in the route handler.

    Concrete implementations: HybridRetriever (default), VectroRetriever,
    GraphRetriever (Sprint 15).
    """

    def search(
        self,
        query: str,
        top_k: int = 10,
        q_vec: "np.ndarray | None" = None,
    ) -> list[object]:
        """Return *top_k* results ranked by relevance.

        Each result must expose ``.content``, ``.source``, and ``.metadata``.
        The optional *q_vec* parameter allows callers to pass a pre-computed
        embedding to avoid re-encoding.
        """
        ...
