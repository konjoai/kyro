from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_index: "BM25Index | None" = None


@dataclass
class BM25Result:
    score: float
    content: str
    source: str
    metadata: dict


class BM25Index:
    """In-memory BM25 index backed by rank-bm25.BM25Okapi."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._corpus: list[str] = []
        self._sources: list[str] = []
        self._metadatas: list[dict] = []
        self._bm25: object | None = None

    def build(self, contents: list[str], sources: list[str], metadatas: list[dict]) -> None:
        """Build the BM25 index from *contents*.

        Must be called before :py:meth:`search`.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError("rank-bm25 is required: pip install rank-bm25") from e

        self._corpus = list(contents)
        self._sources = list(sources)
        self._metadatas = list(metadatas)
        tokenised = [c.lower().split() for c in contents]
        self._bm25 = BM25Okapi(tokenised, k1=self._k1, b=self._b)
        logger.info("BM25Index: built on %d documents", len(contents))

    def search(self, query: str, top_k: int = 20) -> list[BM25Result]:
        if self._bm25 is None:
            raise RuntimeError("BM25Index.build() must be called before search()")

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)  # type: ignore[union-attr]

        # argsort descending, take top_k
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            BM25Result(
                score=float(scores[i]),
                content=self._corpus[i],
                source=self._sources[i],
                metadata=self._metadatas[i],
            )
            for i in indices
        ]

    @property
    def built(self) -> bool:
        return self._bm25 is not None


def get_sparse_index() -> BM25Index:
    """Return the module-level singleton BM25 index (not auto-built; call .build() first)."""
    global _index
    if _index is None:
        _index = BM25Index()
    return _index
