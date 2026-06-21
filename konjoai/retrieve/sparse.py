"""BM25 sparse lexical retrieval index."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_index: BM25Index | None = None


@dataclass
class BM25Result:
    """A corpus chunk with its BM25 score."""

    score: float
    content: str
    source: str
    metadata: dict


class BM25Index:
    """In-memory BM25 index backed by rank-bm25.BM25Okapi.

    Persistence
    -----------
    Call :meth:`save` after :meth:`build` to write the index to disk.
    Call :meth:`load` to restore it; skip a re-build on startup.
    The pickle file contains the tokenised corpus and BM25Okapi object.
    """

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

    def save(self, path: str | Path) -> None:
        """Persist the index to *path* (pickle).

        Saves the corpus, sources, metadatas, BM25 object, and k1/b parameters.
        Raises ``RuntimeError`` if the index has not been built yet.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25Index.build() must be called before save()")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "corpus": self._corpus,
            "sources": self._sources,
            "metadatas": self._metadatas,
            "bm25": self._bm25,
            "k1": self._k1,
            "b": self._b,
        }
        with p.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BM25Index: saved %d docs to %s", len(self._corpus), path)

    def load(self, path: str | Path) -> bool:
        """Restore the index from a previously saved pickle.

        Returns ``True`` on success, ``False`` if the file does not exist or
        is unreadable (caller should fall back to rebuild).
        """
        p = Path(path)
        if not p.exists():
            logger.debug("BM25Index: no saved index at %s", path)
            return False
        try:
            with p.open("rb") as f:
                payload = pickle.load(f)  # noqa: S301  # trusted local file
            self._corpus = payload["corpus"]
            self._sources = payload["sources"]
            self._metadatas = payload["metadatas"]
            self._bm25 = payload["bm25"]
            self._k1 = payload["k1"]
            self._b = payload["b"]
            logger.info("BM25Index: loaded %d docs from %s", len(self._corpus), path)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("BM25Index: failed to load from %s: %s", path, exc)
            return False

    def search(self, query: str, top_k: int = 20) -> list[BM25Result]:
        """Return the top-*k* BM25-scored chunks for *query*, best-first."""
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
    """Return the module-level singleton BM25 index.

    If ``bm25_persist_path`` is configured in :class:`konjoai.config.Settings`
    and a saved index file exists, the index is loaded automatically.
    """
    global _index
    if _index is None:
        _index = BM25Index()
        try:
            from konjoai.config import get_settings

            path = get_settings().bm25_persist_path
            if path:
                _index.load(path)
        except Exception:  # noqa: BLE001
            pass  # config unavailable during testing — leave index empty
    return _index
