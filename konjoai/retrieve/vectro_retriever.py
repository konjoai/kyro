"""VectroRetrieverAdapter — Vectro SIMD hybrid search behind konjo-core's HybridResult interface.

Wraps :class:`vectro.python.retriever.VectroRetriever` so it drops in as a
replacement for the existing BM25 + Qdrant dense pipeline.

Graceful-degradation stack (tried in order):
    1. **Rust bindings** — ``vectro_py`` available → build in-memory
       ``EmbeddingDataset``, use SIMD-accelerated ``hybrid_search_py``.
    2. **Pure-Python fallback** — ``vectro_py`` unavailable → numpy cosine
       similarity + ``rank_bm25`` BM25 entirely in Python (~10× slower).
    3. **Passthrough** — corpus too large (> ``MAX_CORPUS_POINTS``) or
       collection empty → delegates to the existing :func:`hybrid_search`.

Corpus is loaded lazily from Qdrant on the first :meth:`search` call and
cached in memory.  Call :meth:`rebuild` to flush the cache (e.g. after
re-ingesting documents).

Design contract:
    - Returns :class:`konjoai.retrieve.hybrid.HybridResult` objects so it is a
      drop-in for callers of :func:`konjoai.retrieve.hybrid.hybrid_search`.
    - Never raises; always returns a (possibly empty) list.
    - Thread-safe for read access (no mutable state during search).
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

from konjoai.retrieve.hybrid import HybridResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Maximum collection size for in-memory Vectro retrieval.
# Above this threshold the adapter falls back to Qdrant ANN + BM25.
MAX_CORPUS_POINTS = 200_000

# ── Rust-bindings availability check ────────────────────────────────────────
_VECTRO_PY_AVAILABLE: bool = False
try:
    from vectro.python.retriever import VectroRetriever  # type: ignore[import-not-found]
    from vectro_py import EmbeddingDataset, PyEmbedding  # type: ignore[import-not-found]
    _VECTRO_PY_AVAILABLE = True
    logger.debug("VectroRetrieverAdapter: vectro_py Rust bindings available — using SIMD path")
except ImportError:
    logger.debug("VectroRetrieverAdapter: vectro_py unavailable — using numpy fallback")


# ── Module-level singleton ───────────────────────────────────────────────────
_adapter: VectroRetrieverAdapter | None = None
_adapter_lock = threading.Lock()


class VectroRetrieverAdapter:
    """Wraps Vectro retrieval behind konjo-core's :class:`HybridResult` interface.

    Parameters
    ----------
    alpha:
        Dense score weight for hybrid fusion (0 = BM25 only, 1 = dense only).
        Default ``0.7`` matches konjo-core's ``hybrid_alpha`` setting.
    """

    def __init__(self, alpha: float = 0.7) -> None:
        self._alpha = float(np.clip(alpha, 0.0, 1.0))
        self._corpus_vectors: np.ndarray | None = None   # (N, dim) float32
        self._corpus_texts: list[str] = []
        self._corpus_sources: list[str] = []
        self._corpus_ids: list[str] = []
        self._retriever: VectroRetriever | None = None  # Rust-backed, if available
        self._bm25: object | None = None                  # rank_bm25 fallback
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 20) -> list[HybridResult]:
        """Return up to *top_k* results fused from dense + BM25.

        Falls back to :func:`~konjoai.retrieve.hybrid.hybrid_search` if the
        corpus is too large or could not be loaded.
        """
        self._ensure_corpus_loaded()

        n = len(self._corpus_texts)
        if n == 0:
            logger.debug("VectroRetrieverAdapter: empty corpus — falling back")
            return self._fallback(query, top_k)

        if n > MAX_CORPUS_POINTS:
            logger.debug(
                "VectroRetrieverAdapter: corpus (%d) > MAX (%d) — falling back",
                n, MAX_CORPUS_POINTS,
            )
            return self._fallback(query, top_k)

        if _VECTRO_PY_AVAILABLE and self._retriever is not None:
            return self._search_vectro(query, top_k)
        return self._search_numpy(query, top_k)

    def rebuild(self) -> None:
        """Flush the cached corpus so it is re-loaded on the next :meth:`search`."""
        with self._lock:
            self._corpus_vectors = None
            self._corpus_texts = []
            self._corpus_sources = []
            self._corpus_ids = []
            self._retriever = None
            self._bm25 = None
        logger.info("VectroRetrieverAdapter: corpus cache cleared")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_corpus_loaded(self) -> None:
        """Load the in-memory corpus once, double-checked under the lock."""
        if self._corpus_vectors is not None:
            return
        with self._lock:
            if self._corpus_vectors is not None:
                return  # another thread beat us
            self._load_corpus()

    def _load_corpus(self) -> None:
        """Scroll Qdrant to populate the in-memory corpus."""
        try:
            from konjoai.embed.encoder import get_encoder
            from konjoai.store.qdrant import get_store

            store = get_store()
            total = store.count()
            if total == 0:
                logger.debug("VectroRetrieverAdapter: collection empty")
                self._corpus_vectors = np.empty((0, get_encoder().dim), dtype=np.float32)
                return
            if total > MAX_CORPUS_POINTS:
                logger.info(
                    "VectroRetrieverAdapter: collection has %d points (> %d); "
                    "skipping in-memory load",
                    total, MAX_CORPUS_POINTS,
                )
                self._corpus_vectors = np.empty((0,), dtype=np.float32)  # sentinel
                return

            vecs, texts, sources, ids = store.scroll_all()
            self._corpus_vectors = vecs
            self._corpus_texts = texts
            self._corpus_sources = sources
            self._corpus_ids = ids
            logger.info(
                "VectroRetrieverAdapter: loaded corpus n=%d dim=%d",
                len(texts), vecs.shape[1] if vecs.ndim == 2 else 0,
            )

            if _VECTRO_PY_AVAILABLE:
                self._build_vectro_retriever(get_encoder())
            else:
                self._build_bm25()
        except Exception as exc:  # noqa: BLE001
            logger.warning("VectroRetrieverAdapter: corpus load failed: %s", exc)
            self._corpus_vectors = np.empty((0,), dtype=np.float32)  # sentinel

    def _build_vectro_retriever(self, encoder: object) -> None:
        """Build the Rust-backed :class:`VectroRetriever` from the loaded corpus."""
        try:
            ds = EmbeddingDataset()
            for doc_id, vec in zip(self._corpus_ids, self._corpus_vectors):
                ds.add_embedding(PyEmbedding(doc_id, vec))

            def _embed_fn(query: str) -> np.ndarray:
                return encoder.encode([query])[0].astype(np.float32)

            self._retriever = VectroRetriever(
                dataset=ds,
                texts=self._corpus_texts,
                ids=self._corpus_ids,
                embed_fn=_embed_fn,
                alpha=self._alpha,
            )
            logger.info("VectroRetrieverAdapter: Rust VectroRetriever ready (%d docs)", len(self._corpus_ids))
        except Exception as exc:  # noqa: BLE001
            logger.warning("VectroRetrieverAdapter: Rust retriever build failed: %s — using numpy", exc)
            self._retriever = None
            self._build_bm25()

    def _build_bm25(self) -> None:
        """Build rank_bm25 index for the pure-Python fallback path."""
        try:
            from rank_bm25 import BM25Okapi  # type: ignore[import-not-found]
            tokenised = [t.lower().split() for t in self._corpus_texts]
            self._bm25 = BM25Okapi(tokenised)
            logger.debug("VectroRetrieverAdapter: BM25 fallback index ready")
        except ImportError:
            logger.debug("VectroRetrieverAdapter: rank_bm25 unavailable; BM25 component skipped")

    def _search_vectro(self, query: str, top_k: int) -> list[HybridResult]:
        """Search using the Rust-backed VectroRetriever."""
        try:
            raw = self._retriever.retrieve(query, k=top_k)  # type: ignore[union-attr]
            return [
                HybridResult(
                    rrf_score=float(r.combined_score),
                    content=r.text,
                    source=self._corpus_sources[self._corpus_ids.index(r.id)]
                    if r.id in self._corpus_ids else "",
                    metadata={
                        "dense_score": float(r.dense_score),
                        "bm25_score": float(r.bm25_score),
                        "retriever": "vectro_rust",
                    },
                )
                for r in raw
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("VectroRetrieverAdapter: Rust search failed: %s — falling back", exc)
            return self._fallback(query, top_k)

    def _search_numpy(self, query: str, top_k: int) -> list[HybridResult]:
        """Pure-Python cosine + BM25 fallback when Rust bindings are absent."""
        try:
            from konjoai.embed.encoder import get_encoder
            enc = get_encoder()
            q_vec = enc.encode([query])[0].astype(np.float32)

            # Dense cosine similarity
            norms = np.linalg.norm(self._corpus_vectors, axis=1)  # type: ignore[arg-type]
            q_norm = float(np.linalg.norm(q_vec))
            if q_norm == 0.0 or norms.max() == 0.0:
                dense_scores = np.zeros(len(self._corpus_texts), dtype=np.float32)
            else:
                dense_scores = (self._corpus_vectors @ q_vec) / (norms * q_norm + 1e-9)  # type: ignore[operator]

            # BM25 sparse scores
            bm25_scores = np.zeros(len(self._corpus_texts), dtype=np.float32)
            if self._bm25 is not None:
                raw_bm25 = self._bm25.get_scores(query.lower().split())  # type: ignore[union-attr]
                bm25_max = float(max(raw_bm25)) if raw_bm25.max() > 0 else 1.0
                bm25_scores = np.array(raw_bm25, dtype=np.float32) / bm25_max

            combined = self._alpha * dense_scores + (1.0 - self._alpha) * bm25_scores
            top_idx = np.argsort(combined)[::-1][:top_k]

            return [
                HybridResult(
                    rrf_score=float(combined[i]),
                    content=self._corpus_texts[i],
                    source=self._corpus_sources[i],
                    metadata={
                        "dense_score": float(dense_scores[i]),
                        "bm25_score": float(bm25_scores[i]),
                        "retriever": "vectro_numpy",
                    },
                )
                for i in top_idx
                if float(combined[i]) > 0.0
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("VectroRetrieverAdapter: numpy search failed: %s — falling back", exc)
            return self._fallback(query, top_k)

    @staticmethod
    def _fallback(query: str, top_k: int) -> list[HybridResult]:
        """Delegate to the existing Qdrant ANN + BM25 RRF pipeline."""
        from konjoai.retrieve.hybrid import hybrid_search
        return hybrid_search(query, top_k_dense=top_k, top_k_sparse=top_k)


def get_vectro_retriever() -> VectroRetrieverAdapter:
    """Return the module-level singleton :class:`VectroRetrieverAdapter`."""
    global _adapter
    if _adapter is None:
        with _adapter_lock:
            if _adapter is None:
                from konjoai.config import get_settings
                s = get_settings()
                _adapter = VectroRetrieverAdapter(alpha=s.vectro_retriever_alpha)
    return _adapter
