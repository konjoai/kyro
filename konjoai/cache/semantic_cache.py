"""Semantic cache for konjo-core query pipeline.

Design:
  - Two-level lookup: exact string match (O(1) dict) then cosine similarity scan (O(n)).
  - LRU eviction keeps hot entries resident; cold entries are dropped at max_size.
  - Thread-safe: all mutations hold _lock.
  - K3: disabled when cache_enabled=False → caller gets None from get_semantic_cache().
  - K4: q_vec must be float32, shape (1, dim). Asserted at store() boundary.
  - K5: uses only numpy (already required) and stdlib collections.OrderedDict.
  - K6: cache_hit field on QueryResponse is False by default; cache path sets it to True.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SemanticCacheEntry:
    question: str
    question_vec: np.ndarray      # float32, shape (1, dim)
    response: object              # QueryResponse — typed as object to avoid circular import
    created_at: float = field(default_factory=time.monotonic)
    hit_count: int = 0


class SemanticCache:
    """In-process semantic cache with exact + cosine-similarity lookup and LRU eviction.

    Args:
        max_size: Maximum number of entries before LRU eviction kicks in.
        threshold: Cosine similarity threshold for a semantic cache hit (0.0–1.0).
    """

    def __init__(self, max_size: int = 500, threshold: float = 0.95) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")

        self._max_size = max_size
        self._threshold = threshold
        self._lock = threading.Lock()
        # Keyed by normalised question string for exact-match fast path
        self._exact: dict[str, SemanticCacheEntry] = {}
        # Full LRU order (most-recent at end)
        self._lru: OrderedDict[str, SemanticCacheEntry] = OrderedDict()

        self._total_hits: int = 0
        self._total_misses: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def lookup(self, question: str, q_vec: np.ndarray) -> object | None:
        """Return a cached QueryResponse, or None on cache miss.

        Fast path: exact string match.
        Slow path: scan all entries for cosine similarity >= threshold.
        """
        key = self._normalise(question)
        with self._lock:
            # 1. Exact-match fast path
            entry = self._exact.get(key)
            if entry is not None:
                self._lru.move_to_end(key)
                entry.hit_count += 1
                self._total_hits += 1
                logger.debug("cache hit (exact) key=%s hits=%d", key[:40], self._total_hits)
                return entry.response

            # 2. Semantic similarity scan
            q_norm = self._l2_norm(q_vec)
            best_key: str | None = None
            best_sim: float = -1.0
            for k, e in self._lru.items():
                sim = float(np.dot(q_norm, self._l2_norm(e.question_vec)))
                if sim > best_sim:
                    best_sim = sim
                    best_key = k

            if best_key is not None and best_sim >= self._threshold:
                entry = self._lru[best_key]
                self._lru.move_to_end(best_key)
                entry.hit_count += 1
                self._total_hits += 1
                logger.debug(
                    "cache hit (semantic) sim=%.4f key=%s hits=%d",
                    best_sim, best_key[:40], self._total_hits,
                )
                return entry.response

            self._total_misses += 1
            return None

    def store(self, question: str, q_vec: np.ndarray, response: object) -> None:
        """Insert a question/response pair into the cache.

        K4: asserts q_vec is float32.
        Evicts the least-recently-used entry if max_size is exceeded.
        """
        assert q_vec.dtype == np.float32, (
            f"SemanticCache.store: q_vec must be float32, got {q_vec.dtype}"
        )
        key = self._normalise(question)
        entry = SemanticCacheEntry(
            question=question,
            question_vec=q_vec.copy(),   # own the array
            response=response,
        )
        with self._lock:
            if key in self._lru:
                # Refresh existing entry (updated answer)
                self._lru.move_to_end(key)
                self._lru[key] = entry
                self._exact[key] = entry
                return

            # Insert new entry
            self._lru[key] = entry
            self._exact[key] = entry

            # LRU eviction
            while len(self._lru) > self._max_size:
                evicted_key, _ = self._lru.popitem(last=False)
                self._exact.pop(evicted_key, None)
                logger.debug("cache evict key=%s size=%d", evicted_key[:40], len(self._lru))

    def invalidate(self) -> None:
        """Clear all entries. Called by ingest to prevent stale data (K3)."""
        with self._lock:
            count = len(self._lru)
            self._lru.clear()
            self._exact.clear()
        logger.info("cache invalidated — cleared %d entries", count)

    def stats(self) -> dict:
        """Return a snapshot of cache statistics."""
        with self._lock:
            size = len(self._lru)
        total = self._total_hits + self._total_misses
        hit_rate = self._total_hits / total if total > 0 else 0.0
        return {
            "size": size,
            "max_size": self._max_size,
            "threshold": self._threshold,
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "hit_rate": round(hit_rate, 4),
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise(text: str) -> str:
        """Normalise a question string for exact-match keying."""
        return text.strip().lower()

    @staticmethod
    def _l2_norm(vec: np.ndarray) -> np.ndarray:
        """Return L2-normalised 1-D view of a (1, dim) or (dim,) array."""
        flat = vec.ravel().astype(np.float32)
        norm = np.linalg.norm(flat)
        if norm < 1e-10:
            return flat
        return flat / norm


# ── Module-level singleton ────────────────────────────────────────────────────

_cache: SemanticCache | None = None
_cache_lock = threading.Lock()


def get_semantic_cache() -> SemanticCache | None:
    """Return the module-level SemanticCache, or None if cache_enabled=False.

    Lazily initialises the cache on first call (K3: transparent when disabled).
    """
    global _cache  # noqa: PLW0603
    from konjoai.config import get_settings  # local import avoids circular dep

    settings = get_settings()
    if not settings.cache_enabled:
        return None

    if _cache is None:
        with _cache_lock:
            if _cache is None:
                _cache = SemanticCache(
                    max_size=settings.cache_max_size,
                    threshold=settings.cache_similarity_threshold,
                )
                logger.info(
                    "semantic cache initialised — max_size=%d threshold=%.2f",
                    settings.cache_max_size,
                    settings.cache_similarity_threshold,
                )
    return _cache


def _reset_cache() -> None:
    """Test helper: reset the module-level singleton. Never call in production."""
    global _cache  # noqa: PLW0603
    with _cache_lock:
        _cache = None
