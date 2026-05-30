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
    ttl_seconds: int = 0       # 0 = no expiry for this entry
    last_accessed: float = field(default_factory=time.monotonic)

    def is_expired(self) -> bool:
        """Return True if this entry's TTL has elapsed."""
        return bool(self.ttl_seconds > 0 and time.monotonic() - self.created_at > self.ttl_seconds)

    def access_rate_per_day(self) -> float:
        """Estimated access frequency in hits / day based on age and hit_count."""
        age_days = max((time.monotonic() - self.created_at) / 86400.0, 1 / 1440.0)
        return self.hit_count / age_days

    def days_since_last_access(self) -> float:
        """Days elapsed since this entry was last accessed (created or hit)."""
        return (time.monotonic() - self.last_accessed) / 86400.0


class SemanticCache:
    """In-process semantic cache with exact + cosine-similarity lookup and LRU eviction.

    Args:
        max_size: Maximum number of entries before LRU eviction kicks in.
        threshold: Cosine similarity threshold for a semantic cache hit (0.0–1.0).
    """

    def __init__(
        self,
        max_size: int = 500,
        threshold: float = 0.95,
        ttl_seconds: int = 0,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        if ttl_seconds < 0:
            raise ValueError(f"ttl_seconds must be >= 0, got {ttl_seconds}")

        self._max_size = max_size
        self._threshold = threshold
        self._ttl_seconds = ttl_seconds   # 0 = no expiry (default)
        self._lock = threading.Lock()
        # Keyed by normalised question string for exact-match fast path
        self._exact: dict[str, SemanticCacheEntry] = {}
        # Full LRU order (most-recent at end)
        self._lru: OrderedDict[str, SemanticCacheEntry] = OrderedDict()

        self._total_hits: int = 0
        self._total_misses: int = 0
        self._analytics_buf: object | None = None  # LatencyBuffer, set via set_analytics_buffer()

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
                if entry.is_expired():
                    # Lazy eviction of the stale entry on lookup
                    self._lru.pop(key, None)
                    self._exact.pop(key, None)
                    logger.debug("cache evict (ttl, exact) key=%s", key[:40])
                else:
                    self._lru.move_to_end(key)
                    entry.hit_count += 1
                    entry.last_accessed = time.monotonic()
                    self._total_hits += 1
                    logger.debug("cache hit (exact) key=%s hits=%d", key[:40], self._total_hits)
                    return entry.response

            # 2. Semantic similarity scan — skip expired entries
            q_norm = self._l2_norm(q_vec)
            best_key: str | None = None
            best_sim: float = -1.0
            for k, e in self._lru.items():
                if e.is_expired():
                    continue
                sim = float(np.dot(q_norm, self._l2_norm(e.question_vec)))
                if sim > best_sim:
                    best_sim = sim
                    best_key = k

            if best_key is not None and best_sim >= self._threshold:
                entry = self._lru[best_key]
                self._lru.move_to_end(best_key)
                entry.hit_count += 1
                entry.last_accessed = time.monotonic()
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
            ttl_seconds=self._ttl_seconds,
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

    def expired_count(self) -> int:
        """Return the number of entries that have exceeded their TTL.

        Does not remove them; call :meth:`evict_expired` to do that.
        Returns 0 when ``ttl_seconds == 0`` (no TTL configured).
        """
        if self._ttl_seconds == 0:
            return 0
        with self._lock:
            return sum(1 for e in self._lru.values() if e.is_expired())

    def evict_expired(self) -> int:
        """Remove all entries that have exceeded their TTL. Returns the eviction count."""
        if self._ttl_seconds == 0:
            return 0
        with self._lock:
            stale = [k for k, e in self._lru.items() if e.is_expired()]
            for k in stale:
                self._lru.pop(k, None)
                self._exact.pop(k, None)
        if stale:
            logger.info("cache ttl eviction — removed %d expired entries", len(stale))
        return len(stale)

    def stats(self) -> dict:
        """Return a snapshot of cache statistics."""
        with self._lock:
            size = len(self._lru)
        total = self._total_hits + self._total_misses
        hit_rate = self._total_hits / total if total > 0 else 0.0
        expired = self.expired_count()
        return {
            "size": size,
            "max_size": self._max_size,
            "threshold": self._threshold,
            "ttl_seconds": self._ttl_seconds,
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "hit_rate": round(hit_rate, 4),
            "expired_count": expired,
        }

    # ── Analytics buffer (Sprint 28) ─────────────────────────────────────────

    def set_analytics_buffer(self, buf: object) -> None:
        """Attach a :class:`~konjoai.cache.analytics.LatencyBuffer` to this cache.

        Once attached, call :meth:`record_access` from the request path to
        populate the buffer; retrieve analytics via :meth:`analytics_snapshot`.
        """
        self._analytics_buf = buf

    def record_access(self, latency_ms: float, is_hit: bool, similarity: float = 0.0) -> None:
        """Append one access event to the attached analytics buffer.

        No-op when no buffer has been attached via :meth:`set_analytics_buffer`.
        """
        if self._analytics_buf is not None:
            self._analytics_buf.record(latency_ms, is_hit, similarity)  # type: ignore[union-attr]

    def analytics_snapshot(self) -> list:
        """Return the current buffer contents, or an empty list when no buffer is attached."""
        if self._analytics_buf is None:
            return []
        return self._analytics_buf.snapshot()  # type: ignore[union-attr]

    # ── Batch similarity search (Sprint 28) ──────────────────────────────────

    def top_k_similar(self, q_vec: np.ndarray, k: int = 5) -> list[tuple[str, float, int]]:
        """Return up to *k* entries ranked by cosine similarity (no threshold gate).

        Returns a list of ``(normalised_key, similarity, hit_count)`` tuples
        sorted by similarity descending.  Expired entries are excluded.
        This is intentionally *below* the threshold — useful for search/exploration.
        """
        q_norm = self._l2_norm(q_vec)
        with self._lock:
            scored = [
                (k2, float(np.dot(q_norm, self._l2_norm(e.question_vec))), e.hit_count)
                for k2, e in self._lru.items()
                if not e.is_expired()
            ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    # ── Adaptive TTL (Sprint 28) ─────────────────────────────────────────────

    def adjust_ttls(
        self,
        hot_threshold_per_day: float = 5.0,
        cold_days: float = 3.0,
        extend_factor: float = 2.0,
        reduce_factor: float = 0.5,
        min_ttl: int = 60,
        max_ttl: int = 86400 * 7,
    ) -> dict[str, int]:
        """Adjust per-entry TTLs based on access frequency.

        Hot entries (access_rate_per_day > hot_threshold_per_day) have their
        TTL multiplied by *extend_factor*.  Cold entries (days_since_last_access
        > cold_days) have their TTL multiplied by *reduce_factor*.
        Entries with ``ttl_seconds == 0`` are skipped (no-TTL entries are not managed).

        Returns ``{"extended": n, "reduced": m}`` counts.
        """
        extended = reduced = 0
        with self._lock:
            for entry in self._lru.values():
                if entry.ttl_seconds == 0:
                    continue
                if entry.access_rate_per_day() > hot_threshold_per_day:
                    entry.ttl_seconds = min(max_ttl, max(min_ttl, int(entry.ttl_seconds * extend_factor)))
                    extended += 1
                elif entry.days_since_last_access() > cold_days:
                    entry.ttl_seconds = min(max_ttl, max(min_ttl, int(entry.ttl_seconds * reduce_factor)))
                    reduced += 1
        logger.info("adaptive ttl adjustment — extended=%d reduced=%d", extended, reduced)
        return {"extended": extended, "reduced": reduced}

    def ttl_report(self) -> dict:
        """Return a snapshot of current TTL distribution + adjustment candidates.

        Returns:
            * ``no_ttl``          — entries without a TTL (ttl_seconds == 0)
            * ``buckets``         — list of {label, min_s, max_s, count} TTL histogram
            * ``pending_extend``  — entries that would gain a longer TTL next adjust cycle
            * ``pending_reduce``  — entries that would lose TTL next adjust cycle
            * ``total``           — total non-expired entries
        """
        with self._lock:
            entries = [(e.ttl_seconds, e.access_rate_per_day(), e.days_since_last_access(), e.question)
                       for e in self._lru.values() if not e.is_expired()]

        no_ttl = sum(1 for t, *_ in entries if t == 0)
        ttl_entries = [(t, r, d, q) for t, r, d, q in entries if t > 0]

        # 5 log-scale buckets: <1 min, 1–60 min, 1–24 h, 1–7 d, >7 d
        buckets = [
            {"label": "<1 min",  "min_s": 0,       "max_s": 60,     "count": 0},
            {"label": "1–60 min","min_s": 60,      "max_s": 3600,   "count": 0},
            {"label": "1–24 h",  "min_s": 3600,    "max_s": 86400,  "count": 0},
            {"label": "1–7 d",   "min_s": 86400,   "max_s": 604800, "count": 0},
            {"label": ">7 d",    "min_s": 604800,  "max_s": None,   "count": 0},
        ]
        for t, *_ in ttl_entries:
            for b in buckets:
                if b["max_s"] is None or t < b["max_s"]:
                    b["count"] += 1
                    break

        pending_extend = [q for t, r, _d, q in ttl_entries if r > 5.0][:20]
        pending_reduce = [q for t, _r, d, q in ttl_entries if d > 3.0][:20]

        return {
            "total":          len(entries),
            "no_ttl":         no_ttl,
            "with_ttl":       len(ttl_entries),
            "buckets":        buckets,
            "pending_extend": pending_extend,
            "pending_reduce": pending_reduce,
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

_cache: object | None = None
_cache_lock = threading.Lock()


def get_semantic_cache() -> object | None:
    """Return the active semantic cache, or ``None`` when caching is disabled.

    Sprint 22 — backend selection:
        ``cache_backend == "memory"`` → in-process LRU :class:`SemanticCache`.
        ``cache_backend == "redis"``  → cross-pod :class:`RedisSemanticCache`,
        with **K3 graceful fallback** to the in-memory backend if the
        ``redis`` package is missing or the initial ``PING`` fails.
    """
    global _cache  # noqa: PLW0603
    from konjoai.config import get_settings  # local import avoids circular dep

    settings = get_settings()
    if not settings.cache_enabled:
        return None

    if _cache is not None:
        return _cache

    with _cache_lock:
        if _cache is not None:
            return _cache

        backend = (getattr(settings, "cache_backend", "memory") or "memory").lower()
        if backend == "redis":
            from konjoai.cache.redis_cache import build_redis_cache

            redis_cache = build_redis_cache(
                url=settings.cache_redis_url,
                namespace=settings.cache_redis_namespace,
                max_size=settings.cache_max_size,
                threshold=settings.cache_similarity_threshold,
                ttl_seconds=settings.cache_redis_ttl_seconds,
            )
            if redis_cache is not None:
                _cache = redis_cache
                logger.info(
                    "redis semantic cache initialised — namespace=%s max_size=%d threshold=%.2f ttl=%d",
                    settings.cache_redis_namespace,
                    settings.cache_max_size,
                    settings.cache_similarity_threshold,
                    settings.cache_redis_ttl_seconds,
                )
                return _cache
            logger.warning("redis backend unavailable — falling back to in-memory semantic cache")

        _cache = SemanticCache(
            max_size=settings.cache_max_size,
            threshold=settings.cache_similarity_threshold,
            ttl_seconds=getattr(settings, "cache_ttl_seconds", 0),
        )
        logger.info(
            "in-memory semantic cache initialised — max_size=%d threshold=%.2f",
            settings.cache_max_size,
            settings.cache_similarity_threshold,
        )
        return _cache


def _reset_cache() -> None:
    """Test helper: reset the module-level singleton. Never call in production."""
    global _cache  # noqa: PLW0603
    with _cache_lock:
        _cache = None
