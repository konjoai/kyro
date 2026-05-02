"""Redis-backed semantic cache for Sprint 22.

Design
------
Cross-pod-shared semantic cache with tenant-namespaced keys, suitable for the
Helm HPA topology (2–10 replicas) introduced in Sprint 20.

Storage layout (per tenant)::

    <namespace>:<tenant>:entries   HASH   field=normalized_question  value=pickled blob
    <namespace>:<tenant>:lru       ZSET   member=normalized_question  score=monotonic ts

The blob format is a small dict ``{question, vec_bytes, response, created_at}``
serialised with :func:`pickle`. The cache is intra-cluster only — replicas run
the same code revision, so pickle is appropriate (and avoids the ad-hoc JSON
deserialisation logic that an out-of-band consumer would need).

Konjo invariants
----------------
- **K1** — every Redis call is wrapped in ``_safely``; failures are logged
  and the call returns ``None`` / no-op rather than crashing the request.
- **K3** — when ``cache_backend="redis"`` but the ``redis`` package is
  unavailable, :func:`build_redis_cache` returns ``None`` and callers fall
  back to the in-memory cache transparently.
- **K4** — ``q_vec`` is asserted ``float32`` at the ``store()`` boundary,
  identical to the in-memory cache contract.
- **K5** — ``redis`` is an *optional* dependency, lazily imported.
- **K6** — ``RedisSemanticCache`` implements the same ``lookup``/``store``/
  ``invalidate``/``stats`` contract as the in-memory ``SemanticCache``.
- **K7** — multi-tenancy: tenant prefix is read from the ``ContextVar`` set
  by :mod:`konjoai.auth.tenant`, so a tenant cannot read another tenant's
  cached responses even when the underlying Redis instance is shared.
"""
from __future__ import annotations

import logging
import pickle
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from konjoai.auth.tenant import ANONYMOUS_TENANT, get_current_tenant_id

logger = logging.getLogger(__name__)


@dataclass
class _RedisEntry:
    question: str
    vec_bytes: bytes      # raw float32 buffer, l2-normalised at store time
    response: object
    created_at: float


class RedisSemanticCache:
    """Distributed semantic cache backed by Redis.

    The instance is *stateless* aside from process-local hit/miss counters —
    every persistent operation routes through the injected Redis client. This
    keeps the implementation easy to test against an in-process fake (see
    :class:`tests.unit.test_redis_cache._FakeRedis`).

    Args:
        client: Anything that quacks like :class:`redis.Redis` with
            ``decode_responses=False``. The minimal surface used is
            ``hset``, ``hget``, ``hdel``, ``hgetall``, ``zadd``, ``zrange``,
            ``zrem``, ``zcard``, ``delete``, ``expire``, ``ping``.
        namespace: Top-level key prefix (default ``"kyro:cache"``).
        max_size: LRU eviction ceiling per tenant.
        threshold: Cosine similarity threshold for a semantic cache hit.
        ttl_seconds: When > 0, every entry hash is given a TTL so that an
            offline tenant cannot grow the cache forever.
        tenant_provider: Callable returning the active tenant_id, or ``None``
            for the anonymous bucket. Defaults to the Sprint 17 ContextVar.
    """

    def __init__(
        self,
        *,
        client: Any,
        namespace: str = "kyro:cache",
        max_size: int = 500,
        threshold: float = 0.95,
        ttl_seconds: int = 0,
        tenant_provider: Callable[[], str | None] | None = None,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        if ttl_seconds < 0:
            raise ValueError(f"ttl_seconds must be >= 0, got {ttl_seconds}")

        self._client = client
        self._namespace = namespace.rstrip(":")
        self._max_size = max_size
        self._threshold = threshold
        self._ttl_seconds = ttl_seconds
        self._tenant_provider = tenant_provider or get_current_tenant_id

        self._stats_lock = threading.Lock()
        self._total_hits = 0
        self._total_misses = 0

    # ── Key layout ────────────────────────────────────────────────────────────

    def _tenant(self) -> str:
        tid = self._tenant_provider()
        return tid if tid else ANONYMOUS_TENANT

    def _entries_key(self) -> str:
        return f"{self._namespace}:{self._tenant()}:entries"

    def _lru_key(self) -> str:
        return f"{self._namespace}:{self._tenant()}:lru"

    @staticmethod
    def _normalise(text: str) -> str:
        return text.strip().lower()

    @staticmethod
    def _l2_norm(vec: np.ndarray) -> np.ndarray:
        flat = vec.ravel().astype(np.float32)
        norm = float(np.linalg.norm(flat))
        if norm < 1e-10:
            return flat
        return flat / norm

    # ── Safe Redis call wrapper ───────────────────────────────────────────────

    def _safely(self, op: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — Redis transport, NOT silent
            logger.warning("redis cache %s failed: %s", op, exc)
            return None

    # ── Public API ────────────────────────────────────────────────────────────

    def lookup(self, question: str, q_vec: np.ndarray) -> object | None:
        """Return a cached response, or ``None`` on a miss / Redis error."""
        key = self._normalise(question)
        entries_key = self._entries_key()
        lru_key = self._lru_key()

        # 1. Exact-match fast path.
        raw = self._safely("hget", lambda: self._client.hget(entries_key, key))
        if raw is not None:
            entry = self._unpickle(raw)
            if entry is not None:
                self._safely("zadd", lambda: self._client.zadd(lru_key, {key: time.monotonic()}))
                with self._stats_lock:
                    self._total_hits += 1
                logger.debug("redis cache hit (exact) tenant=%s", self._tenant())
                return entry.response

        # 2. Semantic similarity scan.
        all_entries = self._safely("hgetall", lambda: self._client.hgetall(entries_key))
        if not all_entries:
            with self._stats_lock:
                self._total_misses += 1
            return None

        q_norm = self._l2_norm(q_vec)
        best_key: str | None = None
        best_sim: float = -1.0
        best_entry: _RedisEntry | None = None

        for raw_field, raw_blob in all_entries.items():
            field_str = raw_field.decode("utf-8") if isinstance(raw_field, bytes) else str(raw_field)
            entry = self._unpickle(raw_blob)
            if entry is None:
                continue
            entry_vec = np.frombuffer(entry.vec_bytes, dtype=np.float32)
            sim = float(np.dot(q_norm, entry_vec))
            if sim > best_sim:
                best_sim = sim
                best_key = field_str
                best_entry = entry

        if best_entry is not None and best_key is not None and best_sim >= self._threshold:
            self._safely("zadd", lambda: self._client.zadd(lru_key, {best_key: time.monotonic()}))
            with self._stats_lock:
                self._total_hits += 1
            logger.debug("redis cache hit (semantic) sim=%.4f tenant=%s", best_sim, self._tenant())
            return best_entry.response

        with self._stats_lock:
            self._total_misses += 1
        return None

    def store(self, question: str, q_vec: np.ndarray, response: object) -> None:
        """Persist a question/response pair into Redis. K4: float32 enforced."""
        assert q_vec.dtype == np.float32, (
            f"RedisSemanticCache.store: q_vec must be float32, got {q_vec.dtype}"
        )
        key = self._normalise(question)
        normalised_vec = self._l2_norm(q_vec)
        entry = _RedisEntry(
            question=question,
            vec_bytes=normalised_vec.tobytes(),
            response=response,
            created_at=time.monotonic(),
        )
        blob = pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL)
        entries_key = self._entries_key()
        lru_key = self._lru_key()

        self._safely("hset", lambda: self._client.hset(entries_key, key, blob))
        self._safely("zadd", lambda: self._client.zadd(lru_key, {key: time.monotonic()}))
        if self._ttl_seconds > 0:
            self._safely("expire-entries", lambda: self._client.expire(entries_key, self._ttl_seconds))
            self._safely("expire-lru", lambda: self._client.expire(lru_key, self._ttl_seconds))

        # LRU eviction.
        size = self._safely("zcard", lambda: self._client.zcard(lru_key)) or 0
        overflow = int(size) - self._max_size
        if overflow > 0:
            stale = self._safely("zrange-evict", lambda: self._client.zrange(lru_key, 0, overflow - 1)) or []
            for stale_key in stale:
                stale_str = stale_key.decode("utf-8") if isinstance(stale_key, bytes) else str(stale_key)
                self._safely("hdel", lambda k=stale_str: self._client.hdel(entries_key, k))
                self._safely("zrem", lambda k=stale_str: self._client.zrem(lru_key, k))
                logger.debug("redis cache evict tenant=%s key=%s", self._tenant(), stale_str[:40])

    def invalidate(self) -> None:
        """Drop the active tenant's cache entries (called on /ingest)."""
        entries_key = self._entries_key()
        lru_key = self._lru_key()
        self._safely("delete", lambda: self._client.delete(entries_key, lru_key))
        logger.info("redis cache invalidated for tenant=%s", self._tenant())

    def stats(self) -> dict:
        """Process-local hit/miss counters plus the live tenant size."""
        size_raw = self._safely("zcard-stats", lambda: self._client.zcard(self._lru_key()))
        size = int(size_raw) if size_raw is not None else 0
        with self._stats_lock:
            total = self._total_hits + self._total_misses
            hit_rate = self._total_hits / total if total > 0 else 0.0
            return {
                "size": size,
                "max_size": self._max_size,
                "threshold": self._threshold,
                "total_hits": self._total_hits,
                "total_misses": self._total_misses,
                "hit_rate": round(hit_rate, 4),
                "backend": "redis",
                "tenant": self._tenant(),
                "namespace": self._namespace,
            }

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _unpickle(raw: Any) -> _RedisEntry | None:
        if raw is None:
            return None
        try:
            obj = pickle.loads(raw)
        except (pickle.UnpicklingError, EOFError, ValueError, TypeError) as exc:
            logger.warning("redis cache unpickle failed: %s", exc)
            return None
        if not isinstance(obj, _RedisEntry):
            logger.warning("redis cache unpickle returned unexpected type: %r", type(obj))
            return None
        return obj


def build_redis_cache(
    *,
    url: str,
    namespace: str,
    max_size: int,
    threshold: float,
    ttl_seconds: int,
) -> RedisSemanticCache | None:
    """Construct a :class:`RedisSemanticCache` against a live Redis URL.

    Returns ``None`` (and logs a warning) when the ``redis`` package is not
    installed or the initial ``PING`` fails — callers must then fall back to
    the in-memory cache. This is the **K3 graceful-degradation** seam.
    """
    try:
        import redis  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("redis package not installed — falling back to in-memory cache")
        return None

    try:
        client = redis.Redis.from_url(url)
        client.ping()
    except Exception as exc:  # noqa: BLE001 — connection errors of any flavour
        logger.warning("redis connection to %s failed: %s — falling back to in-memory cache", url, exc)
        return None

    return RedisSemanticCache(
        client=client,
        namespace=namespace,
        max_size=max_size,
        threshold=threshold,
        ttl_seconds=ttl_seconds,
    )
