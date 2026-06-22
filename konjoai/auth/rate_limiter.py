"""Per-tenant, per-endpoint in-memory sliding-window rate limiter (Sprint 18).

Design: sliding-window log (deque of timestamps) keyed by (tenant_id, endpoint).
Thread-safe via threading.Lock (each bucket has its own lock). Pure Python —
no Redis, no external dependencies (K5).

Invariants:
- K3: when rate_limiting_enabled=False the limiter is a no-op.
- K5: zero new hard dependencies.
- K6: all new fields default to sensible values; no existing API breakage.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from konjoai.config import get_settings

__all__ = [
    "RateLimiter",
    "RateLimitExceeded",
    "get_rate_limiter",
]


class RateLimitExceeded(Exception):
    """Raised when a request exceeds the configured rate limit for a bucket."""

    def __init__(self, tenant_id: str, endpoint: str, limit: int, window: int) -> None:
        self.tenant_id = tenant_id
        self.endpoint = endpoint
        self.limit = limit
        self.window = window
        super().__init__(
            f"Rate limit exceeded for tenant='{tenant_id}' endpoint='{endpoint}': max {limit} requests per {window}s"
        )


@dataclass
class _Bucket:
    """Single sliding-window bucket (one tenant × one endpoint)."""

    window_seconds: int
    max_requests: int
    _timestamps: deque[float] = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def check_and_record(self, now: float | None = None) -> None:
        """Record a request at *now* (defaults to ``time.monotonic()``).

        Raises:
            RateLimitExceeded: if the bucket is exhausted.

        Note: tenant_id/endpoint are not stored here; the caller raises with
        context from the RateLimiter level.
        """
        if now is None:
            now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            # Evict expired entries from the left.
            while self._timestamps and self._timestamps[0] <= cutoff:
                self._timestamps.popleft()
            if len(self._timestamps) >= self.max_requests:
                raise _BucketExhausted()
            self._timestamps.append(now)

    @property
    def current_count(self) -> int:
        """Return the number of requests in the current window (thread-safe)."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            while self._timestamps and self._timestamps[0] <= cutoff:
                self._timestamps.popleft()
            return len(self._timestamps)


class _BucketExhausted(Exception):
    """Internal signal — bucket is full (no context attached)."""


class RateLimiter:
    """Sliding-window rate limiter keyed by (tenant_id, endpoint).

    Args:
        max_requests:   Maximum requests allowed per window per (tenant, endpoint).
        window_seconds: Length of the sliding window in seconds.
        enabled:        When False every call is a no-op (K3 degradation).
    """

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: int = 60,
        enabled: bool = True,
    ) -> None:
        if max_requests <= 0:
            raise ValueError(f"max_requests must be > 0, got {max_requests}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {window_seconds}")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.enabled = enabled
        # {(tenant_id, endpoint): _Bucket}
        self._buckets: dict[tuple[str, str], _Bucket] = {}
        self._registry_lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def check(
        self,
        tenant_id: str,
        endpoint: str,
        now: float | None = None,
    ) -> None:
        """Assert the (tenant_id, endpoint) bucket is not exhausted.

        Args:
            tenant_id: Opaque tenant identifier (use ANONYMOUS_TENANT if none).
            endpoint:  Logical endpoint label, e.g. "/query" or "/ingest".
            now:       Override the current time (for deterministic tests).

        Raises:
            RateLimitExceeded: if the bucket is full.
        """
        if not self.enabled:
            return
        bucket = self._get_or_create_bucket(tenant_id, endpoint)
        try:
            bucket.check_and_record(now)
        except _BucketExhausted:
            raise RateLimitExceeded(
                tenant_id=tenant_id,
                endpoint=endpoint,
                limit=self.max_requests,
                window=self.window_seconds,
            )

    def current_count(self, tenant_id: str, endpoint: str) -> int:
        """Return the number of active requests in the window (no side effect)."""
        key = (tenant_id, endpoint)
        with self._registry_lock:
            bucket = self._buckets.get(key)
        if bucket is None:
            return 0
        return bucket.current_count

    def reset(self, tenant_id: str | None = None, endpoint: str | None = None) -> None:
        """Clear buckets. Pass tenant_id and/or endpoint to narrow the reset.

        If both are None, all buckets are cleared (useful in tests).
        """
        with self._registry_lock:
            if tenant_id is None and endpoint is None:
                self._buckets.clear()
                return
            keys_to_delete = [
                k
                for k in self._buckets
                if (tenant_id is None or k[0] == tenant_id) and (endpoint is None or k[1] == endpoint)
            ]
            for k in keys_to_delete:
                del self._buckets[k]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_or_create_bucket(self, tenant_id: str, endpoint: str) -> _Bucket:
        """Return the bucket for ``(tenant_id, endpoint)``, creating it if absent."""
        key = (tenant_id, endpoint)
        with self._registry_lock:
            if key not in self._buckets:
                self._buckets[key] = _Bucket(
                    window_seconds=self.window_seconds,
                    max_requests=self.max_requests,
                )
            return self._buckets[key]


# ── Module-level singleton ────────────────────────────────────────────────────

_limiter_instance: RateLimiter | None = None
_limiter_lock = threading.Lock()


def get_rate_limiter() -> RateLimiter:
    """Return the module-level RateLimiter singleton (lazy, reads settings).

    The singleton is re-created whenever it has not yet been initialised. Call
    ``_reset_singleton()`` in tests to force re-creation.
    """
    global _limiter_instance
    with _limiter_lock:
        if _limiter_instance is None:
            s = get_settings()
            _limiter_instance = RateLimiter(
                max_requests=getattr(s, "rate_limit_requests", 60),
                window_seconds=getattr(s, "rate_limit_window_seconds", 60),
                enabled=getattr(s, "rate_limiting_enabled", False),
            )
        return _limiter_instance


def _reset_singleton() -> None:
    """Force re-creation of the singleton on next call (test helper)."""
    global _limiter_instance
    with _limiter_lock:
        _limiter_instance = None
