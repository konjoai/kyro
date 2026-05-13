"""Per-tenant cost attribution tracker (Sprint 26, P1).

Tracks how many queries each tenant makes, how many hit the cache, and
estimates the compute cost saved.  All operations are thread-safe.

Cost model (configurable via Settings)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cost_per_1k_tokens: float = 0.002  # USD per 1 000 tokens
    avg_response_tokens: int = 256     # tokens per LLM response

    tokens_saved_per_hit = avg_response_tokens
    cost_saved_per_hit   = avg_response_tokens / 1000 * cost_per_1k_tokens

The tracker is populated by the query route when the semantic cache is enabled.
It exposes ``GET /tenants/{tenant_id}/cost_report`` via the tenants router.

K3 — feature-flagged: the tracker is only active when ``cache_enabled=True``
in Settings.  When disabled, ``record()`` is a pure no-op.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

__all__ = [
    "TenantCostTracker",
    "TenantCostReport",
    "get_cost_tracker",
    "_reset_cost_tracker",
]

# Default cost model constants (overridden by Settings).
_DEFAULT_COST_PER_1K_TOKENS: float = 0.002
_DEFAULT_AVG_RESPONSE_TOKENS: int = 256


@dataclass
class TenantCostReport:
    """Snapshot of a single tenant's cost attribution."""

    tenant_id: str
    total_queries: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    tokens_saved: int
    estimated_cost_saved_usd: float
    cost_per_1k_tokens: float
    avg_response_tokens: int

    def as_dict(self) -> dict[str, object]:
        return {
            "tenant_id": self.tenant_id,
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(self.hit_rate, 4),
            "tokens_saved": self.tokens_saved,
            "estimated_cost_saved_usd": round(self.estimated_cost_saved_usd, 6),
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "avg_response_tokens": self.avg_response_tokens,
        }


@dataclass
class _TenantBucket:
    hits: int = 0
    misses: int = 0


class TenantCostTracker:
    """Thread-safe per-tenant cost attribution accumulator.

    Usage::

        tracker = TenantCostTracker()
        tracker.record(tenant_id="acme", hit=True)
        report = tracker.report("acme")
    """

    def __init__(
        self,
        *,
        cost_per_1k_tokens: float = _DEFAULT_COST_PER_1K_TOKENS,
        avg_response_tokens: int = _DEFAULT_AVG_RESPONSE_TOKENS,
    ) -> None:
        self._cost_per_1k = cost_per_1k_tokens
        self._avg_tokens = avg_response_tokens
        self._lock = threading.Lock()
        self._buckets: dict[str, _TenantBucket] = {}

    def record(self, tenant_id: str, *, hit: bool) -> None:
        """Record one query result for a tenant.  No-op for empty tenant IDs."""
        if not tenant_id:
            return
        with self._lock:
            bucket = self._buckets.setdefault(tenant_id, _TenantBucket())
            if hit:
                bucket.hits += 1
            else:
                bucket.misses += 1

    def report(self, tenant_id: str) -> TenantCostReport | None:
        """Return a cost report for the given tenant, or None if unknown."""
        with self._lock:
            bucket = self._buckets.get(tenant_id)
        if bucket is None:
            return None
        total = bucket.hits + bucket.misses
        tokens_saved = bucket.hits * self._avg_tokens
        cost_saved = tokens_saved / 1000.0 * self._cost_per_1k
        return TenantCostReport(
            tenant_id=tenant_id,
            total_queries=total,
            cache_hits=bucket.hits,
            cache_misses=bucket.misses,
            hit_rate=bucket.hits / total if total else 0.0,
            tokens_saved=tokens_saved,
            estimated_cost_saved_usd=cost_saved,
            cost_per_1k_tokens=self._cost_per_1k,
            avg_response_tokens=self._avg_tokens,
        )

    def all_tenants(self) -> list[TenantCostReport]:
        """Return cost reports for every tenant with at least one query."""
        with self._lock:
            tenant_ids = list(self._buckets.keys())
        return [r for tid in tenant_ids if (r := self.report(tid)) is not None]

    def reset(self) -> None:
        """Clear all accumulated data."""
        with self._lock:
            self._buckets.clear()


# ── Module-level singleton ─────────────────────────────────────────────────

_tracker: TenantCostTracker | None = None
_tracker_lock = threading.Lock()


def get_cost_tracker() -> TenantCostTracker:
    """Return the process-wide TenantCostTracker singleton.

    Reads ``cost_per_1k_tokens`` and ``avg_response_tokens`` from Settings on
    first call.  Subsequent calls return the cached instance.
    """
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                try:
                    from konjoai.config import get_settings  # noqa: PLC0415

                    s = get_settings()
                    cost = getattr(s, "cost_per_1k_tokens", _DEFAULT_COST_PER_1K_TOKENS)
                    avg  = getattr(s, "avg_response_tokens", _DEFAULT_AVG_RESPONSE_TOKENS)
                except Exception:  # noqa: BLE001 — settings unavailable in tests
                    cost = _DEFAULT_COST_PER_1K_TOKENS
                    avg  = _DEFAULT_AVG_RESPONSE_TOKENS
                _tracker = TenantCostTracker(cost_per_1k_tokens=cost, avg_response_tokens=avg)
    return _tracker


def _reset_cost_tracker() -> None:
    """Test helper — force a fresh singleton on next ``get_cost_tracker()`` call."""
    global _tracker
    with _tracker_lock:
        _tracker = None
