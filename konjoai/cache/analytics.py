"""Lightweight in-process analytics buffer for the semantic cache (Sprint 28).

Design goals
------------
* Zero new hard dependencies — pure stdlib + numpy (already required).
* Thread-safe: all mutation goes through ``threading.Lock``.
* Bounded memory: ring buffer capped at ``MAX_RECORDS`` (default 10 000).
* Fast path: ``record()`` is O(1); analytics queries are O(n) but only
  called on-demand by the API route, never on the hot cache path.

The buffer is a companion to ``SemanticCache`` — attach one via
``SemanticCache.set_analytics_buffer(buf)``, then call
``cache.record_access(latency_ms, is_hit, similarity)`` from the query
route after each lookup.  Routes that want analytics snapshots call
``cache.analytics_snapshot()``.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class AccessRecord:
    """One cache access event — immutable once created."""

    timestamp: float  # time.monotonic() — not wall-clock; use for intervals only
    latency_ms: float
    is_hit: bool
    similarity: float  # cosine similarity for hits; 0.0 for misses


class LatencyBuffer:
    """Thread-safe ring buffer of recent cache access records.

    Add records with :meth:`record`; read a snapshot with :meth:`snapshot`.
    The buffer never grows beyond ``max_records`` entries.
    """

    def __init__(self, max_records: int = 10_000) -> None:
        self._buf: deque[AccessRecord] = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def record(self, latency_ms: float, is_hit: bool, similarity: float = 0.0) -> None:
        """Append one access record. O(1), never blocks."""
        with self._lock:
            self._buf.append(
                AccessRecord(
                    timestamp=time.monotonic(),
                    latency_ms=latency_ms,
                    is_hit=is_hit,
                    similarity=similarity,
                )
            )

    def snapshot(self) -> list[AccessRecord]:
        """Return a copy of the current buffer contents (oldest first)."""
        with self._lock:
            return list(self._buf)

    def clear(self) -> None:
        """Discard all recorded accesses."""
        with self._lock:
            self._buf.clear()

    @property
    def size(self) -> int:
        """Current number of records in the buffer."""
        with self._lock:
            return len(self._buf)


# ── Analytics computation (pure functions, no external I/O) ────────────────


def _percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile of a pre-sorted list. Returns 0.0 on empty input."""
    if not values:
        return 0.0
    idx = max(0, math.ceil(p / 100.0 * len(values)) - 1)
    return values[idx]


def compute_analytics(records: list[AccessRecord], hours: float = 24.0) -> dict:
    """Derive rich stats from a list of ``AccessRecord`` values.

    Args:
        records: Snapshot from :class:`LatencyBuffer`.
        hours:   Only include records from the last ``hours`` hours.

    Returns a dict with the following keys:

    * ``window_hours``          — the requested window
    * ``total_accesses``        — count of records in the window
    * ``hit_count`` / ``miss_count``
    * ``hit_rate``              — float in [0, 1]
    * ``latency``               — nested dict with ``all``, ``hits``, ``misses``
      each containing ``p50``, ``p90``, ``p99``, ``mean``, ``min``, ``max``
    * ``similarity_distribution`` — 5-bucket histogram of hit similarity scores
    * ``hourly_hit_rate``        — list of {hour_offset, hit_rate, count}
      for the 24 one-hour buckets inside the window (most recent last)
    """
    cutoff = time.monotonic() - hours * 3600.0
    window = [r for r in records if r.timestamp >= cutoff]
    if not window:
        return _empty_analytics(hours)

    hits = [r for r in window if r.is_hit]
    misses = [r for r in window if not r.is_hit]
    total = len(window)
    hit_rate = len(hits) / total if total else 0.0

    all_lat = sorted(r.latency_ms for r in window)
    hit_lat = sorted(r.latency_ms for r in hits)
    miss_lat = sorted(r.latency_ms for r in misses)

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
        return {
            "p50": round(_percentile(vals, 50), 3),
            "p90": round(_percentile(vals, 90), 3),
            "p99": round(_percentile(vals, 99), 3),
            "mean": round(sum(vals) / len(vals), 3),
            "min": round(vals[0], 3),
            "max": round(vals[-1], 3),
        }

    return {
        "window_hours": hours,
        "total_accesses": total,
        "hit_count": len(hits),
        "miss_count": len(misses),
        "hit_rate": round(hit_rate, 4),
        "latency": {"all": _stats(all_lat), "hits": _stats(hit_lat), "misses": _stats(miss_lat)},
        "similarity_distribution": _similarity_distribution(hits),
        "hourly_hit_rate": _hourly_hit_rate(window, hours),
    }


def _similarity_distribution(hits: list[AccessRecord]) -> list[dict]:
    """Return a 5-bucket histogram of hit similarity scores over [0, 1]."""
    sim_histo = [0, 0, 0, 0, 0]
    for r in hits:
        sim_histo[min(4, int(r.similarity * 5))] += 1
    return [{"range": f"{i * 0.2:.1f}–{(i + 1) * 0.2:.1f}", "count": sim_histo[i]} for i in range(5)]


def _hourly_hit_rate(window: list[AccessRecord], hours: float) -> list[dict]:
    """Bucket records into one-hour offsets and return per-bucket hit rates."""
    now = time.monotonic()
    n_buckets = max(1, int(math.ceil(hours)))
    bucket_hits = [0] * n_buckets
    bucket_totals = [0] * n_buckets
    for r in window:
        idx = min(n_buckets - 1, int((now - r.timestamp) / 3600.0))
        reverse_idx = n_buckets - 1 - idx  # 0 = most recent hour
        bucket_totals[reverse_idx] += 1
        if r.is_hit:
            bucket_hits[reverse_idx] += 1
    return [
        {
            "hour_offset": i - (n_buckets - 1),  # negative = past
            "count": bucket_totals[i],
            "hit_rate": round(bucket_hits[i] / bucket_totals[i], 4) if bucket_totals[i] else 0.0,
        }
        for i in range(n_buckets)
    ]


def _empty_analytics(hours: float) -> dict:
    empty_stats = {"p50": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "window_hours": hours,
        "total_accesses": 0,
        "hit_count": 0,
        "miss_count": 0,
        "hit_rate": 0.0,
        "latency": {"all": empty_stats, "hits": empty_stats, "misses": empty_stats},
        "similarity_distribution": [{"range": f"{i * 0.2:.1f}–{(i + 1) * 0.2:.1f}", "count": 0} for i in range(5)],
        "hourly_hit_rate": [],
    }
