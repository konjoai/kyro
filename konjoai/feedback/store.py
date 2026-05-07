"""In-process feedback store — Sprint 25.

Thread-safe bounded deque (ring buffer) with optional TTL pruning.
Same K3/K5 pattern as the audit log: stdlib only, zero new hard deps.

Usage::

    from konjoai.feedback.store import get_feedback_store
    from konjoai.feedback.models import FeedbackEvent, THUMBS_UP

    store = get_feedback_store()
    store.record(FeedbackEvent(
        question_hash="a1b2c3d4e5f6a7b8",
        signal=THUMBS_UP,
        timestamp="2026-05-07T12:00:00Z",
        tenant_id="acme-corp",
    ))
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Iterator

from konjoai.feedback.models import FeedbackEvent

logger = logging.getLogger(__name__)

# ── Singleton management ──────────────────────────────────────────────────────

_store_instance: FeedbackStore | None = None
_store_lock = threading.Lock()


def get_feedback_store(max_events: int = 1000) -> "FeedbackStore":
    """Return (or lazily create) the module-level ``FeedbackStore`` singleton."""
    global _store_instance
    with _store_lock:
        if _store_instance is None:
            _store_instance = FeedbackStore(max_events=max_events)
        return _store_instance


def _reset_singleton() -> None:
    """Discard the singleton — test helper only, never call from production code."""
    global _store_instance
    with _store_lock:
        _store_instance = None


# ── Store ─────────────────────────────────────────────────────────────────────


class FeedbackStore:
    """Thread-safe bounded ring buffer for ``FeedbackEvent`` records.

    When ``max_events`` is reached the oldest entry is evicted (LRU).

    Invariants:
    - K1: ``record()`` never raises — errors are logged as warnings.
    - K3: querying an empty or disabled store always succeeds (returns empty).
    - K5: pure stdlib — ``collections.deque``, ``threading.Lock``.
    """

    def __init__(self, max_events: int = 1000) -> None:
        if max_events < 1:
            raise ValueError(f"max_events must be >= 1, got {max_events}")
        self._max_events = max_events
        self._events: deque[FeedbackEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    # ── Write ─────────────────────────────────────────────────────────────────

    def record(self, event: FeedbackEvent) -> None:
        """Append a feedback event to the ring buffer.

        K1: any unexpected error is caught and logged as a warning so that the
        caller's request path is never affected.
        """
        try:
            with self._lock:
                self._events.append(event)
        except Exception as exc:  # pragma: no cover
            logger.warning("FeedbackStore.record failed: %s", exc)

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        *,
        limit: int = 100,
        tenant_id: str | None = None,
        signal: str | None = None,
        question_hash: str | None = None,
    ) -> list[FeedbackEvent]:
        """Return recent feedback events matching the optional filters.

        Events are returned newest-first.  ``limit`` is capped at 1000.
        """
        limit = min(limit, 1000)
        with self._lock:
            snapshot: list[FeedbackEvent] = list(self._events)

        # Apply filters
        if tenant_id is not None:
            snapshot = [e for e in snapshot if e.tenant_id == tenant_id]
        if signal is not None:
            snapshot = [e for e in snapshot if e.signal == signal]
        if question_hash is not None:
            snapshot = [e for e in snapshot if e.question_hash == question_hash]

        # Newest first
        snapshot.reverse()
        return snapshot[:limit]

    def summary(self, *, tenant_id: str | None = None) -> dict:
        """Return aggregate feedback statistics.

        Returns a dict with keys:
        - ``total``          — total events (after optional tenant filter)
        - ``thumbs_up``      — count of THUMBS_UP events
        - ``thumbs_down``    — count of THUMBS_DOWN events
        - ``avg_relevance``  — mean relevance_score across events that have one
        - ``by_question``    — dict mapping question_hash → {thumbs_up, thumbs_down}
        """
        from konjoai.feedback.models import THUMBS_DOWN, THUMBS_UP

        with self._lock:
            snapshot: list[FeedbackEvent] = list(self._events)

        if tenant_id is not None:
            snapshot = [e for e in snapshot if e.tenant_id == tenant_id]

        thumbs_up = sum(1 for e in snapshot if e.signal == THUMBS_UP)
        thumbs_down = sum(1 for e in snapshot if e.signal == THUMBS_DOWN)
        scores = [e.relevance_score for e in snapshot if e.relevance_score is not None]
        avg_relevance = sum(scores) / len(scores) if scores else None

        by_question: dict[str, dict[str, int]] = {}
        for ev in snapshot:
            q = ev.question_hash
            if q not in by_question:
                by_question[q] = {THUMBS_UP: 0, THUMBS_DOWN: 0}
            by_question[q][ev.signal] = by_question[q].get(ev.signal, 0) + 1

        result: dict = {
            "total": len(snapshot),
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
        }
        if avg_relevance is not None:
            result["avg_relevance"] = round(avg_relevance, 4)
        result["by_question"] = by_question
        return result

    # ── Capacity ──────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Current number of stored events."""
        with self._lock:
            return len(self._events)

    @property
    def max_events(self) -> int:
        """Maximum capacity (ring buffer ceiling)."""
        return self._max_events

    def clear(self) -> None:
        """Discard all stored events — mainly for tests."""
        with self._lock:
            self._events.clear()

    # ── Iterator ─────────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[FeedbackEvent]:
        with self._lock:
            return iter(list(self._events))

    def __len__(self) -> int:
        return self.size
