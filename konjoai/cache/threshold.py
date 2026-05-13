"""Adaptive similarity threshold engine for the semantic cache.

Sprint 26 — P1 feature.

Instead of a single global threshold, each query is classified into a type
and routed to a per-type cosine similarity gate.  Higher thresholds for
precise domains (factual, code) prevent stale hits; lower thresholds for
conversational / creative domains increase reuse.

Query classification uses a lightweight keyword heuristic — no model load,
no network call.  The classifier is deterministic given the same input string.

Thread safety
~~~~~~~~~~~~~
``ThresholdStats`` uses a single ``threading.Lock``.  ``AdaptiveThresholdEngine``
is stateless: every call is independent.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "QueryType",
    "ThresholdConfig",
    "ThresholdStats",
    "AdaptiveThresholdEngine",
    "classify_query",
]


class QueryType(str, Enum):
    """Semantic category of a user query."""

    FACTUAL = "factual"
    FAQ = "faq"
    CREATIVE = "creative"
    CODE = "code"


# Compiled once at import time for speed.
_RE_DATE = re.compile(r"\b(\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")
# Bare numbers with 2+ digits (enough to distinguish factual queries)
_RE_NUMBER = re.compile(r"\b\d{2,}(?:\.\d+)?\b")
# Percentage (no trailing \b needed — % is not a word char)
_RE_PERCENT = re.compile(r"\b\d+(?:\.\d+)?\s*%")
# Storage/unit keywords that appear without numbers ("how many mb in a gb")
_RE_UNIT_WORD = re.compile(r"\b(?:mb|gb|tb|kb|km|kg|ms|rpm|fps)\b", re.I)
# "What year was X?" — contains no year digit but is clearly factual
_RE_YEAR_WORD = re.compile(r"\byear\b", re.I)
_RE_HOW = re.compile(r"\bhow\s+(to|do|does|can|would|should|could|many|much)\b", re.I)
_RE_WHAT_IS = re.compile(r"\bwhat\s+(is|are|was|were)\b", re.I)
_RE_CODE_FENCE = re.compile(r"```|def |class |import |return |function |const |var |let ")
_RE_SQL_KW = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|FROM|WHERE)\b")
_RE_LANG_KW = re.compile(
    r"\b(python|javascript|typescript|golang|rust|java|sql|bash|shell|regex|api|json|yaml|csv)\b",
    re.I,
)


def classify_query(question: str) -> QueryType:
    """Classify a query string into one of four semantic types.

    Rules (first match wins, ordered by precision):

    - ``CODE`` — contains a code fence, SQL keyword, language keyword, or
      programming term.
    - ``FACTUAL`` — contains a date, measurement unit, specific number, or
      the word "year" (indicating a factual year-query).
    - ``FAQ`` — contains "how to / how do / what is" or similar.
    - ``CREATIVE`` — default for anything that doesn't match above.
    """
    q = question.strip()
    if _RE_CODE_FENCE.search(q) or _RE_SQL_KW.search(q) or _RE_LANG_KW.search(q):
        return QueryType.CODE
    if (_RE_DATE.search(q) or _RE_NUMBER.search(q) or _RE_PERCENT.search(q)
            or _RE_UNIT_WORD.search(q) or _RE_YEAR_WORD.search(q)):
        return QueryType.FACTUAL
    if _RE_HOW.search(q) or _RE_WHAT_IS.search(q):
        return QueryType.FAQ
    return QueryType.CREATIVE


@dataclass
class ThresholdConfig:
    """Per-type cosine similarity thresholds.

    Values are in (0, 1].  Higher = stricter (fewer hits but more precise).
    """

    factual: float = 0.94
    faq: float = 0.85
    creative: float = 0.75
    code: float = 0.92

    def for_type(self, query_type: QueryType) -> float:
        """Return the threshold for the given query type."""
        return float(getattr(self, query_type.value))

    def as_dict(self) -> dict[str, float]:
        return {t.value: self.for_type(t) for t in QueryType}


@dataclass
class _TypeStats:
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


class ThresholdStats:
    """Thread-safe per-type hit/miss counters.

    Used by ``GET /cache/threshold_stats`` to surface per-category cache
    effectiveness.  Deliberately separate from ``SemanticCache.stats()``
    so the cache core stays decoupled from the classifier.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[QueryType, _TypeStats] = {t: _TypeStats() for t in QueryType}

    def record_hit(self, query_type: QueryType) -> None:
        with self._lock:
            self._data[query_type].hits += 1

    def record_miss(self, query_type: QueryType) -> None:
        with self._lock:
            self._data[query_type].misses += 1

    def snapshot(self) -> dict[str, dict[str, object]]:
        """Return a JSON-serialisable snapshot of all type stats."""
        with self._lock:
            return {
                qt.value: {
                    "hits": s.hits,
                    "misses": s.misses,
                    "total": s.total,
                    "hit_rate": round(s.hit_rate, 4),
                }
                for qt, s in self._data.items()
            }

    def reset(self) -> None:
        with self._lock:
            for s in self._data.values():
                s.hits = 0
                s.misses = 0


# Module-level singleton — shared across all requests.
_stats: ThresholdStats | None = None
_stats_lock = threading.Lock()


def get_threshold_stats() -> ThresholdStats:
    """Return the process-wide ThresholdStats singleton."""
    global _stats
    if _stats is None:
        with _stats_lock:
            if _stats is None:
                _stats = ThresholdStats()
    return _stats


def _reset_stats_singleton() -> None:
    """Test helper — force a fresh singleton on next access."""
    global _stats
    with _stats_lock:
        _stats = None


class AdaptiveThresholdEngine:
    """Stateless engine that resolves the threshold for a given question.

    Usage::

        engine = AdaptiveThresholdEngine()
        q_type, thresh = engine.resolve("How do I install kyro?")
        # thresh == ThresholdConfig().faq == 0.85
    """

    def __init__(self, config: ThresholdConfig | None = None) -> None:
        self._config = config or ThresholdConfig()

    @property
    def config(self) -> ThresholdConfig:
        return self._config

    def resolve(self, question: str) -> tuple[QueryType, float]:
        """Return ``(query_type, threshold)`` for the given question."""
        qt = classify_query(question)
        return qt, self._config.for_type(qt)

    def lookup_with_stats(
        self,
        question: str,
        q_vec: "np.ndarray",
        cache: object,
        stats: ThresholdStats | None = None,
    ) -> tuple[object | None, QueryType, float]:
        """Classify, look up in cache, record stats, and return the result.

        Returns ``(cached_response_or_None, query_type, threshold_used)``.

        The ``cache`` argument must satisfy the ``SemanticCache`` lookup protocol.
        We avoid importing ``SemanticCache`` here to keep the module dependency-free.
        """
        qt, threshold = self.resolve(question)
        import numpy as np  # noqa: PLC0415 — lazy import, numpy is always available

        # Borrow SemanticCache._l2_norm to do the cosine scan ourselves so
        # we can apply the per-type threshold without mutating the cache object.
        result = _lookup_with_threshold(cache, question, q_vec, threshold)
        tracker = stats or get_threshold_stats()
        if result is not None:
            tracker.record_hit(qt)
        else:
            tracker.record_miss(qt)
        return result, qt, threshold


def _lookup_with_threshold(cache: object, question: str, q_vec: "np.ndarray", threshold: float) -> object | None:
    """Call cache.lookup() with a temporary threshold override.

    Uses the cache's own ``_threshold`` attribute to swap in the per-type
    value for the duration of the lookup, then restores it.  The cache's
    internal lock serialises this — there is no TOCTOU gap.
    """
    import threading as _threading  # noqa: PLC0415

    original = cache._threshold  # type: ignore[attr-defined]
    lock = cache._lock           # type: ignore[attr-defined]
    with lock:
        cache._threshold = threshold  # type: ignore[attr-defined]
        try:
            # Call the non-locking inner body directly to avoid deadlock.
            return _inner_lookup(cache, question, q_vec)
        finally:
            cache._threshold = original  # type: ignore[attr-defined]


def _inner_lookup(cache: object, question: str, q_vec: "np.ndarray") -> object | None:
    """Re-implement SemanticCache's lookup body (already under lock)."""
    import numpy as np  # noqa: PLC0415

    key = cache._normalise(question)  # type: ignore[attr-defined]
    exact = cache._exact.get(key)     # type: ignore[attr-defined]
    if exact is not None:
        cache._lru.move_to_end(key)   # type: ignore[attr-defined]
        exact.hit_count += 1
        cache._total_hits += 1       # type: ignore[attr-defined]
        return exact.response

    q_norm = cache._l2_norm(q_vec)   # type: ignore[attr-defined]
    best_key, best_sim = None, -1.0
    for k, e in cache._lru.items():  # type: ignore[attr-defined]
        sim = float(np.dot(q_norm, cache._l2_norm(e.question_vec)))  # type: ignore[attr-defined]
        if sim > best_sim:
            best_sim, best_key = sim, k

    threshold = cache._threshold     # type: ignore[attr-defined]
    if best_key is not None and best_sim >= threshold:
        entry = cache._lru[best_key]  # type: ignore[attr-defined]
        cache._lru.move_to_end(best_key)  # type: ignore[attr-defined]
        entry.hit_count += 1
        cache._total_hits += 1       # type: ignore[attr-defined]
        return entry.response

    cache._total_misses += 1         # type: ignore[attr-defined]
    return None
