"""Semantic cache poisoning guard — Sprint 28.

Three-layer defence against cache write attacks:
  1. Per-tenant write rate limit — sliding window, pure stdlib (K5).
  2. Q-A coherence check — cosine similarity between question and answer embeddings
     must exceed ``min_qa_coherence``.  Only active when embed_fn is wired in
     (``cache_poisoning_check_coherence=True``).
  3. Response length anomaly detection — flags outliers beyond ``length_sigma``
     std-devs from a running Welford mean.  Informational: records but does not block.

K1: coherence embed failures are caught; they never block the store.
K3: guard is only instantiated when ``cache_poisoning_guard_enabled=True``.
K5: stdlib only — ``collections.deque``, ``threading.Lock``, ``hashlib``, ``time``.
K7: rate-limit counters are per-tenant; reports carry tenant_id.
OWASP: question text is never stored — only its 16-hex SHA-256 prefix.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "AnomalyDetector",
    "PoisoningGuard",
    "PoisoningReport",
    "PoisoningReportStore",
    "WriteRateLimiter",
    "get_poisoning_guard",
    "get_poisoning_report_store",
    "_reset_singletons",
]


# ── Rate limiter ───────────────────────────────────────────────────────────────


class WriteRateLimiter:
    """Per-tenant sliding-window rate limiter for cache write calls.

    Args:
        max_writes: Maximum writes allowed within ``window_seconds``.
        window_seconds: Length of the sliding window.
    """

    def __init__(self, max_writes: int = 100, window_seconds: float = 60.0) -> None:
        if max_writes < 1:
            raise ValueError(f"max_writes must be >= 1, got {max_writes}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {window_seconds}")
        self._max = max_writes
        self._window = window_seconds
        self._timestamps: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, tenant_id: str) -> bool:
        """Return True and record the attempt, or False if the tenant is over-limit."""
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            q = self._timestamps.setdefault(tenant_id, deque())
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self._max:
                return False
            q.append(now)
            return True

    def current_count(self, tenant_id: str) -> int:
        """Return the number of writes recorded for this tenant within the current window."""
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            q = self._timestamps.get(tenant_id, deque())
            return sum(1 for t in q if t >= cutoff)


# ── Cosine similarity ──────────────────────────────────────────────────────────


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two float32 arrays (any shape — flattened internally)."""
    a_flat = a.ravel().astype(np.float32)
    b_flat = b.ravel().astype(np.float32)
    a_norm = float(np.linalg.norm(a_flat))
    b_norm = float(np.linalg.norm(b_flat))
    if a_norm < 1e-10 or b_norm < 1e-10:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (a_norm * b_norm))


# ── Anomaly detector ───────────────────────────────────────────────────────────


class AnomalyDetector:
    """Running length statistics for cache responses using Welford's online algorithm.

    Flags responses whose length deviates more than ``sigma_threshold`` std-devs
    from the running mean.  Requires at least ``min_observations`` samples before
    any outlier is reported.

    Args:
        sigma_threshold: Number of std-devs beyond which a response is an outlier.
        min_observations: Minimum recorded samples before flagging begins.
    """

    def __init__(self, sigma_threshold: float = 3.0, min_observations: int = 10) -> None:
        if sigma_threshold <= 0:
            raise ValueError(f"sigma_threshold must be > 0, got {sigma_threshold}")
        self._sigma = sigma_threshold
        self._min_obs = min_observations
        self._n: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0
        self._lock = threading.Lock()

    def record(self, response_text: str) -> None:
        """Update running statistics with a new response."""
        length = len(response_text)
        with self._lock:
            self._n += 1
            delta = length - self._mean
            self._mean += delta / self._n
            self._m2 += delta * (length - self._mean)

    def is_length_outlier(self, response_text: str) -> bool:
        """Return True if this response length is a statistical outlier."""
        with self._lock:
            n, mean, m2 = self._n, self._mean, self._m2
        if n < self._min_obs:
            return False
        std = (m2 / n) ** 0.5
        if std < 1.0:
            return False
        return abs(len(response_text) - mean) > self._sigma * std

    @property
    def n_observations(self) -> int:
        """Number of responses recorded so far."""
        with self._lock:
            return self._n


# ── Poisoning report store ─────────────────────────────────────────────────────


@dataclass
class PoisoningReport:
    """An immutable record of a detected or user-reported cache poisoning event."""

    tenant_id: str
    question_hash: str    # 16-hex SHA-256 prefix — no raw text (OWASP)
    reason: str           # e.g. "rate_limit_exceeded", "low_coherence:0.12"
    timestamp: float = field(default_factory=time.time)


class PoisoningReportStore:
    """Thread-safe bounded ring buffer for poisoning event reports.

    Args:
        max_reports: Maximum events retained; oldest are evicted automatically.
    """

    def __init__(self, max_reports: int = 500) -> None:
        self._reports: deque[PoisoningReport] = deque(maxlen=max_reports)
        self._lock = threading.Lock()

    def record(self, tenant_id: str, question_hash: str, reason: str) -> None:
        """Append a poisoning report. Never raises (K1)."""
        try:
            with self._lock:
                self._reports.append(
                    PoisoningReport(
                        tenant_id=tenant_id,
                        question_hash=question_hash,
                        reason=reason,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("poisoning report store: failed to record: %s", exc)

    def query(
        self,
        *,
        tenant_id: str | None = None,
        limit: int = 100,
    ) -> list[PoisoningReport]:
        """Return the most recent reports, optionally filtered by tenant."""
        with self._lock:
            reports: list[PoisoningReport] = list(self._reports)
        if tenant_id is not None:
            reports = [r for r in reports if r.tenant_id == tenant_id]
        return reports[-limit:]

    def count(self, *, tenant_id: str | None = None) -> int:
        """Return the total number of reports, optionally scoped to a tenant."""
        with self._lock:
            reports = list(self._reports)
        if tenant_id is not None:
            return sum(1 for r in reports if r.tenant_id == tenant_id)
        return len(reports)

    def clear(self) -> None:
        """Remove all stored reports. Intended for tests."""
        with self._lock:
            self._reports.clear()


# ── Poisoning guard ────────────────────────────────────────────────────────────


class PoisoningGuard:
    """Pre-store gate combining rate-limiting, coherence checking, and anomaly detection.

    Args:
        min_qa_coherence: Cosine similarity floor for Q-A coherence (layer 2).
        max_writes_per_minute: Per-tenant write ceiling in a 60-second window (layer 1).
        length_sigma: Std-dev threshold for length anomaly flagging (layer 3).
        min_anomaly_observations: Minimum samples before anomaly detection activates.
        embed_fn: Optional ``(text) -> ndarray`` callable. When provided, layer 2
            is active. When absent, layer 2 is skipped.
        report_store: Injected store; defaults to the module-level singleton.
    """

    def __init__(
        self,
        min_qa_coherence: float = 0.3,
        max_writes_per_minute: int = 100,
        length_sigma: float = 3.0,
        min_anomaly_observations: int = 10,
        embed_fn: Callable[[str], np.ndarray] | None = None,
        report_store: PoisoningReportStore | None = None,
    ) -> None:
        self._min_coherence = min_qa_coherence
        self._rate_limiter = WriteRateLimiter(max_writes=max_writes_per_minute)
        self._anomaly = AnomalyDetector(
            sigma_threshold=length_sigma,
            min_observations=min_anomaly_observations,
        )
        self._embed_fn = embed_fn
        self._report_store = report_store or get_poisoning_report_store()

    def validate(
        self,
        question: str,
        q_vec: np.ndarray,
        answer_text: str,
        tenant_id: str,
    ) -> bool:
        """Return True to proceed with the cache store; False to block it.

        Evaluation order:
          Layer 1 (rate limit)  — False blocks + records the event.
          Layer 2 (coherence)   — False blocks + records the event (when embed_fn set).
          Layer 3 (anomaly)     — records only; never blocks.
        """
        q_hash = hashlib.sha256(question.encode()).hexdigest()[:16]

        # Layer 1 — rate limit
        if not self._rate_limiter.is_allowed(tenant_id):
            logger.warning(
                "cache poisoning guard: rate limit exceeded tenant=%s q_hash=%s",
                tenant_id,
                q_hash,
            )
            self._report_store.record(tenant_id, q_hash, "rate_limit_exceeded")
            return False

        # Layer 2 — Q-A coherence (only when an embedding function is wired in)
        if self._embed_fn is not None:
            try:
                a_vec = np.asarray(self._embed_fn(answer_text[:2048]), dtype=np.float32)
                score = _cosine_similarity(q_vec, a_vec)
                if score < self._min_coherence:
                    logger.warning(
                        "cache poisoning guard: low coherence score=%.3f threshold=%.3f q_hash=%s",
                        score,
                        self._min_coherence,
                        q_hash,
                    )
                    self._report_store.record(
                        tenant_id, q_hash, f"low_coherence:{score:.3f}"
                    )
                    return False
            except Exception as exc:  # noqa: BLE001
                # K1: embed failure must never prevent the store
                logger.warning(
                    "cache poisoning guard: coherence check failed (store allowed): %s", exc
                )

        # Layer 3 — anomaly detection (informational; never blocks)
        if self._anomaly.is_length_outlier(answer_text):
            logger.warning(
                "cache poisoning guard: length anomaly detected len=%d q_hash=%s",
                len(answer_text),
                q_hash,
            )
            self._report_store.record(
                tenant_id, q_hash, f"length_anomaly:{len(answer_text)}"
            )
        self._anomaly.record(answer_text)

        return True


# ── Singletons ─────────────────────────────────────────────────────────────────


_report_store: PoisoningReportStore | None = None
_report_store_lock = threading.Lock()

_guard: PoisoningGuard | None = None
_guard_lock = threading.Lock()


def get_poisoning_report_store() -> PoisoningReportStore:
    """Return the module-level PoisoningReportStore singleton."""
    global _report_store  # noqa: PLW0603
    if _report_store is not None:
        return _report_store
    with _report_store_lock:
        if _report_store is None:
            from konjoai.config import get_settings  # noqa: PLC0415

            settings = get_settings()
            _report_store = PoisoningReportStore(
                max_reports=getattr(settings, "cache_poisoning_max_reports", 500),
            )
    return _report_store  # type: ignore[return-value]


def _make_embed_fn(
    settings: object,
) -> Callable[[str], np.ndarray] | None:
    """Build the coherence embed_fn when cache_poisoning_check_coherence is True."""
    if not getattr(settings, "cache_poisoning_check_coherence", False):
        return None
    try:
        from konjoai.embed.encoder import get_encoder  # noqa: PLC0415

        encoder = get_encoder()

        def _encode(text: str) -> np.ndarray:
            return encoder.encode_query(text)  # type: ignore[no-any-return]

        return _encode
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "poisoning guard: encoder unavailable — coherence check disabled: %s", exc
        )
        return None


def get_poisoning_guard() -> PoisoningGuard:
    """Return the module-level PoisoningGuard singleton."""
    global _guard  # noqa: PLW0603
    if _guard is not None:
        return _guard
    with _guard_lock:
        if _guard is None:
            from konjoai.config import get_settings  # noqa: PLC0415

            settings = get_settings()
            _guard = PoisoningGuard(
                min_qa_coherence=getattr(settings, "cache_poisoning_min_coherence", 0.3),
                max_writes_per_minute=getattr(
                    settings, "cache_poisoning_max_writes_per_minute", 100
                ),
                length_sigma=getattr(settings, "cache_poisoning_length_sigma", 3.0),
                embed_fn=_make_embed_fn(settings),
                report_store=get_poisoning_report_store(),
            )
    return _guard  # type: ignore[return-value]


def _reset_singletons() -> None:
    """Reset module-level singletons. Test helper — never call in production."""
    global _report_store, _guard  # noqa: PLW0603
    _report_store = None
    _guard = None
