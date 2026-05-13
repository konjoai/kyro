"""OpenTelemetry span emission for cache operations (Sprint 26, P1).

Every cache lookup/store emits a span named ``cache.<operation>`` with the
following attributes when OTel is available and enabled:

    cache.lookup   — fired on every lookup attempt
    cache.hit      — fired when the lookup returns a cached value
    cache.miss     — fired when the lookup returns None
    cache.store    — fired on every store call

Span attributes
~~~~~~~~~~~~~~~
    kyro.tenant_id          str   — active tenant or "__anonymous__"
    kyro.similarity_score   float — best cosine score found (-1 when empty)
    kyro.threshold_used     float — threshold applied for this lookup
    kyro.latency_ms         float — wall-clock time of the cache operation
    kyro.query_type         str   — query type from the adaptive classifier
    kyro.tokens_saved       int   — 0 on miss; estimated tokens saved on hit

K3 (graceful degradation)
~~~~~~~~~~~~~~~~~~~~~~~~~
When the ``opentelemetry-sdk`` package is absent, or when ``otel_enabled``
is False in Settings, every function in this module is a no-op.  The caller
never branches on OTel availability.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager

import numpy as np

from konjoai.auth.tenant import get_current_tenant_id

logger = logging.getLogger(__name__)

__all__ = ["cache_span", "emit_cache_lookup"]

# ── Optional OTel import ────────────────────────────────────────────────────

try:
    from opentelemetry import trace as _otel_trace  # type: ignore[import-untyped]
    from opentelemetry.trace import NonRecordingSpan  # type: ignore[import-untyped]
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False

# Estimated average tokens per cached response (used for tokens_saved).
# Operators can override via the ``cache_avg_response_tokens`` setting.
_DEFAULT_AVG_TOKENS = 256


@contextmanager
def cache_span(
    operation: str,
    *,
    tracer_name: str = "kyro.cache",
    enabled: bool = True,
) -> Generator[object, None, None]:
    """Context manager that wraps a block in an OTel span named ``cache.<operation>``.

    Yields the span object (or a sentinel no-op object when OTel is absent).
    Callers can set attributes on the yielded object via ``span.set_attribute()``.

    Example::

        with cache_span("lookup") as span:
            result = cache.lookup(question, q_vec)
            span.set_attribute("kyro.result", "hit" if result else "miss")
    """
    if not _HAS_OTEL or not enabled:
        yield _NoopSpan()
        return

    from konjoai.telemetry import get_tracer  # noqa: PLC0415 — avoid circular at module load

    tracer_wrapper = get_tracer()
    if not tracer_wrapper.available:
        yield _NoopSpan()
        return

    # tracer_wrapper._tracer is the raw OTel tracer.
    raw_tracer = getattr(tracer_wrapper, "_tracer", None)
    if raw_tracer is None:
        yield _NoopSpan()
        return

    with raw_tracer.start_as_current_span(f"cache.{operation}") as span:
        yield span


class _NoopSpan:
    """Minimal no-op span for when OTel is absent or disabled."""

    def set_attribute(self, key: str, value: object) -> None:  # noqa: D102
        pass

    def record_exception(self, exc: BaseException) -> None:  # noqa: D102
        pass


def emit_cache_lookup(
    *,
    question: str,
    q_vec: np.ndarray,
    result: object | None,
    similarity_score: float,
    threshold_used: float,
    latency_ms: float,
    query_type: str = "unknown",
    avg_response_tokens: int = _DEFAULT_AVG_TOKENS,
    otel_enabled: bool = False,
) -> None:
    """Emit ``cache.lookup`` + ``cache.hit``/``cache.miss`` OTel spans.

    Designed to be called from the request path after the cache lookup
    resolves.  All parameters are keyword-only to avoid positional confusion.

    ``otel_enabled`` maps directly to ``Settings.otel_enabled``; when False
    this function is a pure no-op (zero CPU cost beyond the call itself).
    """
    if not _HAS_OTEL or not otel_enabled:
        return

    tenant_id = get_current_tenant_id() or "__anonymous__"
    hit = result is not None
    tokens_saved = avg_response_tokens if hit else 0
    outcome = "hit" if hit else "miss"

    try:
        from konjoai.telemetry import get_tracer  # noqa: PLC0415

        tracer_wrapper = get_tracer()
        if not tracer_wrapper.available:
            return
        raw_tracer = getattr(tracer_wrapper, "_tracer", None)
        if raw_tracer is None:
            return

        with raw_tracer.start_as_current_span(f"cache.lookup") as lookup_span:
            lookup_span.set_attribute("kyro.tenant_id", tenant_id)
            lookup_span.set_attribute("kyro.similarity_score", round(float(similarity_score), 4))
            lookup_span.set_attribute("kyro.threshold_used", round(float(threshold_used), 4))
            lookup_span.set_attribute("kyro.latency_ms", round(float(latency_ms), 3))
            lookup_span.set_attribute("kyro.query_type", query_type)
            lookup_span.set_attribute("kyro.tokens_saved", tokens_saved)

            with raw_tracer.start_as_current_span(f"cache.{outcome}") as result_span:
                result_span.set_attribute("kyro.tenant_id", tenant_id)
                result_span.set_attribute("kyro.outcome", outcome)
                result_span.set_attribute("kyro.tokens_saved", tokens_saved)
    except Exception as exc:  # noqa: BLE001 — OTel must never break the request path
        logger.warning("cache span emission failed: %s", exc)


def emit_cache_store(
    *,
    question: str,
    latency_ms: float,
    otel_enabled: bool = False,
) -> None:
    """Emit a ``cache.store`` OTel span.  No-op when OTel is absent/disabled."""
    if not _HAS_OTEL or not otel_enabled:
        return

    tenant_id = get_current_tenant_id() or "__anonymous__"
    try:
        from konjoai.telemetry import get_tracer  # noqa: PLC0415

        tracer_wrapper = get_tracer()
        if not tracer_wrapper.available:
            return
        raw_tracer = getattr(tracer_wrapper, "_tracer", None)
        if raw_tracer is None:
            return

        with raw_tracer.start_as_current_span("cache.store") as span:
            span.set_attribute("kyro.tenant_id", tenant_id)
            span.set_attribute("kyro.latency_ms", round(float(latency_ms), 3))
    except Exception as exc:  # noqa: BLE001
        logger.warning("cache.store span emission failed: %s", exc)
