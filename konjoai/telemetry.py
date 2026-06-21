"""Pipeline telemetry — per-step latency tracking plus optional OTel + Prometheus observability.

Existing behaviour (K6 — backward-compatible):
    StepTiming, PipelineTelemetry, timed() are unchanged.

Sprint 16 additions:
    KyroMetrics  — Prometheus counters/histograms; no-op when prometheus-client absent.
    KyroTracer   — OTel span wrapper; no-op when opentelemetry-sdk absent or endpoint unset.
    get_metrics() / get_tracer() — module-level singletons.
    record_pipeline_metrics()    — push a completed PipelineTelemetry into Prometheus.

All new features are guarded by _HAS_PROMETHEUS / _HAS_OTEL and by otel_enabled (K3, K5).
"""
from __future__ import annotations

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Optional dep guards ────────────────────────────────────────────────────────

try:
    from prometheus_client import (  # type: ignore[import-untyped]
        Counter,
        Histogram,
        generate_latest,
    )
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False

try:
    from opentelemetry import trace as _otel_trace  # type: ignore[import-untyped]
    from opentelemetry.sdk.trace import TracerProvider as _TracerProvider  # type: ignore[import-untyped]
    from opentelemetry.sdk.trace.export import BatchSpanProcessor as _BatchSpanProcessor  # type: ignore[import-untyped]
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


# ── Existing pipeline timing (unchanged) ──────────────────────────────────────

@dataclass
class StepTiming:
    """Latency record for a single pipeline step."""

    step: str
    duration_ms: float
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Return ``{"duration_ms": ..., **metadata}`` for JSON serialization."""
        d: dict = {"duration_ms": round(self.duration_ms, 3)}
        d.update(self.metadata)
        return d


@dataclass
class PipelineTelemetry:
    """Collects step timings for a single request.

    Usage::

        tel = PipelineTelemetry()
        with timed(tel, "hybrid_search", top_k=10):
            results = hybrid_search(question)
        print(tel.as_dict())
    """

    steps: list[StepTiming] = field(default_factory=list)

    def record(self, step: str, duration_ms: float, **metadata: object) -> None:
        """Append a finished step timing. Called by the timed() context manager."""
        self.steps.append(StepTiming(step=step, duration_ms=duration_ms, metadata=dict(metadata)))

    def total_ms(self) -> float:
        """Sum of all recorded step durations in milliseconds."""
        return sum(s.duration_ms for s in self.steps)

    def as_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON serialization.

        Returns::

            {
                "steps": {"hybrid_search": {"duration_ms": 23.4, "top_k": 10}, ...},
                "total_ms": 87.2
            }
        """
        return {
            "steps": {s.step: s.as_dict() for s in self.steps},
            "total_ms": round(self.total_ms(), 3),
        }


@contextmanager
def timed(
    telemetry: PipelineTelemetry,
    step: str,
    **metadata: object,
) -> Generator[None, None, None]:
    """Context manager: measure wall-clock duration of a block and record it.

    Args:
        telemetry: The PipelineTelemetry instance to accumulate into.
        step: Human-readable step name (e.g. "hybrid_search", "rerank").
        **metadata: Arbitrary key-value pairs attached to the step record
            (e.g. top_k=10, model="gpt-4o-mini").

    Example::

        tel = PipelineTelemetry()
        with timed(tel, "rerank", top_k=5):
            results = reranker.rerank(query, candidates, top_k=5)
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - t0) * 1_000
        telemetry.record(step, duration_ms, **metadata)
        logger.debug("step=%s duration_ms=%.3f metadata=%s", step, duration_ms, metadata)


# ── Sprint 16: Prometheus metrics ─────────────────────────────────────────────

class KyroMetrics:
    """Prometheus counters + histograms for the Kyro pipeline.

    Instantiated as a module-level singleton via get_metrics(). All methods
    are unconditionally safe — they silently no-op when prometheus-client is
    absent or when enabled=False (K3).
    """

    #: Prometheus label buckets (ms) covering sub-ms cache to slow LLM calls.
    _LATENCY_BUCKETS = (1, 5, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, float("inf"))

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled and _HAS_PROMETHEUS
        if self._enabled:
            self._query_total = Counter(
                "kyro_query_total",
                "Total queries processed by the Kyro pipeline",
                ["intent"],
            )
            self._query_errors_total = Counter(
                "kyro_query_errors_total",
                "Total pipeline errors",
                ["step"],
            )
            self._query_latency_ms = Histogram(
                "kyro_query_latency_ms",
                "Per-step pipeline latency in milliseconds",
                ["step"],
                buckets=self._LATENCY_BUCKETS,
            )
            self._cache_hits_total = Counter(
                "kyro_cache_hits_total",
                "Total semantic cache hits",
                [],
            )

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True when prometheus-client is installed and metrics are enabled."""
        return self._enabled

    def record_step(self, step: str, duration_ms: float) -> None:
        """Observe a step latency histogram."""
        if self._enabled:
            self._query_latency_ms.labels(step=step).observe(duration_ms)

    def inc_query(self, intent: str = "retrieval") -> None:
        """Increment the query counter for a given intent label."""
        if self._enabled:
            self._query_total.labels(intent=intent).inc()

    def inc_error(self, step: str = "unknown") -> None:
        """Increment the error counter for a given pipeline step."""
        if self._enabled:
            self._query_errors_total.labels(step=step).inc()

    def inc_cache_hit(self) -> None:
        """Increment the semantic cache hit counter."""
        if self._enabled:
            self._cache_hits_total.inc()

    def exposition(self) -> str:
        """Return Prometheus text exposition format, or empty string when unavailable."""
        if self._enabled and _HAS_PROMETHEUS:
            return generate_latest().decode("utf-8")
        return ""


# ── Sprint 16: OTel tracer ────────────────────────────────────────────────────

@contextmanager
def _noop_span() -> Generator[None, None, None]:
    """Context manager that does nothing — stand-in when OTel is absent."""
    yield


class KyroTracer:
    """Thin wrapper around an OpenTelemetry tracer.

    When opentelemetry-sdk is not installed or no endpoint is configured,
    start_span() returns a _noop_span() so callers never branch on availability.
    """

    def __init__(self, endpoint: str = "", service_name: str = "kyro") -> None:
        self._tracer = None
        if _HAS_OTEL and endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import-untyped]
                    OTLPSpanExporter,
                )
                provider = _TracerProvider()
                exporter = OTLPSpanExporter(endpoint=endpoint)
                provider.add_span_processor(_BatchSpanProcessor(exporter))
                _otel_trace.set_tracer_provider(provider)
                self._tracer = _otel_trace.get_tracer(service_name)
                logger.info("OTel tracer initialised: endpoint=%s service=%s", endpoint, service_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("OTel tracer init failed (non-fatal): %s", exc)

    @property
    def available(self) -> bool:
        """True when OTel is installed and a tracer was successfully initialised."""
        return _HAS_OTEL and self._tracer is not None

    def start_span(self, name: str) -> Generator[None, None, None]:  # type: ignore[return]
        """Return an OTel span context manager, or a no-op if OTel is unavailable."""
        if self.available and self._tracer is not None:
            return self._tracer.start_as_current_span(name)  # type: ignore[return-value]
        return _noop_span()


# ── Module-level singletons ───────────────────────────────────────────────────

_metrics: KyroMetrics | None = None
_tracer: KyroTracer | None = None


def get_metrics() -> KyroMetrics:
    """Return the module-level KyroMetrics singleton.

    Initialised on first call. Reads otel_enabled from Settings so the
    Prometheus backend is only activated when the operator opts in (K3).
    """
    global _metrics
    if _metrics is None:
        try:
            from konjoai.config import get_settings
            enabled = get_settings().otel_enabled
        except Exception:  # noqa: BLE001
            enabled = False
        _metrics = KyroMetrics(enabled=enabled)
    return _metrics


def get_tracer() -> KyroTracer:
    """Return the module-level KyroTracer singleton.

    Reads otel_endpoint and otel_service_name from Settings on first call.
    """
    global _tracer
    if _tracer is None:
        try:
            from konjoai.config import get_settings
            s = get_settings()
            _tracer = KyroTracer(endpoint=s.otel_endpoint, service_name=s.otel_service_name)
        except Exception:  # noqa: BLE001
            _tracer = KyroTracer()
    return _tracer


# ── Convenience helper ────────────────────────────────────────────────────────

def record_pipeline_metrics(
    telemetry: PipelineTelemetry,
    intent: str = "retrieval",
    *,
    enabled: bool = True,
) -> None:
    """Push a completed PipelineTelemetry into Prometheus.

    This is the single call-site in the /query route — callers pass
    ``enabled=settings.otel_enabled`` to honour the K3 feature flag.
    No-ops when Prometheus is absent regardless of ``enabled``.
    """
    if not enabled:
        return
    m = get_metrics()
    m.inc_query(intent)
    for step in telemetry.steps:
        m.record_step(step.step, step.duration_ms)
