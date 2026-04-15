"""Pipeline telemetry — per-step latency tracking with zero new dependencies.

All timing uses time.perf_counter() (stdlib). No structlog, no metrics backend.
Telemetry data flows into QueryResponse.telemetry as a plain dict.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StepTiming:
    """Latency record for a single pipeline step."""

    step: str
    duration_ms: float
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
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
