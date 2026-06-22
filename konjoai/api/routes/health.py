"""Health and observability routes.

GET /health   — liveness probe (existing, re-exported here for router grouping)
GET /metrics  — Prometheus text exposition (Sprint 16; K3-gated on otel_enabled)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

logger = logging.getLogger(__name__)

router = APIRouter(tags=["observability"])


@router.get("/metrics")
def prometheus_metrics() -> Response:
    """Return Prometheus text exposition.

    Requires:
      - ``otel_enabled=true`` in settings (K3 gate)
      - ``prometheus-client`` package installed (K5 graceful degradation)

    Returns 404 when ``otel_enabled=false``, 503 when prometheus-client is absent.
    """
    from konjoai.config import get_settings
    from konjoai.telemetry import get_metrics

    settings = get_settings()
    if not settings.otel_enabled:
        raise HTTPException(
            status_code=404,
            detail="Prometheus metrics not enabled. Set otel_enabled=true to activate.",
        )

    m = get_metrics()
    if not m.available:
        raise HTTPException(
            status_code=503,
            detail="prometheus-client is not installed. pip install prometheus-client>=0.19",
        )

    return Response(
        content=m.exposition(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
