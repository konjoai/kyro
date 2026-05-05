"""Audit log query API — Sprint 24.

Two read-only endpoints surface the in-process audit log for operators:

    GET /audit/events   — paginated, filterable event list
    GET /audit/stats    — per-event-type counts

Both return HTTP 404 when ``audit_enabled=False`` so that the existence of the
endpoint cannot be used as an oracle to determine whether auditing is active.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from konjoai.audit.models import AuditEvent
from konjoai.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/audit", tags=["audit"])


# ── Response models ───────────────────────────────────────────────────────────


class AuditEventOut(BaseModel):
    """JSON-serialisable view of a single ``AuditEvent``."""

    event_type: str
    timestamp: str
    endpoint: str
    status_code: int
    latency_ms: float
    tenant_id: str | None = None
    client_ip: str | None = None
    question_hash: str | None = None
    intent: str | None = None
    cache_hit: bool | None = None
    result_count: int | None = None
    path_hash: str | None = None
    chunks_indexed: int | None = None
    chunks_deduplicated: int | None = None
    reason: str | None = None

    @classmethod
    def from_event(cls, event: AuditEvent) -> "AuditEventOut":
        return cls(**event.as_dict())


class AuditEventsResponse(BaseModel):
    events: list[AuditEventOut]
    total: int
    audit_enabled: bool


class AuditStatsResponse(BaseModel):
    stats: dict[str, int]
    audit_enabled: bool


# ── Guards ────────────────────────────────────────────────────────────────────


def _require_audit_enabled() -> None:
    """Raise HTTP 404 when audit logging is disabled (K3)."""
    settings = get_settings()
    if not getattr(settings, "audit_enabled", False):
        raise HTTPException(
            status_code=404,
            detail="Audit logging is disabled. Set AUDIT_ENABLED=true to enable.",
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/events", response_model=AuditEventsResponse)
def list_audit_events(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of events to return."),
    tenant_id: str | None = Query(None, description="Filter by tenant identifier."),
    event_type: str | None = Query(None, description="Filter by event type (query/ingest/agent_query/auth_failure/rate_limited)."),
    _: None = Depends(_require_audit_enabled),
) -> AuditEventsResponse:
    """Return recent audit log entries, optionally filtered by tenant or event type.

    Returns HTTP 404 when audit logging is disabled (K3: the endpoint's
    existence cannot reveal whether auditing is active).
    """
    from konjoai.audit import get_audit_logger

    audit = get_audit_logger()
    events = audit.query_events(limit=limit, tenant_id=tenant_id, event_type=event_type)
    return AuditEventsResponse(
        events=[AuditEventOut.from_event(e) for e in events],
        total=len(events),
        audit_enabled=True,
    )


@router.get("/stats", response_model=AuditStatsResponse)
def audit_stats(
    _: None = Depends(_require_audit_enabled),
) -> AuditStatsResponse:
    """Return per-event-type counts from the audit log.

    Returns HTTP 404 when audit logging is disabled.
    """
    from konjoai.audit import get_audit_logger

    audit = get_audit_logger()
    return AuditStatsResponse(stats=audit.stats(), audit_enabled=True)
