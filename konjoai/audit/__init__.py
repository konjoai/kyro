"""Kyro audit logging — immutable event trail for compliance and observability.

All user-supplied strings (question text, document paths) are hashed before
storage.  The audit log never contains PII in plaintext form.

Usage::

    from konjoai.audit import get_audit_logger, AuditEvent, QUERY

    audit = get_audit_logger()
    audit.log(AuditEvent(
        event_type=QUERY,
        timestamp="2026-04-28T12:00:00Z",
        endpoint="/query",
        status_code=200,
        latency_ms=123.4,
        tenant_id="acme-corp",
        question_hash=hash_text("What is the refund policy?"),
        intent="retrieval",
        cache_hit=False,
        result_count=5,
    ))
"""
from __future__ import annotations

from konjoai.audit.logger import (
    AuditLogger,
    InMemoryBackend,
    JsonLinesBackend,
    _reset_singleton,
    get_audit_logger,
)
from konjoai.audit.models import (
    AGENT_QUERY,
    AUTH_FAILURE,
    INGEST,
    QUERY,
    RATE_LIMITED,
    AuditEvent,
    hash_text,
)

__all__ = [
    "AuditEvent",
    "AuditLogger",
    "InMemoryBackend",
    "JsonLinesBackend",
    "get_audit_logger",
    "_reset_singleton",
    "hash_text",
    "QUERY",
    "INGEST",
    "AGENT_QUERY",
    "AUTH_FAILURE",
    "RATE_LIMITED",
]
