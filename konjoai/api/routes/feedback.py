"""Feedback collection API — Sprint 25.

Two endpoints expose user relevance feedback for RAG response quality monitoring:

    POST /feedback          — submit a thumbs-up / thumbs-down signal
    GET  /feedback/summary  — aggregate statistics (total, thumbs_up/down, avg_relevance)

Both return HTTP 404 when ``feedback_enabled=False`` (K3: endpoint existence
cannot reveal whether feedback collection is active).

OWASP: raw question text is NEVER accepted or stored — the client must supply
the ``question_hash`` from the ``POST /query`` response (or compute it locally
using the same ``hash_text()`` helper).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator

from konjoai.audit.models import hash_text
from konjoai.config import get_settings
from konjoai.feedback.models import VALID_SIGNALS, FeedbackEvent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/feedback", tags=["feedback"])


# ── Request / response models ─────────────────────────────────────────────────


class FeedbackRequest(BaseModel):
    """Client-submitted feedback for a single query response.

    ``question_hash`` must be the 16-hex-char SHA-256 prefix returned in the
    audit log or computed via ``hash_text(question)``.  Raw question text is
    NEVER accepted here.
    """

    question_hash: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="hash_text(question) — the first 16 hex chars of SHA-256.",
    )
    signal: str = Field(
        ...,
        description=f"Relevance signal: one of {sorted(VALID_SIGNALS)}.",
    )
    relevance_score: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Optional continuous relevance score in [0.0, 1.0].",
    )
    comment: str | None = Field(
        None,
        max_length=1000,
        description="Optional free-text comment (stored as hash only — no PII).",
    )
    model: str | None = Field(
        None,
        max_length=128,
        description="Model name that produced the answer being rated.",
    )
    latency_ms: float | None = Field(
        None,
        ge=0.0,
        description="Pipeline latency (ms) from the original query response.",
    )

    @field_validator("signal")
    @classmethod
    def _validate_signal(cls, v: str) -> str:
        if v not in VALID_SIGNALS:
            raise ValueError(
                f"signal must be one of {sorted(VALID_SIGNALS)}, got {v!r}"
            )
        return v


class FeedbackResponse(BaseModel):
    """Confirmation of a recorded feedback event."""

    recorded: bool
    question_hash: str
    signal: str
    timestamp: str


class FeedbackSummaryResponse(BaseModel):
    """Aggregate feedback statistics."""

    total: int
    thumbs_up: int
    thumbs_down: int
    avg_relevance: float | None = None
    by_question: dict[str, dict[str, int]]
    feedback_enabled: bool


# ── K3 gate ───────────────────────────────────────────────────────────────────


def _require_feedback_enabled() -> None:
    """Raise HTTP 404 when feedback collection is disabled (K3)."""
    settings = get_settings()
    if not getattr(settings, "feedback_enabled", False):
        raise HTTPException(
            status_code=404,
            detail="Feedback collection is disabled. Set FEEDBACK_ENABLED=true to enable.",
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(
    req: FeedbackRequest,
    request: Request,
    _: None = Depends(_require_feedback_enabled),
) -> FeedbackResponse:
    """Record a user relevance feedback event.

    Accepts a ``question_hash`` (not raw question text) and a binary signal
    (``thumbs_up`` | ``thumbs_down``) with an optional continuous
    ``relevance_score`` in [0.0, 1.0].

    Returns HTTP 404 when ``feedback_enabled=False``.
    Returns HTTP 422 for invalid ``signal`` values.
    Returns HTTP 201 on success.
    """
    from konjoai.feedback.store import get_feedback_store

    settings = get_settings()
    now = datetime.now(timezone.utc).isoformat()

    # Derive tenant from request state (set by JWT auth middleware when enabled)
    tenant_id: str | None = getattr(request.state, "tenant_id", None)
    client_ip: str | None = (
        request.client.host if request.client else None
    )

    # Hash the optional comment — OWASP: never store free-text user input
    comment_hash: str | None = (
        hash_text(req.comment) if req.comment else None
    )

    event = FeedbackEvent(
        question_hash=req.question_hash,
        signal=req.signal,
        timestamp=now,
        relevance_score=req.relevance_score,
        tenant_id=tenant_id,
        client_ip=client_ip,
        comment_hash=comment_hash,
        model=req.model,
        latency_ms=req.latency_ms,
    )

    store = get_feedback_store(max_events=settings.feedback_max_events)
    store.record(event)

    logger.debug(
        "feedback recorded: signal=%s q_hash=%s tenant=%s",
        req.signal,
        req.question_hash,
        tenant_id,
    )

    return FeedbackResponse(
        recorded=True,
        question_hash=req.question_hash,
        signal=req.signal,
        timestamp=now,
    )


@router.get("/summary", response_model=FeedbackSummaryResponse)
def feedback_summary(
    tenant_id: str | None = Query(None, description="Filter summary by tenant."),
    _: None = Depends(_require_feedback_enabled),
) -> FeedbackSummaryResponse:
    """Return aggregate feedback statistics.

    Returns HTTP 404 when ``feedback_enabled=False``.

    Response fields:
    - ``total``         — total feedback events (after optional tenant filter)
    - ``thumbs_up``     — count of positive signals
    - ``thumbs_down``   — count of negative signals
    - ``avg_relevance`` — mean relevance_score (omitted when no scores present)
    - ``by_question``   — per-question_hash breakdown
    """
    from konjoai.feedback.store import get_feedback_store

    settings = get_settings()
    store = get_feedback_store(max_events=settings.feedback_max_events)
    stats = store.summary(tenant_id=tenant_id)

    return FeedbackSummaryResponse(
        total=stats["total"],
        thumbs_up=stats["thumbs_up"],
        thumbs_down=stats["thumbs_down"],
        avg_relevance=stats.get("avg_relevance"),
        by_question=stats.get("by_question", {}),
        feedback_enabled=True,
    )
