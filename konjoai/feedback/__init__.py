"""Kyro feedback collection — Sprint 25.

Lightweight in-process feedback store for capturing user relevance signals
(thumbs-up / thumbs-down) after query responses.

Usage::

    from konjoai.feedback import get_feedback_store, FeedbackEvent, THUMBS_UP

    store = get_feedback_store()
    store.record(FeedbackEvent(
        question_hash=hash_text("What is the refund policy?"),
        signal=THUMBS_UP,
        timestamp="2026-05-07T12:00:00Z",
        tenant_id="acme-corp",
    ))
"""
from __future__ import annotations

from konjoai.feedback.models import (
    THUMBS_DOWN,
    THUMBS_UP,
    VALID_SIGNALS,
    FeedbackEvent,
)
from konjoai.feedback.store import (
    FeedbackStore,
    _reset_singleton,
    get_feedback_store,
)

__all__ = [
    "FeedbackEvent",
    "FeedbackStore",
    "get_feedback_store",
    "_reset_singleton",
    "THUMBS_UP",
    "THUMBS_DOWN",
    "VALID_SIGNALS",
]
