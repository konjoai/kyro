"""Feedback event model — Sprint 25.

OWASP PII rule: raw question text is NEVER stored.
Only SHA-256 hashes of user-supplied strings are written to the feedback store.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ── Signal constants ──────────────────────────────────────────────────────────

THUMBS_UP = "thumbs_up"
THUMBS_DOWN = "thumbs_down"

# Valid signal values
VALID_SIGNALS = frozenset({THUMBS_UP, THUMBS_DOWN})


@dataclass
class FeedbackEvent:
    """Immutable record of a single user feedback submission.

    All user-supplied string fields (question text, answer text) are hashed
    before storage — raw PII is never written (OWASP Top-10 A02).

    ``question_hash`` is the 16-hex-char SHA-256 prefix of the original query,
    matching the same hash emitted by the audit log so operators can correlate
    feedback with query audit events.
    """

    # ── Required fields ───────────────────────────────────────────────────────
    question_hash: str        # hash_text(question) — NEVER raw text
    signal: str               # THUMBS_UP | THUMBS_DOWN
    timestamp: str            # ISO 8601 UTC — set by FeedbackStore, not caller

    # ── Optional context ──────────────────────────────────────────────────────
    relevance_score: float | None = None  # caller-supplied 0.0–1.0 relevance
    tenant_id: str | None = None          # tenant identifier from JWT / API key
    client_ip: str | None = None          # client IP for abuse detection
    comment_hash: str | None = None       # hash_text(comment) when provided
    model: str | None = None              # model that produced the answer
    latency_ms: float | None = None       # pipeline latency from the original response

    def as_dict(self) -> dict:
        """Return a JSON-serialisable dict (None fields omitted)."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
