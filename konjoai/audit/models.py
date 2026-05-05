"""Audit event model.

OWASP PII rule: raw question / document text is NEVER stored.
Only SHA-256 hashes of user-supplied strings are written to the audit log.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


# ── Event type constants ──────────────────────────────────────────────────────

QUERY = "query"
INGEST = "ingest"
AGENT_QUERY = "agent_query"
AUTH_FAILURE = "auth_failure"
RATE_LIMITED = "rate_limited"


def hash_text(text: str) -> str:
    """Return the first 16 hex chars of SHA-256(text) — enough for correlation,
    not enough to reconstruct the original string."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


@dataclass
class AuditEvent:
    """Immutable record of a single auditable API interaction.

    All user-supplied string fields are hashed before storage.  The audit log
    never contains raw question text, document paths, or user identifiers in
    plaintext form (OWASP Top-10 A02 — Cryptographic Failures mitigation).
    """

    # ── Required fields ───────────────────────────────────────────────────────
    event_type: str       # QUERY | INGEST | AGENT_QUERY | AUTH_FAILURE | RATE_LIMITED
    timestamp: str        # ISO 8601 UTC — set by AuditLogger, not caller
    endpoint: str         # "/query" | "/ingest" | "/agent/query" | …
    status_code: int      # HTTP status code of the response
    latency_ms: float     # Wall-clock latency in milliseconds

    # ── Optional context ──────────────────────────────────────────────────────
    tenant_id: str | None = None   # tenant identifier from JWT / API key
    client_ip: str | None = None   # client IP address

    # ── Query / agent specific ────────────────────────────────────────────────
    question_hash: str | None = None  # hash_text(question) — NEVER raw text
    intent: str | None = None         # "retrieval" | "chat" | "aggregation"
    cache_hit: bool | None = None
    result_count: int | None = None   # number of source docs returned

    # ── Ingest specific ───────────────────────────────────────────────────────
    path_hash: str | None = None      # hash_text(path) — NEVER raw path
    chunks_indexed: int | None = None
    chunks_deduplicated: int | None = None

    # ── Auth / security specific ──────────────────────────────────────────────
    reason: str | None = None  # "invalid_token" | "rate_limited" | "brute_force"

    def as_dict(self) -> dict:
        """Return a JSON-serialisable dict (None fields omitted)."""
        return {k: v for k, v in self.__dict__.items() if v is not None}
