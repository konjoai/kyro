"""Static API-key authentication layer (Sprint 18).

Users may authenticate using either a Bearer JWT (Sprint 17) *or* a static
API key passed via the ``X-API-Key`` request header. API keys are stored as
SHA-256 hex digests in the settings list ``api_keys``, so the plaintext key
never touches the config file.

Format:  ``X-API-Key: <plaintext-key>``
Config:  ``api_keys: list[str]`` — each entry is a 64-char SHA-256 hex digest
         of the accepted plaintext key.

Key → tenant mapping: a key entry may optionally embed a tenant prefix using
the format ``<sha256hex>:<tenant_id>``. If no tenant is embedded the
``ANONYMOUS_TENANT`` sentinel is returned.

Design notes:
- Hashing is done with ``hashlib.sha256`` (stdlib, no extra deps — K5).
- Comparison is done with ``hmac.compare_digest`` to prevent timing attacks.
- Feature-flagged off by default via ``api_key_auth_enabled=False`` (K3).
"""

from __future__ import annotations

import hashlib
import hmac
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "APIKeyResult",
    "verify_api_key",
    "hash_api_key",
]

# ── Public constants ──────────────────────────────────────────────────────────

_SEPARATOR = ":"


# ── Data contract ─────────────────────────────────────────────────────────────


class APIKeyResult:
    """Resolved identity from a valid API key."""

    __slots__ = ("tenant_id", "key_hash")

    def __init__(self, tenant_id: str, key_hash: str) -> None:
        self.tenant_id = tenant_id
        self.key_hash = key_hash  # the SHA-256 hex digest that matched

    def __repr__(self) -> str:
        return f"APIKeyResult(tenant_id={self.tenant_id!r}, key_hash={self.key_hash[:8]}…)"


# ── Hash helper (public so operators can pre-hash keys) ───────────────────────


def hash_api_key(plaintext: str) -> str:
    """Return the SHA-256 hex digest of *plaintext*.

    Use this to generate the digest values that go into ``api_keys`` config.

    >>> hash_api_key("my-secret-key")  # doctest: +SKIP
    'a3f1...'
    """
    return hashlib.sha256(plaintext.encode()).hexdigest()


# ── Verification ──────────────────────────────────────────────────────────────


def verify_api_key(
    plaintext: str,
    registered_entries: list[str],
) -> APIKeyResult | None:
    """Check *plaintext* against *registered_entries*.

    Args:
        plaintext:          The raw key supplied by the caller.
        registered_entries: Items from ``settings.api_keys``; each entry is
                            either ``<sha256hex>`` or ``<sha256hex>:<tenant_id>``.

    Returns:
        ``APIKeyResult`` on a match, ``None`` if no entry matches.
    """
    if not plaintext:
        return None
    candidate_digest = hash_api_key(plaintext)
    for entry in registered_entries:
        parts = entry.split(_SEPARATOR, 1)
        stored_digest = parts[0].strip()
        tenant_id = parts[1].strip() if len(parts) == 2 else _anonymous_tenant()
        # Constant-time comparison to prevent timing attacks.
        if hmac.compare_digest(candidate_digest.lower(), stored_digest.lower()):
            logger.debug("API key matched for tenant=%s", tenant_id)
            return APIKeyResult(tenant_id=tenant_id, key_hash=stored_digest)
    return None


def _anonymous_tenant() -> str:
    """Lazy import to avoid circular dependencies with tenant.py."""
    from konjoai.auth.tenant import ANONYMOUS_TENANT

    return ANONYMOUS_TENANT
