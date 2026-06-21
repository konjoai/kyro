"""JWT decode and tenant claims extraction (Sprint 17).

Guards against missing PyJWT with _HAS_JWT (K5: no new hard dep).
When multi_tenancy_enabled=True but PyJWT is absent, decode_token raises
RuntimeError immediately so the operator knows to install the dep.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Optional dep guard ────────────────────────────────────────────────────────

try:
    import jwt as _jwt  # PyJWT>=2.8

    _HAS_JWT = True
except ImportError:
    _HAS_JWT = False


# ── Data contract ─────────────────────────────────────────────────────────────


@dataclass
class TenantClaims:
    """Decoded JWT payload with tenant identity and optional roles."""

    tenant_id: str
    roles: list[str] = field(default_factory=list)
    exp: int | None = None


# ── Public API ────────────────────────────────────────────────────────────────


def decode_token(
    token: str,
    secret: str,
    algorithm: str = "HS256",
    tenant_id_claim: str = "sub",
) -> TenantClaims:
    """Decode and verify a Bearer JWT token.

    Args:
        token:           Raw token string (without "Bearer " prefix).
        secret:          HMAC secret key (must not be empty).
        algorithm:       JWT algorithm, e.g. "HS256" (default).
        tenant_id_claim: JWT claim to use as tenant_id (default "sub").

    Returns:
        TenantClaims with tenant_id, roles, and expiry.

    Raises:
        RuntimeError: If PyJWT is not installed.
        ValueError:   If secret is empty, token is expired, signature is
                      invalid, or the tenant_id claim is missing.
    """
    if not _HAS_JWT:
        raise RuntimeError("PyJWT is required for JWT authentication: pip install PyJWT>=2.8")
    if not secret:
        raise ValueError(
            "JWT secret key is empty — set jwt_secret_key in settings or the JWT_SECRET_KEY environment variable."
        )
    try:
        payload: dict = _jwt.decode(token, secret, algorithms=[algorithm])
    except _jwt.ExpiredSignatureError as exc:
        raise ValueError("JWT token has expired") from exc
    except _jwt.InvalidTokenError as exc:
        raise ValueError(f"JWT token is invalid: {exc}") from exc

    raw_id = payload.get(tenant_id_claim)
    if not raw_id:
        raise ValueError(f"JWT token is missing required claim '{tenant_id_claim}'")

    return TenantClaims(
        tenant_id=str(raw_id),
        roles=list(payload.get("roles", [])),
        exp=payload.get("exp"),
    )
