"""konjoai.auth — JWT authentication, API-key auth, rate limiting, and brute-force protection.

Sprint 17 additions:
    TenantClaims     — decoded JWT payload
    decode_token     — decode + verify a Bearer token
    get_tenant_id    — FastAPI dependency: returns tenant_id or None
    get_current_tenant_id / set_current_tenant_id — context-var accessors
    ANONYMOUS_TENANT — sentinel string for unauthenticated requests

Sprint 18 additions:
    APIKeyResult     — resolved identity from a valid API key
    verify_api_key   — check plaintext key against hash registry
    hash_api_key     — produce SHA-256 hex digest for a plaintext key
    RateLimiter      — per-(tenant, endpoint) sliding-window rate limiter
    RateLimitExceeded — raised when limit is exceeded
    get_rate_limiter — module-level singleton accessor
    BruteForceGuard  — IP-based failed-attempt tracker with lockout
    IPLockedOut      — raised when IP is locked out
    get_brute_force_guard — module-level singleton accessor
    check_rate_limit — FastAPI dependency: 429 on rate-limit breach
"""

from konjoai.auth.api_key import APIKeyResult, hash_api_key, verify_api_key
from konjoai.auth.brute_force import BruteForceGuard, IPLockedOut, get_brute_force_guard
from konjoai.auth.jwt_auth import _HAS_JWT, TenantClaims, decode_token
from konjoai.auth.rate_limiter import RateLimiter, RateLimitExceeded, get_rate_limiter
from konjoai.auth.tenant import (
    ANONYMOUS_TENANT,
    get_current_tenant_id,
    set_current_tenant_id,
)

__all__ = [
    # Sprint 17
    "TenantClaims",
    "decode_token",
    "_HAS_JWT",
    "ANONYMOUS_TENANT",
    "get_current_tenant_id",
    "set_current_tenant_id",
    # Sprint 18 — API keys
    "APIKeyResult",
    "verify_api_key",
    "hash_api_key",
    # Sprint 18 — Rate limiting
    "RateLimiter",
    "RateLimitExceeded",
    "get_rate_limiter",
    # Sprint 18 — Brute-force
    "BruteForceGuard",
    "IPLockedOut",
    "get_brute_force_guard",
]
