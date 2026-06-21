"""FastAPI dependencies for auth, rate limiting, and brute-force protection (Sprint 17/18).

get_tenant_id:
    Injected via ``Depends()`` into any route that needs tenant isolation.
    Sprint 17: JWT Bearer auth, K3 pass-through.
    Sprint 18: API-key auth accepted *as an alternative* to JWT Bearer auth.
              Brute-force protection applied before decode attempt.

check_rate_limit:
    Injected via ``Depends()`` into any route that needs rate limiting.
    Sprint 18: sliding-window per-(tenant, endpoint); no-op when disabled.

The context var is reset after the response is returned by yielding in a
generator-style dependency so FastAPI handles cleanup correctly.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from konjoai.auth.brute_force import get_brute_force_guard
from konjoai.auth.jwt_auth import decode_token
from konjoai.config import get_settings

logger = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)


async def get_tenant_id(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> AsyncGenerator[str | None, None]:
    """Resolve tenant_id from Bearer JWT or X-API-Key header; set request-scoped ContextVar.

    Evaluation order (short-circuit on first success):
      1. K3 pass-through — if multi_tenancy_enabled=False AND api_key_auth_enabled=False → yield None.
      2. X-API-Key header — if present and api_key_auth_enabled=True, verify hash lookup.
      3. Bearer JWT      — if present and multi_tenancy_enabled=True, decode & verify.
      4. Fallthrough error — 401 if neither credential supplied but at least one auth mode enabled.

    Brute-force protection:
      When brute_force_enabled=True, a failed JWT or API-key decode increments the
      failure counter for the request IP. On lockout the 429 response is returned.

    Args:
        request:     The incoming HTTP request (injected by FastAPI).
        credentials: Auto-extracted Bearer credentials (injected by FastAPI).

    Yields:
        tenant_id string or None when both auth modes are disabled.

    Raises:
        HTTPException 401: Missing credentials, invalid/expired JWT or API key.
        HTTPException 429: IP is locked out due to repeated failures.
        HTTPException 503: jwt_secret_key not configured.
    """
    async for result in _resolve_tenant_id(request, credentials):
        yield result


async def _resolve_tenant_id(
    request: Request | None,
    credentials: HTTPAuthorizationCredentials | None,
) -> AsyncGenerator[str | None, None]:
    """Core tenant resolution logic, callable from both FastAPI dep and unit tests.

    Accepts ``request=None`` for unit tests that do not construct a full Request.
    All HTTP-boundary logic belongs here; ``get_tenant_id`` delegates to this.
    """
    from konjoai.auth.api_key import verify_api_key
    from konjoai.auth.tenant import _current_tenant_id, set_current_tenant_id

    settings = get_settings()
    mt_enabled = settings.multi_tenancy_enabled
    ak_enabled = getattr(settings, "api_key_auth_enabled", False)

    if not mt_enabled and not ak_enabled:
        # Both auth modes disabled — K3 pass-through; no context var set.
        yield None
        return

    # ── Brute-force check ─────────────────────────────────────────────────────
    client_ip = _get_client_ip(request)
    guard = get_brute_force_guard()
    try:
        guard.check_ip(client_ip)
    except Exception as exc:  # IPLockedOut
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed authentication attempts. {exc}",
        )

    # ── Try X-API-Key (Sprint 18) ─────────────────────────────────────────────
    api_key_header: str | None = None
    if request is not None:
        api_key_header = request.headers.get("X-API-Key")

    if ak_enabled and api_key_header is not None:
        result = verify_api_key(api_key_header, list(getattr(settings, "api_keys", [])))
        if result is not None:
            guard.record_success(client_ip)
            ctx_token = set_current_tenant_id(result.tenant_id)
            logger.debug("API-key auth: tenant_id=%s", result.tenant_id)
            try:
                yield result.tenant_id
            finally:
                _current_tenant_id.reset(ctx_token)
            return
        else:
            # Invalid API key — record failure.
            guard.record_failure(client_ip)
            # If JWT is not enabled or no JWT credentials, abort now.
            if not mt_enabled or credentials is None:
                raise HTTPException(status_code=401, detail="Invalid API key")

    # ── Try Bearer JWT (Sprint 17) ────────────────────────────────────────────
    if mt_enabled:
        if credentials is None:
            raise HTTPException(
                status_code=401,
                detail="Bearer token required (multi_tenancy_enabled=true)",
            )

        if not settings.jwt_secret_key:
            logger.error("multi_tenancy_enabled=True but jwt_secret_key is not configured")
            raise HTTPException(
                status_code=503,
                detail="JWT authentication is not configured (jwt_secret_key missing)",
            )

        try:
            claims = decode_token(
                credentials.credentials,
                settings.jwt_secret_key,
                settings.jwt_algorithm,
                settings.tenant_id_claim,
            )
        except (ValueError, RuntimeError) as exc:
            guard.record_failure(client_ip)
            raise HTTPException(status_code=401, detail=str(exc)) from exc

        guard.record_success(client_ip)
        ctx_token = set_current_tenant_id(claims.tenant_id)
        logger.debug("JWT auth: tenant_id=%s", claims.tenant_id)

        try:
            yield claims.tenant_id
        finally:
            _current_tenant_id.reset(ctx_token)
        return

    # If we get here: api_key_auth is enabled but no valid key and no JWT mode.
    raise HTTPException(
        status_code=401,
        detail="Authentication required",
    )


async def check_rate_limit(
    request: Request,
    tenant_id: str | None = Depends(get_tenant_id),
) -> AsyncGenerator[None, None]:
    """FastAPI dependency that enforces per-(tenant, endpoint) rate limits.

    Yields nothing; raises HTTPException 429 when the limit is exceeded.

    Inject via: ``_: None = Depends(check_rate_limit)``
    """
    from konjoai.auth.rate_limiter import RateLimitExceeded, get_rate_limiter
    from konjoai.auth.tenant import ANONYMOUS_TENANT

    limiter = get_rate_limiter()
    effective_tenant = tenant_id if tenant_id is not None else ANONYMOUS_TENANT
    endpoint = request.url.path
    try:
        limiter.check(effective_tenant, endpoint)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
        )
    yield


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_client_ip(request: Request | None) -> str:
    """Extract the best-effort client IP from the request.

    Prefers X-Forwarded-For (first hop) when set; falls back to direct
    connection address (ASGI scope ``client`` tuple).
    Returns "unknown" when no request is available (direct test calls).
    """
    if request is None:
        return "unknown"
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    client = getattr(request, "client", None)
    if client and client[0]:
        return client[0]
    return "unknown"
