"""Tenant context propagation via Python contextvars.

The ContextVar _current_tenant_id is the single source of truth for the
active tenant within any async task or thread spawned by asyncio.to_thread.
Python propagates ContextVar state into threads created by asyncio.to_thread,
so tenant scoping in QdrantStore.search() and upsert() is automatic once
set_current_tenant_id() has been called by the FastAPI dependency.

Usage (internal):
    token = set_current_tenant_id("acme-corp")
    try:
        ...  # all QdrantStore calls here pick up "acme-corp"
    finally:
        _current_tenant_id.reset(token)  # restore previous value
"""

from __future__ import annotations

from contextvars import ContextVar, Token

ANONYMOUS_TENANT: str = "__anonymous__"

_current_tenant_id: ContextVar[str | None] = ContextVar("kyro_tenant_id", default=None)


def get_current_tenant_id() -> str | None:
    """Return the tenant_id for the current async task / thread, or None."""
    return _current_tenant_id.get()


def set_current_tenant_id(tenant_id: str | None) -> Token:
    """Set the tenant_id for the current context; returns a reset Token.

    The caller MUST call ``_current_tenant_id.reset(token)`` in a finally
    block to restore the previous value (handled by get_tenant_id dep).
    """
    return _current_tenant_id.set(tenant_id)
