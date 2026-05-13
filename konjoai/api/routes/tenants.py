"""Per-tenant cost attribution API routes (Sprint 26, P1).

``GET /tenants/{tenant_id}/cost_report``
    Returns estimated cost savings for a specific tenant.  Counts are derived
    from real cache hit/miss events accumulated during the process lifetime.

``GET /tenants``
    Returns cost reports for all tenants with at least one query.

Both endpoints require ``cache_enabled=True`` (K3: 404 otherwise).
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from konjoai.config import get_settings
from konjoai.services.cost_attribution import TenantCostReport, get_cost_tracker

router = APIRouter(prefix="/tenants", tags=["tenants"])


@router.get("")
async def list_tenants() -> dict[str, object]:
    """Return cost attribution reports for every tracked tenant.

    Requires ``cache_enabled=True``.
    """
    _require_cache_enabled()
    tracker = get_cost_tracker()
    reports = tracker.all_tenants()
    return {
        "tenants": [r.as_dict() for r in reports],
        "count": len(reports),
    }


@router.get("/{tenant_id}/cost_report")
async def tenant_cost_report(tenant_id: str) -> dict[str, object]:
    """Return the cost attribution report for a single tenant.

    Path parameter ``tenant_id`` must match the JWT ``sub`` claim (or the
    value injected by the auth middleware when multi-tenancy is enabled).

    Returns HTTP 404 when:
    - the cache is disabled; or
    - no queries have been recorded for this tenant yet.
    """
    _require_cache_enabled()
    tracker = get_cost_tracker()
    report: TenantCostReport | None = tracker.report(tenant_id)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"no cost attribution data for tenant '{tenant_id}'",
        )
    return report.as_dict()


# ── Helpers ────────────────────────────────────────────────────────────────


def _require_cache_enabled() -> None:
    if not get_settings().cache_enabled:
        raise HTTPException(
            status_code=404,
            detail="cache is not enabled (set CACHE_ENABLED=true)",
        )
