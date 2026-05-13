"""Cache management API routes (Sprint 26, P1).

``GET /cache/threshold_stats``
    Returns per-query-type hit/miss counts for the adaptive threshold engine.
    Feature-gated: requires ``cache_enabled=True`` in Settings.

All routes return HTTP 404 when ``cache_enabled`` is False (K3).
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from konjoai.cache.threshold import get_threshold_stats
from konjoai.config import get_settings

router = APIRouter(prefix="/cache", tags=["cache"])


@router.get("/threshold_stats")
async def threshold_stats() -> dict[str, object]:
    """Return per-query-type cache hit/miss rates from the adaptive threshold engine.

    Each entry contains ``hits``, ``misses``, ``total``, and ``hit_rate`` for
    the four query types: ``factual``, ``faq``, ``creative``, and ``code``.

    Requires ``cache_enabled=True``.  Returns HTTP 404 when the cache is off.
    """
    settings = get_settings()
    if not settings.cache_enabled:
        raise HTTPException(
            status_code=404,
            detail="cache is not enabled (set CACHE_ENABLED=true)",
        )
    return {"threshold_stats": get_threshold_stats().snapshot()}
