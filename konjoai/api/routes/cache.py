"""Cache management API routes (Sprint 26–28).

Endpoints
---------
GET    /cache/threshold_stats  — per-type hit/miss rates (Sprint 26)
POST   /cache/warm             — seed the cache from a list of Q/A pairs (Sprint 27)
GET    /cache/expired_count    — how many entries have exceeded their TTL (Sprint 27)
DELETE /cache/expired          — evict all expired entries (Sprint 27)
GET    /cache/clusters         — k-means topic clustering of cached queries (Sprint 27)
POST   /cache/search           — batch top-k similarity search (Sprint 28)
GET    /cache/analytics        — latency percentiles, hit-rate trends, similarity dist (Sprint 28)
GET    /cache/ttl_report       — TTL distribution + adaptive-TTL candidates (Sprint 28)
POST   /cache/ttl/adjust       — trigger one adaptive-TTL adjustment cycle (Sprint 28)

All routes return HTTP 404 when ``cache_enabled`` is False (K3).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from konjoai.cache.analytics import LatencyBuffer, compute_analytics
from konjoai.cache.semantic_cache import SemanticCache, get_semantic_cache
from konjoai.cache.threshold import get_threshold_stats
from konjoai.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cache", tags=["cache"])


# ── Helpers ────────────────────────────────────────────────────────────────────


def _require_memory_cache() -> SemanticCache:
    """Return the active in-memory SemanticCache, or raise HTTP 404."""
    settings = get_settings()
    if not settings.cache_enabled:
        raise HTTPException(
            status_code=404,
            detail="cache is not enabled (set CACHE_ENABLED=true)",
        )
    cache = get_semantic_cache()
    if cache is None or not isinstance(cache, SemanticCache):
        raise HTTPException(
            status_code=501,
            detail="operation only supported on the in-memory cache backend",
        )
    return cache


def _get_encoder():
    """Lazy-import the encoder; raises 503 when the embedding model is unavailable."""
    try:
        from konjoai.embed.encoder import get_encoder  # noqa: PLC0415

        return get_encoder()
    except Exception as exc:  # noqa: BLE001
        logger.warning("encoder unavailable for cache warm: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"embedding encoder unavailable: {exc}",
        ) from exc


# ── Sprint 26: threshold stats ─────────────────────────────────────────────────


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


# ── Sprint 27: cache warming ───────────────────────────────────────────────────


class WarmPair(BaseModel):
    """A single question/answer pair for cache warming."""

    question: str = Field(..., min_length=1, max_length=2048)
    answer: str = Field(..., min_length=1, max_length=32768)


class WarmRequest(BaseModel):
    """Batch of Q/A pairs to seed into the semantic cache."""

    pairs: list[WarmPair] = Field(..., min_length=1)


class WarmResponse(BaseModel):
    """Result of a cache warming request."""

    warmed: int
    skipped_duplicates: int
    skipped_errors: int
    total_submitted: int


@router.post("/warm", response_model=WarmResponse)
async def warm_cache(body: WarmRequest) -> WarmResponse:
    """Pre-populate the semantic cache from a list of question/answer pairs.

    Useful for seeding the cache from historical query logs before go-live,
    or for warming a freshly deployed instance with known FAQ answers.

    Each pair is embedded using the same encoder the live query path uses, so
    subsequent semantically similar questions will hit the warmed entries at
    the configured similarity threshold.

    Duplicate questions (same normalised key already in the cache) are skipped
    without overwriting the existing entry.  The response reports exactly how
    many pairs were inserted vs skipped.
    """
    settings = get_settings()
    cache = _require_memory_cache()
    max_batch = getattr(settings, "cache_warm_max_batch", 500)
    if len(body.pairs) > max_batch:
        raise HTTPException(
            status_code=422,
            detail=f"batch exceeds cache_warm_max_batch={max_batch}; split into smaller requests",
        )

    encoder = _get_encoder()

    class _Resp:
        """Minimal stand-in that satisfies cache.store()'s response.answer lookup."""
        def __init__(self, answer: str) -> None:
            self.answer = answer

    warmed = skipped_dup = skipped_err = 0

    # Embed in a thread to avoid blocking the event loop
    questions = [p.question for p in body.pairs]
    try:
        vecs: np.ndarray = await asyncio.to_thread(
            encoder.encode, questions  # type: ignore[arg-type]
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("cache warm: batch encode failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"encoder error: {exc}") from exc

    for i, pair in enumerate(body.pairs):
        key = SemanticCache._normalise(pair.question)
        # Exact-key duplicate check — don't overwrite live entries
        with cache._lock:  # type: ignore[attr-defined]
            already = key in cache._exact  # type: ignore[attr-defined]
        if already:
            skipped_dup += 1
            continue
        try:
            q_vec = vecs[i : i + 1].astype(np.float32)  # shape (1, dim)
            cache.store(pair.question, q_vec, _Resp(pair.answer))
            warmed += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("cache warm: store failed for question=%r: %s", pair.question[:60], exc)
            skipped_err += 1

    logger.info(
        "cache warm complete — warmed=%d dup=%d err=%d",
        warmed, skipped_dup, skipped_err,
    )
    return WarmResponse(
        warmed=warmed,
        skipped_duplicates=skipped_dup,
        skipped_errors=skipped_err,
        total_submitted=len(body.pairs),
    )


# ── Sprint 27: TTL expiry endpoints ───────────────────────────────────────────


@router.get("/expired_count")
async def expired_count() -> dict[str, object]:
    """Return the number of cache entries that have exceeded their TTL.

    Returns ``0`` when the cache has no TTL configured (``ttl_seconds=0``).
    Does not remove the entries; call ``DELETE /cache/expired`` to evict them.
    """
    cache = _require_memory_cache()
    count = await asyncio.to_thread(cache.expired_count)
    return {
        "expired_count": count,
        "ttl_seconds": cache._ttl_seconds,  # type: ignore[attr-defined]
    }


@router.delete("/expired")
async def evict_expired() -> dict[str, object]:
    """Evict all cache entries that have exceeded their TTL.

    Returns the number of entries removed.  Idempotent — safe to call
    repeatedly; subsequent calls return ``0`` once all stale entries are gone.
    Returns ``0`` when no TTL is configured.
    """
    cache = _require_memory_cache()
    removed = await asyncio.to_thread(cache.evict_expired)
    return {"evicted": removed, "ttl_seconds": cache._ttl_seconds}  # type: ignore[attr-defined]


# ── Sprint 27: query clustering ────────────────────────────────────────────────


@router.get("/clusters")
async def query_clusters(
    k: int = Query(default=5, ge=2, le=20, description="Number of clusters"),
) -> dict[str, object]:
    """Identify the top-k topic clusters in the cached query embeddings.

    Runs k-means (Lloyd's algorithm, 20 iterations) on the L2-normalised
    question vectors stored in the in-memory cache.  Returns each cluster's
    centroid similarity, hit rate, query count, and up to 5 representative
    questions.

    Requires at least ``2×k`` cache entries (returns HTTP 422 otherwise).
    Computation is O(n × k × iters); safe for caches up to ~10 000 entries.
    """
    cache = _require_memory_cache()

    with cache._lock:  # type: ignore[attr-defined]
        entries = [
            (e.question, e.question_vec, e.hit_count)
            for e in cache._lru.values()  # type: ignore[attr-defined]
            if not e.is_expired()
        ]

    n = len(entries)
    if n < k * 2:
        raise HTTPException(
            status_code=422,
            detail=f"need at least {k * 2} non-expired entries to form {k} clusters (have {n})",
        )

    clusters = await asyncio.to_thread(_kmeans_cluster, entries, k)
    return {"k": k, "n_entries": n, "clusters": clusters}


def _kmeans_cluster(
    entries: list[tuple[str, np.ndarray, int]],
    k: int,
    max_iter: int = 20,
) -> list[dict[str, object]]:
    """Lloyd's k-means on L2-normalised embeddings. Pure numpy. Max 50 lines."""
    questions = [e[0] for e in entries]
    hit_counts = np.array([e[2] for e in entries], dtype=np.float32)

    # Stack + normalise all embeddings to unit sphere
    vecs = np.vstack([SemanticCache._l2_norm(e[1]) for e in entries])  # (n, dim)

    # Initialise centroids via k-means++ seeding
    rng = np.random.default_rng(seed=42)
    centroids = [vecs[rng.integers(len(vecs))].copy()]
    for _ in range(k - 1):
        dists = np.array([min(float(1 - np.dot(c, v)) for c in centroids) for v in vecs])
        dists = np.clip(dists, 0, None)
        total = dists.sum()
        probs = dists / total if total > 0 else np.ones(len(vecs)) / len(vecs)
        centroids.append(vecs[rng.choice(len(vecs), p=probs)].copy())
    centroids_arr = np.array(centroids)  # (k, dim)

    # Lloyd iterations
    labels = np.zeros(len(vecs), dtype=int)
    for _ in range(max_iter):
        sims = vecs @ centroids_arr.T  # (n, k) cosine similarities
        new_labels = np.argmax(sims, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for ci in range(k):
            mask = labels == ci
            if mask.any():
                c = vecs[mask].mean(axis=0)
                norm = np.linalg.norm(c)
                centroids_arr[ci] = c / norm if norm > 1e-10 else c

    # Build output — one dict per cluster
    result: list[dict[str, object]] = []
    for ci in range(k):
        mask = labels == ci
        if not mask.any():
            continue
        cluster_qs = [questions[i] for i in np.where(mask)[0]]
        cluster_hits = hit_counts[mask]
        member_sims = float((vecs[mask] @ centroids_arr[ci]).mean())
        result.append({
            "cluster_id": ci,
            "size": int(mask.sum()),
            "avg_hit_count": round(float(cluster_hits.mean()), 2),
            "avg_centroid_similarity": round(member_sims, 4),
            "representative_questions": cluster_qs[:5],
        })

    result.sort(key=lambda c: c["size"], reverse=True)
    return result


# ── Sprint 28: batch similarity search ────────────────────────────────────────


class SearchQuery(BaseModel):
    """Batch similarity search request."""

    queries: list[str] = Field(..., min_length=1, max_length=100)
    top_k: int = Field(default=3, ge=1, le=20)


@router.post("/search")
async def batch_search(body: SearchQuery) -> dict[str, object]:
    """Return top-k cached matches for each query in a single round-trip.

    Unlike ``lookup()``, this endpoint does *not* apply the similarity threshold —
    it returns the closest entries regardless of whether they would constitute a
    cache hit.  This makes it useful for exploration, analytics, and debugging
    ("what does the cache know about X?").

    Each element of the ``results`` list corresponds to the query at the same
    index and contains up to ``top_k`` matches sorted by similarity descending.
    """
    cache = _require_memory_cache()
    encoder = _get_encoder()

    questions = body.queries
    try:
        vecs: np.ndarray = await asyncio.to_thread(encoder.encode, questions)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"encoder error: {exc}") from exc

    results = []
    with cache._lock:  # type: ignore[attr-defined]
        lru_snapshot = {
            k: (e.question, e.response, e.hit_count)
            for k, e in cache._lru.items()  # type: ignore[attr-defined]
            if not e.is_expired()
        }

    for i, q in enumerate(questions):
        q_norm = SemanticCache._l2_norm(vecs[i : i + 1].astype(np.float32))
        scored = []
        for k2, (orig_q, resp, hits) in lru_snapshot.items():
            entry_vec = cache._lru.get(k2)  # type: ignore[attr-defined]
            if entry_vec is None:
                continue
            sim = float(np.dot(q_norm, SemanticCache._l2_norm(entry_vec.question_vec)))
            answer = resp.answer if hasattr(resp, "answer") else str(resp)
            scored.append({
                "question": orig_q,
                "answer":   answer[:256],
                "similarity": round(sim, 4),
                "hit_count":  hits,
            })
        scored.sort(key=lambda m: m["similarity"], reverse=True)
        results.append({"query_index": i, "query": q, "matches": scored[: body.top_k]})

    return {"results": results, "threshold": cache._threshold}  # type: ignore[attr-defined]


# ── Sprint 28: cache analytics ────────────────────────────────────────────────


def _get_or_create_buffer(cache: SemanticCache) -> LatencyBuffer:
    """Return the existing LatencyBuffer attached to *cache*, creating one if absent."""
    buf = cache._analytics_buf  # type: ignore[attr-defined]
    if buf is None:
        buf = LatencyBuffer()
        cache.set_analytics_buffer(buf)
    return buf  # type: ignore[return-value]


@router.get("/analytics")
async def cache_analytics(
    hours: float = Query(default=24.0, ge=0.1, le=720.0, description="Window in hours"),
) -> dict[str, object]:
    """Rich analytics for the last ``hours`` hours of cache activity.

    Returns latency percentiles (p50/p90/p99) for hits and misses separately,
    a similarity distribution histogram for hits, and an hourly hit-rate
    breakdown so you can spot time-of-day patterns.

    The analytics buffer is auto-created on first access.  Populate it by
    calling ``cache.record_access(latency_ms, is_hit, similarity)`` from the
    query route after each cache lookup.
    """
    cache = _require_memory_cache()
    buf = _get_or_create_buffer(cache)
    records = await asyncio.to_thread(buf.snapshot)
    result = await asyncio.to_thread(compute_analytics, records, hours)
    result["buffer_size"] = buf.size
    return result


# ── Sprint 28: adaptive TTL ───────────────────────────────────────────────────


@router.get("/ttl_report")
async def ttl_report() -> dict[str, object]:
    """Return a snapshot of current TTL distribution and adaptive-TTL candidates.

    Identifies entries that would be extended (hot entries, access rate > 5/day)
    or reduced (cold entries, not accessed in > 3 days) in the next adjustment
    cycle.  This is a read-only view — call ``POST /cache/ttl/adjust`` to act.

    Returns ``0`` counts and empty lists when ``ttl_seconds=0`` (no TTL configured).
    """
    cache = _require_memory_cache()
    report = await asyncio.to_thread(cache.ttl_report)
    return report


class TtlAdjustRequest(BaseModel):
    """Optional overrides for an adaptive-TTL adjustment cycle."""

    hot_threshold_per_day: float = Field(default=5.0, ge=0.1)
    cold_days: float = Field(default=3.0, ge=0.1)
    extend_factor: float = Field(default=2.0, ge=1.0, le=10.0)
    reduce_factor: float = Field(default=0.5, ge=0.01, le=1.0)
    min_ttl: int = Field(default=60, ge=1)
    max_ttl: int = Field(default=86400 * 7, ge=60)


@router.post("/ttl/adjust")
async def ttl_adjust(body: TtlAdjustRequest | None = None) -> dict[str, object]:
    """Trigger one adaptive-TTL adjustment cycle.

    Hot entries (access_rate_per_day > hot_threshold_per_day) have their TTL
    multiplied by extend_factor.  Cold entries (days_since_last_access > cold_days)
    have their TTL multiplied by reduce_factor.  Both are clamped to
    [min_ttl, max_ttl].  Entries with ``ttl_seconds == 0`` are not managed.

    Returns ``{"extended": n, "reduced": m}`` — the number of entries modified.
    """
    cache = _require_memory_cache()
    params = body or TtlAdjustRequest()
    result = await asyncio.to_thread(
        cache.adjust_ttls,
        params.hot_threshold_per_day,
        params.cold_days,
        params.extend_factor,
        params.reduce_factor,
        params.min_ttl,
        params.max_ttl,
    )
    return result
