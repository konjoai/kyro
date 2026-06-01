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
from konjoai.cache.federation import FederatedLookup, PeerRegistry, get_federated_lookup, get_peer_registry
from konjoai.cache.poisoning import get_poisoning_report_store
from konjoai.cache.rewriter import QueryRewriter, get_rewriter
from konjoai.cache.semantic_cache import SemanticCache, get_semantic_cache
from konjoai.cache.suspicious import get_flag_store, scan_for_suspicious
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


# ── Poisoning guard routes (Sprint 28, from parallel branch) ─────────────────


def _require_poisoning_guard_enabled() -> None:
    """Raise HTTP 404 when the poisoning guard is not enabled (K3)."""
    settings = get_settings()
    if not settings.cache_enabled or not settings.cache_poisoning_guard_enabled:
        raise HTTPException(
            status_code=404,
            detail="cache poisoning guard is not enabled (set CACHE_POISONING_GUARD_ENABLED=true)",
        )


class ReportPoisoningRequest(BaseModel):
    """Body for a manual cache-poisoning report."""

    question_hash: str = Field(
        ..., min_length=8, max_length=64,
        description="16-hex SHA-256 prefix of the question (OWASP — no raw text).",
    )
    reason: str = Field(..., min_length=1, max_length=256)
    tenant_id: str | None = Field(None)


class ReportPoisoningResponse(BaseModel):
    """Confirmation of a poisoning report submission."""

    recorded: bool
    report_hash: str


@router.post("/report_poisoning", status_code=201, response_model=ReportPoisoningResponse)
async def report_poisoning(body: ReportPoisoningRequest) -> ReportPoisoningResponse:
    """Manually report a suspected semantic cache poisoning event.

    Appends a ``PoisoningReport`` to the in-process report ring buffer.
    The ``question_hash`` field must be the 16-hex SHA-256 prefix emitted by
    the query audit log — raw question text is rejected (OWASP).

    Requires ``cache_enabled=True`` **and** ``cache_poisoning_guard_enabled=True``.
    Returns HTTP 404 when either flag is off (K3).
    """
    _require_poisoning_guard_enabled()
    tenant = body.tenant_id or "anonymous"
    store = get_poisoning_report_store()
    await asyncio.to_thread(store.record, tenant, body.question_hash, body.reason)
    report_hash = hashlib.sha256(
        f"{tenant}:{body.question_hash}:{body.reason}".encode()
    ).hexdigest()[:16]
    return ReportPoisoningResponse(recorded=True, report_hash=report_hash)


@router.get("/poisoning_reports")
async def poisoning_reports(
    tenant_id: str | None = Query(default=None, description="Filter by tenant ID."),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, object]:
    """Return recent cache poisoning reports (oldest-first, filtered by tenant).

    Requires ``cache_enabled=True`` **and** ``cache_poisoning_guard_enabled=True``.
    """
    _require_poisoning_guard_enabled()
    store = get_poisoning_report_store()
    reports = await asyncio.to_thread(store.query, tenant_id=tenant_id, limit=limit)
    return {
        "count": len(reports),
        "reports": [
            {"tenant_id": r.tenant_id, "question_hash": r.question_hash,
             "reason": r.reason, "timestamp": r.timestamp}
            for r in reports
        ],
    }


# ── Sprint 29: query rewriting ────────────────────────────────────────────────


class RewriteRequest(BaseModel):
    """Preview a rewrite without touching the cache."""

    question: str = Field(..., min_length=1, max_length=2048)


@router.post("/rewrite")
async def preview_rewrite(body: RewriteRequest) -> dict[str, object]:
    """Preview how query rewriting would normalise a question.

    Returns the original question, the rewritten form, and a per-step trace
    showing exactly what each transformation changed.  Does not modify the
    cache.

    Enabled regardless of ``cache_query_rewrite_enabled`` — always useful for
    debugging.  The rewrite pipeline used is the one configured by
    ``cache_query_rewrite_steps``.
    """
    rewriter: QueryRewriter = await asyncio.to_thread(get_rewriter)
    result = await asyncio.to_thread(rewriter.explain, body.question)
    return {
        "original":  result.original,
        "rewritten": result.rewritten,
        "changed":   result.changed,
        "steps": [
            {"name": s.name, "before": s.before, "after": s.after, "changed": s.changed}
            for s in result.steps
        ],
    }


@router.get("/rewrite/config")
async def rewrite_config() -> dict[str, object]:
    """Return the active query-rewrite configuration."""
    settings = get_settings()
    return {
        "enabled":    getattr(settings, "cache_query_rewrite_enabled", False),
        "steps":      getattr(settings, "cache_query_rewrite_steps", []),
        "step_names": (await asyncio.to_thread(get_rewriter)).step_names,
    }


# ── Sprint 29: suspicious entry detection ─────────────────────────────────────


@router.get("/suspicious")
async def list_suspicious(
    k: int = Query(default=5, ge=2, le=20, description="Cluster count for outlier detection"),
    z: float = Query(default=2.0, ge=1.0, le=5.0, description="Outlier threshold in standard deviations"),
    auto_flag: bool = Query(default=True, description="Automatically save detected entries to the flag store"),
) -> dict[str, object]:
    """Scan the cache for suspicious entries using three detection signals.

    **Embedding outlier** — entries whose vector is ≥ z σ from its cluster
    centroid, suggesting injection of off-topic content.

    **Hit-count anomaly** — entries hit far more than typical, which an
    adversary could use to keep a poisoned answer "warm" in the LRU.

    **Answer length anomaly** — entries with answers ≥ z σ away from the
    mean length (extremely short or extremely long answers).

    Requires at least 2×k non-expired cache entries.  Returns ``[]`` otherwise.
    When ``auto_flag=True`` (default), found entries are recorded in the flag
    store so they can be reviewed via ``POST /cache/suspicious/{hash}/approve``
    or ``/reject``.
    """
    cache = _require_memory_cache()
    findings = await asyncio.to_thread(scan_for_suspicious, cache, k, z)
    if auto_flag and findings:
        store = get_flag_store()
        for f in findings:
            await asyncio.to_thread(
                store.flag,
                f["entry_hash"], f["question"], f["reason"], f["score"], f["signal"],
            )
    return {"count": len(findings), "suspicious": findings}


@router.get("/suspicious/flagged")
async def list_flagged() -> dict[str, object]:
    """Return all entries currently in the flag store (all statuses)."""
    store = get_flag_store()
    flags = await asyncio.to_thread(store.all_flags)
    return {
        "count": len(flags),
        "flags": [
            {
                "entry_hash": f.entry_hash,
                "question":   f.question,
                "reason":     f.reason,
                "score":      f.score,
                "signal":     f.signal,
                "status":     f.status,
            }
            for f in flags
        ],
    }


@router.post("/suspicious/{entry_hash}/approve")
async def approve_suspicious(entry_hash: str) -> dict[str, object]:
    """Mark a flagged entry as safe (approved).

    The entry remains in the cache.  Its flag status changes to ``"approved"``
    so it will not be highlighted in future scans.
    """
    store = get_flag_store()
    ok = await asyncio.to_thread(store.resolve, entry_hash, "approve")
    if not ok:
        raise HTTPException(status_code=404, detail=f"no flag found for hash {entry_hash!r}")
    return {"entry_hash": entry_hash, "status": "approved"}


@router.post("/suspicious/{entry_hash}/reject")
async def reject_suspicious(entry_hash: str) -> dict[str, object]:
    """Reject (evict) a flagged suspicious entry from the cache.

    The entry is removed from the live ``SemanticCache`` and the flag is
    marked ``"rejected"``.  If the entry has already been evicted (e.g. by
    LRU), the flag is still updated without error.
    """
    store = get_flag_store()
    ok = await asyncio.to_thread(store.resolve, entry_hash, "reject")
    if not ok:
        raise HTTPException(status_code=404, detail=f"no flag found for hash {entry_hash!r}")
    # Best-effort eviction from live cache (entry may already be gone)
    try:
        cache = _require_memory_cache()
        with cache._lock:  # type: ignore[attr-defined]
            # The flag store uses normalised keys; scan exact dict for match
            to_delete = [k for k in cache._exact  # type: ignore[attr-defined]
                         if hashlib.sha256(k.encode()).hexdigest()[:16] == entry_hash]
        for k in to_delete:
            with cache._lock:  # type: ignore[attr-defined]
                cache._lru.pop(k, None)  # type: ignore[attr-defined]
                cache._exact.pop(k, None)  # type: ignore[attr-defined]
    except HTTPException:
        pass  # cache disabled — eviction is a no-op
    return {"entry_hash": entry_hash, "status": "rejected"}


# ── Sprint 29: cache federation ───────────────────────────────────────────────


class RegisterPeerRequest(BaseModel):
    """Body for registering a federation peer."""

    url: str = Field(..., min_length=7, description="Base URL of the peer kyro instance (no trailing slash).")
    name: str = Field(default="", max_length=64, description="Human-readable label for the peer.")
    auth_token: str | None = Field(default=None, description="Bearer token sent to the peer on lookup.")


@router.post("/federate", status_code=201)
async def register_peer(body: RegisterPeerRequest) -> dict[str, object]:
    """Register a peer kyro instance for federated cache sharing.

    On a cache miss the local instance can query healthy peers (via
    ``POST /cache/search``) before invoking the LLM.  Enable federation with
    ``CACHE_FEDERATION_ENABLED=true``.

    Returns the assigned ``peer_id`` which can be used to deregister or
    inspect the peer later.
    """
    registry = get_peer_registry()
    node = await asyncio.to_thread(registry.register, body.url, name=body.name, auth_token=body.auth_token)
    return {
        "peer_id":   node.peer_id,
        "url":       node.url,
        "name":      node.name,
        "registered_at": node.registered_at,
    }


@router.get("/peers")
async def list_peers() -> dict[str, object]:
    """Return all registered federation peers with availability + hit-contribution stats."""
    lookup = get_federated_lookup()
    status = await asyncio.to_thread(lookup.peer_status)
    return {"count": len(status), "peers": status}


@router.delete("/peers/{peer_id}")
async def remove_peer(peer_id: str) -> dict[str, object]:
    """Deregister a federation peer."""
    registry = get_peer_registry()
    removed = await asyncio.to_thread(registry.remove, peer_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"peer {peer_id!r} not found")
    return {"peer_id": peer_id, "removed": True}


@router.post("/peers/health_check")
async def check_peers_health(
    timeout: float = Query(default=3.0, ge=0.1, le=30.0),
) -> dict[str, object]:
    """Trigger a synchronous health check against all registered peers.

    Returns a dict mapping peer_id → healthy (bool).  Also updates each
    peer's availability score used by the federated lookup logic.
    """
    registry = get_peer_registry()
    results = await asyncio.to_thread(registry.check_all_health, timeout)
    return {"results": results, "checked": len(results)}
