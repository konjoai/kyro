"""Sprint 27 — cache warming, TTL expiry, and query clustering tests."""
from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from konjoai.api.routes.cache import _kmeans_cluster
from konjoai.api.routes.cache import router as cache_router
from konjoai.cache.semantic_cache import SemanticCache, SemanticCacheEntry

# ── Helpers ─────────────────────────────────────────────────────────────────


def _vec(seed: int, dim: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / (np.linalg.norm(v) + 1e-10)).reshape(1, dim).astype(np.float32)


@dataclass
class _Resp:
    answer: str


def _filled_cache(n: int, threshold: float = 0.95, ttl: int = 0) -> SemanticCache:
    cache = SemanticCache(max_size=200, threshold=threshold, ttl_seconds=ttl)
    for i in range(n):
        cache.store(f"question {i}", _vec(i), _Resp(f"answer {i}"))
    return cache


@dataclass
class _Settings:
    cache_enabled: bool = True
    cache_warm_max_batch: int = 500
    cache_max_size: int = 200
    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 0


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(cache_router)
    return app


# ── TTL — SemanticCacheEntry ─────────────────────────────────────────────────


class TestSemanticCacheEntryTTL:
    def test_no_ttl_never_expires(self) -> None:
        e = SemanticCacheEntry(
            question="q", question_vec=_vec(0), response=_Resp("a"), ttl_seconds=0
        )
        assert e.is_expired() is False

    def test_future_ttl_not_expired(self) -> None:
        e = SemanticCacheEntry(
            question="q", question_vec=_vec(0), response=_Resp("a"), ttl_seconds=3600
        )
        assert e.is_expired() is False

    def test_past_ttl_is_expired(self) -> None:
        e = SemanticCacheEntry(
            question="q", question_vec=_vec(0), response=_Resp("a"),
            created_at=time.monotonic() - 100, ttl_seconds=10,
        )
        assert e.is_expired() is True


# ── TTL — SemanticCache ───────────────────────────────────────────────────────


class TestSemanticCacheTTL:
    def test_constructor_rejects_negative_ttl(self) -> None:
        with pytest.raises(ValueError):
            SemanticCache(max_size=10, threshold=0.95, ttl_seconds=-1)

    def test_ttl_zero_means_no_expiry(self) -> None:
        cache = SemanticCache(max_size=10, threshold=0.95, ttl_seconds=0)
        v = _vec(1)
        cache.store("hello", v, _Resp("world"))
        assert cache.lookup("hello", v) is not None

    def test_expired_entry_skipped_on_lookup(self) -> None:
        cache = SemanticCache(max_size=10, threshold=0.95, ttl_seconds=1)
        v = _vec(2)
        # Manually insert an already-expired entry
        key = SemanticCache._normalise("stale question")
        entry = SemanticCacheEntry(
            question="stale question", question_vec=v,
            response=_Resp("old answer"),
            created_at=time.monotonic() - 200,
            ttl_seconds=1,
        )
        with cache._lock:
            cache._lru[key] = entry
            cache._exact[key] = entry
        assert cache.lookup("stale question", v) is None

    def test_exact_match_expired_evicted_lazily(self) -> None:
        cache = SemanticCache(max_size=10, threshold=0.95, ttl_seconds=1)
        v = _vec(3)
        key = SemanticCache._normalise("timed out")
        entry = SemanticCacheEntry(
            question="timed out", question_vec=v,
            response=_Resp("stale"),
            created_at=time.monotonic() - 200,
            ttl_seconds=1,
        )
        with cache._lock:
            cache._lru[key] = entry
            cache._exact[key] = entry
        # Lookup triggers lazy eviction
        assert cache.lookup("timed out", v) is None
        with cache._lock:
            assert key not in cache._lru

    def test_expired_count_zero_when_ttl_disabled(self) -> None:
        cache = _filled_cache(5, ttl=0)
        assert cache.expired_count() == 0

    def test_expired_count_reflects_stale_entries(self) -> None:
        cache = SemanticCache(max_size=20, threshold=0.95, ttl_seconds=10)
        # Insert 3 live entries
        for i in range(3):
            cache.store(f"live {i}", _vec(i), _Resp("ok"))
        # Insert 2 already-expired entries directly
        for i in range(3, 5):
            key = SemanticCache._normalise(f"stale {i}")
            entry = SemanticCacheEntry(
                question=f"stale {i}", question_vec=_vec(i),
                response=_Resp("old"),
                created_at=time.monotonic() - 200,
                ttl_seconds=10,
            )
            with cache._lock:
                cache._lru[key] = entry
                cache._exact[key] = entry
        assert cache.expired_count() == 2

    def test_evict_expired_removes_stale_entries(self) -> None:
        cache = SemanticCache(max_size=20, threshold=0.95, ttl_seconds=10)
        cache.store("fresh", _vec(0), _Resp("ok"))
        key = SemanticCache._normalise("expired")
        entry = SemanticCacheEntry(
            question="expired", question_vec=_vec(1),
            response=_Resp("stale"),
            created_at=time.monotonic() - 200,
            ttl_seconds=10,
        )
        with cache._lock:
            cache._lru[key] = entry
            cache._exact[key] = entry
        removed = cache.evict_expired()
        assert removed == 1
        assert cache.expired_count() == 0
        with cache._lock:
            assert "fresh" in str(list(cache._exact.keys()))

    def test_evict_expired_noop_when_ttl_zero(self) -> None:
        cache = _filled_cache(5, ttl=0)
        assert cache.evict_expired() == 0

    def test_stats_includes_ttl_and_expired_count(self) -> None:
        cache = SemanticCache(max_size=10, threshold=0.95, ttl_seconds=60)
        s = cache.stats()
        assert "ttl_seconds" in s
        assert s["ttl_seconds"] == 60
        assert "expired_count" in s


# ── TTL — API routes ──────────────────────────────────────────────────────────


class TestTTLRoutes:
    def _client_with_cache(self, cache: SemanticCache) -> TestClient:
        app = _make_app()
        client = TestClient(app)
        return client, cache

    def test_expired_count_returns_zero_when_no_ttl(self) -> None:
        cache = _filled_cache(5)
        app = _make_app()
        client = TestClient(app)
        with (
            patch("konjoai.api.routes.cache.get_settings", return_value=_Settings()),
            patch("konjoai.api.routes.cache.get_semantic_cache", return_value=cache),
        ):
            resp = client.get("/cache/expired_count")
        assert resp.status_code == 200
        assert resp.json()["expired_count"] == 0

    def test_expired_count_reflects_stale(self) -> None:
        cache = SemanticCache(max_size=20, threshold=0.95, ttl_seconds=10)
        key = SemanticCache._normalise("stale")
        entry = SemanticCacheEntry(
            question="stale", question_vec=_vec(0),
            response=_Resp("old"),
            created_at=time.monotonic() - 200,
            ttl_seconds=10,
        )
        with cache._lock:
            cache._lru[key] = entry
            cache._exact[key] = entry
        app = _make_app()
        client = TestClient(app)
        with (
            patch("konjoai.api.routes.cache.get_settings", return_value=_Settings()),
            patch("konjoai.api.routes.cache.get_semantic_cache", return_value=cache),
        ):
            resp = client.get("/cache/expired_count")
        assert resp.status_code == 200
        assert resp.json()["expired_count"] == 1

    def test_delete_expired_evicts_and_reports_count(self) -> None:
        cache = SemanticCache(max_size=20, threshold=0.95, ttl_seconds=10)
        for i in range(2):
            key = SemanticCache._normalise(f"stale {i}")
            entry = SemanticCacheEntry(
                question=f"stale {i}", question_vec=_vec(i),
                response=_Resp("old"),
                created_at=time.monotonic() - 200,
                ttl_seconds=10,
            )
            with cache._lock:
                cache._lru[key] = entry
                cache._exact[key] = entry
        app = _make_app()
        client = TestClient(app)
        with (
            patch("konjoai.api.routes.cache.get_settings", return_value=_Settings()),
            patch("konjoai.api.routes.cache.get_semantic_cache", return_value=cache),
        ):
            resp = client.delete("/cache/expired")
        assert resp.status_code == 200
        assert resp.json()["evicted"] == 2

    def test_cache_disabled_returns_404(self) -> None:
        app = _make_app()
        client = TestClient(app)
        with patch("konjoai.api.routes.cache.get_settings",
                   return_value=_Settings(cache_enabled=False)):
            r1 = client.get("/cache/expired_count")
            r2 = client.delete("/cache/expired")
        assert r1.status_code == 404
        assert r2.status_code == 404


# ── Cache warming — route ─────────────────────────────────────────────────────


def _stub_encoder(questions: list[str]) -> np.ndarray:
    """Deterministic per-question embeddings for test isolation."""
    dim = 32
    out = np.zeros((len(questions), dim), dtype=np.float32)
    for i, q in enumerate(questions):
        rng = np.random.default_rng(seed=abs(hash(q)) % (2 ** 31))
        v = rng.standard_normal(dim).astype(np.float32)
        out[i] = v / (np.linalg.norm(v) + 1e-10)
    return out


class TestCacheWarmRoute:
    def _patched_client(self, cache: SemanticCache) -> TestClient:
        app = _make_app()
        client = TestClient(app)
        return client

    def test_warm_stores_pairs(self) -> None:
        cache = SemanticCache(max_size=100, threshold=0.95)
        app = _make_app()
        client = TestClient(app)
        pairs = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(5)]
        with (
            patch("konjoai.api.routes.cache.get_settings", return_value=_Settings()),
            patch("konjoai.api.routes.cache.get_semantic_cache", return_value=cache),
            patch("konjoai.api.routes.cache._get_encoder") as mock_enc,
        ):
            mock_enc.return_value.encode = _stub_encoder
            resp = client.post("/cache/warm", json={"pairs": pairs})
        assert resp.status_code == 200
        body = resp.json()
        assert body["warmed"] == 5
        assert body["skipped_duplicates"] == 0
        assert body["skipped_errors"] == 0
        assert body["total_submitted"] == 5
        assert len(cache._lru) == 5

    def test_warm_skips_duplicates(self) -> None:
        cache = SemanticCache(max_size=100, threshold=0.95)
        # Pre-populate one entry
        cache.store("q0", _stub_encoder(["q0"]), _Resp("existing"))
        pairs = [{"question": "q0", "answer": "new"}, {"question": "q1", "answer": "a1"}]
        app = _make_app()
        client = TestClient(app)
        with (
            patch("konjoai.api.routes.cache.get_settings", return_value=_Settings()),
            patch("konjoai.api.routes.cache.get_semantic_cache", return_value=cache),
            patch("konjoai.api.routes.cache._get_encoder") as mock_enc,
        ):
            mock_enc.return_value.encode = _stub_encoder
            resp = client.post("/cache/warm", json={"pairs": pairs})
        assert resp.status_code == 200
        body = resp.json()
        assert body["warmed"] == 1
        assert body["skipped_duplicates"] == 1

    def test_warm_rejects_batch_exceeding_limit(self) -> None:
        cache = SemanticCache(max_size=100, threshold=0.95)
        pairs = [{"question": f"q{i}", "answer": "a"} for i in range(10)]
        settings = _Settings(cache_warm_max_batch=5)
        app = _make_app()
        client = TestClient(app)
        with (
            patch("konjoai.api.routes.cache.get_settings", return_value=settings),
            patch("konjoai.api.routes.cache.get_semantic_cache", return_value=cache),
        ):
            resp = client.post("/cache/warm", json={"pairs": pairs})
        assert resp.status_code == 422
        assert "cache_warm_max_batch" in resp.json()["detail"]

    def test_warm_returns_404_when_cache_disabled(self) -> None:
        app = _make_app()
        client = TestClient(app)
        with patch("konjoai.api.routes.cache.get_settings",
                   return_value=_Settings(cache_enabled=False)):
            resp = client.post("/cache/warm", json={"pairs": [{"question": "x", "answer": "y"}]})
        assert resp.status_code == 404

    def test_warm_validates_empty_pairs_rejected(self) -> None:
        app = _make_app()
        client = TestClient(app)
        with patch("konjoai.api.routes.cache.get_settings", return_value=_Settings()):
            resp = client.post("/cache/warm", json={"pairs": []})
        assert resp.status_code == 422

    def test_warm_validates_empty_question_rejected(self) -> None:
        app = _make_app()
        client = TestClient(app)
        with patch("konjoai.api.routes.cache.get_settings", return_value=_Settings()):
            resp = client.post("/cache/warm", json={"pairs": [{"question": "", "answer": "a"}]})
        assert resp.status_code == 422


# ── Query clustering ──────────────────────────────────────────────────────────


class TestKmeansCluster:
    def _entries(self, n: int) -> list[tuple[str, np.ndarray, int]]:
        return [(f"q{i}", _vec(i), i % 3) for i in range(n)]

    def test_returns_k_clusters(self) -> None:
        entries = self._entries(20)
        result = _kmeans_cluster(entries, k=3)
        assert len(result) == 3

    def test_all_entries_assigned(self) -> None:
        entries = self._entries(30)
        result = _kmeans_cluster(entries, k=4)
        total = sum(c["size"] for c in result)
        assert total == 30

    def test_representative_questions_capped_at_5(self) -> None:
        entries = self._entries(50)
        result = _kmeans_cluster(entries, k=3)
        for cluster in result:
            assert len(cluster["representative_questions"]) <= 5

    def test_sorted_by_size_descending(self) -> None:
        entries = self._entries(40)
        result = _kmeans_cluster(entries, k=5)
        sizes = [c["size"] for c in result]
        assert sizes == sorted(sizes, reverse=True)

    def test_avg_centroid_similarity_in_range(self) -> None:
        entries = self._entries(30)
        result = _kmeans_cluster(entries, k=3)
        for cluster in result:
            assert 0.0 <= cluster["avg_centroid_similarity"] <= 1.0 + 1e-5

    def test_cluster_ids_are_present(self) -> None:
        entries = self._entries(20)
        result = _kmeans_cluster(entries, k=3)
        assert all("cluster_id" in c for c in result)


class TestClusterRoute:
    def test_cluster_route_returns_clusters(self) -> None:
        cache = _filled_cache(30)
        app = _make_app()
        client = TestClient(app)
        with (
            patch("konjoai.api.routes.cache.get_settings", return_value=_Settings()),
            patch("konjoai.api.routes.cache.get_semantic_cache", return_value=cache),
        ):
            resp = client.get("/cache/clusters?k=3")
        assert resp.status_code == 200
        body = resp.json()
        assert body["k"] == 3
        assert body["n_entries"] == 30
        assert len(body["clusters"]) == 3

    def test_cluster_route_422_when_too_few_entries(self) -> None:
        cache = _filled_cache(5)
        app = _make_app()
        client = TestClient(app)
        with (
            patch("konjoai.api.routes.cache.get_settings", return_value=_Settings()),
            patch("konjoai.api.routes.cache.get_semantic_cache", return_value=cache),
        ):
            resp = client.get("/cache/clusters?k=5")
        assert resp.status_code == 422
        assert "entries" in resp.json()["detail"]

    def test_cluster_route_rejects_k_out_of_range(self) -> None:
        cache = _filled_cache(100)
        app = _make_app()
        client = TestClient(app)
        with (
            patch("konjoai.api.routes.cache.get_settings", return_value=_Settings()),
            patch("konjoai.api.routes.cache.get_semantic_cache", return_value=cache),
        ):
            r1 = client.get("/cache/clusters?k=1")
            r2 = client.get("/cache/clusters?k=21")
        assert r1.status_code == 422
        assert r2.status_code == 422

    def test_cluster_route_404_when_disabled(self) -> None:
        app = _make_app()
        client = TestClient(app)
        with patch("konjoai.api.routes.cache.get_settings",
                   return_value=_Settings(cache_enabled=False)):
            resp = client.get("/cache/clusters?k=3")
        assert resp.status_code == 404
