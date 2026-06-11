"""Sprint 28 — batch search, analytics buffer, and adaptive TTL tests."""
from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from konjoai.api.routes.cache import router as cache_router
from konjoai.cache.analytics import AccessRecord, LatencyBuffer, compute_analytics
from konjoai.cache.semantic_cache import SemanticCache, SemanticCacheEntry

# ── Helpers ──────────────────────────────────────────────────────────────────


def _vec(seed: int, dim: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / (np.linalg.norm(v) + 1e-10)).reshape(1, dim).astype(np.float32)


@dataclass
class _Resp:
    answer: str


@dataclass
class _Settings:
    cache_enabled: bool = True
    cache_warm_max_batch: int = 500
    cache_max_size: int = 200
    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 0


def _filled_cache(n: int, threshold: float = 0.95, ttl: int = 0) -> SemanticCache:
    cache = SemanticCache(max_size=200, threshold=threshold, ttl_seconds=ttl)
    for i in range(n):
        cache.store(f"question {i}", _vec(i), _Resp(f"answer {i}"))
    return cache


def _stub_encode(questions: list[str]) -> np.ndarray:
    dim = 32
    out = np.zeros((len(questions), dim), dtype=np.float32)
    for i, q in enumerate(questions):
        rng = np.random.default_rng(seed=abs(hash(q)) % (2 ** 31))
        v = rng.standard_normal(dim).astype(np.float32)
        out[i] = v / (np.linalg.norm(v) + 1e-10)
    return out


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(cache_router)
    return app


# ── LatencyBuffer ─────────────────────────────────────────────────────────────


class TestLatencyBuffer:
    def test_empty_snapshot(self) -> None:
        buf = LatencyBuffer()
        assert buf.snapshot() == []
        assert buf.size == 0

    def test_record_and_retrieve(self) -> None:
        buf = LatencyBuffer()
        buf.record(5.0, True, 0.97)
        buf.record(200.0, False, 0.0)
        snap = buf.snapshot()
        assert len(snap) == 2
        assert snap[0].is_hit is True
        assert snap[0].latency_ms == 5.0
        assert snap[0].similarity == 0.97
        assert snap[1].is_hit is False

    def test_bounded_at_max_records(self) -> None:
        buf = LatencyBuffer(max_records=5)
        for i in range(10):
            buf.record(float(i), True)
        assert buf.size == 5

    def test_oldest_entries_evicted_first(self) -> None:
        buf = LatencyBuffer(max_records=3)
        for i in range(5):
            buf.record(float(i), True)
        snap = buf.snapshot()
        assert snap[0].latency_ms == 2.0  # oldest kept = index 2

    def test_clear_empties_buffer(self) -> None:
        buf = LatencyBuffer()
        buf.record(1.0, True)
        buf.clear()
        assert buf.size == 0


# ── compute_analytics ─────────────────────────────────────────────────────────


class TestComputeAnalytics:
    def _records(self, n_hits: int, n_misses: int, age_secs: float = 0.0) -> list[AccessRecord]:
        ts = time.monotonic() - age_secs
        hits  = [AccessRecord(ts, 5.0 + i, True,  0.90 + 0.01*i) for i in range(n_hits)]
        misses= [AccessRecord(ts, 300.0,   False, 0.0)  for _ in range(n_misses)]
        return hits + misses

    def test_empty_records_returns_zeros(self) -> None:
        result = compute_analytics([])
        assert result["total_accesses"] == 0
        assert result["hit_rate"] == 0.0

    def test_hit_rate_correct(self) -> None:
        records = self._records(6, 4)
        result = compute_analytics(records)
        assert result["total_accesses"] == 10
        assert result["hit_count"] == 6
        assert result["miss_count"] == 4
        assert result["hit_rate"] == pytest.approx(0.6)

    def test_latency_percentiles_present(self) -> None:
        records = self._records(5, 0)
        result = compute_analytics(records)
        for key in ("p50", "p90", "p99", "mean", "min", "max"):
            assert key in result["latency"]["hits"]
            assert key in result["latency"]["all"]

    def test_old_records_excluded_by_window(self) -> None:
        old = [AccessRecord(time.monotonic() - 7300, 5.0, True, 0.9)]
        fresh = [AccessRecord(time.monotonic() - 10, 10.0, False, 0.0)]
        result = compute_analytics(old + fresh, hours=1.0)
        assert result["total_accesses"] == 1
        assert result["hit_count"] == 0

    def test_similarity_distribution_has_5_buckets(self) -> None:
        records = self._records(10, 0)
        result = compute_analytics(records)
        assert len(result["similarity_distribution"]) == 5

    def test_hourly_hit_rate_present(self) -> None:
        records = self._records(5, 5)
        result = compute_analytics(records, hours=4.0)
        assert "hourly_hit_rate" in result
        assert isinstance(result["hourly_hit_rate"], list)


# ── SemanticCache analytics methods ──────────────────────────────────────────


class TestSemanticCacheAnalytics:
    def test_record_access_noop_without_buffer(self) -> None:
        cache = _filled_cache(5)
        cache.record_access(5.0, True, 0.95)  # must not raise

    def test_record_access_populates_buffer(self) -> None:
        cache = _filled_cache(5)
        buf = LatencyBuffer()
        cache.set_analytics_buffer(buf)
        cache.record_access(5.0, True, 0.95)
        cache.record_access(300.0, False)
        assert buf.size == 2

    def test_analytics_snapshot_empty_without_buffer(self) -> None:
        cache = _filled_cache(5)
        assert cache.analytics_snapshot() == []

    def test_analytics_snapshot_returns_buffer_contents(self) -> None:
        cache = _filled_cache(5)
        buf = LatencyBuffer()
        cache.set_analytics_buffer(buf)
        buf.record(8.0, True, 0.91)
        snap = cache.analytics_snapshot()
        assert len(snap) == 1
        assert snap[0].latency_ms == 8.0


# ── top_k_similar ─────────────────────────────────────────────────────────────


class TestTopKSimilar:
    def test_returns_at_most_k_results(self) -> None:
        cache = _filled_cache(20)
        results = cache.top_k_similar(_vec(0), k=5)
        assert len(results) <= 5

    def test_results_sorted_by_similarity_descending(self) -> None:
        cache = _filled_cache(20)
        results = cache.top_k_similar(_vec(0), k=10)
        sims = [r[1] for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_each_result_has_key_sim_hits(self) -> None:
        cache = _filled_cache(10)
        for key, sim, hits in cache.top_k_similar(_vec(3), k=3):
            assert isinstance(key, str)
            assert -1.0 <= sim <= 1.0 + 1e-5
            assert isinstance(hits, int)

    def test_exact_match_is_top_result(self) -> None:
        cache = SemanticCache(max_size=50, threshold=0.95)
        v = _vec(99)
        cache.store("unique question", v, _Resp("answer"))
        results = cache.top_k_similar(v, k=3)
        assert results[0][0] == SemanticCache._normalise("unique question")
        assert results[0][1] == pytest.approx(1.0, abs=1e-4)

    def test_excludes_expired_entries(self) -> None:
        cache = SemanticCache(max_size=20, threshold=0.95, ttl_seconds=10)
        v = _vec(7)
        key = SemanticCache._normalise("stale")
        entry = SemanticCacheEntry(
            question="stale", question_vec=v, response=_Resp("old"),
            created_at=time.monotonic() - 200, ttl_seconds=10,
        )
        with cache._lock:
            cache._lru[key] = entry
            cache._exact[key] = entry
        results = cache.top_k_similar(v, k=5)
        assert all(k2 != key for k2, *_ in results)


# ── Adaptive TTL ──────────────────────────────────────────────────────────────


class TestAdaptiveTTL:
    def _hot_entry(self, question: str, vec: np.ndarray, ttl: int = 3600) -> SemanticCacheEntry:
        e = SemanticCacheEntry(
            question=question, question_vec=vec, response=_Resp("ans"),
            created_at=time.monotonic() - 3600,  # 1 hour old
            hit_count=20,                          # 20 hits in 1 hour = 480/day
            ttl_seconds=ttl,
        )
        return e

    def _cold_entry(self, question: str, vec: np.ndarray, ttl: int = 3600) -> SemanticCacheEntry:
        return SemanticCacheEntry(
            question=question, question_vec=vec, response=_Resp("ans"),
            created_at=time.monotonic() - 86400 * 4,  # 4 days old, never hit
            hit_count=0, ttl_seconds=ttl,
            last_accessed=time.monotonic() - 86400 * 4,
        )

    def test_hot_entries_get_extended(self) -> None:
        cache = SemanticCache(max_size=20, threshold=0.95, ttl_seconds=3600)
        key = SemanticCache._normalise("hot q")
        entry = self._hot_entry("hot q", _vec(0))
        with cache._lock:
            cache._lru[key] = entry
            cache._exact[key] = entry
        result = cache.adjust_ttls(hot_threshold_per_day=5.0, extend_factor=2.0)
        assert result["extended"] >= 1
        assert entry.ttl_seconds == 7200

    def test_cold_entries_get_reduced(self) -> None:
        cache = SemanticCache(max_size=20, threshold=0.95, ttl_seconds=3600)
        key = SemanticCache._normalise("cold q")
        entry = self._cold_entry("cold q", _vec(1))
        with cache._lock:
            cache._lru[key] = entry
            cache._exact[key] = entry
        result = cache.adjust_ttls(cold_days=3.0, reduce_factor=0.5)
        assert result["reduced"] >= 1
        assert entry.ttl_seconds == 1800

    def test_no_ttl_entries_skipped(self) -> None:
        cache = _filled_cache(5, ttl=0)  # no TTL entries
        result = cache.adjust_ttls()
        assert result["extended"] == 0
        assert result["reduced"] == 0

    def test_ttl_clamped_to_min_max(self) -> None:
        cache = SemanticCache(max_size=20, threshold=0.95)
        key = SemanticCache._normalise("q")
        entry = SemanticCacheEntry(
            question="q", question_vec=_vec(0), response=_Resp("a"),
            created_at=time.monotonic() - 3600, hit_count=100, ttl_seconds=30,
        )
        with cache._lock:
            cache._lru[key] = entry
            cache._exact[key] = entry
        cache.adjust_ttls(hot_threshold_per_day=1.0, extend_factor=100.0, max_ttl=120)
        assert entry.ttl_seconds == 120

    def test_ttl_report_no_ttl_entries(self) -> None:
        cache = _filled_cache(10, ttl=0)
        report = cache.ttl_report()
        assert report["no_ttl"] == 10
        assert report["with_ttl"] == 0

    def test_ttl_report_buckets_sum_to_with_ttl(self) -> None:
        cache = _filled_cache(10, ttl=3600)
        report = cache.ttl_report()
        assert sum(b["count"] for b in report["buckets"]) == report["with_ttl"]

    def test_entry_last_accessed_updates_on_hit(self) -> None:
        cache = SemanticCache(max_size=10, threshold=0.95)
        v = _vec(0)
        cache.store("q", v, _Resp("a"))
        key = SemanticCache._normalise("q")
        before = cache._lru[key].last_accessed
        time.sleep(0.01)
        cache.lookup("q", v)
        after = cache._lru[key].last_accessed
        assert after > before


# ── Route tests ───────────────────────────────────────────────────────────────


class TestBatchSearchRoute:
    def test_returns_results_for_each_query(self) -> None:
        cache = _filled_cache(20)
        app = _make_app()
        client = TestClient(app)
        with (
            patch("konjoai.api.routes.cache.get_settings", return_value=_Settings()),
            patch("konjoai.api.routes.cache.get_semantic_cache", return_value=cache),
            patch("konjoai.api.routes.cache._get_encoder") as mock_enc,
        ):
            mock_enc.return_value.encode = _stub_encode
            resp = client.post("/cache/search", json={"queries": ["q1", "q2", "q3"], "top_k": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["results"]) == 3
        for r in body["results"]:
            assert "query_index" in r
            assert len(r["matches"]) <= 3

    def test_matches_sorted_by_similarity(self) -> None:
        cache = _filled_cache(15)
        app = _make_app()
        client = TestClient(app)
        with (
            patch("konjoai.api.routes.cache.get_settings", return_value=_Settings()),
            patch("konjoai.api.routes.cache.get_semantic_cache", return_value=cache),
            patch("konjoai.api.routes.cache._get_encoder") as mock_enc,
        ):
            mock_enc.return_value.encode = _stub_encode
            resp = client.post("/cache/search", json={"queries": ["hello"], "top_k": 5})
        matches = resp.json()["results"][0]["matches"]
        sims = [m["similarity"] for m in matches]
        assert sims == sorted(sims, reverse=True)

    def test_returns_404_when_disabled(self) -> None:
        app = _make_app()
        client = TestClient(app)
        with patch("konjoai.api.routes.cache.get_settings",
                   return_value=_Settings(cache_enabled=False)):
            resp = client.post("/cache/search", json={"queries": ["x"], "top_k": 3})
        assert resp.status_code == 404


class TestAnalyticsRoute:
    def test_returns_analytics_shape(self) -> None:
        cache = _filled_cache(10)
        buf = LatencyBuffer()
        cache.set_analytics_buffer(buf)
        for i in range(5):
            buf.record(10.0 + i, True, 0.9)
            buf.record(300.0, False)
        app = _make_app()
        client = TestClient(app)
        with (
            patch("konjoai.api.routes.cache.get_settings", return_value=_Settings()),
            patch("konjoai.api.routes.cache.get_semantic_cache", return_value=cache),
        ):
            resp = client.get("/cache/analytics?hours=1")
        assert resp.status_code == 200
        body = resp.json()
        assert "hit_rate" in body
        assert "latency" in body
        assert "similarity_distribution" in body
        assert "hourly_hit_rate" in body

    def test_returns_404_when_disabled(self) -> None:
        app = _make_app()
        client = TestClient(app)
        with patch("konjoai.api.routes.cache.get_settings",
                   return_value=_Settings(cache_enabled=False)):
            assert client.get("/cache/analytics").status_code == 404


class TestTTLReportRoute:
    def test_report_shape(self) -> None:
        cache = _filled_cache(8, ttl=3600)
        app = _make_app()
        client = TestClient(app)
        with (
            patch("konjoai.api.routes.cache.get_settings", return_value=_Settings()),
            patch("konjoai.api.routes.cache.get_semantic_cache", return_value=cache),
        ):
            resp = client.get("/cache/ttl_report")
        assert resp.status_code == 200
        body = resp.json()
        for key in ("total", "no_ttl", "with_ttl", "buckets", "pending_extend", "pending_reduce"):
            assert key in body

    def test_ttl_adjust_returns_counts(self) -> None:
        cache = _filled_cache(8, ttl=3600)
        app = _make_app()
        client = TestClient(app)
        with (
            patch("konjoai.api.routes.cache.get_settings", return_value=_Settings()),
            patch("konjoai.api.routes.cache.get_semantic_cache", return_value=cache),
        ):
            resp = client.post("/cache/ttl/adjust")
        assert resp.status_code == 200
        body = resp.json()
        assert "extended" in body
        assert "reduced" in body
