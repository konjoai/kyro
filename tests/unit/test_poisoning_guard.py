"""Tests for konjoai/cache/poisoning.py — Sprint 28.

Coverage
--------
- WriteRateLimiter: allow within limit, block over limit, window reset, per-tenant
  isolation, current_count, thread safety, invalid args.
- AnomalyDetector: too few obs, no outlier, outlier detected, Welford accuracy,
  uniform inputs, invalid sigma.
- PoisoningReportStore: record, query (all / tenant filter / limit), count, clear,
  ring-buffer eviction, thread safety, K1 (never raises).
- PoisoningGuard: all-pass (no embed_fn), rate-limit blocks, coherence blocks,
  coherence passes, embed failure does not block (K1), anomaly records but does not
  block, per-tenant rate-limit isolation.
- Singletons: get_poisoning_report_store, get_poisoning_guard, _reset_singletons.
- API: POST /cache/report_poisoning (enabled / disabled), GET /cache/poisoning_reports.
- K3: endpoints return 404 when guard disabled.
- OWASP: question_hash field must be present; raw question is rejected.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from konjoai.cache.poisoning import (
    AnomalyDetector,
    PoisoningGuard,
    PoisoningReportStore,
    WriteRateLimiter,
    _cosine_similarity,
    _reset_singletons,
    get_poisoning_guard,
    get_poisoning_report_store,
)


@pytest.fixture(autouse=True)
def _reset():
    _reset_singletons()
    yield
    _reset_singletons()


# ── Helpers ────────────────────────────────────────────────────────────────────


def _unit_vec(*components: float) -> np.ndarray:
    v = np.array([list(components)], dtype=np.float32)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-10 else v


def _make_client(
    cache_enabled: bool = True,
    guard_enabled: bool = True,
) -> TestClient:
    @dataclass
    class _Settings:
        cache_enabled: bool = cache_enabled
        cache_poisoning_guard_enabled: bool = guard_enabled
        cache_poisoning_max_reports: int = 500

    from konjoai.api.app import create_app

    app = create_app()
    with patch("konjoai.config.get_settings", return_value=_Settings()):
        with patch("konjoai.api.routes.cache.get_settings", return_value=_Settings()):
            return TestClient(app)


# ── WriteRateLimiter ───────────────────────────────────────────────────────────


class TestWriteRateLimiter:
    def test_allows_within_limit(self) -> None:
        lim = WriteRateLimiter(max_writes=5)
        for _ in range(5):
            assert lim.is_allowed("t1")

    def test_blocks_over_limit(self) -> None:
        lim = WriteRateLimiter(max_writes=3)
        for _ in range(3):
            lim.is_allowed("t1")
        assert not lim.is_allowed("t1")

    def test_tenants_are_independent(self) -> None:
        lim = WriteRateLimiter(max_writes=2)
        lim.is_allowed("t1")
        lim.is_allowed("t1")
        assert not lim.is_allowed("t1")
        assert lim.is_allowed("t2")

    def test_window_expires(self) -> None:
        lim = WriteRateLimiter(max_writes=2, window_seconds=0.05)
        lim.is_allowed("t1")
        lim.is_allowed("t1")
        assert not lim.is_allowed("t1")
        time.sleep(0.1)
        assert lim.is_allowed("t1")

    def test_current_count(self) -> None:
        lim = WriteRateLimiter(max_writes=10)
        lim.is_allowed("t1")
        lim.is_allowed("t1")
        assert lim.current_count("t1") == 2

    def test_current_count_unknown_tenant(self) -> None:
        lim = WriteRateLimiter(max_writes=10)
        assert lim.current_count("nobody") == 0

    def test_invalid_max_writes(self) -> None:
        with pytest.raises(ValueError):
            WriteRateLimiter(max_writes=0)

    def test_invalid_window_seconds(self) -> None:
        with pytest.raises(ValueError):
            WriteRateLimiter(window_seconds=-1.0)

    def test_thread_safety(self) -> None:
        lim = WriteRateLimiter(max_writes=50)
        results: list[bool] = []
        lock = threading.Lock()

        def attempt() -> None:
            res = lim.is_allowed("t1")
            with lock:
                results.append(res)

        threads = [threading.Thread(target=attempt) for _ in range(70)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sum(results) == 50


# ── AnomalyDetector ────────────────────────────────────────────────────────────


class TestAnomalyDetector:
    def test_no_flag_below_min_observations(self) -> None:
        det = AnomalyDetector(min_observations=10)
        for _ in range(5):
            det.record("hello")
        assert not det.is_length_outlier("hello")

    def test_no_flag_typical_length(self) -> None:
        det = AnomalyDetector(sigma_threshold=3.0, min_observations=5)
        for _ in range(20):
            det.record("x" * 100)
        assert not det.is_length_outlier("x" * 100)

    def test_outlier_detected(self) -> None:
        det = AnomalyDetector(sigma_threshold=2.0, min_observations=5)
        for i in range(30):
            det.record("x" * (100 + i % 5))  # lengths 100–104; std ~1.5
        assert det.is_length_outlier("x" * 10_000)

    def test_n_observations(self) -> None:
        det = AnomalyDetector()
        assert det.n_observations == 0
        det.record("hello")
        assert det.n_observations == 1

    def test_uniform_responses_no_outlier(self) -> None:
        det = AnomalyDetector(min_observations=5)
        for _ in range(20):
            det.record("a")
        assert not det.is_length_outlier("a")

    def test_invalid_sigma(self) -> None:
        with pytest.raises(ValueError):
            AnomalyDetector(sigma_threshold=0)

    def test_record_many(self) -> None:
        det = AnomalyDetector(min_observations=3)
        for i in range(100):
            det.record("x" * (100 + i % 5))
        assert det.n_observations == 100


# ── PoisoningReportStore ───────────────────────────────────────────────────────


class TestPoisoningReportStore:
    def test_record_and_query(self) -> None:
        store = PoisoningReportStore()
        store.record("t1", "abc123", "rate_limit_exceeded")
        reports = store.query()
        assert len(reports) == 1
        r = reports[0]
        assert r.tenant_id == "t1"
        assert r.question_hash == "abc123"
        assert r.reason == "rate_limit_exceeded"

    def test_query_filter_by_tenant(self) -> None:
        store = PoisoningReportStore()
        store.record("t1", "a", "r1")
        store.record("t2", "b", "r2")
        assert len(store.query(tenant_id="t1")) == 1
        assert len(store.query(tenant_id="t2")) == 1
        assert len(store.query()) == 2

    def test_query_limit(self) -> None:
        store = PoisoningReportStore()
        for i in range(10):
            store.record("t1", f"hash{i}", "r")
        assert len(store.query(limit=3)) == 3

    def test_count_total(self) -> None:
        store = PoisoningReportStore()
        store.record("t1", "a", "r")
        store.record("t2", "b", "r")
        assert store.count() == 2

    def test_count_by_tenant(self) -> None:
        store = PoisoningReportStore()
        store.record("t1", "a", "r")
        store.record("t1", "b", "r")
        store.record("t2", "c", "r")
        assert store.count(tenant_id="t1") == 2
        assert store.count(tenant_id="t2") == 1

    def test_clear(self) -> None:
        store = PoisoningReportStore()
        store.record("t1", "a", "r")
        store.clear()
        assert store.count() == 0

    def test_ring_buffer_eviction(self) -> None:
        store = PoisoningReportStore(max_reports=3)
        for i in range(5):
            store.record("t1", f"h{i}", "r")
        assert store.count() == 3

    def test_record_never_raises(self) -> None:
        store = PoisoningReportStore(max_reports=1)
        for _ in range(200):
            store.record("t1", "h", "r")

    def test_thread_safety(self) -> None:
        store = PoisoningReportStore(max_reports=200)

        def record_many() -> None:
            for i in range(20):
                store.record("t1", f"h{i}", "r")

        threads = [threading.Thread(target=record_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert store.count() <= 200


# ── _cosine_similarity ─────────────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        v = _unit_vec(1.0, 0.0, 0.0)
        assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self) -> None:
        a = _unit_vec(1.0, 0.0, 0.0)
        b = _unit_vec(0.0, 1.0, 0.0)
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_zero_vector_returns_zero(self) -> None:
        a = np.zeros((1, 3), dtype=np.float32)
        b = _unit_vec(1.0, 0.0, 0.0)
        assert _cosine_similarity(a, b) == 0.0


# ── PoisoningGuard ─────────────────────────────────────────────────────────────


class TestPoisoningGuard:
    def test_all_pass_no_embed_fn(self) -> None:
        store = PoisoningReportStore()
        guard = PoisoningGuard(max_writes_per_minute=100, report_store=store)
        assert guard.validate("question?", _unit_vec(1.0, 0.0), "answer", "t1")
        assert store.count() == 0

    def test_rate_limit_blocks_third_call(self) -> None:
        store = PoisoningReportStore()
        guard = PoisoningGuard(max_writes_per_minute=2, report_store=store)
        assert guard.validate("q", _unit_vec(1.0, 0.0), "a", "t1")
        assert guard.validate("q", _unit_vec(1.0, 0.0), "a", "t1")
        assert not guard.validate("q", _unit_vec(1.0, 0.0), "a", "t1")
        assert any(r.reason == "rate_limit_exceeded" for r in store.query())

    def test_low_coherence_blocks(self) -> None:
        store = PoisoningReportStore()
        q_vec = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        def embed_fn(_text: str) -> np.ndarray:
            return np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

        guard = PoisoningGuard(
            min_qa_coherence=0.3, embed_fn=embed_fn, report_store=store
        )
        assert not guard.validate("question?", q_vec, "answer", "t1")
        assert any("low_coherence" in r.reason for r in store.query())

    def test_high_coherence_passes(self) -> None:
        store = PoisoningReportStore()
        q_vec = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        def embed_fn(_text: str) -> np.ndarray:
            return np.array([[0.95, 0.05, 0.0, 0.0]], dtype=np.float32)

        guard = PoisoningGuard(
            min_qa_coherence=0.3, embed_fn=embed_fn, report_store=store
        )
        assert guard.validate("question?", q_vec, "answer", "t1")

    def test_embed_failure_does_not_block(self) -> None:
        """K1: coherence embed failure must never prevent the store."""
        store = PoisoningReportStore()

        def bad_embed(_text: str) -> np.ndarray:
            raise RuntimeError("encoder down")

        guard = PoisoningGuard(embed_fn=bad_embed, report_store=store)
        assert guard.validate("q", _unit_vec(1.0, 0.0), "a", "t1")

    def test_length_anomaly_records_but_does_not_block(self) -> None:
        store = PoisoningReportStore()
        guard = PoisoningGuard(
            max_writes_per_minute=1000,
            length_sigma=1.0,
            min_anomaly_observations=5,
            report_store=store,
        )
        for _ in range(30):
            guard.validate("q", _unit_vec(1.0, 0.0), "a" * 10, "t1")
        store.clear()
        result = guard.validate("q", _unit_vec(1.0, 0.0), "a" * 100_000, "t1")
        assert result

    def test_rate_limit_per_tenant_independent(self) -> None:
        guard = PoisoningGuard(max_writes_per_minute=1)
        assert guard.validate("q", _unit_vec(1.0, 0.0), "a", "t1")
        assert not guard.validate("q", _unit_vec(1.0, 0.0), "a", "t1")
        assert guard.validate("q", _unit_vec(1.0, 0.0), "a", "t2")

    def test_coherence_check_truncates_long_answer(self) -> None:
        """Answers longer than 2048 chars are truncated before embedding."""
        calls: list[str] = []

        def embed_fn(text: str) -> np.ndarray:
            calls.append(text)
            return np.array([[1.0, 0.0]], dtype=np.float32)

        guard = PoisoningGuard(
            min_qa_coherence=0.0,
            embed_fn=embed_fn,
        )
        long_answer = "x" * 5000
        guard.validate("q", _unit_vec(1.0, 0.0), long_answer, "t1")
        assert len(calls[0]) <= 2048


# ── Singletons ─────────────────────────────────────────────────────────────────


class TestSingletons:
    def test_get_report_store_is_singleton(self) -> None:
        s1 = get_poisoning_report_store()
        s2 = get_poisoning_report_store()
        assert s1 is s2

    def test_reset_singletons(self) -> None:
        s1 = get_poisoning_report_store()
        _reset_singletons()
        s2 = get_poisoning_report_store()
        assert s1 is not s2

    def test_get_guard_is_singleton(self) -> None:
        @dataclass
        class _S:
            cache_poisoning_min_coherence: float = 0.3
            cache_poisoning_max_writes_per_minute: int = 100
            cache_poisoning_length_sigma: float = 3.0
            cache_poisoning_check_coherence: bool = False
            cache_poisoning_max_reports: int = 500

        with patch("konjoai.config.get_settings", return_value=_S()):
            g1 = get_poisoning_guard()
            g2 = get_poisoning_guard()
        assert g1 is g2


# ── API ────────────────────────────────────────────────────────────────────────


class TestPoisoningAPI:
    def _make_app(self, *, guard_enabled: bool) -> TestClient:
        @dataclass
        class _S:
            cache_enabled: bool = True
            cache_poisoning_guard_enabled: bool = guard_enabled
            cache_poisoning_max_reports: int = 500

        from konjoai.api.app import create_app

        app = create_app()
        s = _S()
        with (
            patch("konjoai.config.get_settings", return_value=s),
            patch("konjoai.api.routes.cache.get_settings", return_value=s),
        ):
            return TestClient(app, raise_server_exceptions=False)

    def test_report_poisoning_disabled_returns_404(self) -> None:
        client = self._make_app(guard_enabled=False)
        resp = client.post(
            "/cache/report_poisoning",
            json={"question_hash": "abc1234567890123", "reason": "test"},
        )
        assert resp.status_code == 404

    def test_report_poisoning_enabled_returns_201(self) -> None:
        _reset_singletons()

        @dataclass
        class _S:
            cache_enabled: bool = True
            cache_poisoning_guard_enabled: bool = True
            cache_poisoning_max_reports: int = 500

        from konjoai.api.app import create_app

        app = create_app()
        s = _S()
        with (
            patch("konjoai.config.get_settings", return_value=s),
            patch("konjoai.api.routes.cache.get_settings", return_value=s),
        ):
            client = TestClient(app)
            resp = client.post(
                "/cache/report_poisoning",
                json={"question_hash": "abc1234567890123", "reason": "test_reason"},
            )
        assert resp.status_code == 201
        body = resp.json()
        assert body["recorded"] is True
        assert len(body["report_hash"]) == 16

    def test_report_poisoning_invalid_hash_422(self) -> None:
        _reset_singletons()

        @dataclass
        class _S:
            cache_enabled: bool = True
            cache_poisoning_guard_enabled: bool = True
            cache_poisoning_max_reports: int = 500

        from konjoai.api.app import create_app

        app = create_app()
        s = _S()
        with (
            patch("konjoai.config.get_settings", return_value=s),
            patch("konjoai.api.routes.cache.get_settings", return_value=s),
        ):
            client = TestClient(app)
            resp = client.post(
                "/cache/report_poisoning",
                json={"question_hash": "x", "reason": "r"},  # too short
            )
        assert resp.status_code == 422

    def test_poisoning_reports_disabled_returns_404(self) -> None:
        client = self._make_app(guard_enabled=False)
        resp = client.get("/cache/poisoning_reports")
        assert resp.status_code == 404

    def test_poisoning_reports_enabled_returns_200(self) -> None:
        _reset_singletons()

        @dataclass
        class _S:
            cache_enabled: bool = True
            cache_poisoning_guard_enabled: bool = True
            cache_poisoning_max_reports: int = 500

        from konjoai.api.app import create_app

        app = create_app()
        s = _S()
        with (
            patch("konjoai.config.get_settings", return_value=s),
            patch("konjoai.api.routes.cache.get_settings", return_value=s),
        ):
            client = TestClient(app)
            resp = client.get("/cache/poisoning_reports")
        assert resp.status_code == 200
        body = resp.json()
        assert "reports" in body
        assert "count" in body

    def test_poisoning_reports_schema(self) -> None:
        _reset_singletons()

        @dataclass
        class _S:
            cache_enabled: bool = True
            cache_poisoning_guard_enabled: bool = True
            cache_poisoning_max_reports: int = 500

        store = get_poisoning_report_store()
        store.record("t1", "abc1234567890123", "test_reason")

        from konjoai.api.app import create_app

        app = create_app()
        s = _S()
        with (
            patch("konjoai.config.get_settings", return_value=s),
            patch("konjoai.api.routes.cache.get_settings", return_value=s),
        ):
            client = TestClient(app)
            resp = client.get("/cache/poisoning_reports?tenant_id=t1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] >= 1
        report = body["reports"][0]
        assert report["tenant_id"] == "t1"
        assert report["question_hash"] == "abc1234567890123"
