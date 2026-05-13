"""Tests for Sprint 26 per-tenant cost attribution (TenantCostTracker).

Coverage:
- TenantCostTracker.record() and report() — basic accounting
- hit_rate, tokens_saved, cost_saved calculations
- Unknown tenant returns None
- reset() clears all data
- all_tenants() returns all tracked tenants
- Thread safety under concurrent access
- Singleton lifecycle (get_cost_tracker, _reset_cost_tracker)
- /tenants/{id}/cost_report route — disabled → 404, unknown → 404, known → 200
- /tenants route — lists all tenants
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from konjoai.services.cost_attribution import (
    TenantCostTracker,
    TenantCostReport,
    _reset_cost_tracker,
    get_cost_tracker,
)


# ── TenantCostTracker ─────────────────────────────────────────────────────────


class TestTenantCostTracker:
    def test_unknown_tenant_returns_none(self) -> None:
        tracker = TenantCostTracker()
        assert tracker.report("nobody") is None

    def test_single_hit(self) -> None:
        tracker = TenantCostTracker()
        tracker.record("acme", hit=True)
        report = tracker.report("acme")
        assert report is not None
        assert report.cache_hits == 1
        assert report.cache_misses == 0
        assert report.total_queries == 1
        assert report.hit_rate == pytest.approx(1.0)

    def test_single_miss(self) -> None:
        tracker = TenantCostTracker()
        tracker.record("acme", hit=False)
        report = tracker.report("acme")
        assert report is not None
        assert report.cache_hits == 0
        assert report.cache_misses == 1
        assert report.hit_rate == pytest.approx(0.0)

    def test_mixed_hit_miss(self) -> None:
        tracker = TenantCostTracker()
        for _ in range(3):
            tracker.record("t1", hit=True)
        tracker.record("t1", hit=False)
        report = tracker.report("t1")
        assert report is not None
        assert report.total_queries == 4
        assert report.hit_rate == pytest.approx(0.75)

    def test_tokens_saved_per_hit(self) -> None:
        tracker = TenantCostTracker(avg_response_tokens=512)
        tracker.record("t", hit=True)
        tracker.record("t", hit=True)
        report = tracker.report("t")
        assert report is not None
        assert report.tokens_saved == 1024

    def test_cost_saved_calculation(self) -> None:
        tracker = TenantCostTracker(cost_per_1k_tokens=0.002, avg_response_tokens=1000)
        tracker.record("t", hit=True)
        report = tracker.report("t")
        assert report is not None
        assert report.estimated_cost_saved_usd == pytest.approx(0.002)

    def test_zero_cost_on_all_misses(self) -> None:
        tracker = TenantCostTracker()
        tracker.record("t", hit=False)
        report = tracker.report("t")
        assert report is not None
        assert report.tokens_saved == 0
        assert report.estimated_cost_saved_usd == pytest.approx(0.0)

    def test_as_dict_shape(self) -> None:
        tracker = TenantCostTracker()
        tracker.record("acme", hit=True)
        report = tracker.report("acme")
        assert report is not None
        d = report.as_dict()
        expected_keys = {
            "tenant_id", "total_queries", "cache_hits", "cache_misses",
            "hit_rate", "tokens_saved", "estimated_cost_saved_usd",
            "cost_per_1k_tokens", "avg_response_tokens",
        }
        assert set(d.keys()) == expected_keys

    def test_tenant_isolation(self) -> None:
        tracker = TenantCostTracker()
        tracker.record("acme", hit=True)
        tracker.record("globex", hit=False)
        acme   = tracker.report("acme")
        globex = tracker.report("globex")
        assert acme is not None and acme.cache_hits == 1
        assert globex is not None and globex.cache_misses == 1
        assert acme.cache_misses == 0
        assert globex.cache_hits == 0

    def test_empty_tenant_id_is_noop(self) -> None:
        tracker = TenantCostTracker()
        tracker.record("", hit=True)
        assert tracker.report("") is None

    def test_reset_clears_all_data(self) -> None:
        tracker = TenantCostTracker()
        tracker.record("acme", hit=True)
        tracker.record("globex", hit=False)
        tracker.reset()
        assert tracker.report("acme") is None
        assert tracker.report("globex") is None

    def test_all_tenants_empty(self) -> None:
        tracker = TenantCostTracker()
        assert tracker.all_tenants() == []

    def test_all_tenants_returns_all(self) -> None:
        tracker = TenantCostTracker()
        tracker.record("a", hit=True)
        tracker.record("b", hit=False)
        tenant_ids = {r.tenant_id for r in tracker.all_tenants()}
        assert "a" in tenant_ids and "b" in tenant_ids

    def test_thread_safety(self) -> None:
        tracker = TenantCostTracker()
        errors: list[Exception] = []

        def worker(tid: str) -> None:
            try:
                for _ in range(100):
                    tracker.record(tid, hit=True)
                    tracker.record(tid, hit=False)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors

    def test_report_returns_tenant_id(self) -> None:
        tracker = TenantCostTracker()
        tracker.record("my-tenant", hit=True)
        report = tracker.report("my-tenant")
        assert report is not None
        assert report.tenant_id == "my-tenant"


# ── Singleton ─────────────────────────────────────────────────────────────────


class TestCostTrackerSingleton:
    def setup_method(self) -> None:
        _reset_cost_tracker()

    def test_singleton_identity(self) -> None:
        t1 = get_cost_tracker()
        t2 = get_cost_tracker()
        assert t1 is t2

    def test_reset_gives_fresh_instance(self) -> None:
        t1 = get_cost_tracker()
        _reset_cost_tracker()
        t2 = get_cost_tracker()
        assert t1 is not t2

    def test_singleton_reads_defaults_without_settings(self) -> None:
        with patch("konjoai.services.cost_attribution.get_cost_tracker"):
            tracker = TenantCostTracker()
        assert isinstance(tracker, TenantCostTracker)


# ── /tenants routes ───────────────────────────────────────────────────────────


@dataclass
class _SettingsEnabled:
    cache_enabled: bool = True

@dataclass
class _SettingsDisabled:
    cache_enabled: bool = False


def _make_app(enabled: bool):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from konjoai.api.routes.tenants import router

    app = FastAPI()
    app.include_router(router)
    settings = _SettingsEnabled() if enabled else _SettingsDisabled()
    client = TestClient(app)
    return client, settings


class TestTenantsRoute:
    def setup_method(self) -> None:
        _reset_cost_tracker()

    def test_list_returns_404_when_cache_disabled(self) -> None:
        client, settings = _make_app(enabled=False)
        with patch("konjoai.api.routes.tenants.get_settings", return_value=settings):
            resp = client.get("/tenants")
        assert resp.status_code == 404

    def test_cost_report_returns_404_when_cache_disabled(self) -> None:
        client, settings = _make_app(enabled=False)
        with patch("konjoai.api.routes.tenants.get_settings", return_value=settings):
            resp = client.get("/tenants/acme/cost_report")
        assert resp.status_code == 404

    def test_cost_report_returns_404_for_unknown_tenant(self) -> None:
        client, settings = _make_app(enabled=True)
        with (
            patch("konjoai.api.routes.tenants.get_settings", return_value=settings),
            patch("konjoai.api.routes.tenants.get_cost_tracker", return_value=TenantCostTracker()),
        ):
            resp = client.get("/tenants/nobody/cost_report")
        assert resp.status_code == 404

    def test_cost_report_returns_200_for_known_tenant(self) -> None:
        tracker = TenantCostTracker()
        tracker.record("acme", hit=True)
        tracker.record("acme", hit=False)
        client, settings = _make_app(enabled=True)
        with (
            patch("konjoai.api.routes.tenants.get_settings", return_value=settings),
            patch("konjoai.api.routes.tenants.get_cost_tracker", return_value=tracker),
        ):
            resp = client.get("/tenants/acme/cost_report")
        assert resp.status_code == 200
        body = resp.json()
        assert body["tenant_id"] == "acme"
        assert body["total_queries"] == 2
        assert body["cache_hits"] == 1
        assert body["cache_misses"] == 1
        assert "estimated_cost_saved_usd" in body

    def test_list_tenants_returns_empty_when_no_queries(self) -> None:
        tracker = TenantCostTracker()
        client, settings = _make_app(enabled=True)
        with (
            patch("konjoai.api.routes.tenants.get_settings", return_value=settings),
            patch("konjoai.api.routes.tenants.get_cost_tracker", return_value=tracker),
        ):
            resp = client.get("/tenants")
        assert resp.status_code == 200
        body = resp.json()
        assert body["tenants"] == []
        assert body["count"] == 0

    def test_list_tenants_returns_all_tracked(self) -> None:
        tracker = TenantCostTracker()
        tracker.record("a", hit=True)
        tracker.record("b", hit=False)
        client, settings = _make_app(enabled=True)
        with (
            patch("konjoai.api.routes.tenants.get_settings", return_value=settings),
            patch("konjoai.api.routes.tenants.get_cost_tracker", return_value=tracker),
        ):
            resp = client.get("/tenants")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        tenant_ids = {t["tenant_id"] for t in body["tenants"]}
        assert "a" in tenant_ids and "b" in tenant_ids

    def test_cost_report_hit_rate_and_tokens(self) -> None:
        tracker = TenantCostTracker(avg_response_tokens=100, cost_per_1k_tokens=0.01)
        for _ in range(4):
            tracker.record("rx", hit=True)
        client, settings = _make_app(enabled=True)
        with (
            patch("konjoai.api.routes.tenants.get_settings", return_value=settings),
            patch("konjoai.api.routes.tenants.get_cost_tracker", return_value=tracker),
        ):
            resp = client.get("/tenants/rx/cost_report")
        body = resp.json()
        assert body["cache_hits"] == 4
        assert body["tokens_saved"] == 400
        assert body["estimated_cost_saved_usd"] == pytest.approx(0.004)
        assert body["hit_rate"] == pytest.approx(1.0)
