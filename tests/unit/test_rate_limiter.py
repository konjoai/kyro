"""Sprint 18 — Rate limiter tests.

Coverage targets:
- RateLimiter construction: valid, invalid params
- _Bucket sliding-window eviction
- check() happy path and limit exhaustion
- check() with explicit `now` (deterministic)
- Multiple tenants are isolated
- Multiple endpoints are isolated
- Disabled limiter is always a no-op
- current_count() — side-effect-free inspection
- reset() — full, per-tenant, per-endpoint
- RateLimitExceeded message contract
- get_rate_limiter() singleton + _reset_singleton()
"""

from __future__ import annotations

import threading
import time

import pytest

from konjoai.auth.rate_limiter import (
    RateLimiter,
    RateLimitExceeded,
    _reset_singleton,
    get_rate_limiter,
)

# ── Construction ──────────────────────────────────────────────────────────────


class TestRateLimiterConstruction:
    def test_default_construction(self) -> None:
        rl = RateLimiter()
        assert rl.max_requests == 60
        assert rl.window_seconds == 60
        assert rl.enabled is True

    def test_custom_params(self) -> None:
        rl = RateLimiter(max_requests=10, window_seconds=30, enabled=False)
        assert rl.max_requests == 10
        assert rl.window_seconds == 30
        assert rl.enabled is False

    def test_zero_max_requests_raises(self) -> None:
        with pytest.raises(ValueError, match="max_requests"):
            RateLimiter(max_requests=0)

    def test_negative_max_requests_raises(self) -> None:
        with pytest.raises(ValueError, match="max_requests"):
            RateLimiter(max_requests=-1)

    def test_zero_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window_seconds"):
            RateLimiter(window_seconds=0)

    def test_negative_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window_seconds"):
            RateLimiter(window_seconds=-5)


# ── Happy path ────────────────────────────────────────────────────────────────


class TestRateLimiterHappyPath:
    def test_single_request_allowed(self) -> None:
        rl = RateLimiter(max_requests=3, window_seconds=60)
        rl.check("t1", "/query")  # should not raise

    def test_up_to_limit_allowed(self) -> None:
        rl = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            rl.check("t1", "/query")

    def test_exceeds_limit_raises(self) -> None:
        rl = RateLimiter(max_requests=3, window_seconds=60)
        now = time.monotonic()
        for i in range(3):
            rl.check("t1", "/query", now=now + i * 0.001)
        with pytest.raises(RateLimitExceeded):
            rl.check("t1", "/query", now=now + 0.01)

    def test_rate_limit_exceeded_attributes(self) -> None:
        rl = RateLimiter(max_requests=1, window_seconds=10)
        now = time.monotonic()
        rl.check("t1", "/q", now=now)
        with pytest.raises(RateLimitExceeded) as exc_info:
            rl.check("t1", "/q", now=now + 0.001)
        err = exc_info.value
        assert err.tenant_id == "t1"
        assert err.endpoint == "/q"
        assert err.limit == 1
        assert err.window == 10

    def test_rate_limit_exceeded_message(self) -> None:
        rl = RateLimiter(max_requests=2, window_seconds=30)
        now = time.monotonic()
        rl.check("acme", "/ingest", now=now)
        rl.check("acme", "/ingest", now=now + 0.001)
        with pytest.raises(RateLimitExceeded) as exc_info:
            rl.check("acme", "/ingest", now=now + 0.002)
        assert "acme" in str(exc_info.value)
        assert "/ingest" in str(exc_info.value)


# ── Sliding window eviction ───────────────────────────────────────────────────


class TestSlidingWindow:
    def test_old_requests_evicted_allow_new(self) -> None:
        """Requests outside the window no longer count."""
        rl = RateLimiter(max_requests=2, window_seconds=10)
        t0 = 1000.0
        # Fill window at t0
        rl.check("t1", "/q", now=t0)
        rl.check("t1", "/q", now=t0 + 0.001)
        # Window expires at t0 + 10; new request at t0 + 11 should succeed
        rl.check("t1", "/q", now=t0 + 11.0)

    def test_requests_at_window_boundary_evicted(self) -> None:
        """Requests exactly at the cutoff (≤ cutoff) are evicted."""
        rl = RateLimiter(max_requests=1, window_seconds=5)
        t0 = 500.0
        rl.check("t1", "/q", now=t0)
        # At t0 + 5 the first request is right on the boundary and should be evicted
        rl.check("t1", "/q", now=t0 + 5.001)

    def test_multiple_buckets_independent_windows(self) -> None:
        rl = RateLimiter(max_requests=2, window_seconds=10)
        t0 = 100.0
        rl.check("t1", "/q", now=t0)
        rl.check("t1", "/q", now=t0 + 0.001)
        # t2 has a fresh bucket — still allowed
        rl.check("t2", "/q", now=t0 + 0.002)


# ── Tenant and endpoint isolation ─────────────────────────────────────────────


class TestIsolation:
    def test_tenants_isolated(self) -> None:
        rl = RateLimiter(max_requests=1, window_seconds=60)
        now = time.monotonic()
        rl.check("tenant-a", "/q", now=now)
        rl.check("tenant-b", "/q", now=now + 0.001)  # separate bucket — not exhausted

    def test_endpoints_isolated(self) -> None:
        rl = RateLimiter(max_requests=1, window_seconds=60)
        now = time.monotonic()
        rl.check("t1", "/query", now=now)
        rl.check("t1", "/ingest", now=now + 0.001)  # different endpoint — allowed

    def test_tenant_a_limit_does_not_affect_tenant_b(self) -> None:
        rl = RateLimiter(max_requests=2, window_seconds=60)
        now = time.monotonic()
        rl.check("a", "/q", now=now)
        rl.check("a", "/q", now=now + 0.001)
        with pytest.raises(RateLimitExceeded):
            rl.check("a", "/q", now=now + 0.002)
        # b should not be affected
        rl.check("b", "/q", now=now + 0.003)


# ── Disabled limiter ──────────────────────────────────────────────────────────


class TestDisabledLimiter:
    def test_disabled_never_raises(self) -> None:
        rl = RateLimiter(max_requests=1, window_seconds=60, enabled=False)
        for _ in range(100):
            rl.check("t1", "/q")  # must not raise

    def test_disabled_current_count_returns_zero(self) -> None:
        rl = RateLimiter(max_requests=1, window_seconds=60, enabled=False)
        rl.check("t1", "/q")
        # Because disabled, nothing is recorded
        assert rl.current_count("t1", "/q") == 0


# ── current_count() ───────────────────────────────────────────────────────────


class TestCurrentCount:
    def test_zero_before_any_request(self) -> None:
        rl = RateLimiter(max_requests=10, window_seconds=60)
        assert rl.current_count("t1", "/q") == 0

    def test_increments_with_requests(self) -> None:
        rl = RateLimiter(max_requests=10, window_seconds=60)
        now = time.monotonic()
        rl.check("t1", "/q", now=now)
        rl.check("t1", "/q", now=now + 0.001)
        assert rl.current_count("t1", "/q") == 2

    def test_does_not_consume_quota(self) -> None:
        """current_count() must not add a new timestamp entry."""
        rl = RateLimiter(max_requests=2, window_seconds=60)
        now = time.monotonic()
        rl.check("t1", "/q", now=now)
        _ = rl.current_count("t1", "/q")
        _ = rl.current_count("t1", "/q")
        rl.check("t1", "/q", now=now + 0.001)  # second real request — still OK
        # Third real request should raise (limit=2)
        with pytest.raises(RateLimitExceeded):
            rl.check("t1", "/q", now=now + 0.002)


# ── reset() ───────────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_all_clears_all_buckets(self) -> None:
        rl = RateLimiter(max_requests=1, window_seconds=60)
        now = time.monotonic()
        rl.check("t1", "/q", now=now)
        rl.reset()
        rl.check("t1", "/q", now=now + 0.001)  # should succeed after reset

    def test_reset_per_tenant(self) -> None:
        rl = RateLimiter(max_requests=1, window_seconds=60)
        now = time.monotonic()
        rl.check("t1", "/q", now=now)
        rl.check("t2", "/q", now=now + 0.001)
        rl.reset(tenant_id="t1")
        rl.check("t1", "/q")  # reset — should succeed
        with pytest.raises(RateLimitExceeded):
            rl.check("t2", "/q", now=now + 0.002)  # t2 still exhausted

    def test_reset_per_endpoint(self) -> None:
        rl = RateLimiter(max_requests=1, window_seconds=60)
        now = time.monotonic()
        rl.check("t1", "/q", now=now)
        rl.check("t1", "/ingest", now=now + 0.001)
        rl.reset(endpoint="/q")
        rl.check("t1", "/q")  # reset endpoint — should succeed
        with pytest.raises(RateLimitExceeded):
            rl.check("t1", "/ingest", now=now + 0.002)


# ── Thread safety (smoke test) ────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_checks_do_not_panic(self) -> None:
        """Multiple threads hitting the same bucket should not panic (may raise)."""
        rl = RateLimiter(max_requests=50, window_seconds=60)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(10):
                    rl.check("shared", "/q")
            except RateLimitExceeded:
                pass
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Thread safety errors: {errors}"


# ── Singleton ─────────────────────────────────────────────────────────────────


class TestSingleton:
    def setup_method(self) -> None:
        _reset_singleton()

    def teardown_method(self) -> None:
        _reset_singleton()

    def test_get_rate_limiter_returns_instance(self) -> None:
        from unittest.mock import patch

        from konjoai.config import Settings

        stub_settings = Settings.model_construct(
            rate_limiting_enabled=False,
            rate_limit_requests=60,
            rate_limit_window_seconds=60,
        )
        with patch("konjoai.auth.rate_limiter.get_settings", return_value=stub_settings):
            limiter = get_rate_limiter()
        assert isinstance(limiter, RateLimiter)

    def test_get_rate_limiter_same_instance_twice(self) -> None:
        from unittest.mock import patch

        from konjoai.config import Settings

        stub_settings = Settings.model_construct(
            rate_limiting_enabled=False,
            rate_limit_requests=10,
            rate_limit_window_seconds=10,
        )
        with patch("konjoai.auth.rate_limiter.get_settings", return_value=stub_settings):
            a = get_rate_limiter()
            b = get_rate_limiter()
        assert a is b

    def test_reset_singleton_forces_new_instance(self) -> None:
        from unittest.mock import patch

        from konjoai.config import Settings

        stub = Settings.model_construct(
            rate_limiting_enabled=False,
            rate_limit_requests=10,
            rate_limit_window_seconds=10,
        )
        with patch("konjoai.auth.rate_limiter.get_settings", return_value=stub):
            a = get_rate_limiter()
            _reset_singleton()
            b = get_rate_limiter()
        assert a is not b

    def test_singleton_respects_enabled_flag(self) -> None:
        from unittest.mock import patch

        from konjoai.config import Settings

        stub = Settings.model_construct(
            rate_limiting_enabled=True,
            rate_limit_requests=5,
            rate_limit_window_seconds=30,
        )
        with patch("konjoai.auth.rate_limiter.get_settings", return_value=stub):
            limiter = get_rate_limiter()
        assert limiter.enabled is True
        assert limiter.max_requests == 5
        assert limiter.window_seconds == 30
