"""Sprint 18 — Brute-force protection tests.

Coverage targets:
- BruteForceGuard construction: valid, invalid params
- check_ip: allows clean IPs
- check_ip: raises IPLockedOut after max_attempts failures
- record_failure: counts correctly within window
- record_failure: old failures outside window evicted
- record_success: clears failure record
- is_locked: True / False states
- failure_count: accurate counting
- reset: per-IP and full clear
- Disabled guard is always a no-op
- IPLockedOut message contract
- Thread-safety smoke test
- get_brute_force_guard singleton + _reset_singleton
- get_tenant_id dep: locked-out IP returns 429
- get_tenant_id dep: brute-force counter incremented on JWT failure
- get_tenant_id dep: brute-force counter cleared on JWT success
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from konjoai.auth.brute_force import (
    BruteForceGuard,
    IPLockedOut,
    _reset_singleton,
    get_brute_force_guard,
)

# ── Construction ──────────────────────────────────────────────────────────────


class TestBruteForceGuardConstruction:
    def test_default_construction(self) -> None:
        g = BruteForceGuard()
        assert g.max_attempts == 5
        assert g.window_seconds == 60
        assert g.lockout_seconds == 300
        assert g.enabled is True

    def test_custom_params(self) -> None:
        g = BruteForceGuard(
            max_attempts=3,
            window_seconds=30,
            lockout_seconds=120,
            enabled=False,
        )
        assert g.max_attempts == 3
        assert g.window_seconds == 30
        assert g.lockout_seconds == 120
        assert g.enabled is False

    def test_zero_max_attempts_raises(self) -> None:
        with pytest.raises(ValueError, match="max_attempts"):
            BruteForceGuard(max_attempts=0)

    def test_negative_max_attempts_raises(self) -> None:
        with pytest.raises(ValueError, match="max_attempts"):
            BruteForceGuard(max_attempts=-1)

    def test_zero_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window_seconds"):
            BruteForceGuard(window_seconds=0)

    def test_zero_lockout_raises(self) -> None:
        with pytest.raises(ValueError, match="lockout_seconds"):
            BruteForceGuard(lockout_seconds=0)


# ── check_ip ──────────────────────────────────────────────────────────────────


class TestCheckIp:
    def test_clean_ip_passes(self) -> None:
        g = BruteForceGuard(max_attempts=3, window_seconds=60, lockout_seconds=300)
        g.check_ip("1.2.3.4")  # must not raise

    def test_locked_ip_raises(self) -> None:
        g = BruteForceGuard(max_attempts=2, window_seconds=60, lockout_seconds=300)
        now = time.monotonic()
        g.record_failure("1.2.3.4", now=now)
        g.record_failure("1.2.3.4", now=now + 0.001)
        with pytest.raises(IPLockedOut):
            g.check_ip("1.2.3.4", now=now + 0.002)

    def test_different_ips_independent(self) -> None:
        g = BruteForceGuard(max_attempts=2, window_seconds=60, lockout_seconds=300)
        now = time.monotonic()
        g.record_failure("1.1.1.1", now=now)
        g.record_failure("1.1.1.1", now=now + 0.001)
        g.check_ip("2.2.2.2", now=now + 0.002)  # different IP — not locked


# ── IPLockedOut ───────────────────────────────────────────────────────────────


class TestIPLockedOut:
    def test_attributes(self) -> None:
        g = BruteForceGuard(max_attempts=1, window_seconds=10, lockout_seconds=60)
        now = time.monotonic()
        g.record_failure("9.9.9.9", now=now)
        with pytest.raises(IPLockedOut) as exc_info:
            g.check_ip("9.9.9.9", now=now + 0.001)
        err = exc_info.value
        assert err.ip == "9.9.9.9"
        assert err.lockout_seconds == 60
        assert err.retry_after > now

    def test_message_contains_ip(self) -> None:
        g = BruteForceGuard(max_attempts=1, window_seconds=10, lockout_seconds=30)
        now = time.monotonic()
        g.record_failure("5.5.5.5", now=now)
        with pytest.raises(IPLockedOut) as exc_info:
            g.check_ip("5.5.5.5", now=now + 0.001)
        assert "5.5.5.5" in str(exc_info.value)


# ── record_failure ────────────────────────────────────────────────────────────


class TestRecordFailure:
    def test_single_failure_does_not_lock(self) -> None:
        g = BruteForceGuard(max_attempts=3, window_seconds=60, lockout_seconds=120)
        now = time.monotonic()
        g.record_failure("10.0.0.1", now=now)
        assert not g.is_locked("10.0.0.1", now=now + 0.001)

    def test_failures_at_limit_cause_lockout(self) -> None:
        g = BruteForceGuard(max_attempts=3, window_seconds=60, lockout_seconds=120)
        now = time.monotonic()
        for i in range(3):
            g.record_failure("10.0.0.2", now=now + i * 0.001)
        assert g.is_locked("10.0.0.2", now=now + 0.01)

    def test_old_failures_evicted(self) -> None:
        g = BruteForceGuard(max_attempts=2, window_seconds=10, lockout_seconds=120)
        t0 = time.monotonic()
        g.record_failure("10.0.0.3", now=t0)
        # First failure is now outside the window
        g.record_failure("10.0.0.3", now=t0 + 11.0)
        # Only one failure within window — not locked
        assert not g.is_locked("10.0.0.3", now=t0 + 11.0 + 0.001)

    def test_failure_count_accurate(self) -> None:
        g = BruteForceGuard(max_attempts=5, window_seconds=60, lockout_seconds=120)
        now = time.monotonic()
        for i in range(3):
            g.record_failure("10.0.0.4", now=now + i * 0.001)
        assert g.failure_count("10.0.0.4") == 3


# ── record_success ────────────────────────────────────────────────────────────


class TestRecordSuccess:
    def test_success_clears_failure_count(self) -> None:
        g = BruteForceGuard(max_attempts=3, window_seconds=60, lockout_seconds=120)
        now = time.monotonic()
        g.record_failure("10.0.0.5", now=now)
        g.record_failure("10.0.0.5", now=now + 0.001)
        g.record_success("10.0.0.5")
        assert g.failure_count("10.0.0.5") == 0

    def test_success_unlocks_locked_ip(self) -> None:
        g = BruteForceGuard(max_attempts=2, window_seconds=60, lockout_seconds=300)
        now = time.monotonic()
        g.record_failure("10.0.0.6", now=now)
        g.record_failure("10.0.0.6", now=now + 0.001)
        assert g.is_locked("10.0.0.6")
        g.record_success("10.0.0.6")
        assert not g.is_locked("10.0.0.6")

    def test_success_on_unknown_ip_is_noop(self) -> None:
        g = BruteForceGuard(max_attempts=3, window_seconds=60, lockout_seconds=120)
        g.record_success("99.99.99.99")  # must not raise


# ── is_locked ─────────────────────────────────────────────────────────────────


class TestIsLocked:
    def test_false_before_any_failure(self) -> None:
        g = BruteForceGuard()
        assert not g.is_locked("0.0.0.0")

    def test_false_after_reset(self) -> None:
        g = BruteForceGuard(max_attempts=1, window_seconds=60, lockout_seconds=120)
        now = time.monotonic()
        g.record_failure("1.1.1.1", now=now)
        g.reset("1.1.1.1")
        assert not g.is_locked("1.1.1.1")


# ── failure_count ─────────────────────────────────────────────────────────────


class TestFailureCount:
    def test_zero_for_unknown_ip(self) -> None:
        g = BruteForceGuard()
        assert g.failure_count("9.8.7.6") == 0

    def test_returns_correct_count(self) -> None:
        g = BruteForceGuard(max_attempts=10, window_seconds=60, lockout_seconds=120)
        now = time.monotonic()
        for i in range(4):
            g.record_failure("1.2.3.4", now=now + i * 0.001)
        assert g.failure_count("1.2.3.4") == 4


# ── reset ─────────────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_all(self) -> None:
        g = BruteForceGuard(max_attempts=1, window_seconds=60, lockout_seconds=120)
        now = time.monotonic()
        g.record_failure("a.a.a.a", now=now)
        g.record_failure("b.b.b.b", now=now + 0.001)
        g.reset()
        assert g.failure_count("a.a.a.a") == 0
        assert g.failure_count("b.b.b.b") == 0

    def test_reset_specific_ip(self) -> None:
        g = BruteForceGuard(max_attempts=1, window_seconds=60, lockout_seconds=120)
        now = time.monotonic()
        g.record_failure("c.c.c.c", now=now)
        g.record_failure("d.d.d.d", now=now + 0.001)
        g.reset("c.c.c.c")
        assert g.failure_count("c.c.c.c") == 0
        assert g.failure_count("d.d.d.d") == 1


# ── Disabled guard ────────────────────────────────────────────────────────────


class TestDisabledGuard:
    def test_disabled_check_ip_never_raises(self) -> None:
        g = BruteForceGuard(max_attempts=1, window_seconds=60, lockout_seconds=120, enabled=False)
        for i in range(10):
            g.record_failure("evil", now=float(i))
        g.check_ip("evil")  # must not raise

    def test_disabled_record_failure_noop(self) -> None:
        g = BruteForceGuard(max_attempts=1, window_seconds=60, lockout_seconds=120, enabled=False)
        g.record_failure("x")
        # Enabled check returns no record, count stays 0
        assert g.failure_count("x") == 0

    def test_disabled_record_success_noop(self) -> None:
        g = BruteForceGuard(max_attempts=1, window_seconds=60, lockout_seconds=120, enabled=False)
        g.record_success("x")  # must not raise

    def test_disabled_is_locked_false(self) -> None:
        g = BruteForceGuard(max_attempts=1, window_seconds=60, lockout_seconds=120, enabled=False)
        assert not g.is_locked("x")


# ── Thread safety (smoke test) ────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_failures_do_not_panic(self) -> None:
        g = BruteForceGuard(max_attempts=100, window_seconds=60, lockout_seconds=300)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(10):
                    g.record_failure("shared-ip")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(5)]
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

    def test_get_brute_force_guard_returns_instance(self) -> None:
        from konjoai.config import Settings

        stub = Settings.model_construct(
            brute_force_enabled=False,
            brute_force_max_attempts=5,
            brute_force_window_seconds=60,
            brute_force_lockout_seconds=300,
        )
        with patch("konjoai.auth.brute_force.get_settings", return_value=stub):
            g = get_brute_force_guard()
        assert isinstance(g, BruteForceGuard)

    def test_singleton_same_instance(self) -> None:
        from konjoai.config import Settings

        stub = Settings.model_construct(
            brute_force_enabled=False,
            brute_force_max_attempts=5,
            brute_force_window_seconds=60,
            brute_force_lockout_seconds=300,
        )
        with patch("konjoai.auth.brute_force.get_settings", return_value=stub):
            a = get_brute_force_guard()
            b = get_brute_force_guard()
        assert a is b

    def test_reset_singleton_forces_recreation(self) -> None:
        from konjoai.config import Settings

        stub = Settings.model_construct(
            brute_force_enabled=False,
            brute_force_max_attempts=5,
            brute_force_window_seconds=60,
            brute_force_lockout_seconds=300,
        )
        with patch("konjoai.auth.brute_force.get_settings", return_value=stub):
            a = get_brute_force_guard()
            _reset_singleton()
            b = get_brute_force_guard()
        assert a is not b

    def test_singleton_respects_settings(self) -> None:
        from konjoai.config import Settings

        stub = Settings.model_construct(
            brute_force_enabled=True,
            brute_force_max_attempts=3,
            brute_force_window_seconds=30,
            brute_force_lockout_seconds=120,
        )
        with patch("konjoai.auth.brute_force.get_settings", return_value=stub):
            g = get_brute_force_guard()
        assert g.enabled is True
        assert g.max_attempts == 3
        assert g.window_seconds == 30
        assert g.lockout_seconds == 120


# ── Integration with get_tenant_id dep ───────────────────────────────────────


@dataclass
class _BFSettingsStub:
    multi_tenancy_enabled: bool = True
    jwt_secret_key: str = "secret"
    jwt_algorithm: str = "HS256"
    tenant_id_claim: str = "sub"
    api_key_auth_enabled: bool = False
    api_keys: list = None  # type: ignore[assignment]
    brute_force_enabled: bool = True
    brute_force_max_attempts: int = 3
    brute_force_window_seconds: int = 60
    brute_force_lockout_seconds: int = 300

    def __post_init__(self) -> None:
        if self.api_keys is None:
            self.api_keys = []


class _FakeRequest:
    def __init__(self, ip: str = "10.0.0.1") -> None:
        self.headers: dict[str, str] = {}
        self.client = (ip, 0)
        self.url = MagicMock()
        self.url.path = "/query"


class TestGetTenantIdBruteForce:
    async def _collect(self, gen):
        return await gen.__anext__()

    @pytest.mark.asyncio
    async def test_locked_ip_returns_429(self) -> None:
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        from konjoai.auth.deps import _resolve_tenant_id

        stub = _BFSettingsStub()
        req = _FakeRequest(ip="6.6.6.6")
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="tok")

        # Simulate a locked-out guard
        mock_guard = MagicMock()
        mock_guard.check_ip.side_effect = IPLockedOut("6.6.6.6", 300, time.monotonic() + 300)

        with patch("konjoai.auth.deps.get_settings", return_value=stub):
            with patch("konjoai.auth.deps.get_brute_force_guard", return_value=mock_guard):
                gen = _resolve_tenant_id(request=req, credentials=creds)
                with pytest.raises(HTTPException) as exc_info:
                    await self._collect(gen)
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_failed_jwt_increments_brute_force(self) -> None:
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        from konjoai.auth.deps import _resolve_tenant_id

        stub = _BFSettingsStub()
        req = _FakeRequest(ip="7.7.7.7")
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="bad_tok")

        mock_guard = MagicMock()
        mock_guard.check_ip.return_value = None

        with patch("konjoai.auth.deps.get_settings", return_value=stub):
            with patch("konjoai.auth.deps.get_brute_force_guard", return_value=mock_guard):
                with patch("konjoai.auth.deps.decode_token", side_effect=ValueError("bad")):
                    gen = _resolve_tenant_id(request=req, credentials=creds)
                    with pytest.raises(HTTPException):
                        await self._collect(gen)

        mock_guard.record_failure.assert_called_once_with("7.7.7.7")

    @pytest.mark.asyncio
    async def test_successful_jwt_clears_brute_force(self) -> None:
        from fastapi.security import HTTPAuthorizationCredentials

        from konjoai.auth.deps import _resolve_tenant_id
        from konjoai.auth.jwt_auth import TenantClaims

        stub = _BFSettingsStub()
        req = _FakeRequest(ip="8.8.8.8")
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="good_tok")
        fake_claims = TenantClaims(tenant_id="happy-tenant")

        mock_guard = MagicMock()
        mock_guard.check_ip.return_value = None

        with patch("konjoai.auth.deps.get_settings", return_value=stub):
            with patch("konjoai.auth.deps.get_brute_force_guard", return_value=mock_guard):
                with patch("konjoai.auth.deps.decode_token", return_value=fake_claims):
                    gen = _resolve_tenant_id(request=req, credentials=creds)
                    result = await self._collect(gen)

        assert result == "happy-tenant"
        mock_guard.record_success.assert_called_once_with("8.8.8.8")
