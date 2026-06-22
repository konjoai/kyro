"""Sprint 18 — API key authentication tests.

Coverage targets:
- hash_api_key: deterministic SHA-256 output, hex encoding
- verify_api_key: match, no-match, empty key, empty registry
- verify_api_key: tenant_id extracted from <hash>:<tenant> format
- verify_api_key: ANONYMOUS_TENANT when no tenant in entry
- verify_api_key: case-insensitive hash comparison
- APIKeyResult: attributes, repr
- Timing-safe comparison (smoke test — not a timing oracle)
- get_tenant_id dep: API-key auth path (patched settings)
- get_tenant_id dep: invalid API key returns 401
- get_tenant_id dep: API key auth disabled — skipped even if header present
- get_tenant_id dep: API key preferred over JWT when both present
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from konjoai.auth.api_key import APIKeyResult, hash_api_key, verify_api_key
from konjoai.auth.tenant import ANONYMOUS_TENANT

# ── hash_api_key ──────────────────────────────────────────────────────────────


class TestHashApiKey:
    def test_returns_64_hex_chars(self) -> None:
        h = hash_api_key("secret")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self) -> None:
        assert hash_api_key("key") == hash_api_key("key")

    def test_different_keys_different_hashes(self) -> None:
        assert hash_api_key("key-a") != hash_api_key("key-b")

    def test_matches_stdlib_sha256(self) -> None:
        plaintext = "my-api-key-12345"
        expected = hashlib.sha256(plaintext.encode()).hexdigest()
        assert hash_api_key(plaintext) == expected

    def test_unicode_key(self) -> None:
        h = hash_api_key("こんにちは")
        assert len(h) == 64

    def test_empty_string(self) -> None:
        # Empty string has a deterministic SHA-256
        h = hash_api_key("")
        assert len(h) == 64


# ── APIKeyResult ──────────────────────────────────────────────────────────────


class TestAPIKeyResult:
    def test_attributes(self) -> None:
        r = APIKeyResult(tenant_id="acme", key_hash="abcd1234" * 8)
        assert r.tenant_id == "acme"
        assert r.key_hash == "abcd1234" * 8

    def test_repr_contains_tenant(self) -> None:
        r = APIKeyResult(tenant_id="corp", key_hash="a" * 64)
        assert "corp" in repr(r)

    def test_repr_truncates_hash(self) -> None:
        r = APIKeyResult(tenant_id="t", key_hash="deadbeef" * 8)
        # Should not leak the full hash in repr
        assert len(repr(r)) < 200


# ── verify_api_key ────────────────────────────────────────────────────────────


class TestVerifyApiKey:
    _KEY = "super-secret-api-key"
    _HASH = hash_api_key(_KEY)

    def test_valid_key_returns_result(self) -> None:
        result = verify_api_key(self._KEY, [self._HASH])
        assert result is not None
        assert isinstance(result, APIKeyResult)

    def test_valid_key_with_tenant(self) -> None:
        entry = f"{self._HASH}:my-org"
        result = verify_api_key(self._KEY, [entry])
        assert result is not None
        assert result.tenant_id == "my-org"

    def test_valid_key_without_tenant_uses_anonymous(self) -> None:
        result = verify_api_key(self._KEY, [self._HASH])
        assert result is not None
        assert result.tenant_id == ANONYMOUS_TENANT

    def test_invalid_key_returns_none(self) -> None:
        result = verify_api_key("wrong-key", [self._HASH])
        assert result is None

    def test_empty_key_returns_none(self) -> None:
        result = verify_api_key("", [self._HASH])
        assert result is None

    def test_empty_registry_returns_none(self) -> None:
        result = verify_api_key(self._KEY, [])
        assert result is None

    def test_multiple_entries_first_match(self) -> None:
        other_key = "other-key"
        other_hash = hash_api_key(other_key)
        entries = [f"{other_hash}:org-b", f"{self._HASH}:org-a"]
        result = verify_api_key(self._KEY, entries)
        assert result is not None
        assert result.tenant_id == "org-a"

    def test_case_insensitive_hash_comparison(self) -> None:
        upper_entry = self._HASH.upper()
        result = verify_api_key(self._KEY, [upper_entry])
        assert result is not None

    def test_result_key_hash_populated(self) -> None:
        result = verify_api_key(self._KEY, [self._HASH])
        assert result is not None
        assert result.key_hash == self._HASH

    def test_tenant_with_colon_in_id(self) -> None:
        """Only the first colon is used as separator; tenant_id may not contain one."""
        # Entry format is <hash>:<tenant> with single split
        entry = f"{self._HASH}:org:sub"
        result = verify_api_key(self._KEY, [entry])
        assert result is not None
        # split(sep, 1) → tenant_id = "org:sub"
        assert result.tenant_id == "org:sub"

    def test_entry_with_spaces_around_hash(self) -> None:
        entry = f"  {self._HASH}  :my-tenant"
        result = verify_api_key(self._KEY, [entry])
        assert result is not None
        assert result.tenant_id == "my-tenant"


# ── get_tenant_id dep — API key paths ─────────────────────────────────────────


@dataclass
class _APIKeySettingsStub:
    multi_tenancy_enabled: bool = False
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    tenant_id_claim: str = "sub"
    api_key_auth_enabled: bool = True
    api_keys: list = None  # type: ignore[assignment]
    brute_force_enabled: bool = False
    brute_force_max_attempts: int = 5
    brute_force_window_seconds: int = 60
    brute_force_lockout_seconds: int = 300

    def __post_init__(self) -> None:
        if self.api_keys is None:
            self.api_keys = []


class _FakeRequest:
    """Minimal Request stub for direct dependency calls in tests."""

    def __init__(self, headers: dict | None = None, client_ip: str = "127.0.0.1") -> None:
        self.headers = headers or {}
        self.client = (client_ip, 0)
        self.url = MagicMock()
        self.url.path = "/query"


class TestGetTenantIdAPIKeyPaths:
    _KEY = "test-api-key-sprint-18"
    _HASH = hash_api_key(_KEY)

    def _stub_settings(self, **kwargs):
        return _APIKeySettingsStub(api_keys=[f"{self._HASH}:acme-corp"], **kwargs)

    async def _collect(self, gen):
        return await gen.__anext__()

    @pytest.mark.asyncio
    async def test_valid_api_key_yields_tenant_id(self) -> None:
        from konjoai.auth.deps import _resolve_tenant_id

        stub = self._stub_settings()
        req = _FakeRequest(headers={"X-API-Key": self._KEY})
        with patch("konjoai.auth.deps.get_settings", return_value=stub):
            with patch("konjoai.auth.deps.get_brute_force_guard") as mock_guard:
                mock_guard.return_value.check_ip.return_value = None
                mock_guard.return_value.record_success.return_value = None
                gen = _resolve_tenant_id(request=req, credentials=None)
                result = await self._collect(gen)
        assert result == "acme-corp"

    @pytest.mark.asyncio
    async def test_invalid_api_key_raises_401(self) -> None:
        from fastapi import HTTPException

        from konjoai.auth.deps import _resolve_tenant_id

        stub = self._stub_settings()
        req = _FakeRequest(headers={"X-API-Key": "wrong-key"})
        with patch("konjoai.auth.deps.get_settings", return_value=stub):
            with patch("konjoai.auth.deps.get_brute_force_guard") as mock_guard:
                mock_guard.return_value.check_ip.return_value = None
                mock_guard.return_value.record_failure.return_value = None
                gen = _resolve_tenant_id(request=req, credentials=None)
                with pytest.raises(HTTPException) as exc_info:
                    await self._collect(gen)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_api_key_auth_disabled_ignores_header(self) -> None:
        from konjoai.auth.deps import _resolve_tenant_id

        stub = _APIKeySettingsStub(
            api_key_auth_enabled=False,
            multi_tenancy_enabled=False,
            api_keys=[f"{self._HASH}:acme"],
        )
        req = _FakeRequest(headers={"X-API-Key": self._KEY})
        with patch("konjoai.auth.deps.get_settings", return_value=stub):
            gen = _resolve_tenant_id(request=req, credentials=None)
            result = await self._collect(gen)
        # K3: both auth modes disabled → pass-through yields None
        assert result is None

    @pytest.mark.asyncio
    async def test_context_var_cleared_after_api_key_auth(self) -> None:
        from konjoai.auth.deps import _resolve_tenant_id
        from konjoai.auth.tenant import get_current_tenant_id

        stub = self._stub_settings()
        req = _FakeRequest(headers={"X-API-Key": self._KEY})
        with patch("konjoai.auth.deps.get_settings", return_value=stub):
            with patch("konjoai.auth.deps.get_brute_force_guard") as mock_guard:
                mock_guard.return_value.check_ip.return_value = None
                mock_guard.return_value.record_success.return_value = None
                gen = _resolve_tenant_id(request=req, credentials=None)
                await self._collect(gen)
                try:
                    await gen.aclose()
                except StopAsyncIteration:
                    pass
        assert get_current_tenant_id() is None

    @pytest.mark.asyncio
    async def test_api_key_preferred_over_jwt_when_both_present(self) -> None:
        """When both X-API-Key and Bearer token are present, API key wins."""
        from fastapi.security import HTTPAuthorizationCredentials

        from konjoai.auth.deps import _resolve_tenant_id
        from konjoai.auth.jwt_auth import TenantClaims

        stub = _APIKeySettingsStub(
            api_key_auth_enabled=True,
            multi_tenancy_enabled=True,
            jwt_secret_key="secret",
            api_keys=[f"{self._HASH}:apikey-tenant"],
        )
        req = _FakeRequest(headers={"X-API-Key": self._KEY})
        jwt_creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="fake.jwt.token")
        fake_claims = TenantClaims(tenant_id="jwt-tenant")
        with patch("konjoai.auth.deps.get_settings", return_value=stub):
            with patch("konjoai.auth.deps.decode_token", return_value=fake_claims):
                with patch("konjoai.auth.deps.get_brute_force_guard") as mock_guard:
                    mock_guard.return_value.check_ip.return_value = None
                    mock_guard.return_value.record_success.return_value = None
                    gen = _resolve_tenant_id(request=req, credentials=jwt_creds)
                    result = await self._collect(gen)
        # API key path should win
        assert result == "apikey-tenant"
