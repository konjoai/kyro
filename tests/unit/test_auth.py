"""Sprint 17 — Multi-tenancy + JWT auth tests.

Coverage targets:
- TenantClaims dataclass contract
- decode_token: valid token, expired, invalid signature, missing claim, no PyJWT
- _HAS_JWT flag
- get_current_tenant_id / set_current_tenant_id context var
- ANONYMOUS_TENANT constant
- Context var isolation between concurrent tasks
- get_tenant_id FastAPI dependency: K3 pass-through, 401, 503 paths
- QdrantStore tenant_id payload injection (upsert) and filter (search)
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

import konjoai.auth.jwt_auth as _jwt_module
from konjoai.auth.jwt_auth import _HAS_JWT, TenantClaims, decode_token

try:
    import qdrant_client as _qc  # noqa: F401
    _HAS_QDRANT = True
except ImportError:
    _HAS_QDRANT = False
from konjoai.auth.tenant import (
    ANONYMOUS_TENANT,
    _current_tenant_id,
    get_current_tenant_id,
    set_current_tenant_id,
)

# ── TenantClaims ──────────────────────────────────────────────────────────────


class TestTenantClaims:
    def test_required_tenant_id(self) -> None:
        c = TenantClaims(tenant_id="acme")
        assert c.tenant_id == "acme"

    def test_roles_default_empty(self) -> None:
        c = TenantClaims(tenant_id="acme")
        assert c.roles == []

    def test_exp_default_none(self) -> None:
        c = TenantClaims(tenant_id="acme")
        assert c.exp is None

    def test_full_construction(self) -> None:
        c = TenantClaims(tenant_id="corp", roles=["admin", "read"], exp=9999999999)
        assert c.tenant_id == "corp"
        assert "admin" in c.roles
        assert c.exp == 9999999999

    def test_roles_are_list(self) -> None:
        c = TenantClaims(tenant_id="x", roles=["r1"])
        assert isinstance(c.roles, list)


# ── _HAS_JWT flag ─────────────────────────────────────────────────────────────


class TestHasJwtFlag:
    def test_has_jwt_is_bool(self) -> None:
        assert isinstance(_HAS_JWT, bool)


# ── decode_token — error paths (no PyJWT required) ────────────────────────────


class TestDecodeTokenErrors:
    def test_raises_runtime_error_when_jwt_absent(self) -> None:
        with patch.object(_jwt_module, "_HAS_JWT", False):
            with pytest.raises(RuntimeError, match="PyJWT"):
                decode_token("tok", "secret")

    def test_raises_value_error_when_secret_empty(self) -> None:
        with patch.object(_jwt_module, "_HAS_JWT", True):
            with pytest.raises(ValueError, match="secret"):
                decode_token("tok", "")


# ── decode_token — happy paths (requires PyJWT) ───────────────────────────────


@pytest.mark.skipif(not _HAS_JWT, reason="PyJWT not installed")
class TestDecodeTokenWithJWT:
    _SECRET = "test-secret-key-for-kyro-unit-tests"

    def _make_token(self, payload: dict) -> str:
        import jwt
        return jwt.encode(payload, self._SECRET, algorithm="HS256")

    def test_valid_token_returns_claims(self) -> None:
        tok = self._make_token({"sub": "tenant-a", "roles": ["read"]})
        claims = decode_token(tok, self._SECRET)
        assert claims.tenant_id == "tenant-a"
        assert "read" in claims.roles

    def test_tenant_id_from_sub_claim(self) -> None:
        tok = self._make_token({"sub": "my-org"})
        claims = decode_token(tok, self._SECRET)
        assert claims.tenant_id == "my-org"

    def test_custom_tenant_id_claim(self) -> None:
        tok = self._make_token({"tenant": "enterprise-co"})
        claims = decode_token(tok, self._SECRET, tenant_id_claim="tenant")
        assert claims.tenant_id == "enterprise-co"

    def test_roles_empty_when_not_in_token(self) -> None:
        tok = self._make_token({"sub": "t1"})
        claims = decode_token(tok, self._SECRET)
        assert claims.roles == []

    def test_roles_extracted(self) -> None:
        tok = self._make_token({"sub": "t1", "roles": ["admin", "writer"]})
        claims = decode_token(tok, self._SECRET)
        assert set(claims.roles) == {"admin", "writer"}

    def test_expired_token_raises_value_error(self) -> None:
        import time
        tok = self._make_token({"sub": "t1", "exp": int(time.time()) - 3600})
        with pytest.raises(ValueError, match="expired"):
            decode_token(tok, self._SECRET)

    def test_wrong_secret_raises_value_error(self) -> None:
        tok = self._make_token({"sub": "t1"})
        with pytest.raises(ValueError, match="invalid"):
            decode_token(tok, "wrong-secret")

    def test_missing_tenant_id_claim_raises_value_error(self) -> None:
        tok = self._make_token({"role": "admin"})  # no "sub"
        with pytest.raises(ValueError, match="missing"):
            decode_token(tok, self._SECRET)

    def test_exp_field_populated_when_present(self) -> None:
        import time
        future = int(time.time()) + 3600
        tok = self._make_token({"sub": "t1", "exp": future})
        claims = decode_token(tok, self._SECRET)
        assert claims.exp == future


# ── Context var: get/set ──────────────────────────────────────────────────────


class TestTenantContextVar:
    def setup_method(self) -> None:
        _current_tenant_id.set(None)

    def test_default_is_none(self) -> None:
        assert get_current_tenant_id() is None

    def test_set_and_get_roundtrip(self) -> None:
        token = set_current_tenant_id("acme-corp")
        try:
            assert get_current_tenant_id() == "acme-corp"
        finally:
            _current_tenant_id.reset(token)

    def test_set_none_clears_context(self) -> None:
        tok1 = set_current_tenant_id("x")
        tok2 = set_current_tenant_id(None)
        try:
            assert get_current_tenant_id() is None
        finally:
            _current_tenant_id.reset(tok2)
            _current_tenant_id.reset(tok1)

    def test_reset_restores_previous_value(self) -> None:
        tok1 = set_current_tenant_id("outer")
        tok2 = set_current_tenant_id("inner")
        assert get_current_tenant_id() == "inner"
        _current_tenant_id.reset(tok2)
        assert get_current_tenant_id() == "outer"
        _current_tenant_id.reset(tok1)

    def test_anonymous_tenant_constant(self) -> None:
        assert ANONYMOUS_TENANT == "__anonymous__"
        assert isinstance(ANONYMOUS_TENANT, str)

    def test_context_var_isolated_per_task(self) -> None:
        """Context vars are not shared across concurrent asyncio tasks."""
        results: list[str | None] = []

        async def task_a() -> None:
            set_current_tenant_id("tenant-a")
            await asyncio.sleep(0)
            results.append(get_current_tenant_id())

        async def task_b() -> None:
            # b never sets the var — should see None (not tenant-a)
            await asyncio.sleep(0)
            results.append(get_current_tenant_id())

        async def run() -> None:
            await asyncio.gather(task_a(), task_b())

        asyncio.run(run())
        assert "tenant-a" in results
        assert None in results  # task_b sees its own context


# ── get_tenant_id FastAPI dependency ─────────────────────────────────────────


@dataclass
class _AuthSettingsStub:
    multi_tenancy_enabled: bool = False
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    tenant_id_claim: str = "sub"


class TestGetTenantIdDep:
    """Test the get_tenant_id async generator dependency in isolation.

    Sprint 18: tests call the internal ``_resolve_tenant_id`` helper directly
    (accepts ``request=None``) so unit tests do not need to construct a full
    FastAPI Request object.  ``get_tenant_id`` is the public FastAPI dep that
    delegates to ``_resolve_tenant_id``; its non-optional Request param is
    required by FastAPI's DI system and cannot be Optional.
    """

    def _stub_creds(self, token: str):
        from fastapi.security import HTTPAuthorizationCredentials
        return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)

    async def _collect(self, gen):
        """Drive an async generator to its first yield and return it."""
        return await gen.__anext__()

    @pytest.mark.asyncio
    async def test_returns_none_when_multi_tenancy_disabled(self) -> None:
        from konjoai.auth.deps import _resolve_tenant_id
        with patch("konjoai.auth.deps.get_settings", return_value=_AuthSettingsStub(multi_tenancy_enabled=False)):
            gen = _resolve_tenant_id(request=None, credentials=None)
            result = await self._collect(gen)
        assert result is None

    @pytest.mark.asyncio
    async def test_raises_401_when_enabled_and_no_credentials(self) -> None:
        from fastapi import HTTPException

        from konjoai.auth.deps import _resolve_tenant_id
        stub = _AuthSettingsStub(multi_tenancy_enabled=True, jwt_secret_key="s")
        with patch("konjoai.auth.deps.get_settings", return_value=stub):
            gen = _resolve_tenant_id(request=None, credentials=None)
            with pytest.raises(HTTPException) as exc_info:
                await self._collect(gen)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_raises_503_when_secret_not_configured(self) -> None:
        from fastapi import HTTPException

        from konjoai.auth.deps import _resolve_tenant_id
        stub = _AuthSettingsStub(multi_tenancy_enabled=True, jwt_secret_key="")
        creds = self._stub_creds("sometoken")
        with patch("konjoai.auth.deps.get_settings", return_value=stub):
            gen = _resolve_tenant_id(request=None, credentials=creds)
            with pytest.raises(HTTPException) as exc_info:
                await self._collect(gen)
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_raises_401_when_token_invalid(self) -> None:
        from fastapi import HTTPException

        from konjoai.auth.deps import _resolve_tenant_id
        stub = _AuthSettingsStub(multi_tenancy_enabled=True, jwt_secret_key="real-secret")
        creds = self._stub_creds("not-a-valid-jwt")
        with patch("konjoai.auth.deps.get_settings", return_value=stub):
            with patch("konjoai.auth.deps.decode_token", side_effect=ValueError("bad token")):
                gen = _resolve_tenant_id(request=None, credentials=creds)
                with pytest.raises(HTTPException) as exc_info:
                    await self._collect(gen)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_yields_tenant_id_and_sets_context_var(self) -> None:
        from konjoai.auth.deps import _resolve_tenant_id
        stub = _AuthSettingsStub(multi_tenancy_enabled=True, jwt_secret_key="s")
        creds = self._stub_creds("valid.jwt.token")
        fake_claims = TenantClaims(tenant_id="enterprise-a")
        with patch("konjoai.auth.deps.get_settings", return_value=stub):
            with patch("konjoai.auth.deps.decode_token", return_value=fake_claims):
                gen = _resolve_tenant_id(request=None, credentials=creds)
                tenant_id = await self._collect(gen)
        assert tenant_id == "enterprise-a"

    @pytest.mark.asyncio
    async def test_context_var_cleared_after_generator_exhausted(self) -> None:
        from konjoai.auth.deps import _resolve_tenant_id
        stub = _AuthSettingsStub(multi_tenancy_enabled=True, jwt_secret_key="s")
        creds = self._stub_creds("valid.jwt.token")
        fake_claims = TenantClaims(tenant_id="cleanup-test")
        with patch("konjoai.auth.deps.get_settings", return_value=stub):
            with patch("konjoai.auth.deps.decode_token", return_value=fake_claims):
                gen = _resolve_tenant_id(request=None, credentials=creds)
                await self._collect(gen)
                try:
                    await gen.aclose()  # trigger finally block
                except StopAsyncIteration:
                    pass
        # After cleanup the context var should be back to None
        assert get_current_tenant_id() is None


# ── QdrantStore tenant integration (unit-level mocking) ───────────────────────


@pytest.mark.skipif(not _HAS_QDRANT, reason="qdrant-client not installed")
class TestQdrantStoreTenantScoping:
    def test_upsert_adds_tenant_id_to_payload(self) -> None:
        """When tenant context is set, upsert injects tenant_id into payload."""
        import numpy as np

        from konjoai.store.qdrant import QdrantStore

        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []

        store = QdrantStore.__new__(QdrantStore)
        store._client = mock_client
        store._collection = "test_col"
        store._dim = 4

        embeddings = np.ones((1, 4), dtype=np.float32)
        contents = ["hello world"]
        sources = ["doc.md"]
        metadatas = [{}]

        tok = set_current_tenant_id("tenant-xyz")
        try:
            with patch("konjoai.config.get_settings") as gs:
                gs.return_value.vectro_quantize = False
                store.upsert(embeddings, contents, sources, metadatas)
        finally:
            _current_tenant_id.reset(tok)

        call_args = mock_client.upsert.call_args
        points = call_args.kwargs.get("points") or call_args.args[1] if call_args.args else call_args.kwargs["points"]
        assert points[0].payload.get("tenant_id") == "tenant-xyz"

    def test_upsert_no_tenant_id_when_context_unset(self) -> None:
        """Without tenant context, upsert does not add tenant_id to payload."""
        import numpy as np

        from konjoai.store.qdrant import QdrantStore

        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []

        store = QdrantStore.__new__(QdrantStore)
        store._client = mock_client
        store._collection = "test_col"
        store._dim = 4

        _current_tenant_id.set(None)

        embeddings = np.ones((1, 4), dtype=np.float32)
        with patch("konjoai.config.get_settings") as gs:
            gs.return_value.vectro_quantize = False
            store.upsert(embeddings, ["chunk"], ["src.md"], [{}])

        call_args = mock_client.upsert.call_args
        points = call_args.kwargs.get("points") or call_args.args[1] if call_args.args else call_args.kwargs["points"]
        assert "tenant_id" not in points[0].payload

    def test_search_applies_filter_when_tenant_set(self) -> None:
        """When tenant context is set, search passes a Qdrant Filter."""
        import numpy as np

        from konjoai.store.qdrant import QdrantStore

        mock_client = MagicMock()
        mock_client.query_points.return_value.points = []

        store = QdrantStore.__new__(QdrantStore)
        store._client = mock_client
        store._collection = "test_col"
        store._dim = 4

        tok = set_current_tenant_id("filter-tenant")
        try:
            store.search(np.ones((1, 4), dtype=np.float32), top_k=5)
        finally:
            _current_tenant_id.reset(tok)

        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs.get("query_filter") is not None

    def test_search_no_filter_when_tenant_unset(self) -> None:
        """Without tenant context, search passes no filter (backward compat)."""
        import numpy as np

        from konjoai.store.qdrant import QdrantStore

        mock_client = MagicMock()
        mock_client.query_points.return_value.points = []

        store = QdrantStore.__new__(QdrantStore)
        store._client = mock_client
        store._collection = "test_col"
        store._dim = 4

        _current_tenant_id.set(None)
        store.search(np.ones((1, 4), dtype=np.float32), top_k=5)

        call_kwargs = mock_client.query_points.call_args.kwargs
        assert call_kwargs.get("query_filter") is None
