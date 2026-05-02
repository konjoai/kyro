"""Sprint 22 — RedisSemanticCache + cache backend selection.

These tests use an in-process fake Redis (a small dict + sorted-list
implementation) so they exercise the full RedisSemanticCache code path
without requiring the optional ``redis`` package to be installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from unittest.mock import patch

import numpy as np
import pytest

from konjoai.auth.tenant import set_current_tenant_id
from konjoai.cache import (
    RedisSemanticCache,
    SemanticCache,
    build_redis_cache,
    get_semantic_cache,
)
from konjoai.cache.semantic_cache import _reset_cache


# ── Fake Redis client ────────────────────────────────────────────────────────


class _FakeRedis:
    """Minimal in-process Redis stand-in covering the surface used by
    :class:`RedisSemanticCache`.
    """

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, bytes]] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.expires: dict[str, int] = {}
        self.calls: list[tuple[str, tuple]] = []

    # Hashes
    def hset(self, key: str, field: str, value: bytes) -> int:
        self.calls.append(("hset", (key, field)))
        bucket = self.hashes.setdefault(key, {})
        new = field not in bucket
        bucket[field] = value
        return 1 if new else 0

    def hget(self, key: str, field: str) -> bytes | None:
        self.calls.append(("hget", (key, field)))
        return self.hashes.get(key, {}).get(field)

    def hdel(self, key: str, field: str) -> int:
        self.calls.append(("hdel", (key, field)))
        bucket = self.hashes.get(key, {})
        return 1 if bucket.pop(field, None) is not None else 0

    def hgetall(self, key: str) -> dict[bytes, bytes]:
        self.calls.append(("hgetall", (key,)))
        return {k.encode(): v for k, v in self.hashes.get(key, {}).items()}

    # ZSets
    def zadd(self, key: str, mapping: dict[str, float]) -> int:
        self.calls.append(("zadd", (key, tuple(mapping.items()))))
        bucket = self.zsets.setdefault(key, {})
        added = 0
        for member, score in mapping.items():
            if member not in bucket:
                added += 1
            bucket[member] = score
        return added

    def zrange(self, key: str, start: int, end: int) -> list[bytes]:
        self.calls.append(("zrange", (key, start, end)))
        items = sorted(self.zsets.get(key, {}).items(), key=lambda kv: kv[1])
        if end == -1:
            sliced = items[start:]
        else:
            sliced = items[start : end + 1]
        return [m.encode() for m, _ in sliced]

    def zrem(self, key: str, member: str) -> int:
        self.calls.append(("zrem", (key, member)))
        return 1 if self.zsets.get(key, {}).pop(member, None) is not None else 0

    def zcard(self, key: str) -> int:
        return len(self.zsets.get(key, {}))

    def delete(self, *keys: str) -> int:
        self.calls.append(("delete", keys))
        n = 0
        for k in keys:
            if k in self.hashes:
                del self.hashes[k]
                n += 1
            if k in self.zsets:
                del self.zsets[k]
                n += 1
        return n

    def expire(self, key: str, seconds: int) -> int:
        self.expires[key] = seconds
        return 1

    def ping(self) -> bool:
        return True


@dataclass
class _StubResp:
    answer: str

    def model_copy(self, *, update: dict) -> "_StubResp":  # mimic pydantic
        return _StubResp(answer=update.get("answer", self.answer))


def _vec(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


# ── Construction & guardrails ────────────────────────────────────────────────


class TestRedisCacheConstruction:
    def test_rejects_non_unit_threshold(self) -> None:
        with pytest.raises(ValueError):
            RedisSemanticCache(client=_FakeRedis(), threshold=0.0)

    def test_rejects_zero_max_size(self) -> None:
        with pytest.raises(ValueError):
            RedisSemanticCache(client=_FakeRedis(), max_size=0)

    def test_rejects_negative_ttl(self) -> None:
        with pytest.raises(ValueError):
            RedisSemanticCache(client=_FakeRedis(), ttl_seconds=-1)


# ── Lookup / store roundtrip ─────────────────────────────────────────────────


class TestRedisCacheRoundtrip:
    def test_exact_match_roundtrip(self) -> None:
        cache = RedisSemanticCache(client=_FakeRedis())
        v = _vec(1)
        cache.store("What is X?", v, _StubResp(answer="X is the unknown."))
        hit = cache.lookup("What is X?", v)
        assert hit is not None
        assert isinstance(hit, _StubResp)
        assert hit.answer == "X is the unknown."

    def test_cosine_match_above_threshold(self) -> None:
        cache = RedisSemanticCache(client=_FakeRedis(), threshold=0.95)
        v = _vec(2)
        cache.store("Question alpha", v, _StubResp(answer="alpha-answer"))
        # Slightly perturbed but very close vector
        v2 = v.copy()
        v2[0] += 0.001
        v2 = (v2 / np.linalg.norm(v2)).astype(np.float32)
        hit = cache.lookup("Question beta", v2)
        assert hit is not None
        assert hit.answer == "alpha-answer"

    def test_cosine_match_below_threshold_misses(self) -> None:
        cache = RedisSemanticCache(client=_FakeRedis(), threshold=0.99)
        cache.store("alpha", _vec(3), _StubResp(answer="A"))
        miss = cache.lookup("orthogonal", _vec(99))
        assert miss is None

    def test_lookup_question_normalised(self) -> None:
        cache = RedisSemanticCache(client=_FakeRedis())
        v = _vec(4)
        cache.store("  Hello WORLD  ", v, _StubResp(answer="hi"))
        assert cache.lookup("hello world", v).answer == "hi"

    def test_store_rejects_non_float32(self) -> None:
        cache = RedisSemanticCache(client=_FakeRedis())
        with pytest.raises(AssertionError):
            cache.store("q", np.zeros(4, dtype=np.float64), _StubResp(answer="a"))

    def test_lookup_returns_none_on_empty_cache(self) -> None:
        cache = RedisSemanticCache(client=_FakeRedis())
        assert cache.lookup("anything", _vec(5)) is None


# ── LRU eviction ─────────────────────────────────────────────────────────────


class TestRedisCacheLRU:
    def test_eviction_drops_oldest(self) -> None:
        client = _FakeRedis()
        cache = RedisSemanticCache(client=client, max_size=2, threshold=0.999)
        cache.store("first", _vec(10), _StubResp(answer="1"))
        cache.store("second", _vec(11), _StubResp(answer="2"))
        cache.store("third", _vec(12), _StubResp(answer="3"))

        # Only two entries should remain after the third store.
        entries = client.hashes.get("kyro:cache:__anonymous__:entries", {})
        assert len(entries) == 2
        assert "first" not in entries

    def test_lookup_refreshes_lru(self) -> None:
        client = _FakeRedis()
        cache = RedisSemanticCache(client=client, max_size=2, threshold=0.999)
        cache.store("a", _vec(20), _StubResp(answer="A"))
        cache.store("b", _vec(21), _StubResp(answer="B"))
        # Touch "a" so "b" becomes oldest.
        cache.lookup("a", _vec(20))
        cache.store("c", _vec(22), _StubResp(answer="C"))
        entries = client.hashes.get("kyro:cache:__anonymous__:entries", {})
        assert "a" in entries
        assert "b" not in entries
        assert "c" in entries


# ── Tenant scoping ───────────────────────────────────────────────────────────


class TestRedisCacheTenantScoping:
    def test_tenant_isolated_buckets(self) -> None:
        client = _FakeRedis()
        cache = RedisSemanticCache(client=client)
        v = _vec(30)

        token = set_current_tenant_id("acme")
        try:
            cache.store("shared question", v, _StubResp(answer="acme-answer"))
        finally:
            from konjoai.auth.tenant import _current_tenant_id
            _current_tenant_id.reset(token)

        token = set_current_tenant_id("globex")
        try:
            assert cache.lookup("shared question", v) is None  # different tenant
            cache.store("shared question", v, _StubResp(answer="globex-answer"))
            assert cache.lookup("shared question", v).answer == "globex-answer"
        finally:
            from konjoai.auth.tenant import _current_tenant_id
            _current_tenant_id.reset(token)

        # Acme's entry must still be intact.
        token = set_current_tenant_id("acme")
        try:
            assert cache.lookup("shared question", v).answer == "acme-answer"
        finally:
            from konjoai.auth.tenant import _current_tenant_id
            _current_tenant_id.reset(token)

    def test_anonymous_tenant_when_unset(self) -> None:
        client = _FakeRedis()
        cache = RedisSemanticCache(client=client)
        cache.store("q", _vec(40), _StubResp(answer="anon"))
        assert "kyro:cache:__anonymous__:entries" in client.hashes


# ── Invalidate / stats ───────────────────────────────────────────────────────


class TestRedisCacheInvalidateAndStats:
    def test_invalidate_drops_only_active_tenant(self) -> None:
        client = _FakeRedis()
        cache = RedisSemanticCache(client=client)

        token_a = set_current_tenant_id("a")
        try:
            cache.store("qa", _vec(50), _StubResp(answer="A"))
        finally:
            from konjoai.auth.tenant import _current_tenant_id
            _current_tenant_id.reset(token_a)

        token_b = set_current_tenant_id("b")
        try:
            cache.store("qb", _vec(51), _StubResp(answer="B"))
            cache.invalidate()
            assert cache.lookup("qb", _vec(51)) is None
        finally:
            from konjoai.auth.tenant import _current_tenant_id
            _current_tenant_id.reset(token_b)

        token_a = set_current_tenant_id("a")
        try:
            assert cache.lookup("qa", _vec(50)).answer == "A"
        finally:
            from konjoai.auth.tenant import _current_tenant_id
            _current_tenant_id.reset(token_a)

    def test_stats_track_hits_and_misses(self) -> None:
        cache = RedisSemanticCache(client=_FakeRedis())
        v = _vec(60)
        cache.store("q", v, _StubResp(answer="Y"))
        cache.lookup("q", v)
        cache.lookup("q", v)
        cache.lookup("not present", _vec(99))
        s = cache.stats()
        assert s["total_hits"] == 2
        assert s["total_misses"] == 1
        assert s["backend"] == "redis"
        assert s["hit_rate"] == round(2 / 3, 4)


# ── TTL ──────────────────────────────────────────────────────────────────────


class TestRedisCacheTTL:
    def test_ttl_zero_does_not_call_expire(self) -> None:
        client = _FakeRedis()
        cache = RedisSemanticCache(client=client, ttl_seconds=0)
        cache.store("q", _vec(70), _StubResp(answer="A"))
        assert client.expires == {}

    def test_ttl_positive_sets_expiry_on_both_keys(self) -> None:
        client = _FakeRedis()
        cache = RedisSemanticCache(client=client, ttl_seconds=120)
        cache.store("q", _vec(71), _StubResp(answer="A"))
        assert client.expires["kyro:cache:__anonymous__:entries"] == 120
        assert client.expires["kyro:cache:__anonymous__:lru"] == 120


# ── Safe-call wrapper (K1: log, never crash a request) ────────────────────────


class _BrokenRedis(_FakeRedis):
    def hget(self, *_a, **_kw):
        raise RuntimeError("connection lost")

    def hset(self, *_a, **_kw):
        raise RuntimeError("connection lost")

    def hgetall(self, *_a, **_kw):
        raise RuntimeError("connection lost")

    def zadd(self, *_a, **_kw):
        raise RuntimeError("connection lost")


class TestRedisCacheGracefulDegradation:
    def test_lookup_returns_none_when_redis_errors(self) -> None:
        cache = RedisSemanticCache(client=_BrokenRedis())
        assert cache.lookup("q", _vec(80)) is None

    def test_store_does_not_raise_on_redis_error(self) -> None:
        cache = RedisSemanticCache(client=_BrokenRedis())
        # Must not raise — request paths must stay alive even when Redis is down.
        cache.store("q", _vec(81), _StubResp(answer="A"))


# ── build_redis_cache: K3 fallback when redis missing or PING fails ──────────


class TestBuildRedisCache:
    def test_returns_none_when_redis_package_missing(self) -> None:
        import builtins
        real_import = builtins.__import__

        def _fail_redis(name: str, *args, **kwargs):
            if name == "redis":
                raise ImportError("simulated missing dep")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", _fail_redis):
            result = build_redis_cache(
                url="redis://x:6379/0",
                namespace="kyro:cache",
                max_size=10,
                threshold=0.9,
                ttl_seconds=0,
            )
        assert result is None

    def test_returns_none_when_ping_fails(self) -> None:
        import sys
        import types

        # Build a synthetic ``redis`` module exposing Redis.from_url(...).ping() -> raise.
        fake_module = types.ModuleType("redis")

        class _Boom:
            @classmethod
            def from_url(cls, _url):  # noqa: D401
                instance = cls()
                return instance

            def ping(self):
                raise ConnectionError("nope")

        fake_module.Redis = _Boom  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"redis": fake_module}):
            result = build_redis_cache(
                url="redis://x:6379/0",
                namespace="kyro:cache",
                max_size=10,
                threshold=0.9,
                ttl_seconds=0,
            )
        assert result is None


# ── Factory dispatch in get_semantic_cache() ─────────────────────────────────


@dataclass
class _Settings:
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.95
    cache_max_size: int = 64
    cache_backend: str = "memory"
    cache_redis_url: str = "redis://localhost:6379/0"
    cache_redis_namespace: str = "kyro:cache"
    cache_redis_ttl_seconds: int = 0


class TestSemanticCacheFactoryDispatch:
    def setup_method(self) -> None:
        _reset_cache()

    def teardown_method(self) -> None:
        _reset_cache()

    def test_returns_none_when_disabled(self) -> None:
        with patch(
            "konjoai.config.get_settings",
            return_value=_Settings(cache_enabled=False),
        ):
            assert get_semantic_cache() is None

    def test_returns_in_memory_when_backend_memory(self) -> None:
        with patch(
            "konjoai.config.get_settings",
            return_value=_Settings(cache_backend="memory"),
        ):
            cache = get_semantic_cache()
        assert isinstance(cache, SemanticCache)

    def test_returns_redis_backend_when_configured(self) -> None:
        fake_cache = RedisSemanticCache(client=_FakeRedis())
        with (
            patch(
                "konjoai.config.get_settings",
                return_value=_Settings(cache_backend="redis"),
            ),
            patch(
                "konjoai.cache.redis_cache.build_redis_cache",
                return_value=fake_cache,
            ) as build_mock,
        ):
            cache = get_semantic_cache()
        assert cache is fake_cache
        assert build_mock.called

    def test_falls_back_to_memory_when_redis_build_returns_none(self) -> None:
        with (
            patch(
                "konjoai.config.get_settings",
                return_value=_Settings(cache_backend="redis"),
            ),
            patch(
                "konjoai.cache.redis_cache.build_redis_cache",
                return_value=None,
            ),
        ):
            cache = get_semantic_cache()
        assert isinstance(cache, SemanticCache)

    def test_unknown_backend_falls_back_to_memory(self) -> None:
        with patch(
            "konjoai.config.get_settings",
            return_value=_Settings(cache_backend="cassandra"),
        ):
            cache = get_semantic_cache()
        assert isinstance(cache, SemanticCache)


# ── Backend protocol parity ───────────────────────────────────────────────────


class TestBackendContractParity:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda: SemanticCache(max_size=8, threshold=0.95),
            lambda: RedisSemanticCache(client=_FakeRedis(), max_size=8, threshold=0.95),
        ],
    )
    def test_implements_lookup_store_invalidate_stats(self, factory) -> None:
        cache = factory()
        for op in ("lookup", "store", "invalidate", "stats"):
            assert callable(getattr(cache, op)), f"missing {op} on {type(cache).__name__}"

    def test_both_backends_roundtrip_identically(self) -> None:
        v = _vec(200)
        for cache in (
            SemanticCache(max_size=4, threshold=0.95),
            RedisSemanticCache(client=_FakeRedis(), max_size=4, threshold=0.95),
        ):
            cache.store("hello", v, _StubResp(answer="hi"))
            hit = cache.lookup("hello", v)
            assert hit is not None and hit.answer == "hi"
            cache.invalidate()
            assert cache.lookup("hello", v) is None


# Silence the unused-import warning for the typing helper.
assert Iterable is not None
