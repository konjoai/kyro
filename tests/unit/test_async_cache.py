"""Sprint 23 — AsyncSemanticCache + singleflight stampede protection.

These tests pin down:
- pass-through behaviour (lookup/store/invalidate/stats)
- singleflight collapse: N concurrent identical misses → 1 compute call
- error propagation: every waiter sees the exception, in-flight slot freed
- tenant scoping: same question, different tenants → separate compute calls
- singleflight=False short-circuit: each caller computes independently
- stats expose stampedes_collapsed / inflight_peak / inflight_now
- offload_to_thread=False keeps everything on the loop
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass

import numpy as np
import pytest

from konjoai.auth.tenant import _current_tenant_id, set_current_tenant_id
from konjoai.cache import AsyncSemanticCache, SemanticCache
from konjoai.cache.async_cache import _inflight_key
from konjoai.cache.async_cache import wrap as async_wrap


def _vec(seed: int, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)


@dataclass
class _StubResp:
    answer: str

    def model_copy(self, *, update: dict) -> _StubResp:
        return _StubResp(answer=update.get("answer", self.answer))


# ── Pass-through behaviour ────────────────────────────────────────────────


class TestPassThrough:
    @pytest.mark.asyncio
    async def test_lookup_then_store_roundtrip(self) -> None:
        cache = AsyncSemanticCache(SemanticCache(max_size=8, threshold=0.95))
        v = _vec(1)
        assert await cache.lookup("hi", v) is None
        await cache.store("hi", v, _StubResp(answer="hello"))
        hit = await cache.lookup("hi", v)
        assert hit is not None and hit.answer == "hello"

    @pytest.mark.asyncio
    async def test_invalidate_clears_backend(self) -> None:
        backend = SemanticCache(max_size=4, threshold=0.95)
        cache = AsyncSemanticCache(backend)
        v = _vec(2)
        await cache.store("q", v, _StubResp(answer="A"))
        assert await cache.lookup("q", v) is not None
        await cache.invalidate()
        assert await cache.lookup("q", v) is None

    @pytest.mark.asyncio
    async def test_stats_extends_backend_stats_with_telemetry(self) -> None:
        cache = AsyncSemanticCache(SemanticCache(max_size=4, threshold=0.95))
        s = await cache.stats()
        # Backend stats fields preserved.
        for key in ("size", "max_size", "threshold", "total_hits", "total_misses", "hit_rate"):
            assert key in s
        # Async wrapper telemetry added.
        for key in ("singleflight_enabled", "stampedes_collapsed", "inflight_peak", "inflight_now"):
            assert key in s
        assert s["singleflight_enabled"] is True
        assert s["stampedes_collapsed"] == 0
        assert s["inflight_now"] == 0


# ── Singleflight collapse ─────────────────────────────────────────────────


class TestSingleflightCollapse:
    @pytest.mark.asyncio
    async def test_n_concurrent_misses_collapse_to_one_compute(self) -> None:
        cache = AsyncSemanticCache(SemanticCache(max_size=16, threshold=0.95))
        v = _vec(10)

        compute_calls = 0
        compute_started = asyncio.Event()
        compute_can_finish = asyncio.Event()
        lock = threading.Lock()

        async def compute() -> _StubResp:
            nonlocal compute_calls
            with lock:
                compute_calls += 1
            compute_started.set()
            await compute_can_finish.wait()
            return _StubResp(answer="THE answer")

        # Fan out 8 concurrent callers. The first will start computing;
        # the other 7 should suspend on the same in-flight future.
        tasks = [asyncio.create_task(cache.get_or_compute("the question", v, compute)) for _ in range(8)]

        await compute_started.wait()
        # At this point exactly one compute is in flight.
        assert compute_calls == 1
        # And the inflight peak should be 1 (only the leader holds the slot).
        assert len(cache._inflight) == 1

        compute_can_finish.set()
        results = await asyncio.gather(*tasks)

        # Every caller saw the exact same answer.
        assert all(r.answer == "THE answer" for r in results)
        # compute() ran exactly once across all 8 callers.
        assert compute_calls == 1

        s = await cache.stats()
        assert s["stampedes_collapsed"] == 7  # 8 callers — 1 leader = 7 followers
        assert s["inflight_now"] == 0
        assert s["inflight_peak"] >= 1

    @pytest.mark.asyncio
    async def test_repeat_question_after_first_resolves_is_a_real_cache_hit(self) -> None:
        cache = AsyncSemanticCache(SemanticCache(max_size=8, threshold=0.95))
        v = _vec(11)

        async def compute() -> _StubResp:
            return _StubResp(answer="X")

        first = await cache.get_or_compute("q", v, compute)
        assert first.answer == "X"
        # The next caller hits the cache directly — singleflight not consulted.
        s_before = await cache.stats()
        second = await cache.get_or_compute("q", v, compute)
        assert second.answer == "X"
        s_after = await cache.stats()
        # No new stampede collapse (it never went into the singleflight path).
        assert s_after["stampedes_collapsed"] == s_before["stampedes_collapsed"]


# ── Error propagation ─────────────────────────────────────────────────────


class TestErrorPropagation:
    @pytest.mark.asyncio
    async def test_compute_exception_propagates_to_every_waiter(self) -> None:
        # offload_to_thread=False keeps all I/O on the event loop so the
        # five tasks run sequentially before the gate opens — avoids a race
        # where async.to_thread() lookups complete in arbitrary order and
        # some tasks miss the singleflight window.
        cache = AsyncSemanticCache(SemanticCache(max_size=4, threshold=0.95), offload_to_thread=False)
        v = _vec(20)

        compute_started = asyncio.Event()
        gate = asyncio.Event()
        attempts = 0

        async def compute() -> _StubResp:
            nonlocal attempts
            attempts += 1
            compute_started.set()
            await gate.wait()
            raise RuntimeError("LLM exploded")

        tasks = [asyncio.create_task(cache.get_or_compute("boom", v, compute)) for _ in range(5)]
        await compute_started.wait()
        gate.set()

        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(r, RuntimeError) for r in results)
        assert all("LLM exploded" in str(r) for r in results)
        # compute() was only invoked by the leader.
        assert attempts == 1
        # In-flight slot was freed so a retry can proceed.
        assert len(cache._inflight) == 0

    @pytest.mark.asyncio
    async def test_after_error_next_caller_can_retry(self) -> None:
        cache = AsyncSemanticCache(SemanticCache(max_size=4, threshold=0.95))
        v = _vec(21)
        attempts = 0

        async def flaky() -> _StubResp:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("transient")
            return _StubResp(answer="recovered")

        with pytest.raises(RuntimeError):
            await cache.get_or_compute("flaky", v, flaky)
        # Retry succeeds — the in-flight slot is gone.
        result = await cache.get_or_compute("flaky", v, flaky)
        assert result.answer == "recovered"
        assert attempts == 2


# ── Tenant scoping ────────────────────────────────────────────────────────


class TestTenantScoping:
    def test_inflight_key_includes_tenant(self) -> None:
        assert _inflight_key("Q", "acme") != _inflight_key("Q", "globex")
        assert _inflight_key("Q", None) == _inflight_key("Q", None)
        # Same tenant, different surrounding whitespace/case → same key.
        assert _inflight_key("  Hello  ", "t1") == _inflight_key("hello", "t1")

    @pytest.mark.asyncio
    async def test_two_tenants_each_compute_independently(self) -> None:
        cache = AsyncSemanticCache(SemanticCache(max_size=16, threshold=0.95))
        v = _vec(30)
        compute_calls = 0
        gate = asyncio.Event()
        compute_started = asyncio.Event()

        async def compute() -> _StubResp:
            nonlocal compute_calls
            compute_calls += 1
            if compute_calls >= 2:
                compute_started.set()
            await gate.wait()
            return _StubResp(answer="answer")

        async def fire(tenant: str) -> object:
            token = set_current_tenant_id(tenant)
            try:
                return await cache.get_or_compute("shared question", v, compute)
            finally:
                _current_tenant_id.reset(token)

        t_a = asyncio.create_task(fire("acme"))
        t_b = asyncio.create_task(fire("globex"))
        # Wait until both leaders have actually started computing — by then
        # they must each hold a separate in-flight slot.
        await compute_started.wait()
        assert len(cache._inflight) == 2
        gate.set()
        await asyncio.gather(t_a, t_b)
        # NOT collapsed — different tenants got their own compute call.
        assert compute_calls == 2


# ── singleflight=False ────────────────────────────────────────────────────


class TestSingleflightDisabled:
    @pytest.mark.asyncio
    async def test_each_caller_computes_independently(self) -> None:
        # offload_to_thread=False + asyncio.sleep(0) in compute is the only
        # deterministic way to guarantee all three lookups happen before any
        # store.  With offload_to_thread=True the thread pool may serialise
        # lookup→store for task 1 before task 2's lookup runs, so task 2 sees
        # a cache hit and skips compute — making attempts < 3 intermittently.
        cache = AsyncSemanticCache(
            SemanticCache(max_size=8, threshold=0.95),
            singleflight=False,
            offload_to_thread=False,
        )
        v = _vec(40)
        attempts = 0

        async def compute() -> _StubResp:
            nonlocal attempts
            attempts += 1
            # Yield so the other tasks can reach their lookup before we store.
            await asyncio.sleep(0)
            return _StubResp(answer=f"v{attempts}")

        # Fire 3 concurrent identical misses with singleflight off.
        await asyncio.gather(*[cache.get_or_compute("q", v, compute) for _ in range(3)])
        # All 3 invoked compute.
        assert attempts == 3
        # The cache was written by each one in turn — last writer wins.
        last = await cache.lookup("q", v)
        assert last is not None and last.answer.startswith("v")


# ── offload_to_thread=False stays on the loop ────────────────────────────


class TestOffloadFlag:
    @pytest.mark.asyncio
    async def test_offload_off_does_not_use_to_thread(self, monkeypatch) -> None:
        cache = AsyncSemanticCache(
            SemanticCache(max_size=4, threshold=0.95),
            offload_to_thread=False,
        )
        called = {"to_thread": 0}
        import konjoai.cache.async_cache as ac

        original = ac.asyncio.to_thread

        async def _spy(fn, *args, **kwargs):  # pragma: no cover — should not run
            called["to_thread"] += 1
            return original(fn, *args, **kwargs)

        monkeypatch.setattr(ac.asyncio, "to_thread", _spy)

        v = _vec(50)
        await cache.store("q", v, _StubResp(answer="A"))
        hit = await cache.lookup("q", v)
        assert hit.answer == "A"
        assert called["to_thread"] == 0


# ── wrap() factory parity ────────────────────────────────────────────────


class TestWrapFactory:
    @pytest.mark.asyncio
    async def test_wrap_returns_async_cache(self) -> None:
        backend = SemanticCache(max_size=4, threshold=0.95)
        cache = async_wrap(backend)
        assert isinstance(cache, AsyncSemanticCache)
        assert cache.backend is backend
        assert cache.singleflight_enabled is True
        s = await cache.stats()
        assert s["singleflight_enabled"] is True


# ── Configure pytest-asyncio mode ────────────────────────────────────────

# pytest-asyncio mode=auto would let us drop the per-test marker, but the
# project's conftest doesn't enable it globally. Each async test above is
# explicitly marked.
