"""Async semantic cache wrapper with singleflight stampede protection.

Sprint 23 — *Know* the problem, then *Outline*, *Nail*, *Justify*, *Optimize*.

Know
----
At scale, ``konjoai.cache.SemanticCache.lookup`` is called from inside
``await asyncio.to_thread(...)`` in the request path. Two pain points:

1. **Thread hop per lookup.** Every cache call pays a thread-pool round trip
   even though the in-memory backend is microsecond-fast. The Redis backend
   does real I/O — also wants a coroutine, not a thread.
2. **Cache stampede / thundering herd.** When N concurrent requests ask the
   same fresh question, they all miss together, they all hit the LLM
   together, and N-1 of those calls are wasted as soon as the first one's
   answer lands in the cache. The Redis fan-out introduced in Sprint 22
   makes this *worse*, not better, because pods now race each other instead
   of just intra-pod workers.

Outline
-------
``AsyncSemanticCache`` wraps any object that quacks like the
``SemanticCache`` / ``RedisSemanticCache`` contract (``lookup``, ``store``,
``invalidate``, ``stats``) and exposes ``async`` versions plus the key
new primitive: :meth:`get_or_compute`. ``get_or_compute`` is singleflight-
safe — concurrent callers asking the same normalised question collapse to
a single ``compute`` invocation; the rest await the in-flight result and
all read the cached answer the moment it lands.

Konjo invariants
----------------
- **K1**: ``compute`` exceptions propagate to every waiter; the in-flight
  slot is cleared so the next caller can retry.
- **K3**: ``singleflight=False`` makes the wrapper a thin async adapter
  with zero coordination overhead — useful when you've already deduped
  upstream.
- **K5**: pure stdlib (``asyncio``). No new hard dependency.
- **K6**: purely additive — the synchronous ``SemanticCache`` /
  ``RedisSemanticCache`` contracts are unchanged.
- **K7**: the in-flight dict is keyed inside the active tenant's namespace
  by re-using the backend's key normalisation — see :func:`_inflight_key`.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class _SyncBackend(Protocol):
    """Minimal contract every sync cache backend in the package satisfies."""

    def lookup(self, question: str, q_vec: np.ndarray) -> object | None:
        """Return a cached response, or None on a miss."""
        ...

    def store(self, question: str, q_vec: np.ndarray, response: object) -> None:
        """Store a response keyed by the question and its vector."""
        ...

    def invalidate(self) -> None:
        """Clear all entries for the active tenant."""
        ...

    def stats(self) -> dict:
        """Return hit/miss telemetry for the backend."""
        ...


def _inflight_key(question: str, tenant: str | None) -> str:
    """Identical to the lower-cased / stripped key the sync backends use,
    namespaced by the active tenant so two tenants asking the same string do
    not share a singleflight slot.
    """
    return f"{tenant or '__anonymous__'}::{(question or '').strip().lower()}"


class AsyncSemanticCache:
    """Async wrapper + singleflight gate for any sync cache backend.

    Args:
        backend: An instance of ``SemanticCache`` or ``RedisSemanticCache``
            (or anything else satisfying :class:`_SyncBackend`).
        singleflight: When True (default), concurrent ``get_or_compute``
            calls for the same normalised question collapse to one compute.
        offload_to_thread: When True, all backend calls are wrapped in
            ``asyncio.to_thread`` to keep the event loop responsive when the
            backend does I/O (e.g. Redis). Defaults to True.
        tenant_provider: Callable returning the active tenant_id string or
            None. Defaults to ``konjoai.auth.tenant.get_current_tenant_id``
            so the singleflight key matches the backend's tenant scoping.
    """

    def __init__(
        self,
        backend: _SyncBackend,
        *,
        singleflight: bool = True,
        offload_to_thread: bool = True,
        tenant_provider: Callable[[], str | None] | None = None,
    ) -> None:
        self._backend = backend
        self._singleflight = singleflight
        self._offload = offload_to_thread

        if tenant_provider is None:
            from konjoai.auth.tenant import get_current_tenant_id

            tenant_provider = get_current_tenant_id
        self._tenant_provider = tenant_provider

        self._inflight: dict[str, asyncio.Future[object | None]] = {}
        self._inflight_lock = asyncio.Lock()

        # Telemetry counters — process-local; Sprint-22 Redis stats keep the
        # cross-pod numbers.
        self._stampedes_collapsed = 0
        self._inflight_max = 0

    # ── Backend property (handy for tests / introspection) ────────────────

    @property
    def backend(self) -> _SyncBackend:
        return self._backend

    @property
    def singleflight_enabled(self) -> bool:
        return self._singleflight

    # ── Async lookup / store / invalidate ────────────────────────────────

    async def lookup(self, question: str, q_vec: np.ndarray) -> object | None:
        """Await the backend lookup, offloading to a thread when configured."""
        if self._offload:
            return await asyncio.to_thread(self._backend.lookup, question, q_vec)
        return self._backend.lookup(question, q_vec)

    async def store(self, question: str, q_vec: np.ndarray, response: object) -> None:
        """Await the backend store, offloading to a thread when configured."""
        if self._offload:
            await asyncio.to_thread(self._backend.store, question, q_vec, response)
            return
        self._backend.store(question, q_vec, response)

    async def invalidate(self) -> None:
        """Await the backend invalidate, offloading to a thread when configured."""
        if self._offload:
            await asyncio.to_thread(self._backend.invalidate)
            return
        self._backend.invalidate()

    async def stats(self) -> dict:
        """Return backend stats merged with this wrapper's singleflight telemetry."""
        if self._offload:
            base = await asyncio.to_thread(self._backend.stats)
        else:
            base = self._backend.stats()
        base.update(
            {
                "singleflight_enabled": self._singleflight,
                "stampedes_collapsed": self._stampedes_collapsed,
                "inflight_peak": self._inflight_max,
                "inflight_now": len(self._inflight),
            }
        )
        return base

    # ── The singleflight primitive ───────────────────────────────────────

    async def get_or_compute(
        self,
        question: str,
        q_vec: np.ndarray,
        compute: Callable[[], Awaitable[object]],
    ) -> object:
        """Return a cached value, or compute exactly once across concurrent callers.

        Semantics:
            1. Try the backend ``lookup`` first — fast path.
            2. On miss, atomically register an in-flight ``Future`` for the
               normalised question. If one already exists, await it (this is
               the stampede collapse).
            3. The first caller invokes ``compute()``. The result is stored
               in the backend and used to resolve the future.
            4. On exception, the future raises to every waiter and the
               in-flight slot is removed so the next caller can retry.

        ``compute`` is an async-callable; pass a tiny coroutine that does
        the LLM call, e.g. ``lambda: asyncio.to_thread(generator.generate, ...)``.
        """
        cached = await self.lookup(question, q_vec)
        if cached is not None:
            return cached

        if not self._singleflight:
            value = await compute()
            await self.store(question, q_vec, value)
            return value

        key = _inflight_key(question, self._tenant_provider())
        loop = asyncio.get_running_loop()

        async with self._inflight_lock:
            existing = self._inflight.get(key)
            if existing is not None:
                # Stampede collapse — somebody else is already computing this.
                self._stampedes_collapsed += 1
                logger.debug(
                    "singleflight collapse: waiting on inflight key=%s peers=%d",
                    key[:64], len(self._inflight),
                )
                future = existing
                # Release the lock while we await the existing future.
            else:
                future = loop.create_future()
                self._inflight[key] = future
                self._inflight_max = max(self._inflight_max, len(self._inflight))

        if existing is not None:
            return await future

        # We are the leader — actually do the work.
        try:
            try:
                value = await compute()
            except BaseException as exc:  # noqa: BLE001 — re-raised to waiters
                future.set_exception(exc)
                raise
            await self.store(question, q_vec, value)
            future.set_result(value)
            return value
        finally:
            # Always free the slot so retries are possible after errors.
            async with self._inflight_lock:
                self._inflight.pop(key, None)


def wrap(
    backend: _SyncBackend,
    *,
    singleflight: bool = True,
    offload_to_thread: bool = True,
) -> AsyncSemanticCache:
    """Convenience factory mirroring :class:`AsyncSemanticCache`.

    >>> from konjoai.cache import SemanticCache, async_cache
    >>> cache = async_cache.wrap(SemanticCache(max_size=500, threshold=0.95))
    """
    return AsyncSemanticCache(
        backend, singleflight=singleflight, offload_to_thread=offload_to_thread
    )
