"""Streaming response cache — store SSE chunk sequences and replay on cache hit.

Design
------
- ``StreamingResponseCache`` wraps a private :class:`SemanticCache` instance
  (separate from the regular query cache) to avoid key collisions.
- On a cache miss the caller records tokens via individual :class:`StreamChunk`
  objects and calls :meth:`StreamingResponseCache.store` after the stream ends.
- On a cache hit :meth:`StreamingResponseCache.replay` yields SSE frames:
  - Each non-final frame: ``data: {"token": "...", "done": false, "cache_hit": true}``
  - Final frame: ``data: {...metadata..., "done": true, "cache_hit": true}``
- ``replay_delay_ms > 0`` inserts a fixed sleep between chunks (faithful pacing).
  ``replay_delay_ms == 0`` (default) emits all chunks without sleeping.
- K3: returns ``None`` from :func:`get_streaming_cache` when either
  ``cache_enabled`` or ``cache_stream_enabled`` is ``False``.
- K4: inner :class:`SemanticCache` asserts ``q_vec`` is float32 at ``store()``.
- K5: stdlib only (``asyncio``, ``json``, ``threading``, ``time``,
  ``dataclasses``); numpy already required.
- K6: additive — no change to existing cache or query-route contracts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from konjoai.cache.semantic_cache import SemanticCache

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "StreamChunk",
    "StreamingCacheEntry",
    "StreamingResponseCache",
    "get_streaming_cache",
    "_reset_singleton",
]


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class StreamChunk:
    """A single token emitted during a streaming response.

    Args:
        token: The text fragment yielded by the generator.
        delay_ms: Milliseconds elapsed before this chunk was emitted during
            the original stream.  Used when ``replay_delay_ms == 0``.
    """

    token: str
    delay_ms: float = 0.0


@dataclass
class StreamingCacheEntry:
    """Stored representation of a complete streaming response.

    Args:
        question: The original question string (for diagnostics only).
        chunks: Ordered list of tokens and their inter-chunk timing.
        final_metadata: Payload for the SSE ``done`` frame
            (``model``, ``sources``, ``intent``).
        created_at: Monotonic timestamp at store time.
    """

    question: str
    chunks: list[StreamChunk]
    final_metadata: dict
    created_at: float = field(default_factory=time.monotonic)


# ── Cache ─────────────────────────────────────────────────────────────────────


class StreamingResponseCache:
    """Cache that stores and replays complete SSE streaming responses.

    Wraps a private :class:`SemanticCache` instance so streaming entries
    never collide with regular (non-streaming) query cache entries.

    Args:
        inner_cache: Dedicated ``SemanticCache`` for streaming entries.
        replay_delay_ms: Fixed inter-chunk delay in milliseconds during replay.
            ``0`` (default) replays without sleeping.
        max_chunks: Safety cap — entries with more chunks are silently dropped.
    """

    def __init__(
        self,
        inner_cache: SemanticCache,
        replay_delay_ms: float = 0.0,
        max_chunks: int = 10_000,
    ) -> None:
        self._cache = inner_cache
        self.replay_delay_ms = replay_delay_ms
        self.max_chunks = max_chunks

    # ── Public API ────────────────────────────────────────────────────────────

    def lookup(self, question: str, q_vec: np.ndarray) -> StreamingCacheEntry | None:
        """Return a cached streaming entry, or ``None`` on miss."""
        result = self._cache.lookup(question, q_vec)
        if isinstance(result, StreamingCacheEntry):
            return result
        return None

    def store(
        self,
        question: str,
        q_vec: np.ndarray,
        chunks: list[StreamChunk],
        final_metadata: dict,
    ) -> None:
        """Persist a streaming response for future replay.

        Silently skips entries that exceed ``max_chunks`` (K1 — no raises).
        """
        if not chunks:
            return
        if len(chunks) > self.max_chunks:
            logger.warning(
                "streaming cache: skipping store — chunk count %d exceeds max_chunks %d",
                len(chunks),
                self.max_chunks,
            )
            return
        entry = StreamingCacheEntry(
            question=question,
            chunks=list(chunks),
            final_metadata=dict(final_metadata),
        )
        try:
            self._cache.store(question, q_vec, entry)
        except Exception:  # noqa: BLE001
            logger.warning("streaming cache: store failed", exc_info=True)

    async def replay(self, entry: StreamingCacheEntry) -> AsyncGenerator[str, None]:
        """Yield SSE frames for a streaming cache hit.

        If ``replay_delay_ms > 0`` a fixed sleep is inserted before each chunk.
        Otherwise chunks are emitted without sleeping.
        """
        fixed_delay = self.replay_delay_ms / 1000.0 if self.replay_delay_ms > 0 else 0.0
        for chunk in entry.chunks:
            if fixed_delay > 0:
                await asyncio.sleep(fixed_delay)
            yield f"data: {json.dumps({'token': chunk.token, 'done': False, 'cache_hit': True})}\n\n"
        final = {**entry.final_metadata, "done": True, "cache_hit": True}
        yield f"data: {json.dumps(final)}\n\n"

    def stats(self) -> dict:
        """Return inner cache statistics plus streaming-specific fields."""
        base = self._cache.stats()
        return {
            **base,
            "replay_delay_ms": self.replay_delay_ms,
            "max_chunks": self.max_chunks,
        }


# ── Singleton factory ─────────────────────────────────────────────────────────

_streaming_cache: StreamingResponseCache | None = None
_streaming_cache_lock = threading.Lock()


def get_streaming_cache() -> StreamingResponseCache | None:
    """Return the active streaming cache, or ``None`` when disabled.

    K3: returns ``None`` when ``cache_enabled=False`` or
    ``cache_stream_enabled=False`` — caller treats absence as a no-op.
    """
    global _streaming_cache  # noqa: PLW0603

    from konjoai.config import get_settings  # local import avoids circular dep

    settings = get_settings()
    if not settings.cache_enabled or not getattr(settings, "cache_stream_enabled", False):
        return None

    if _streaming_cache is not None:
        return _streaming_cache

    with _streaming_cache_lock:
        if _streaming_cache is not None:
            return _streaming_cache

        inner = SemanticCache(
            max_size=settings.cache_max_size,
            threshold=settings.cache_similarity_threshold,
            ttl_seconds=getattr(settings, "cache_ttl_seconds", 0),
        )
        _streaming_cache = StreamingResponseCache(
            inner_cache=inner,
            replay_delay_ms=float(getattr(settings, "cache_stream_replay_delay_ms", 0.0)),
            max_chunks=int(getattr(settings, "cache_stream_max_chunks", 10_000)),
        )
        logger.info(
            "streaming response cache initialised — max_size=%d threshold=%.2f replay_delay_ms=%.1f",
            settings.cache_max_size,
            settings.cache_similarity_threshold,
            _streaming_cache.replay_delay_ms,
        )
    return _streaming_cache


def _reset_singleton() -> None:
    """Test helper: reset the module-level singleton. Never call in production."""
    global _streaming_cache  # noqa: PLW0603
    with _streaming_cache_lock:
        _streaming_cache = None
