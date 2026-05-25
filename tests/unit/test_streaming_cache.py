"""Tests for konjoai/cache/streaming.py — Sprint 29.

Coverage
--------
- StreamChunk: construction, defaults.
- StreamingCacheEntry: construction, fields, default created_at.
- StreamingResponseCache:
  - lookup miss (empty cache).
  - lookup exact hit.
  - lookup semantic hit (similar q_vec over threshold).
  - lookup miss below threshold.
  - lookup returns None when inner cache holds non-StreamingCacheEntry object.
  - store + roundtrip.
  - store silently skips empty chunk list.
  - store silently skips chunk list exceeding max_chunks.
  - store exception is logged as warning, never raises (K1).
  - replay — all chunks yielded in order (no delay when replay_delay_ms=0).
  - replay — cache_hit=True in every frame.
  - replay — final frame carries metadata keys.
  - replay — fixed delay applied when replay_delay_ms > 0.
  - replay — zero delay emits without sleeping.
  - stats returns inner cache stats plus streaming-specific keys.
- Singleton factory:
  - get_streaming_cache returns None when cache_enabled=False.
  - get_streaming_cache returns None when cache_stream_enabled=False.
  - get_streaming_cache returns None when both disabled.
  - get_streaming_cache returns StreamingResponseCache when both enabled.
  - get_streaming_cache is idempotent (same instance on repeated calls).
  - _reset_singleton clears the singleton.
  - singleton respects replay_delay_ms and max_chunks from settings.
- API route integration:
  - POST /query/stream with streaming cache disabled → normal SSE stream.
  - POST /query/stream cache miss → normal SSE stream, response stored.
  - POST /query/stream cache hit → replayed SSE stream with cache_hit=True.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from konjoai.cache.semantic_cache import SemanticCache
from konjoai.cache.streaming import (
    StreamChunk,
    StreamingCacheEntry,
    StreamingResponseCache,
    _reset_singleton,
    get_streaming_cache,
)

# ── Fixtures & helpers ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset():
    _reset_singleton()
    yield
    _reset_singleton()


def _vec(*components: float) -> np.ndarray:
    v = np.array([list(components)], dtype=np.float32)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-10 else v


def _inner(threshold: float = 0.99) -> SemanticCache:
    return SemanticCache(max_size=200, threshold=threshold)


def _entry(question: str = "q", n_chunks: int = 3) -> StreamingCacheEntry:
    chunks = [StreamChunk(token=f"tok{i}", delay_ms=float(i * 10)) for i in range(n_chunks)]
    return StreamingCacheEntry(
        question=question,
        chunks=chunks,
        final_metadata={"model": "m", "sources": [], "intent": "retrieval"},
    )


def _collect_replay(entry: StreamingCacheEntry, delay_ms: float = 0.0) -> list[dict]:
    cache = StreamingResponseCache(inner_cache=_inner(), replay_delay_ms=delay_ms)

    async def _gather():
        frames = []
        async for frame in cache.replay(entry):
            assert frame.startswith("data: ")
            frames.append(json.loads(frame[len("data: ") :].strip()))
        return frames

    return asyncio.run(_gather())


# ── StreamChunk ───────────────────────────────────────────────────────────────


class TestStreamChunk:
    def test_construction(self) -> None:
        c = StreamChunk(token="hello", delay_ms=12.5)
        assert c.token == "hello"
        assert c.delay_ms == 12.5

    def test_default_delay_is_zero(self) -> None:
        c = StreamChunk(token="x")
        assert c.delay_ms == 0.0


# ── StreamingCacheEntry ───────────────────────────────────────────────────────


class TestStreamingCacheEntry:
    def test_construction(self) -> None:
        chunks = [StreamChunk(token="a"), StreamChunk(token="b")]
        e = StreamingCacheEntry(question="q", chunks=chunks, final_metadata={"model": "m"})
        assert e.question == "q"
        assert len(e.chunks) == 2
        assert e.final_metadata["model"] == "m"

    def test_created_at_is_set(self) -> None:
        e = _entry()
        assert e.created_at > 0

    def test_chunks_order_preserved(self) -> None:
        tokens = ["a", "b", "c", "d"]
        chunks = [StreamChunk(token=t) for t in tokens]
        e = StreamingCacheEntry(question="q", chunks=chunks, final_metadata={})
        assert [c.token for c in e.chunks] == tokens


# ── StreamingResponseCache.lookup ─────────────────────────────────────────────


class TestStreamingResponseCacheLookup:
    def test_lookup_miss_empty_cache(self) -> None:
        sc = StreamingResponseCache(inner_cache=_inner())
        assert sc.lookup("question", _vec(1, 0, 0)) is None

    def test_lookup_exact_hit(self) -> None:
        sc = StreamingResponseCache(inner_cache=_inner())
        q = "what is the capital?"
        v = _vec(1, 0, 0)
        entry = _entry(q)
        sc.store(q, v, entry.chunks, entry.final_metadata)
        result = sc.lookup(q, v)
        assert isinstance(result, StreamingCacheEntry)
        assert result.question == q

    def test_lookup_semantic_hit(self) -> None:
        sc = StreamingResponseCache(inner_cache=SemanticCache(max_size=100, threshold=0.90))
        q = "what is the capital?"
        v1 = _vec(1.0, 0.01, 0.0)
        v2 = _vec(1.0, 0.02, 0.0)  # close enough for 0.90 threshold
        entry = _entry(q)
        sc.store(q, v1, entry.chunks, entry.final_metadata)
        result = sc.lookup(q + " (rephrased)", v2)
        assert isinstance(result, StreamingCacheEntry)

    def test_lookup_miss_below_threshold(self) -> None:
        sc = StreamingResponseCache(inner_cache=SemanticCache(max_size=100, threshold=0.99))
        v1 = _vec(1, 0, 0)
        v2 = _vec(0, 1, 0)  # orthogonal — cosine sim = 0
        entry = _entry()
        sc.store("q", v1, entry.chunks, entry.final_metadata)
        assert sc.lookup("other q", v2) is None

    def test_lookup_non_streaming_entry_returns_none(self) -> None:
        # If inner cache holds a non-StreamingCacheEntry (e.g. from a different user),
        # lookup must return None, not leak the wrong type.
        inner = _inner()
        q = "q"
        v = _vec(1, 0, 0)
        inner.store(q, v, {"answer": "plain dict, not a StreamingCacheEntry"})
        sc = StreamingResponseCache(inner_cache=inner)
        assert sc.lookup(q, v) is None


# ── StreamingResponseCache.store ──────────────────────────────────────────────


class TestStreamingResponseCacheStore:
    def test_store_and_roundtrip(self) -> None:
        sc = StreamingResponseCache(inner_cache=_inner())
        q = "round trip?"
        v = _vec(1, 0, 0)
        entry = _entry(q, n_chunks=5)
        sc.store(q, v, entry.chunks, entry.final_metadata)
        result = sc.lookup(q, v)
        assert result is not None
        assert len(result.chunks) == 5
        assert [c.token for c in result.chunks] == [f"tok{i}" for i in range(5)]

    def test_store_skips_empty_chunks(self) -> None:
        sc = StreamingResponseCache(inner_cache=_inner())
        v = _vec(1, 0, 0)
        sc.store("q", v, [], {"model": "m"})
        assert sc.lookup("q", v) is None

    def test_store_skips_when_over_max_chunks(self) -> None:
        sc = StreamingResponseCache(inner_cache=_inner(), max_chunks=2)
        v = _vec(1, 0, 0)
        chunks = [StreamChunk(token=f"t{i}") for i in range(3)]
        sc.store("q", v, chunks, {})
        assert sc.lookup("q", v) is None

    def test_store_exactly_at_max_chunks_is_stored(self) -> None:
        sc = StreamingResponseCache(inner_cache=_inner(), max_chunks=3)
        v = _vec(1, 0, 0)
        chunks = [StreamChunk(token=f"t{i}") for i in range(3)]
        sc.store("q", v, chunks, {"model": "m"})
        result = sc.lookup("q", v)
        assert result is not None
        assert len(result.chunks) == 3

    def test_store_exception_does_not_raise(self) -> None:
        inner = MagicMock(spec=SemanticCache)
        inner.store.side_effect = RuntimeError("disk full")
        inner.lookup.return_value = None
        sc = StreamingResponseCache(inner_cache=inner)
        sc.store("q", _vec(1, 0, 0), [StreamChunk(token="t")], {})  # must not raise

    def test_store_copies_chunks_list(self) -> None:
        sc = StreamingResponseCache(inner_cache=_inner())
        v = _vec(1, 0, 0)
        chunks = [StreamChunk(token="a")]
        sc.store("q", v, chunks, {})
        chunks.append(StreamChunk(token="b"))  # mutate original
        result = sc.lookup("q", v)
        assert result is not None
        assert len(result.chunks) == 1  # stored copy is unaffected


# ── StreamingResponseCache.replay ─────────────────────────────────────────────


class TestStreamingResponseCacheReplay:
    def test_all_chunks_yielded_in_order(self) -> None:
        entry = _entry(n_chunks=4)
        frames = _collect_replay(entry)
        tokens = [f["token"] for f in frames if not f["done"]]
        assert tokens == [f"tok{i}" for i in range(4)]

    def test_final_frame_is_done(self) -> None:
        entry = _entry()
        frames = _collect_replay(entry)
        assert frames[-1]["done"] is True

    def test_cache_hit_true_in_all_frames(self) -> None:
        entry = _entry(n_chunks=3)
        frames = _collect_replay(entry)
        for f in frames:
            assert f["cache_hit"] is True

    def test_final_frame_carries_metadata(self) -> None:
        entry = StreamingCacheEntry(
            question="q",
            chunks=[StreamChunk(token="x")],
            final_metadata={"model": "gpt-4o", "sources": [{"s": 1}], "intent": "retrieval"},
        )
        frames = _collect_replay(entry)
        last = frames[-1]
        assert last["model"] == "gpt-4o"
        assert last["sources"] == [{"s": 1}]
        assert last["intent"] == "retrieval"
        assert last["done"] is True

    def test_replay_no_sleep_when_delay_zero(self) -> None:
        entry = _entry(n_chunks=10)
        import time

        start = time.perf_counter()
        _collect_replay(entry, delay_ms=0.0)
        elapsed_ms = (time.perf_counter() - start) * 1000
        # With no sleep, 10 chunks should replay in well under 100ms.
        assert elapsed_ms < 200

    def test_replay_applies_fixed_delay(self) -> None:
        # Use 10ms delay × 3 chunks → at least 20ms elapsed (between chunks).
        entry = _entry(n_chunks=3)
        import time

        start = time.perf_counter()
        _collect_replay(entry, delay_ms=30.0)
        elapsed_ms = (time.perf_counter() - start) * 1000
        # 3 chunks × 30ms = 90ms minimum. Allow 3× slack for CI.
        assert elapsed_ms >= 50

    def test_replay_yields_correct_chunk_count(self) -> None:
        entry = _entry(n_chunks=7)
        frames = _collect_replay(entry)
        non_final = [f for f in frames if not f["done"]]
        assert len(non_final) == 7

    def test_replay_empty_chunks_yields_only_final_frame(self) -> None:
        entry = StreamingCacheEntry(
            question="q",
            chunks=[],
            final_metadata={"model": "m", "sources": [], "intent": "retrieval"},
        )
        frames = _collect_replay(entry)
        assert len(frames) == 1
        assert frames[0]["done"] is True


# ── StreamingResponseCache.stats ──────────────────────────────────────────────


class TestStreamingResponseCacheStats:
    def test_stats_has_inner_cache_keys(self) -> None:
        sc = StreamingResponseCache(inner_cache=_inner())
        s = sc.stats()
        assert "size" in s
        assert "hit_rate" in s

    def test_stats_has_streaming_specific_keys(self) -> None:
        sc = StreamingResponseCache(inner_cache=_inner(), replay_delay_ms=42.0, max_chunks=999)
        s = sc.stats()
        assert s["replay_delay_ms"] == 42.0
        assert s["max_chunks"] == 999


# ── Singleton factory ─────────────────────────────────────────────────────────


@dataclass
class _SettingsStreamOff:
    cache_enabled: bool = True
    cache_stream_enabled: bool = False
    cache_max_size: int = 100
    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 0
    cache_stream_replay_delay_ms: float = 0.0
    cache_stream_max_chunks: int = 10_000


@dataclass
class _SettingsCacheOff:
    cache_enabled: bool = False
    cache_stream_enabled: bool = True
    cache_max_size: int = 100
    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 0
    cache_stream_replay_delay_ms: float = 0.0
    cache_stream_max_chunks: int = 10_000


@dataclass
class _SettingsBothOn:
    cache_enabled: bool = True
    cache_stream_enabled: bool = True
    cache_max_size: int = 100
    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 0
    cache_stream_replay_delay_ms: float = 25.0
    cache_stream_max_chunks: int = 500


class TestSingletonFactory:
    def test_returns_none_when_cache_disabled(self) -> None:
        with patch("konjoai.config.get_settings", return_value=_SettingsCacheOff()):
            assert get_streaming_cache() is None

    def test_returns_none_when_stream_disabled(self) -> None:
        with patch("konjoai.config.get_settings", return_value=_SettingsStreamOff()):
            assert get_streaming_cache() is None

    def test_returns_none_when_both_disabled(self) -> None:
        @dataclass
        class _Both:
            cache_enabled: bool = False
            cache_stream_enabled: bool = False
            cache_max_size: int = 100
            cache_similarity_threshold: float = 0.95
            cache_ttl_seconds: int = 0
            cache_stream_replay_delay_ms: float = 0.0
            cache_stream_max_chunks: int = 10_000

        with patch("konjoai.config.get_settings", return_value=_Both()):
            assert get_streaming_cache() is None

    def test_returns_instance_when_both_enabled(self) -> None:
        with patch("konjoai.config.get_settings", return_value=_SettingsBothOn()):
            sc = get_streaming_cache()
            assert isinstance(sc, StreamingResponseCache)

    def test_idempotent_returns_same_instance(self) -> None:
        with patch("konjoai.config.get_settings", return_value=_SettingsBothOn()):
            a = get_streaming_cache()
            b = get_streaming_cache()
            assert a is b

    def test_reset_clears_singleton(self) -> None:
        with patch("konjoai.config.get_settings", return_value=_SettingsBothOn()):
            a = get_streaming_cache()
            _reset_singleton()
            b = get_streaming_cache()
            assert a is not b

    def test_singleton_respects_replay_delay_ms(self) -> None:
        with patch("konjoai.config.get_settings", return_value=_SettingsBothOn()):
            sc = get_streaming_cache()
            assert sc is not None
            assert sc.replay_delay_ms == 25.0

    def test_singleton_respects_max_chunks(self) -> None:
        with patch("konjoai.config.get_settings", return_value=_SettingsBothOn()):
            sc = get_streaming_cache()
            assert sc is not None
            assert sc.max_chunks == 500


# ── API route integration ──────────────────────────────────────────────────────


import contextlib  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from konjoai.api.routes.query import router as query_router  # noqa: E402
from konjoai.generate.generator import GenerationResult  # noqa: E402
from konjoai.retrieve.hybrid import HybridResult  # noqa: E402
from konjoai.retrieve.reranker import RerankResult  # noqa: E402
from konjoai.retrieve.router import QueryIntent  # noqa: E402


@dataclass
class _SettingsStreamRoute:
    enable_query_router: bool = True
    enable_hyde: bool = False
    enable_telemetry: bool = True
    use_vectro_retriever: bool = False
    use_colbert: bool = False
    enable_crag: bool = False
    enable_self_rag: bool = False
    enable_query_decomposition: bool = False
    decomposition_max_sub_queries: int = 4
    top_k_dense: int = 5
    top_k_sparse: int = 5
    openai_model: str = "stub-model"
    request_timeout_seconds: float = 30.0
    enable_graph_rag: bool = False
    graph_rag_max_communities: int = 5
    graph_rag_similarity_threshold: float = 0.3
    otel_enabled: bool = False
    audit_enabled: bool = False
    cache_enabled: bool = False
    cache_stream_enabled: bool = False
    cache_multiturn_enabled: bool = False
    cache_poisoning_guard_enabled: bool = False


@dataclass
class _SettingsStreamCacheOn:
    enable_query_router: bool = True
    enable_hyde: bool = False
    enable_telemetry: bool = True
    use_vectro_retriever: bool = False
    use_colbert: bool = False
    enable_crag: bool = False
    enable_self_rag: bool = False
    enable_query_decomposition: bool = False
    decomposition_max_sub_queries: int = 4
    top_k_dense: int = 5
    top_k_sparse: int = 5
    openai_model: str = "stub-model"
    request_timeout_seconds: float = 30.0
    enable_graph_rag: bool = False
    graph_rag_max_communities: int = 5
    graph_rag_similarity_threshold: float = 0.3
    otel_enabled: bool = False
    audit_enabled: bool = False
    cache_enabled: bool = True
    cache_stream_enabled: bool = True
    cache_multiturn_enabled: bool = False
    cache_poisoning_guard_enabled: bool = False
    cache_max_size: int = 100
    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 0
    cache_stream_replay_delay_ms: float = 0.0
    cache_stream_max_chunks: int = 10_000


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(query_router)
    return app


def _rerank_stub() -> list[RerankResult]:
    return [
        RerankResult(content="doc content", source="a.txt", score=0.9, metadata={}),
    ]


class _GeneratorStub:
    _model = "stub-model"

    def generate(self, question: str, context: str) -> GenerationResult:
        _ = (question, context)
        return GenerationResult(
            answer="stub answer",
            model="stub-model",
            usage={"prompt_tokens": 5, "completion_tokens": 3},
        )

    def generate_stream(self, question: str, context: str):
        _ = (question, context)
        yield "hello "
        yield "world"


def _parse_sse(raw: str) -> list[dict]:
    """Parse raw SSE text into list of decoded JSON frames."""
    frames = []
    for line in raw.splitlines():
        if line.startswith("data: "):
            frames.append(json.loads(line[len("data: ") :]))
    return frames


class TestQueryStreamRoute:
    def _make_patches(self, settings, extra: list | None = None):
        """Return a list of patch objects for the base streaming route.

        Source-module patches are required for the locally-imported symbols
        inside _stream_execute (hybrid_search, rerank, get_generator, classify_intent).
        """
        base = [
            patch("konjoai.api.routes.query.get_settings", return_value=settings),
            patch("konjoai.retrieve.router.classify_intent", return_value=QueryIntent.RETRIEVAL),
            patch(
                "konjoai.retrieve.hybrid.hybrid_search",
                return_value=[
                    HybridResult(content="doc", source="a.txt", rrf_score=0.8, metadata={}),
                ],
            ),
            patch("konjoai.retrieve.reranker.rerank", return_value=_rerank_stub()),
            patch("konjoai.generate.generator.get_generator", return_value=_GeneratorStub()),
        ]
        return base + (extra or [])

    @staticmethod
    def _apply(patches):
        """Return a context manager that activates all patches."""
        stack = contextlib.ExitStack()
        for p in patches:
            stack.enter_context(p)
        return stack

    def test_stream_disabled_returns_normal_sse(self) -> None:
        settings = _SettingsStreamRoute()
        app = _make_app()
        with self._apply(self._make_patches(settings)):
            client = TestClient(app)
            resp = client.post("/query/stream", json={"question": "hello"})
        assert resp.status_code == 200
        frames = _parse_sse(resp.text)
        assert any(f.get("done") for f in frames)
        assert not any(f.get("cache_hit") for f in frames)

    def test_stream_cache_miss_produces_normal_sse(self) -> None:
        settings = _SettingsStreamCacheOn()
        stub_vec = np.ones((1, 3), dtype=np.float32)
        stub_sc = StreamingResponseCache(
            inner_cache=SemanticCache(max_size=10, threshold=0.99),
            replay_delay_ms=0.0,
        )
        app = _make_app()
        extra = [
            patch("konjoai.api.routes.query.get_streaming_cache", return_value=stub_sc),
            patch(
                "konjoai.embed.encoder.get_encoder",
                return_value=MagicMock(encode=MagicMock(return_value=stub_vec)),
            ),
        ]
        with self._apply(self._make_patches(settings, extra)):
            client = TestClient(app)
            resp = client.post("/query/stream", json={"question": "what is kyro?"})
        assert resp.status_code == 200
        frames = _parse_sse(resp.text)
        done_frames = [f for f in frames if f.get("done")]
        assert done_frames
        # cache_hit should not be True on a miss (final frame either has no key or False)
        assert not done_frames[0].get("cache_hit")

    def test_stream_cache_hit_returns_replayed_sse(self) -> None:
        settings = _SettingsStreamCacheOn()
        stub_vec = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        question = "what is the meaning of life?"

        # Pre-populate the streaming cache
        inner = SemanticCache(max_size=10, threshold=0.99)
        stub_sc = StreamingResponseCache(inner_cache=inner, replay_delay_ms=0.0)
        stored_chunks = [StreamChunk(token="42"), StreamChunk(token=" always")]
        stub_sc.store(
            question,
            stub_vec,
            stored_chunks,
            {"model": "cached-model", "sources": [], "intent": "retrieval"},
        )

        app = _make_app()
        extra = [
            patch("konjoai.api.routes.query.get_streaming_cache", return_value=stub_sc),
            patch(
                "konjoai.embed.encoder.get_encoder",
                return_value=MagicMock(encode=MagicMock(return_value=stub_vec)),
            ),
        ]
        with self._apply(self._make_patches(settings, extra)):
            client = TestClient(app)
            resp = client.post("/query/stream", json={"question": question})

        assert resp.status_code == 200
        frames = _parse_sse(resp.text)
        assert any(f.get("cache_hit") for f in frames)
        done_frames = [f for f in frames if f.get("done")]
        assert done_frames
        assert done_frames[0].get("cache_hit") is True

    def test_stream_cache_hit_content_matches_stored(self) -> None:
        settings = _SettingsStreamCacheOn()
        stub_vec = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        question = "stream content test"

        inner = SemanticCache(max_size=10, threshold=0.99)
        stub_sc = StreamingResponseCache(inner_cache=inner, replay_delay_ms=0.0)
        stored_chunks = [StreamChunk(token="chunk_A"), StreamChunk(token="chunk_B")]
        stub_sc.store(question, stub_vec, stored_chunks, {"model": "m", "sources": [], "intent": "r"})

        app = _make_app()
        extra = [
            patch("konjoai.api.routes.query.get_streaming_cache", return_value=stub_sc),
            patch(
                "konjoai.embed.encoder.get_encoder",
                return_value=MagicMock(encode=MagicMock(return_value=stub_vec)),
            ),
        ]
        with self._apply(self._make_patches(settings, extra)):
            client = TestClient(app)
            resp = client.post("/query/stream", json={"question": question})

        frames = _parse_sse(resp.text)
        token_frames = [f for f in frames if not f.get("done")]
        tokens = [f["token"] for f in token_frames]
        assert tokens == ["chunk_A", "chunk_B"]
