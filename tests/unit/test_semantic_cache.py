"""Unit tests for konjoai.cache.semantic_cache.

Covers:
    - exact match hit / semantic match hit / miss
    - LRU eviction
    - invalidate
    - disabled cache returns None
    - hit_count tracking
    - stats dict keys and values
    - K4 dtype assertion
    - threshold boundary (just above / just below)
    - thread safety
    - _reset_cache() test helper
    - sub-5 ms cache hit latency (performance regression gate)
"""
from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_vec(dim: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((1, dim)).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_response(answer: str = "answer") -> Any:
    """Minimal stand-in for QueryResponse (avoids FastAPI import overhead)."""
    class _FakeResponse:
        def __init__(self, answer: str) -> None:
            self.answer = answer
            self.cache_hit = False
            self.telemetry = None

        def model_copy(self, *, update: dict) -> "_FakeResponse":
            obj = _FakeResponse(self.answer)
            obj.cache_hit = update.get("cache_hit", self.cache_hit)
            obj.telemetry = update.get("telemetry", self.telemetry)
            return obj

    return _FakeResponse(answer)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset():
    """Reset singleton before every test."""
    from konjoai.cache.semantic_cache import _reset_cache
    _reset_cache()
    yield
    _reset_cache()


@pytest.fixture()
def cache():
    from konjoai.cache.semantic_cache import SemanticCache
    return SemanticCache(max_size=5, threshold=0.95)


# ---------------------------------------------------------------------------
# 1. Exact match hit
# ---------------------------------------------------------------------------

def test_exact_match_hit(cache):
    q = "What is Konjo KORE?"
    v = _rand_vec()
    resp = _make_response()
    cache.store(q, v, resp)

    result = cache.lookup(q, v)
    assert result is resp


def test_exact_match_hit_normalises_whitespace(cache):
    q = "  What is Konjo KORE?  "
    v = _rand_vec()
    resp = _make_response()
    cache.store(q, v, resp)

    result = cache.lookup("what is konjo kore?", v)
    assert result is resp


# ---------------------------------------------------------------------------
# 2. Semantic (cosine) match hit
# ---------------------------------------------------------------------------

def test_semantic_match_hit(cache):
    dim = 128
    rng = np.random.default_rng(42)
    base = rng.standard_normal(dim).astype(np.float32)
    # Slightly perturbed vector — high cosine similarity
    perturbed = base + rng.standard_normal(dim).astype(np.float32) * 0.01
    v1 = (base / np.linalg.norm(base)).reshape(1, -1)
    v2 = (perturbed / np.linalg.norm(perturbed)).reshape(1, -1)

    sim = float(np.dot(v1.ravel(), v2.ravel()))
    assert sim >= 0.95, f"fixture vecs too dissimilar ({sim:.4f}) — adjust perturbation"

    resp = _make_response("semantic hit")
    cache.store("question a", v1, resp)
    result = cache.lookup("question b", v2)
    assert result is resp


# ---------------------------------------------------------------------------
# 3. Semantic miss
# ---------------------------------------------------------------------------

def test_semantic_miss(cache):
    v1 = _rand_vec(dim=64, seed=0)
    v2 = _rand_vec(dim=64, seed=99)   # orthogonal-ish — low similarity
    cache.store("question a", v1, _make_response())
    result = cache.lookup("question b", v2)
    assert result is None


# ---------------------------------------------------------------------------
# 4. LRU eviction
# ---------------------------------------------------------------------------

def test_lru_eviction():
    from konjoai.cache.semantic_cache import SemanticCache
    c = SemanticCache(max_size=3, threshold=0.95)
    entries = []
    for i in range(3):
        v = _rand_vec(seed=i)
        resp = _make_response(f"r{i}")
        c.store(f"q{i}", v, resp)
        entries.append((f"q{i}", v, resp))

    # Insert a 4th entry → q0 should be evicted (LRU)
    v_new = _rand_vec(seed=100)
    c.store("q3", v_new, _make_response("r3"))

    assert c.stats()["size"] == 3
    # q0 was inserted first and not accessed → should be evicted
    assert c.lookup("q0", entries[0][1]) is None
    # q1, q2 should still be present
    assert c.lookup("q1", entries[1][1]) is entries[1][2]
    assert c.lookup("q2", entries[2][1]) is entries[2][2]


def test_lru_access_prevents_eviction():
    from konjoai.cache.semantic_cache import SemanticCache
    c = SemanticCache(max_size=3, threshold=0.95)
    entries = []
    for i in range(3):
        v = _rand_vec(seed=i)
        resp = _make_response(f"r{i}")
        c.store(f"q{i}", v, resp)
        entries.append((f"q{i}", v, resp))

    # Access q0 to make it recently used
    c.lookup("q0", entries[0][1])

    # Insert a 4th entry → q1 (least recently used) should be evicted
    v_new = _rand_vec(seed=100)
    c.store("q3", v_new, _make_response("r3"))

    assert c.lookup("q0", entries[0][1]) is entries[0][2]   # still in cache
    assert c.lookup("q1", entries[1][1]) is None             # evicted


# ---------------------------------------------------------------------------
# 5. Invalidate clears all entries
# ---------------------------------------------------------------------------

def test_invalidate_clears_all(cache):
    for i in range(3):
        cache.store(f"q{i}", _rand_vec(seed=i), _make_response(f"r{i}"))
    assert cache.stats()["size"] == 3
    cache.invalidate()
    assert cache.stats()["size"] == 0
    # Lookups should all miss after invalidation
    for i in range(3):
        assert cache.lookup(f"q{i}", _rand_vec(seed=i)) is None


# ---------------------------------------------------------------------------
# 6. Disabled cache returns None from get_semantic_cache
# ---------------------------------------------------------------------------

def test_disabled_returns_none():
    from konjoai.cache.semantic_cache import get_semantic_cache
    with patch("konjoai.config.get_settings") as mock_cfg:
        mock_cfg.return_value.cache_enabled = False
        result = get_semantic_cache()
    assert result is None


def test_enabled_returns_cache_instance():
    from konjoai.cache.semantic_cache import SemanticCache, get_semantic_cache
    with patch("konjoai.config.get_settings") as mock_cfg:
        mock_cfg.return_value.cache_enabled = True
        mock_cfg.return_value.cache_max_size = 100
        mock_cfg.return_value.cache_similarity_threshold = 0.95
        result = get_semantic_cache()
    assert isinstance(result, SemanticCache)


# ---------------------------------------------------------------------------
# 7. hit_count increments
# ---------------------------------------------------------------------------

def test_hit_count_increments(cache):
    v = _rand_vec()
    resp = _make_response()
    cache.store("q", v, resp)
    cache.lookup("q", v)
    cache.lookup("q", v)
    cache.lookup("q", v)
    # The entry's hit_count should be 3
    from konjoai.cache.semantic_cache import SemanticCache
    key = SemanticCache._normalise("q")
    entry = cache._exact[key]
    assert entry.hit_count == 3


# ---------------------------------------------------------------------------
# 8. stats dict structure
# ---------------------------------------------------------------------------

def test_cache_stats_keys(cache):
    stats = cache.stats()
    expected_keys = {"size", "max_size", "threshold", "total_hits", "total_misses", "hit_rate"}
    assert expected_keys == set(stats.keys())


def test_cache_stats_values(cache):
    v = _rand_vec()
    resp = _make_response()
    cache.store("q", v, resp)
    cache.lookup("q", v)           # hit
    cache.lookup("miss", _rand_vec(seed=99))  # miss

    stats = cache.stats()
    assert stats["size"] == 1
    assert stats["total_hits"] == 1
    assert stats["total_misses"] == 1
    assert stats["hit_rate"] == 0.5


# ---------------------------------------------------------------------------
# 9. K4 dtype assertion
# ---------------------------------------------------------------------------

def test_dtype_assertion_raises(cache):
    v_float64 = np.random.default_rng(0).standard_normal((1, 64)).astype(np.float64)
    with pytest.raises(AssertionError, match="float32"):
        cache.store("q", v_float64, _make_response())


def test_dtype_float32_accepted(cache):
    v = _rand_vec().astype(np.float32)
    cache.store("q", v, _make_response())   # must not raise


# ---------------------------------------------------------------------------
# 10. Threshold boundary
# ---------------------------------------------------------------------------

def test_threshold_boundary_above_hits():
    from konjoai.cache.semantic_cache import SemanticCache
    threshold = 0.90
    c = SemanticCache(max_size=10, threshold=threshold)

    dim = 256
    rng = np.random.default_rng(7)
    base = rng.standard_normal(dim).astype(np.float32)
    base /= np.linalg.norm(base)

    # Build a vector with known cosine similarity ABOVE threshold
    perturb = rng.standard_normal(dim).astype(np.float32)
    perturb -= np.dot(perturb, base) * base    # make orthogonal component
    scale = np.sqrt((1 - 0.95**2) / max(np.dot(perturb, perturb), 1e-10))
    v_close = base * 0.95 + perturb * scale
    v_close = (v_close / np.linalg.norm(v_close)).reshape(1, -1)
    v_base = base.reshape(1, -1)

    sim = float(np.dot(v_base.ravel(), v_close.ravel()))
    assert sim >= threshold, f"fixture similarity {sim:.4f} below threshold {threshold}"

    resp = _make_response()
    c.store("stored question", v_base, resp)
    result = c.lookup("different question", v_close)
    assert result is resp, f"expected semantic hit at sim={sim:.4f}"


def test_threshold_boundary_below_misses():
    from konjoai.cache.semantic_cache import SemanticCache
    c = SemanticCache(max_size=10, threshold=0.99)  # very strict

    v1 = _rand_vec(seed=0)
    v2 = _rand_vec(seed=1)   # random orthogonal-ish
    c.store("q1", v1, _make_response())
    result = c.lookup("q2", v2)
    assert result is None


# ---------------------------------------------------------------------------
# 11. Thread safety
# ---------------------------------------------------------------------------

def test_thread_safety():
    from konjoai.cache.semantic_cache import SemanticCache
    c = SemanticCache(max_size=200, threshold=0.95)

    errors: list[Exception] = []

    def writer(i: int) -> None:
        try:
            v = _rand_vec(seed=i)
            c.store(f"q{i}", v, _make_response(f"r{i}"))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def reader(i: int) -> None:
        try:
            v = _rand_vec(seed=i)
            c.lookup(f"q{i}", v)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(50)]
    threads += [threading.Thread(target=reader, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Thread safety errors: {errors}"
    assert c.stats()["size"] <= 200


# ---------------------------------------------------------------------------
# 12. _reset_cache() helper
# ---------------------------------------------------------------------------

def test_reset_cache_reinitialises_singleton():
    from konjoai.cache.semantic_cache import _reset_cache, get_semantic_cache
    with patch("konjoai.config.get_settings") as mock_cfg:
        mock_cfg.return_value.cache_enabled = True
        mock_cfg.return_value.cache_max_size = 100
        mock_cfg.return_value.cache_similarity_threshold = 0.95
        c1 = get_semantic_cache()
        assert c1 is not None
        _reset_cache()
        c2 = get_semantic_cache()
    # After reset a fresh instance is created
    assert c2 is not c1


# ---------------------------------------------------------------------------
# 13. Cache hit latency < 5 ms (performance regression gate)
# ---------------------------------------------------------------------------

def test_cache_hit_under_5ms(cache):
    """Cached responses must be served in < 5 ms (sub-LLM-call overhead)."""
    dim = 768
    rng = np.random.default_rng(0)
    v = rng.standard_normal((1, dim)).astype(np.float32)
    v /= np.linalg.norm(v)
    cache.store("perf question", v, _make_response("fast answer"))

    # Warm up (avoid cold JIT / import overhead)
    for _ in range(5):
        cache.lookup("perf question", v)

    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        result = cache.lookup("perf question", v)
    elapsed_ms = (time.perf_counter() - t0) * 1000 / N

    assert result is not None, "expected cache hit"
    assert elapsed_ms < 5.0, (
        f"Cache hit took {elapsed_ms:.2f} ms average — must be < 5 ms"
    )


# ---------------------------------------------------------------------------
# 14. Store overwrites existing entry (refresh)
# ---------------------------------------------------------------------------

def test_store_overwrites_existing(cache):
    v = _rand_vec()
    resp1 = _make_response("first answer")
    resp2 = _make_response("updated answer")
    cache.store("q", v, resp1)
    cache.store("q", v, resp2)    # overwrite
    result = cache.lookup("q", v)
    assert result is resp2
    assert cache.stats()["size"] == 1  # no duplicate entry


# ---------------------------------------------------------------------------
# 15. Empty cache always misses
# ---------------------------------------------------------------------------

def test_empty_cache_miss(cache):
    v = _rand_vec()
    result = cache.lookup("anything", v)
    assert result is None
    assert cache.stats()["total_misses"] == 1
