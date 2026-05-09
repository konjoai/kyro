"""K3 tests — Observatory live metrics shape.

Sprint K3 wires real counters from ``konjoai.cache.SemanticCache`` and the
demo's synchronous singleflight gate into ``/metrics``. The dashboard at
``demo/observatory.html`` consumes exactly the fields these tests pin down,
so every test here is a contract — break the shape and the dashboard
silently goes blank.

Konjo gates exercised:
  K1  — ``ask()`` errors (e.g. waiter-timeout) never propagate counter corruption.
  K3  — every metric is a *real* count from the live SemanticCache, not a mock.
  K6  — adding ``metrics()`` is purely additive — ``stats()`` and ``ask()`` still work.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

# demo/server.py is not a package — add the directory to sys.path.
_DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from server import DemoState  # noqa: E402


@pytest.fixture
def state() -> DemoState:
    return DemoState()


# ── 1. Empty-state contract ────────────────────────────────────────────────


def test_metrics_empty_state_has_full_shape(state: DemoState) -> None:
    """Pre-traffic metrics must still expose every field — the dashboard
    renders the cold-start view from this exact payload."""
    m = state.metrics()

    required = {
        "cache_hit_rate",
        "avg_latency_ms",
        "singleflight_ratio",
        "total_queries",
        "top_terms",
        "latency_history",
    }
    assert required.issubset(m.keys()), f"missing keys: {required - m.keys()}"

    assert m["cache_hit_rate"] == 0.0
    assert m["avg_latency_ms"] == 0.0
    assert m["singleflight_ratio"] == 0.0
    assert m["total_queries"] == 0
    assert m["top_terms"] == []
    assert m["latency_history"] == []


# ── 2. Field types ─────────────────────────────────────────────────────────


def test_metrics_field_types_match_dashboard_contract(state: DemoState) -> None:
    """The dashboard JS expects exact types. Floats for ratios, ints for counts,
    list-of-dict for top_terms, list-of-float for latency_history."""
    state.ask("What is the capital of France?")
    state.ask("How fast does shipping arrive?")
    m = state.metrics()

    assert isinstance(m["cache_hit_rate"], float)
    assert isinstance(m["avg_latency_ms"], float)
    assert isinstance(m["singleflight_ratio"], float)
    assert isinstance(m["total_queries"], int)
    assert isinstance(m["top_terms"], list)
    assert isinstance(m["latency_history"], list)

    for entry in m["top_terms"]:
        assert isinstance(entry, dict)
        assert set(entry.keys()) == {"term", "count"}
        assert isinstance(entry["term"], str)
        assert isinstance(entry["count"], int)

    for v in m["latency_history"]:
        assert isinstance(v, float)


# ── 3. Hit rate tracks SemanticCache ───────────────────────────────────────


def test_cache_hit_rate_reflects_real_semantic_cache(state: DemoState) -> None:
    """K3: not mocked — the rate must equal SemanticCache.stats()['hit_rate']."""
    state.ask("What is the capital of France?")           # miss
    state.ask("What is the capital of France?")           # exact hit
    state.ask("What is your refund policy?")              # miss

    m = state.metrics()
    cache_stats = state.cache.stats()
    assert m["cache_hit_rate"] == cache_stats["hit_rate"]
    assert m["total_hits"] == cache_stats["total_hits"]
    assert m["total_misses"] == cache_stats["total_misses"]
    # 1 hit / 3 lookups → 0.3333 (rounded to 4dp by SemanticCache.stats)
    assert m["cache_hit_rate"] == pytest.approx(1 / 3, abs=1e-3)


# ── 4. Latency history bounded + ordered ───────────────────────────────────


def test_latency_history_is_bounded_and_chronological(state: DemoState) -> None:
    """Capacity = 60. Newest sample at the end. Older samples drop off the front."""
    cap = state.LATENCY_HISTORY_LEN
    # Issue 65 queries — first 5 should fall off the front.
    for i in range(cap + 5):
        state.ask(f"unique question number {i}")

    m = state.metrics()
    assert len(m["latency_history"]) == cap
    assert m["history_capacity"] == cap
    assert m["total_queries"] == cap + 5
    # Average latency must equal the mean of the retained window — not all 65.
    expected_avg = sum(m["latency_history"]) / cap
    assert m["avg_latency_ms"] == pytest.approx(expected_avg, abs=1e-3)


# ── 5. Top terms leaderboard ───────────────────────────────────────────────


def test_top_terms_leaderboard_capped_at_ten_and_stop_words_stripped(
    state: DemoState,
) -> None:
    """Stop words and punctuation must not pollute the leaderboard. ``the``,
    ``what``, ``is`` etc. are dropped before counting."""
    state.ask("What is the capital of France?")
    state.ask("What is the capital of Germany?")
    state.ask("What is the capital of the UK?")
    state.ask("What is your refund policy?")

    m = state.metrics()
    terms = {e["term"]: e["count"] for e in m["top_terms"]}

    # No stop word should appear (server._STOP includes "the", "what", "is").
    for stop in ("the", "what", "is", "of"):
        assert stop not in terms

    # Real content terms must be counted accurately.
    assert terms.get("capital") == 3
    assert terms.get("france") == 1
    assert terms.get("germany") == 1
    # Length cap enforced.
    assert len(m["top_terms"]) <= 10


# ── 6. Singleflight ratio under concurrent load ───────────────────────────


def test_singleflight_ratio_counts_real_concurrent_collapses(
    state: DemoState,
) -> None:
    """K3: counters are real — fire 5 threads at the same fresh question and
    expect 4 of them to collapse onto the leader's compute."""
    barrier = threading.Barrier(5)
    results: list[dict] = []

    def fire() -> None:
        barrier.wait()
        results.append(state.ask("a brand new previously unseen question"))

    threads = [threading.Thread(target=fire) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    m = state.metrics()
    assert m["total_queries"] == 5
    assert m["singleflight_collapsed"] == 4
    assert m["singleflight_ratio"] == pytest.approx(4 / 5)
    assert sum(1 for r in results if r.get("collapsed")) == 4


# ── Bonus: sample corpus contract ──────────────────────────────────────────


def test_sample_queries_corpus_has_20_queries_across_3_tenants() -> None:
    """The observatory’s "replay corpus" button consumes this file directly."""
    path = _DEMO_DIR / "sample_queries.json"
    data = json.loads(path.read_text())
    assert len(data["queries"]) == 20
    tenants = {q["tenant_id"] for q in data["queries"]}
    assert len(tenants) == 3
    assert tenants == {t["tenant_id"] for t in data["tenants"]}
