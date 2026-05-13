"""Tests for the Sprint 26 adaptive similarity threshold engine.

Coverage:
- classify_query: all four query types + edge cases
- ThresholdConfig: defaults, for_type(), as_dict()
- ThresholdStats: record_hit/miss, snapshot, reset, thread safety
- AdaptiveThresholdEngine: resolve(), singleton stats
- /cache/threshold_stats route: disabled → 404, enabled → JSON
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest

from konjoai.cache.threshold import (
    AdaptiveThresholdEngine,
    QueryType,
    ThresholdConfig,
    ThresholdStats,
    _reset_stats_singleton,
    classify_query,
    get_threshold_stats,
)


# ── classify_query ────────────────────────────────────────────────────────────


class TestClassifyQuery:
    def test_code_fence_detected(self) -> None:
        assert classify_query("```python\nprint('hello')\n```") == QueryType.CODE

    def test_language_keyword_detected(self) -> None:
        assert classify_query("How do I sort a list in Python?") == QueryType.CODE

    def test_sql_is_code(self) -> None:
        assert classify_query("SELECT * FROM users WHERE id = 1") == QueryType.CODE

    def test_date_is_factual(self) -> None:
        assert classify_query("What happened on 2024-01-15?") == QueryType.FACTUAL

    def test_year_is_factual(self) -> None:
        assert classify_query("What year was Docker first released?") == QueryType.FACTUAL

    def test_measurement_is_factual(self) -> None:
        assert classify_query("How many mb in a gb?") == QueryType.FACTUAL

    def test_percentage_is_factual(self) -> None:
        assert classify_query("What is 45% of 200?") == QueryType.FACTUAL

    def test_how_to_is_faq(self) -> None:
        assert classify_query("How to configure nginx") == QueryType.FAQ

    def test_how_do_is_faq(self) -> None:
        assert classify_query("How do I reset my password?") == QueryType.FAQ

    def test_what_is_is_faq(self) -> None:
        assert classify_query("What is semantic caching?") == QueryType.FAQ

    def test_what_are_is_faq(self) -> None:
        assert classify_query("What are the benefits of RAG?") == QueryType.FAQ

    def test_creative_default(self) -> None:
        assert classify_query("Tell me a story about a robot") == QueryType.CREATIVE

    def test_empty_string_is_creative(self) -> None:
        assert classify_query("") == QueryType.CREATIVE

    def test_single_word_is_creative(self) -> None:
        assert classify_query("hello") == QueryType.CREATIVE

    def test_code_wins_over_how(self) -> None:
        # "How do I use Python?" — code keyword beats FAQ rule
        assert classify_query("How do I use Python?") == QueryType.CODE

    def test_code_wins_over_factual(self) -> None:
        # contains a number but also "API" → code
        assert classify_query("What is the API rate limit for 1000 requests?") == QueryType.CODE


# ── ThresholdConfig ──────────────────────────────────────────────────────────


class TestThresholdConfig:
    def test_defaults(self) -> None:
        cfg = ThresholdConfig()
        assert cfg.factual  == pytest.approx(0.94)
        assert cfg.faq      == pytest.approx(0.85)
        assert cfg.creative == pytest.approx(0.75)
        assert cfg.code     == pytest.approx(0.92)

    def test_for_type_all_types(self) -> None:
        cfg = ThresholdConfig()
        for qt in QueryType:
            v = cfg.for_type(qt)
            assert 0.0 < v <= 1.0

    def test_as_dict_contains_all_types(self) -> None:
        d = ThresholdConfig().as_dict()
        for qt in QueryType:
            assert qt.value in d
        assert len(d) == len(QueryType)

    def test_custom_values(self) -> None:
        cfg = ThresholdConfig(factual=0.99, faq=0.70, creative=0.60, code=0.88)
        assert cfg.for_type(QueryType.FACTUAL)  == pytest.approx(0.99)
        assert cfg.for_type(QueryType.FAQ)      == pytest.approx(0.70)
        assert cfg.for_type(QueryType.CREATIVE) == pytest.approx(0.60)
        assert cfg.for_type(QueryType.CODE)     == pytest.approx(0.88)


# ── ThresholdStats ───────────────────────────────────────────────────────────


class TestThresholdStats:
    def setup_method(self) -> None:
        _reset_stats_singleton()

    def test_starts_at_zero(self) -> None:
        stats = ThresholdStats()
        snap = stats.snapshot()
        for qt in QueryType:
            entry = snap[qt.value]
            assert entry["hits"] == 0
            assert entry["misses"] == 0
            assert entry["hit_rate"] == pytest.approx(0.0)

    def test_record_hit(self) -> None:
        stats = ThresholdStats()
        stats.record_hit(QueryType.FAQ)
        snap = stats.snapshot()
        assert snap["faq"]["hits"] == 1
        assert snap["faq"]["misses"] == 0
        assert snap["faq"]["hit_rate"] == pytest.approx(1.0)

    def test_record_miss(self) -> None:
        stats = ThresholdStats()
        stats.record_miss(QueryType.FACTUAL)
        snap = stats.snapshot()
        assert snap["factual"]["misses"] == 1
        assert snap["factual"]["hit_rate"] == pytest.approx(0.0)

    def test_mixed_hit_rate(self) -> None:
        stats = ThresholdStats()
        stats.record_hit(QueryType.CODE)
        stats.record_hit(QueryType.CODE)
        stats.record_miss(QueryType.CODE)
        snap = stats.snapshot()
        assert snap["code"]["total"] == 3
        # snapshot rounds to 4dp; use abs tolerance to handle 0.6667 vs 0.6666…
        assert snap["code"]["hit_rate"] == pytest.approx(2 / 3, abs=5e-4)

    def test_reset_clears_all(self) -> None:
        stats = ThresholdStats()
        stats.record_hit(QueryType.CREATIVE)
        stats.record_miss(QueryType.FAQ)
        stats.reset()
        snap = stats.snapshot()
        for entry in snap.values():
            assert entry["hits"] == 0
            assert entry["misses"] == 0

    def test_thread_safety(self) -> None:
        stats = ThresholdStats()
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(50):
                    stats.record_hit(QueryType.FAQ)
                    stats.record_miss(QueryType.CREATIVE)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors
        snap = stats.snapshot()
        assert snap["faq"]["hits"] == 400      # 8 × 50
        assert snap["creative"]["misses"] == 400

    def test_singleton_identity(self) -> None:
        s1 = get_threshold_stats()
        s2 = get_threshold_stats()
        assert s1 is s2

    def test_reset_singleton_gives_fresh_instance(self) -> None:
        s1 = get_threshold_stats()
        _reset_stats_singleton()
        s2 = get_threshold_stats()
        assert s1 is not s2


# ── AdaptiveThresholdEngine ──────────────────────────────────────────────────


class TestAdaptiveThresholdEngine:
    def test_resolve_returns_correct_type_and_threshold(self) -> None:
        engine = AdaptiveThresholdEngine()
        qt, thresh = engine.resolve("How to install docker")
        assert qt == QueryType.FAQ
        assert thresh == pytest.approx(0.85)

    def test_resolve_code_question(self) -> None:
        engine = AdaptiveThresholdEngine()
        qt, thresh = engine.resolve("Write a python function to sort a list")
        assert qt == QueryType.CODE
        assert thresh == pytest.approx(0.92)

    def test_custom_config_propagates(self) -> None:
        cfg = ThresholdConfig(faq=0.70)
        engine = AdaptiveThresholdEngine(config=cfg)
        _, thresh = engine.resolve("What is kyro?")
        assert thresh == pytest.approx(0.70)

    def test_config_property_readable(self) -> None:
        engine = AdaptiveThresholdEngine()
        assert isinstance(engine.config, ThresholdConfig)


# ── /cache/threshold_stats route ─────────────────────────────────────────────


@dataclass
class _SettingsEnabled:
    cache_enabled: bool = True

@dataclass
class _SettingsDisabled:
    cache_enabled: bool = False


class TestThresholdStatsRoute:
    def _client(self, enabled: bool):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from konjoai.api.routes.cache import router

        app = FastAPI()
        app.include_router(router)
        settings = _SettingsEnabled() if enabled else _SettingsDisabled()
        with patch("konjoai.api.routes.cache.get_settings", return_value=settings):
            return TestClient(app)

    def test_returns_404_when_cache_disabled(self) -> None:
        client = self._client(enabled=False)
        resp = client.get("/cache/threshold_stats")
        assert resp.status_code == 404

    def test_returns_200_with_all_types_when_enabled(self) -> None:
        _reset_stats_singleton()
        client = self._client(enabled=True)
        with patch("konjoai.api.routes.cache.get_settings", return_value=_SettingsEnabled()):
            resp = client.get("/cache/threshold_stats")
        assert resp.status_code == 200
        body = resp.json()
        assert "threshold_stats" in body
        stats = body["threshold_stats"]
        for qt in QueryType:
            assert qt.value in stats
            entry = stats[qt.value]
            assert "hits" in entry and "misses" in entry and "hit_rate" in entry

    def test_stats_reflect_recorded_events(self) -> None:
        _reset_stats_singleton()
        get_threshold_stats().record_hit(QueryType.FACTUAL)
        get_threshold_stats().record_miss(QueryType.FACTUAL)

        client = self._client(enabled=True)
        with patch("konjoai.api.routes.cache.get_settings", return_value=_SettingsEnabled()):
            resp = client.get("/cache/threshold_stats")
        assert resp.status_code == 200
        factual = resp.json()["threshold_stats"]["factual"]
        assert factual["hits"] >= 1
        assert factual["misses"] >= 1
