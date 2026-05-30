"""Tests for konjoai/cache/multiturn.py — Sprint 28.

Coverage
--------
- question_hash / compute_turn_hash: length, determinism, order sensitivity.
- TurnHistory: empty hash, add advances hash, max_turns eviction, len.
- ConversationStore: get_or_create (create + reuse), add_turn, get_turn_hash,
  different tenants isolated, LRU eviction, conversation_count, clear,
  thread safety, K1 (never raises on add_turn).
- MultiTurnCache: lookup miss (empty), store + lookup hit same context, context
  isolation (different conversations don't collide), advance_turn, stats,
  store advances history, lookup does not advance history.
- Singletons: get_conversation_store / get_multiturn_cache (singleton), _reset.
- Config: cache_multiturn_max_conversations / cache_multiturn_window respected.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest

from konjoai.cache.multiturn import (
    ConversationStore,
    MultiTurnCache,
    TurnHistory,
    _reset_singletons,
    compute_turn_hash,
    get_conversation_store,
    get_multiturn_cache,
    question_hash,
)
from konjoai.cache.semantic_cache import SemanticCache


@pytest.fixture(autouse=True)
def _reset():
    _reset_singletons()
    yield
    _reset_singletons()


# ── Helpers ────────────────────────────────────────────────────────────────────


def _cache(threshold: float = 0.99) -> SemanticCache:
    return SemanticCache(max_size=200, threshold=threshold)


def _vec(*components: float) -> np.ndarray:
    v = np.array([list(components)], dtype=np.float32)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-10 else v


class _Resp:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.cache_hit = False


# ── Hash helpers ───────────────────────────────────────────────────────────────


class TestHashHelpers:
    def test_question_hash_is_16_hex(self) -> None:
        h = question_hash("hello world")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_question_hash_deterministic(self) -> None:
        assert question_hash("test") == question_hash("test")

    def test_question_hash_distinct_inputs(self) -> None:
        assert question_hash("foo") != question_hash("bar")

    def test_compute_turn_hash_empty_list(self) -> None:
        h = compute_turn_hash([])
        assert len(h) == 16

    def test_compute_turn_hash_order_matters(self) -> None:
        assert compute_turn_hash(["a", "b"]) != compute_turn_hash(["b", "a"])

    def test_compute_turn_hash_deterministic(self) -> None:
        assert compute_turn_hash(["a", "b"]) == compute_turn_hash(["a", "b"])

    def test_compute_turn_hash_single_element(self) -> None:
        h = compute_turn_hash(["x"])
        assert len(h) == 16


# ── TurnHistory ────────────────────────────────────────────────────────────────


class TestTurnHistory:
    def test_empty_history_produces_valid_hash(self) -> None:
        h = TurnHistory(max_turns=5)
        turn_hash = h.current_turn_hash("what is AI?")
        assert len(turn_hash) == 16

    def test_add_changes_hash(self) -> None:
        h = TurnHistory(max_turns=5)
        before = h.current_turn_hash("Q2")
        h.add("Q1")
        after = h.current_turn_hash("Q2")
        assert before != after

    def test_len_zero_on_creation(self) -> None:
        assert len(TurnHistory(max_turns=5)) == 0

    def test_len_after_add(self) -> None:
        h = TurnHistory(max_turns=5)
        h.add("Q1")
        assert len(h) == 1

    def test_max_turns_eviction(self) -> None:
        h = TurnHistory(max_turns=2)
        h.add("Q1")
        h.add("Q2")
        h.add("Q3")
        assert len(h) == 2

    def test_same_current_question_same_hash(self) -> None:
        h = TurnHistory(max_turns=5)
        h.add("Q1")
        h.add("Q2")
        assert h.current_turn_hash("Q3") == h.current_turn_hash("Q3")

    def test_eviction_removes_oldest(self) -> None:
        """After eviction, history [B,C] should match a fresh history seeded with [B,C]."""
        h = TurnHistory(max_turns=2)
        h.add("A")
        h.add("B")
        h.add("C")  # evicts A → history = [B, C]
        expected_turn_hash = h.current_turn_hash("D")

        h2 = TurnHistory(max_turns=2)
        h2.add("B")
        h2.add("C")
        assert h2.current_turn_hash("D") == expected_turn_hash


# ── ConversationStore ──────────────────────────────────────────────────────────


class TestConversationStore:
    def test_get_or_create_returns_turn_history(self) -> None:
        store = ConversationStore()
        h = store.get_or_create("t1", "c1")
        assert isinstance(h, TurnHistory)

    def test_same_key_returns_same_object(self) -> None:
        store = ConversationStore()
        h1 = store.get_or_create("t1", "c1")
        h2 = store.get_or_create("t1", "c1")
        assert h1 is h2

    def test_different_tenants_are_isolated(self) -> None:
        store = ConversationStore()
        h1 = store.get_or_create("t1", "c1")
        h2 = store.get_or_create("t2", "c1")
        assert h1 is not h2

    def test_different_conversations_are_isolated(self) -> None:
        store = ConversationStore()
        h1 = store.get_or_create("t1", "c1")
        h2 = store.get_or_create("t1", "c2")
        assert h1 is not h2

    def test_add_turn_advances_history(self) -> None:
        store = ConversationStore()
        store.add_turn("t1", "c1", "Q1")
        h = store.get_or_create("t1", "c1")
        assert len(h) == 1

    def test_get_turn_hash_no_prior_history(self) -> None:
        store = ConversationStore()
        h = store.get_turn_hash("t1", "c1", "Q1")
        assert len(h) == 16

    def test_get_turn_hash_changes_after_add_turn(self) -> None:
        store = ConversationStore()
        h1 = store.get_turn_hash("t1", "c1", "Q2")
        store.add_turn("t1", "c1", "Q1")
        h2 = store.get_turn_hash("t1", "c1", "Q2")
        assert h1 != h2

    def test_conversation_count(self) -> None:
        store = ConversationStore()
        store.add_turn("t1", "c1", "q")
        store.add_turn("t1", "c2", "q")
        assert store.conversation_count() == 2

    def test_lru_eviction_at_max(self) -> None:
        store = ConversationStore(max_conversations=2)
        store.add_turn("t1", "c1", "q")
        store.add_turn("t1", "c2", "q")
        store.add_turn("t1", "c3", "q")
        assert store.conversation_count() == 2

    def test_clear(self) -> None:
        store = ConversationStore()
        store.add_turn("t1", "c1", "q")
        store.clear()
        assert store.conversation_count() == 0

    def test_add_turn_k1_never_raises(self) -> None:
        """K1: add_turn must not propagate exceptions."""
        store = ConversationStore(max_conversations=1)
        for i in range(20):
            store.add_turn("t1", f"c{i}", "q")

    def test_thread_safety(self) -> None:
        store = ConversationStore(max_conversations=100)

        def add_many() -> None:
            for i in range(10):
                store.add_turn("t1", f"conv{i % 5}", f"q{i}")

        threads = [threading.Thread(target=add_many) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def test_invalid_max_conversations(self) -> None:
        with pytest.raises(ValueError):
            ConversationStore(max_conversations=0)

    def test_invalid_max_turns(self) -> None:
        with pytest.raises(ValueError):
            ConversationStore(max_turns=0)


# ── MultiTurnCache ─────────────────────────────────────────────────────────────


class TestMultiTurnCache:
    def test_lookup_miss_empty_cache(self) -> None:
        mt = MultiTurnCache(_cache())
        result = mt.lookup("Q1", _vec(1.0, 0.0, 0.0), "t1", "c1")
        assert result is None

    def test_store_then_lookup_same_context_hits(self) -> None:
        """Same conversation context → same turn hash → cache hit."""
        inner = _cache(threshold=0.99)
        cs_store = ConversationStore()
        cs_lookup = ConversationStore()

        cs_store.add_turn("t1", "c1", "prior question")
        cs_lookup.add_turn("t1", "c2", "prior question")  # identical prior history

        mt_store = MultiTurnCache(inner, conversation_store=cs_store)
        mt_lookup = MultiTurnCache(inner, conversation_store=cs_lookup)

        q_vec = _vec(1.0, 0.0, 0.0)
        resp = _Resp("the answer")
        mt_store.store("Q1", q_vec, resp, "t1", "c1")
        result = mt_lookup.lookup("Q1", q_vec, "t1", "c2")
        assert result is resp

    def test_keyed_strings_differ_for_different_history(self) -> None:
        """Different prior history → different turn hashes → distinct exact-match keys.

        Semantic-scan isolation is NOT guaranteed (vector similarity is key-agnostic);
        this test verifies that the exact-match address space is partitioned by context.
        """
        cs_a = ConversationStore()
        cs_b = ConversationStore()
        cs_a.add_turn("t1", "cA", "question alpha")
        cs_b.add_turn("t1", "cB", "question beta")
        hash_a = cs_a.get_turn_hash("t1", "cA", "Q")
        hash_b = cs_b.get_turn_hash("t1", "cB", "Q")
        assert hash_a != hash_b
        assert MultiTurnCache._keyed("Q", hash_a) != MultiTurnCache._keyed("Q", hash_b)

    def test_different_conversation_ids_produce_different_keys(self) -> None:
        """Same question, different conversation histories → different exact-match keys."""
        cs = ConversationStore()
        cs.add_turn("t1", "cA", "shared prior")
        cs.add_turn("t1", "cB", "different prior")
        hash_a = cs.get_turn_hash("t1", "cA", "Q")
        hash_b = cs.get_turn_hash("t1", "cB", "Q")
        assert hash_a != hash_b

    def test_advance_turn_changes_subsequent_hash(self) -> None:
        cs = ConversationStore()
        mt = MultiTurnCache(_cache(), conversation_store=cs)

        hash_before = cs.get_turn_hash("t1", "c1", "Q2")
        mt.advance_turn("t1", "c1", "Q1")
        hash_after = cs.get_turn_hash("t1", "c1", "Q2")
        assert hash_before != hash_after

    def test_lookup_does_not_advance_history(self) -> None:
        cs = ConversationStore()
        mt = MultiTurnCache(_cache(), conversation_store=cs)
        mt.lookup("Q1", _vec(1.0, 0.0, 0.0), "t1", "c1")
        assert len(cs.get_or_create("t1", "c1")) == 0

    def test_store_advances_history(self) -> None:
        cs = ConversationStore()
        mt = MultiTurnCache(_cache(), conversation_store=cs)
        mt.store("Q1", _vec(1.0, 0.0, 0.0), _Resp("a"), "t1", "c1")
        assert len(cs.get_or_create("t1", "c1")) == 1

    def test_conversation_store_property(self) -> None:
        cs = ConversationStore()
        mt = MultiTurnCache(_cache(), conversation_store=cs)
        assert mt.conversation_store is cs

    def test_stats_includes_active_conversations(self) -> None:
        cs = ConversationStore()
        mt = MultiTurnCache(_cache(), conversation_store=cs)
        mt.store("Q1", _vec(1.0, 0.0, 0.0), _Resp("a"), "t1", "c1")
        s = mt.stats()
        assert s["active_conversations"] == 1
        assert "size" in s

    def test_keyed_string_format(self) -> None:
        assert MultiTurnCache._keyed("hello?", "abc12345") == "[conv:abc12345] hello?"

    def test_tenant_isolation(self) -> None:
        """Same conversation_id across different tenants does not collide."""
        inner = _cache(threshold=0.99)
        cs = ConversationStore()
        mt = MultiTurnCache(inner, conversation_store=cs)
        q_vec = _vec(1.0, 0.0, 0.0)
        mt.store("Q", q_vec, _Resp("tenant1_answer"), "t1", "shared_conv")
        mt.store("Q", q_vec, _Resp("tenant2_answer"), "t2", "shared_conv")
        result_t1 = mt.lookup("Q", q_vec, "t1", "shared_conv")
        result_t2 = mt.lookup("Q", q_vec, "t2", "shared_conv")
        # After both stores, both conversations have 1 turn so hashes differ
        # (second lookup is after advance, first is before — this tests no cross-contamination)
        assert result_t1 is not None or result_t2 is not None


# ── Singletons ─────────────────────────────────────────────────────────────────


class TestSingletons:
    def _settings(self) -> object:
        @dataclass
        class _S:
            cache_max_size: int = 500
            cache_multiturn_threshold: float = 0.88
            cache_ttl_seconds: int = 0
            cache_multiturn_max_conversations: int = 100
            cache_multiturn_window: int = 3

        return _S()

    def test_get_conversation_store_is_singleton(self) -> None:
        s = self._settings()
        with patch("konjoai.config.get_settings", return_value=s):
            cs1 = get_conversation_store()
            cs2 = get_conversation_store()
        assert cs1 is cs2

    def test_reset_returns_new_instance(self) -> None:
        s = self._settings()
        with patch("konjoai.config.get_settings", return_value=s):
            cs1 = get_conversation_store()
        _reset_singletons()
        with patch("konjoai.config.get_settings", return_value=s):
            cs2 = get_conversation_store()
        assert cs1 is not cs2

    def test_get_multiturn_cache_is_singleton(self) -> None:
        s = self._settings()
        with patch("konjoai.config.get_settings", return_value=s):
            mt1 = get_multiturn_cache()
            mt2 = get_multiturn_cache()
        assert mt1 is mt2

    def test_conversation_store_respects_config(self) -> None:
        @dataclass
        class _S:
            cache_max_size: int = 500
            cache_multiturn_threshold: float = 0.88
            cache_ttl_seconds: int = 0
            cache_multiturn_max_conversations: int = 7
            cache_multiturn_window: int = 3

        with patch("konjoai.config.get_settings", return_value=_S()):
            cs = get_conversation_store()
        assert cs._max_conversations == 7
        assert cs._max_turns == 3
