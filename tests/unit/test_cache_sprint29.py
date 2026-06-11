"""Sprint 29 — query rewriting, suspicious entry detection, and cache federation tests."""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from konjoai.api.routes.cache import router as cache_router
from konjoai.cache.federation import FederatedLookup, PeerRegistry, _reset_federation
from konjoai.cache.rewriter import (
    DEFAULT_STEPS,
    QueryRewriter,
    RewriteResult,
    _reset_rewriter,
)
from konjoai.cache.semantic_cache import SemanticCache
from konjoai.cache.suspicious import (
    SuspiciousFlagStore,
    _reset_flag_store,
    get_flag_store,
    scan_for_suspicious,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _vec(seed: int, dim: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / (np.linalg.norm(v) + 1e-10)).reshape(1, dim).astype(np.float32)


@dataclass
class _Resp:
    answer: str


@dataclass
class _Settings:
    cache_enabled: bool = True
    cache_warm_max_batch: int = 500
    cache_max_size: int = 200
    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 0
    cache_query_rewrite_enabled: bool = True
    cache_query_rewrite_steps: list = None
    cache_federation_enabled: bool = True
    cache_federation_timeout_seconds: float = 2.0
    cache_poisoning_guard_enabled: bool = False

    def __post_init__(self) -> None:
        if self.cache_query_rewrite_steps is None:
            self.cache_query_rewrite_steps = list(DEFAULT_STEPS)


def _filled_cache(n: int, answer_len: int = 30) -> SemanticCache:
    cache = SemanticCache(max_size=200, threshold=0.95)
    for i in range(n):
        cache.store(f"question {i}", _vec(i), _Resp("x" * answer_len))
    return cache


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(cache_router)
    return app


# ── QueryRewriter unit tests ──────────────────────────────────────────────────


class TestQueryRewriter:
    def test_lowercase_step(self) -> None:
        r = QueryRewriter(steps=["lowercase"])
        assert r.rewrite("WHAT IS THE SLA?") == "what is the sla?"

    def test_normalize_whitespace_step(self) -> None:
        r = QueryRewriter(steps=["normalize_whitespace"])
        assert r.rewrite("what   is   this") == "what is this"

    def test_expand_contractions_step(self) -> None:
        r = QueryRewriter(steps=["expand_contractions"])
        result = r.rewrite("What's the refund policy?")
        assert "what is" in result.lower()

    def test_strip_fillers_step(self) -> None:
        r = QueryRewriter(steps=["strip_fillers"])
        assert r.rewrite("Tell me about the refund policy") == "the refund policy"

    def test_strip_trailing_question_mark(self) -> None:
        r = QueryRewriter(steps=["strip_trailing_question_mark"])
        assert r.rewrite("What is the SLA?") == "What is the SLA"

    def test_default_pipeline_reduces_surface_form(self) -> None:
        r = QueryRewriter()
        a = r.rewrite("What's the REFUND Policy???")
        b = r.rewrite("what is the refund policy")
        assert a == b, f"Expected same form, got {a!r} vs {b!r}"

    def test_explain_returns_rewrite_result(self) -> None:
        r = QueryRewriter()
        result = r.explain("What's the refund policy?")
        assert isinstance(result, RewriteResult)
        assert result.original == "What's the refund policy?"
        assert len(result.steps) == len(r.step_names)

    def test_explain_marks_changed_steps(self) -> None:
        r = QueryRewriter(steps=["lowercase", "normalize_whitespace"])
        result = r.explain("HELLO   WORLD")
        changed = [s for s in result.steps if s.changed]
        assert len(changed) >= 1

    def test_unknown_steps_silently_skipped(self) -> None:
        r = QueryRewriter(steps=["lowercase", "nonexistent_step_xyz"])
        assert r.rewrite("HELLO") == "hello"
        assert r.step_names == ["lowercase"]

    def test_empty_question_returns_empty(self) -> None:
        r = QueryRewriter()
        assert r.rewrite("") == ""

    def test_step_names_matches_configured_steps(self) -> None:
        r = QueryRewriter(steps=["lowercase", "normalize_whitespace"])
        assert r.step_names == ["lowercase", "normalize_whitespace"]


# ── Rewrite route ─────────────────────────────────────────────────────────────


class TestRewriteRoute:
    def test_preview_rewrite_returns_shape(self) -> None:
        _reset_rewriter()
        app = _make_app()
        client = TestClient(app)
        with patch("konjoai.api.routes.cache.get_rewriter", return_value=QueryRewriter()):
            resp = client.post("/cache/rewrite", json={"question": "What's the refund policy?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "original" in body
        assert "rewritten" in body
        assert "steps" in body

    def test_preview_shows_changed_true_for_contraction(self) -> None:
        _reset_rewriter()
        app = _make_app()
        client = TestClient(app)
        with patch("konjoai.api.routes.cache.get_rewriter", return_value=QueryRewriter()):
            resp = client.post("/cache/rewrite", json={"question": "What's the SLA?"})
        body = resp.json()
        assert body["changed"] is True

    def test_rewrite_config_returns_enabled_flag(self) -> None:
        _reset_rewriter()
        app = _make_app()
        client = TestClient(app)
        with (
            patch("konjoai.api.routes.cache.get_settings", return_value=_Settings()),
            patch("konjoai.api.routes.cache.get_rewriter", return_value=QueryRewriter()),
        ):
            resp = client.get("/cache/rewrite/config")
        assert resp.status_code == 200
        body = resp.json()
        assert "enabled" in body
        assert "steps" in body


# ── SuspiciousFlagStore ───────────────────────────────────────────────────────


class TestSuspiciousFlagStore:
    def setup_method(self) -> None:
        _reset_flag_store()

    def test_flag_and_retrieve_pending(self) -> None:
        store = SuspiciousFlagStore()
        store.flag("abc123", "q", "outlier", 2.5, "embedding_outlier")
        pending = store.pending()
        assert len(pending) == 1
        assert pending[0].entry_hash == "abc123"
        assert pending[0].status == "pending"

    def test_approve_changes_status(self) -> None:
        store = SuspiciousFlagStore()
        store.flag("abc123", "q", "outlier", 2.5, "embedding_outlier")
        ok = store.resolve("abc123", "approve")
        assert ok is True
        assert store.pending() == []
        assert store.get("abc123").status == "approved"

    def test_reject_changes_status(self) -> None:
        store = SuspiciousFlagStore()
        store.flag("abc123", "q", "outlier", 2.5, "embedding_outlier")
        store.resolve("abc123", "reject")
        assert store.get("abc123").status == "rejected"

    def test_resolve_unknown_hash_returns_false(self) -> None:
        store = SuspiciousFlagStore()
        assert store.resolve("nonexistent", "approve") is False

    def test_duplicate_flag_replaces_entry(self) -> None:
        store = SuspiciousFlagStore()
        store.flag("h1", "q", "reason1", 1.5, "embedding_outlier")
        store.flag("h1", "q", "reason2", 3.0, "hit_count_anomaly")
        assert store.get("h1").reason == "reason2"
        assert len(store.all_flags()) == 1

    def test_singleton_returns_same_instance(self) -> None:
        _reset_flag_store()
        a = get_flag_store()
        b = get_flag_store()
        assert a is b


# ── scan_for_suspicious ───────────────────────────────────────────────────────


class TestScanForSuspicious:
    def test_returns_empty_for_small_cache(self) -> None:
        cache = _filled_cache(5)
        result = scan_for_suspicious(cache, k=5, z_threshold=2.0)
        assert result == []

    def test_returns_list_for_adequate_cache(self) -> None:
        cache = _filled_cache(30)
        result = scan_for_suspicious(cache, k=3, z_threshold=1.0)
        assert isinstance(result, list)

    def test_sorted_by_score_descending(self) -> None:
        cache = _filled_cache(30)
        result = scan_for_suspicious(cache, k=3, z_threshold=1.0)
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_result_has_required_keys(self) -> None:
        cache = _filled_cache(30)
        result = scan_for_suspicious(cache, k=3, z_threshold=1.0)
        for item in result:
            assert "entry_hash" in item
            assert "question" in item
            assert "reason" in item
            assert "score" in item
            assert "signal" in item

    def test_answer_length_outlier_detected(self) -> None:
        # Insert one entry with a suspiciously long answer
        cache = SemanticCache(max_size=50, threshold=0.95)
        for i in range(20):
            cache.store(f"q{i}", _vec(i), _Resp("short " * 3))  # ~18 chars
        # Outlier: 10× longer
        cache.store("outlier q", _vec(99), _Resp("x" * 200))
        result = scan_for_suspicious(cache, k=3, z_threshold=2.0)
        signals = [r["signal"] for r in result]
        assert "answer_length_anomaly" in signals


# ── Suspicious routes ─────────────────────────────────────────────────────────


class TestSuspiciousRoutes:
    def setup_method(self) -> None:
        _reset_flag_store()

    def test_list_suspicious_returns_shape(self) -> None:
        cache = _filled_cache(30)
        app = _make_app()
        client = TestClient(app)
        with (
            patch("konjoai.api.routes.cache.get_settings", return_value=_Settings()),
            patch("konjoai.api.routes.cache.get_semantic_cache", return_value=cache),
        ):
            resp = client.get("/cache/suspicious?k=3&z=1.0")
        assert resp.status_code == 200
        body = resp.json()
        assert "count" in body
        assert "suspicious" in body

    def test_approve_missing_hash_404(self) -> None:
        app = _make_app()
        client = TestClient(app)
        with (
            patch("konjoai.api.routes.cache.get_settings", return_value=_Settings()),
            patch("konjoai.api.routes.cache.get_semantic_cache", return_value=_filled_cache(10)),
        ):
            resp = client.post("/cache/suspicious/deadbeef12345678/approve")
        assert resp.status_code == 404

    def test_approve_flagged_entry_returns_200(self) -> None:
        store = get_flag_store()
        store.flag("deadbeef00000001", "q", "reason", 2.5, "embedding_outlier")
        app = _make_app()
        client = TestClient(app)
        with (
            patch("konjoai.api.routes.cache.get_settings", return_value=_Settings()),
            patch("konjoai.api.routes.cache.get_semantic_cache", return_value=_filled_cache(5)),
        ):
            resp = client.post("/cache/suspicious/deadbeef00000001/approve")
        assert resp.status_code == 200
        assert resp.json()["status"] == "approved"

    def test_list_flagged_returns_all_statuses(self) -> None:
        store = get_flag_store()
        store.flag("h1", "q1", "r1", 2.1, "embedding_outlier")
        store.flag("h2", "q2", "r2", 3.0, "hit_count_anomaly")
        store.resolve("h2", "approve")
        app = _make_app()
        client = TestClient(app)
        with (
            patch("konjoai.api.routes.cache.get_settings", return_value=_Settings()),
            patch("konjoai.api.routes.cache.get_semantic_cache", return_value=_filled_cache(5)),
        ):
            resp = client.get("/cache/suspicious/flagged")
        body = resp.json()
        assert body["count"] == 2
        statuses = {f["status"] for f in body["flags"]}
        assert "pending" in statuses
        # h2 was resolved with "approved" action → status "approved"
        assert any(f["status"] == "approved" for f in body["flags"])


# ── PeerRegistry ──────────────────────────────────────────────────────────────


class TestPeerRegistry:
    def setup_method(self) -> None:
        _reset_federation()

    def test_register_and_list_peer(self) -> None:
        registry = PeerRegistry()
        node = registry.register("http://peer1:8000", name="peer1")
        assert node.url == "http://peer1:8000"
        assert len(registry.all_peers()) == 1

    def test_register_strips_trailing_slash(self) -> None:
        registry = PeerRegistry()
        node = registry.register("http://peer1:8000/", name="p")
        assert node.url == "http://peer1:8000"

    def test_remove_existing_peer(self) -> None:
        registry = PeerRegistry()
        node = registry.register("http://peer1:8000", name="p")
        assert registry.remove(node.peer_id) is True
        assert registry.all_peers() == []

    def test_remove_nonexistent_returns_false(self) -> None:
        registry = PeerRegistry()
        assert registry.remove("nonexistent") is False

    def test_health_update_changes_availability(self) -> None:
        registry = PeerRegistry()
        node = registry.register("http://peer1:8000", name="p")
        registry._update_availability(node.peer_id, False)
        updated = registry.get(node.peer_id)
        assert updated.last_healthy is False
        assert updated.availability < 1.0

    def test_availability_recovers_on_success(self) -> None:
        registry = PeerRegistry()
        node = registry.register("http://peer1:8000", name="p")
        # Fail many times, then succeed
        for _ in range(10):
            registry._update_availability(node.peer_id, False)
        registry._update_availability(node.peer_id, True)
        updated = registry.get(node.peer_id)
        assert updated.availability > 0.0

    def test_record_hit_increments_counter(self) -> None:
        registry = PeerRegistry()
        node = registry.register("http://peer1:8000", name="p")
        registry.record_hit(node.peer_id)
        registry.record_hit(node.peer_id)
        assert registry.get(node.peer_id).hits_contributed == 2


# ── FederatedLookup ───────────────────────────────────────────────────────────


class TestFederatedLookup:
    def setup_method(self) -> None:
        _reset_federation()

    def test_returns_none_with_no_peers(self) -> None:
        registry = PeerRegistry()
        lookup = FederatedLookup(registry)
        result = lookup.lookup("any question")
        assert result is None

    def test_peer_status_returns_list(self) -> None:
        registry = PeerRegistry()
        registry.register("http://peer1:8000", name="p1")
        registry.register("http://peer2:8000", name="p2")
        lookup = FederatedLookup(registry)
        status = lookup.peer_status()
        assert len(status) == 2
        for s in status:
            assert "peer_id" in s
            assert "availability" in s
            assert "hits_contributed" in s
            assert "hit_share_pct" in s

    def test_lookup_returns_result_on_peer_hit(self) -> None:
        import httpx as real_httpx  # noqa: PLC0415

        registry = PeerRegistry()
        node = registry.register("http://peer1:8000", name="p1")
        registry._update_availability(node.peer_id, True)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [{"query_index": 0, "query": "q", "matches": [
                {"question": "cached q", "answer": "cached answer", "similarity": 0.97, "hit_count": 3}
            ]}]
        }
        with patch.object(real_httpx, "post", return_value=mock_resp):
            result = FederatedLookup(registry).lookup("q", min_similarity=0.95)

        assert result is not None
        assert result.answer == "cached answer"
        assert result.similarity == pytest.approx(0.97)
        assert registry.get(node.peer_id).hits_contributed == 1

    def test_lookup_skips_peer_below_min_similarity(self) -> None:
        import httpx as real_httpx  # noqa: PLC0415

        registry = PeerRegistry()
        node = registry.register("http://peer1:8000", name="p1")
        registry._update_availability(node.peer_id, True)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [{"query_index": 0, "query": "q", "matches": [
                {"question": "q", "answer": "ans", "similarity": 0.70, "hit_count": 1}
            ]}]
        }
        with patch.object(real_httpx, "post", return_value=mock_resp):
            result = FederatedLookup(registry).lookup("q", min_similarity=0.95)

        assert result is None  # below threshold — not returned


# ── Federation routes ─────────────────────────────────────────────────────────


class TestFederationRoutes:
    def setup_method(self) -> None:
        _reset_federation()

    def test_register_peer_returns_peer_id(self) -> None:
        app = _make_app()
        client = TestClient(app)
        resp = client.post("/cache/federate", json={"url": "http://peer1:8000", "name": "peer1"})
        assert resp.status_code == 201
        body = resp.json()
        assert "peer_id" in body
        assert body["url"] == "http://peer1:8000"

    def test_list_peers_includes_registered(self) -> None:
        app = _make_app()
        client = TestClient(app)
        client.post("/cache/federate", json={"url": "http://peer1:8000", "name": "p1"})
        resp = client.get("/cache/peers")
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    def test_remove_peer_returns_200(self) -> None:
        app = _make_app()
        client = TestClient(app)
        reg = client.post("/cache/federate", json={"url": "http://peer1:8000"}).json()
        resp = client.delete(f"/cache/peers/{reg['peer_id']}")
        assert resp.status_code == 200
        assert resp.json()["removed"] is True

    def test_remove_nonexistent_peer_404(self) -> None:
        app = _make_app()
        client = TestClient(app)
        resp = client.delete("/cache/peers/nonexistent")
        assert resp.status_code == 404
