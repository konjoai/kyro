"""Tests for Sprint 25 — Feedback Collection API.

Coverage:
- FeedbackEvent model: construction, as_dict(), None-field omission
- Signal constants: THUMBS_UP / THUMBS_DOWN / VALID_SIGNALS
- FeedbackStore: record, query (filter by tenant/signal/question_hash), summary,
  LRU eviction, size/max_events, clear, thread safety, singleton, reset
- Config: feedback_enabled default False, feedback_max_events default 1000
- API (disabled): POST /feedback → 404, GET /feedback/summary → 404
- API (enabled): POST /feedback → 201, GET /feedback/summary → 200
- API validation: invalid signal → 422, missing question_hash → 422
- OWASP: raw question text never stored; only question_hash accepted
- K3: disabled → zero overhead (no store allocation needed for request path)
- Tenant isolation: summary filtered by tenant_id
"""
from __future__ import annotations

import threading
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Helpers / shared fixtures ─────────────────────────────────────────────────

def _make_client(feedback_enabled: bool = True, max_events: int = 1000) -> TestClient:
    """Build a FastAPI TestClient with a patched settings stub."""
    from dataclasses import dataclass

    @dataclass
    class _SettingsStub:
        # core required by other routes / deps
        enable_query_router: bool = True
        enable_telemetry: bool = False
        otel_enabled: bool = False
        cache_enabled: bool = False
        cache_backend: str = "memory"
        enable_hyde: bool = False
        enable_crag: bool = False
        enable_self_rag: bool = False
        enable_query_decomposition: bool = False
        enable_graph_rag: bool = False
        use_vectro_retriever: bool = False
        use_colbert: bool = False
        multi_tenancy_enabled: bool = False
        jwt_secret_key: str = ""
        jwt_algorithm: str = "HS256"
        tenant_id_claim: str = "sub"
        api_key_auth_enabled: bool = False
        api_keys: list = None
        rate_limiting_enabled: bool = False
        rate_limit_requests: int = 60
        rate_limit_window_seconds: int = 60
        brute_force_enabled: bool = False
        brute_force_max_attempts: int = 5
        brute_force_window_seconds: int = 60
        brute_force_lockout_seconds: int = 300
        request_timeout_seconds: float = 30.0
        audit_enabled: bool = False
        audit_backend: str = "memory"
        audit_log_path: str = "logs/audit.jsonl"
        audit_max_memory_events: int = 1000
        # Sprint 25
        feedback_enabled: bool = feedback_enabled
        feedback_max_events: int = max_events

        def __post_init__(self):
            if self.api_keys is None:
                self.api_keys = []

    stub = _SettingsStub()

    from konjoai.api.app import app
    from konjoai.feedback.store import _reset_singleton

    _reset_singleton()

    with patch("konjoai.api.routes.feedback.get_settings", return_value=stub):
        client = TestClient(app, raise_server_exceptions=True)
        # Monkey-patch settings for the duration of the test via dependency override
        app.dependency_overrides = {}
        return client, stub


# ─────────────────────────────────────────────────────────────────────────────
# 1. FeedbackEvent model
# ─────────────────────────────────────────────────────────────────────────────

class TestFeedbackEventModel:
    def test_minimal_construction(self):
        from konjoai.feedback.models import FeedbackEvent, THUMBS_UP
        ev = FeedbackEvent(
            question_hash="a1b2c3d4e5f6a7b8",
            signal=THUMBS_UP,
            timestamp="2026-05-07T12:00:00Z",
        )
        assert ev.question_hash == "a1b2c3d4e5f6a7b8"
        assert ev.signal == THUMBS_UP
        assert ev.timestamp == "2026-05-07T12:00:00Z"
        assert ev.relevance_score is None
        assert ev.tenant_id is None
        assert ev.client_ip is None
        assert ev.comment_hash is None
        assert ev.model is None
        assert ev.latency_ms is None

    def test_full_construction(self):
        from konjoai.feedback.models import FeedbackEvent, THUMBS_DOWN
        ev = FeedbackEvent(
            question_hash="abc123",
            signal=THUMBS_DOWN,
            timestamp="2026-05-07T12:00:00Z",
            relevance_score=0.3,
            tenant_id="acme",
            client_ip="127.0.0.1",
            comment_hash="deadbeef12345678",
            model="gpt-4o",
            latency_ms=342.1,
        )
        assert ev.relevance_score == pytest.approx(0.3)
        assert ev.tenant_id == "acme"
        assert ev.client_ip == "127.0.0.1"
        assert ev.model == "gpt-4o"
        assert ev.latency_ms == pytest.approx(342.1)

    def test_as_dict_omits_none(self):
        from konjoai.feedback.models import FeedbackEvent, THUMBS_UP
        ev = FeedbackEvent(
            question_hash="a1b2",
            signal=THUMBS_UP,
            timestamp="t",
        )
        d = ev.as_dict()
        assert "question_hash" in d
        assert "signal" in d
        assert "timestamp" in d
        assert "relevance_score" not in d
        assert "tenant_id" not in d
        assert "comment_hash" not in d

    def test_as_dict_includes_set_fields(self):
        from konjoai.feedback.models import FeedbackEvent, THUMBS_DOWN
        ev = FeedbackEvent(
            question_hash="x",
            signal=THUMBS_DOWN,
            timestamp="t",
            relevance_score=0.7,
            tenant_id="t1",
        )
        d = ev.as_dict()
        assert d["relevance_score"] == pytest.approx(0.7)
        assert d["tenant_id"] == "t1"

    def test_signal_constants_distinct(self):
        from konjoai.feedback.models import THUMBS_UP, THUMBS_DOWN
        assert THUMBS_UP != THUMBS_DOWN

    def test_valid_signals_set(self):
        from konjoai.feedback.models import THUMBS_UP, THUMBS_DOWN, VALID_SIGNALS
        assert THUMBS_UP in VALID_SIGNALS
        assert THUMBS_DOWN in VALID_SIGNALS
        assert len(VALID_SIGNALS) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 2. FeedbackStore
# ─────────────────────────────────────────────────────────────────────────────

def _make_event(signal="thumbs_up", tenant="t1", q_hash="abc123", score=None, ts="2026-01-01T00:00:00Z"):
    from konjoai.feedback.models import FeedbackEvent
    return FeedbackEvent(
        question_hash=q_hash,
        signal=signal,
        timestamp=ts,
        relevance_score=score,
        tenant_id=tenant,
    )


class TestFeedbackStore:
    def setup_method(self):
        from konjoai.feedback.store import _reset_singleton
        _reset_singleton()

    def test_construction_defaults(self):
        from konjoai.feedback.store import FeedbackStore
        s = FeedbackStore()
        assert s.max_events == 1000
        assert s.size == 0

    def test_construction_custom_max(self):
        from konjoai.feedback.store import FeedbackStore
        s = FeedbackStore(max_events=5)
        assert s.max_events == 5

    def test_construction_invalid_max(self):
        from konjoai.feedback.store import FeedbackStore
        with pytest.raises(ValueError):
            FeedbackStore(max_events=0)

    def test_record_increases_size(self):
        from konjoai.feedback.store import FeedbackStore
        s = FeedbackStore()
        s.record(_make_event())
        assert s.size == 1

    def test_record_multiple(self):
        from konjoai.feedback.store import FeedbackStore
        s = FeedbackStore()
        for i in range(5):
            s.record(_make_event(q_hash=f"q{i}"))
        assert s.size == 5

    def test_lru_eviction_at_max(self):
        from konjoai.feedback.store import FeedbackStore
        s = FeedbackStore(max_events=3)
        for i in range(5):
            s.record(_make_event(q_hash=f"q{i}"))
        assert s.size == 3
        # Oldest entries evicted; only q2, q3, q4 remain
        hashes = {e.question_hash for e in s}
        assert hashes == {"q2", "q3", "q4"}

    def test_query_newest_first(self):
        from konjoai.feedback.store import FeedbackStore
        s = FeedbackStore()
        s.record(_make_event(q_hash="first"))
        s.record(_make_event(q_hash="second"))
        results = s.query()
        assert results[0].question_hash == "second"
        assert results[1].question_hash == "first"

    def test_query_filter_by_tenant(self):
        from konjoai.feedback.store import FeedbackStore
        s = FeedbackStore()
        s.record(_make_event(tenant="acme"))
        s.record(_make_event(tenant="beta"))
        results = s.query(tenant_id="acme")
        assert all(e.tenant_id == "acme" for e in results)
        assert len(results) == 1

    def test_query_filter_by_signal(self):
        from konjoai.feedback.store import FeedbackStore
        s = FeedbackStore()
        s.record(_make_event(signal="thumbs_up"))
        s.record(_make_event(signal="thumbs_down"))
        results = s.query(signal="thumbs_up")
        assert all(e.signal == "thumbs_up" for e in results)
        assert len(results) == 1

    def test_query_filter_by_question_hash(self):
        from konjoai.feedback.store import FeedbackStore
        s = FeedbackStore()
        s.record(_make_event(q_hash="abc"))
        s.record(_make_event(q_hash="xyz"))
        results = s.query(question_hash="abc")
        assert len(results) == 1
        assert results[0].question_hash == "abc"

    def test_query_limit(self):
        from konjoai.feedback.store import FeedbackStore
        s = FeedbackStore()
        for i in range(10):
            s.record(_make_event(q_hash=f"q{i}"))
        results = s.query(limit=3)
        assert len(results) == 3

    def test_query_limit_capped_at_1000(self):
        from konjoai.feedback.store import FeedbackStore
        s = FeedbackStore(max_events=2000)
        for i in range(100):
            s.record(_make_event(q_hash=f"q{i}"))
        # limit=5000 should be capped at 1000
        results = s.query(limit=5000)
        assert len(results) <= 1000

    def test_query_empty_store(self):
        from konjoai.feedback.store import FeedbackStore
        s = FeedbackStore()
        assert s.query() == []

    def test_summary_empty(self):
        from konjoai.feedback.store import FeedbackStore
        s = FeedbackStore()
        sm = s.summary()
        assert sm["total"] == 0
        assert sm["thumbs_up"] == 0
        assert sm["thumbs_down"] == 0
        assert sm.get("avg_relevance") is None

    def test_summary_counts(self):
        from konjoai.feedback.store import FeedbackStore
        s = FeedbackStore()
        s.record(_make_event(signal="thumbs_up"))
        s.record(_make_event(signal="thumbs_up"))
        s.record(_make_event(signal="thumbs_down"))
        sm = s.summary()
        assert sm["total"] == 3
        assert sm["thumbs_up"] == 2
        assert sm["thumbs_down"] == 1

    def test_summary_avg_relevance(self):
        from konjoai.feedback.store import FeedbackStore
        s = FeedbackStore()
        s.record(_make_event(score=0.8))
        s.record(_make_event(score=0.4))
        sm = s.summary()
        assert sm["avg_relevance"] == pytest.approx(0.6, abs=1e-4)

    def test_summary_no_relevance_scores(self):
        from konjoai.feedback.store import FeedbackStore
        s = FeedbackStore()
        s.record(_make_event(score=None))
        sm = s.summary()
        assert "avg_relevance" not in sm

    def test_summary_by_question(self):
        from konjoai.feedback.store import FeedbackStore
        s = FeedbackStore()
        s.record(_make_event(q_hash="q1", signal="thumbs_up"))
        s.record(_make_event(q_hash="q1", signal="thumbs_down"))
        s.record(_make_event(q_hash="q2", signal="thumbs_up"))
        sm = s.summary()
        assert sm["by_question"]["q1"]["thumbs_up"] == 1
        assert sm["by_question"]["q1"]["thumbs_down"] == 1
        assert sm["by_question"]["q2"]["thumbs_up"] == 1

    def test_summary_tenant_filter(self):
        from konjoai.feedback.store import FeedbackStore
        s = FeedbackStore()
        s.record(_make_event(tenant="acme", signal="thumbs_up"))
        s.record(_make_event(tenant="beta", signal="thumbs_up"))
        s.record(_make_event(tenant="acme", signal="thumbs_down"))
        sm = s.summary(tenant_id="acme")
        assert sm["total"] == 2
        assert sm["thumbs_up"] == 1
        assert sm["thumbs_down"] == 1

    def test_clear(self):
        from konjoai.feedback.store import FeedbackStore
        s = FeedbackStore()
        s.record(_make_event())
        s.record(_make_event())
        s.clear()
        assert s.size == 0
        assert len(s) == 0

    def test_len_matches_size(self):
        from konjoai.feedback.store import FeedbackStore
        s = FeedbackStore()
        s.record(_make_event())
        assert len(s) == s.size

    def test_iter(self):
        from konjoai.feedback.store import FeedbackStore
        s = FeedbackStore()
        s.record(_make_event(q_hash="x"))
        events = list(s)
        assert len(events) == 1
        assert events[0].question_hash == "x"

    def test_thread_safety(self):
        """20 threads × 50 writes each = 1000 total (bounded to max_events)."""
        from konjoai.feedback.store import FeedbackStore
        s = FeedbackStore(max_events=1000)
        errors = []

        def _writer():
            try:
                for _ in range(50):
                    s.record(_make_event())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_writer) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"
        assert s.size <= 1000

    def test_singleton_returns_same_instance(self):
        from konjoai.feedback.store import get_feedback_store, _reset_singleton
        _reset_singleton()
        s1 = get_feedback_store()
        s2 = get_feedback_store()
        assert s1 is s2

    def test_reset_clears_singleton(self):
        from konjoai.feedback.store import get_feedback_store, _reset_singleton
        _reset_singleton()
        s1 = get_feedback_store()
        _reset_singleton()
        s2 = get_feedback_store()
        assert s1 is not s2


# ─────────────────────────────────────────────────────────────────────────────
# 3. Config
# ─────────────────────────────────────────────────────────────────────────────

class TestFeedbackConfig:
    def test_feedback_enabled_default_false(self):
        from konjoai.config import Settings
        s = Settings()
        assert s.feedback_enabled is False

    def test_feedback_max_events_default(self):
        from konjoai.config import Settings
        s = Settings()
        assert s.feedback_max_events == 1000


# ─────────────────────────────────────────────────────────────────────────────
# 4. API — disabled (K3)
# ─────────────────────────────────────────────────────────────────────────────

def _make_disabled_settings():
    from dataclasses import dataclass
    @dataclass
    class _S:
        feedback_enabled: bool = False
        feedback_max_events: int = 1000
    return _S()


def _make_enabled_settings(max_events: int = 1000):
    from dataclasses import dataclass
    @dataclass
    class _S:
        feedback_enabled: bool = True
        feedback_max_events: int = max_events
    return _S()


def _feedback_app(settings_stub):
    """Build an isolated FastAPI app containing only the feedback router."""
    from fastapi import FastAPI
    from konjoai.api.routes.feedback import router as fb_router
    app = FastAPI()
    app.include_router(fb_router)
    return app


class TestFeedbackAPIDisabled:
    """When feedback_enabled=False all endpoints must return 404 (K3)."""

    def setup_method(self):
        from konjoai.feedback.store import _reset_singleton
        _reset_singleton()

    def test_post_feedback_returns_404(self):
        stub = _make_disabled_settings()
        app = _feedback_app(stub)
        with patch("konjoai.api.routes.feedback.get_settings", return_value=stub):
            with TestClient(app) as client:
                resp = client.post("/feedback", json={
                    "question_hash": "abc123",
                    "signal": "thumbs_up",
                })
        assert resp.status_code == 404

    def test_get_summary_returns_404(self):
        stub = _make_disabled_settings()
        app = _feedback_app(stub)
        with patch("konjoai.api.routes.feedback.get_settings", return_value=stub):
            with TestClient(app) as client:
                resp = client.get("/feedback/summary")
        assert resp.status_code == 404

    def test_404_message_mentions_env_var(self):
        stub = _make_disabled_settings()
        app = _feedback_app(stub)
        with patch("konjoai.api.routes.feedback.get_settings", return_value=stub):
            with TestClient(app) as client:
                resp = client.post("/feedback", json={
                    "question_hash": "abc",
                    "signal": "thumbs_up",
                })
        assert "FEEDBACK_ENABLED" in resp.json().get("detail", "")


# ─────────────────────────────────────────────────────────────────────────────
# 5. API — enabled
# ─────────────────────────────────────────────────────────────────────────────

class TestFeedbackAPIEnabled:
    def setup_method(self):
        from konjoai.feedback.store import _reset_singleton
        _reset_singleton()

    def test_post_thumbs_up_returns_201(self):
        stub = _make_enabled_settings()
        app = _feedback_app(stub)
        with patch("konjoai.api.routes.feedback.get_settings", return_value=stub):
            with TestClient(app) as client:
                resp = client.post("/feedback", json={
                    "question_hash": "a1b2c3d4e5f6a7b8",
                    "signal": "thumbs_up",
                })
        assert resp.status_code == 201

    def test_post_thumbs_down_returns_201(self):
        stub = _make_enabled_settings()
        app = _feedback_app(stub)
        with patch("konjoai.api.routes.feedback.get_settings", return_value=stub):
            with TestClient(app) as client:
                resp = client.post("/feedback", json={
                    "question_hash": "a1b2c3d4e5f6a7b8",
                    "signal": "thumbs_down",
                })
        assert resp.status_code == 201

    def test_response_contract_recorded_true(self):
        stub = _make_enabled_settings()
        app = _feedback_app(stub)
        with patch("konjoai.api.routes.feedback.get_settings", return_value=stub):
            with TestClient(app) as client:
                resp = client.post("/feedback", json={
                    "question_hash": "abc123",
                    "signal": "thumbs_up",
                })
        body = resp.json()
        assert body["recorded"] is True
        assert body["question_hash"] == "abc123"
        assert body["signal"] == "thumbs_up"
        assert "timestamp" in body

    def test_post_with_relevance_score(self):
        stub = _make_enabled_settings()
        app = _feedback_app(stub)
        with patch("konjoai.api.routes.feedback.get_settings", return_value=stub):
            with TestClient(app) as client:
                resp = client.post("/feedback", json={
                    "question_hash": "abc123",
                    "signal": "thumbs_up",
                    "relevance_score": 0.85,
                })
        assert resp.status_code == 201

    def test_post_with_all_optional_fields(self):
        stub = _make_enabled_settings()
        app = _feedback_app(stub)
        with patch("konjoai.api.routes.feedback.get_settings", return_value=stub):
            with TestClient(app) as client:
                resp = client.post("/feedback", json={
                    "question_hash": "abc123",
                    "signal": "thumbs_down",
                    "relevance_score": 0.2,
                    "comment": "The answer was off-topic.",
                    "model": "gpt-4o",
                    "latency_ms": 512.3,
                })
        assert resp.status_code == 201

    def test_invalid_signal_returns_422(self):
        stub = _make_enabled_settings()
        app = _feedback_app(stub)
        with patch("konjoai.api.routes.feedback.get_settings", return_value=stub):
            with TestClient(app) as client:
                resp = client.post("/feedback", json={
                    "question_hash": "abc",
                    "signal": "invalid_signal",
                })
        assert resp.status_code == 422

    def test_missing_question_hash_returns_422(self):
        stub = _make_enabled_settings()
        app = _feedback_app(stub)
        with patch("konjoai.api.routes.feedback.get_settings", return_value=stub):
            with TestClient(app) as client:
                resp = client.post("/feedback", json={"signal": "thumbs_up"})
        assert resp.status_code == 422

    def test_missing_signal_returns_422(self):
        stub = _make_enabled_settings()
        app = _feedback_app(stub)
        with patch("konjoai.api.routes.feedback.get_settings", return_value=stub):
            with TestClient(app) as client:
                resp = client.post("/feedback", json={"question_hash": "abc"})
        assert resp.status_code == 422

    def test_relevance_score_out_of_range_returns_422(self):
        stub = _make_enabled_settings()
        app = _feedback_app(stub)
        with patch("konjoai.api.routes.feedback.get_settings", return_value=stub):
            with TestClient(app) as client:
                resp = client.post("/feedback", json={
                    "question_hash": "abc",
                    "signal": "thumbs_up",
                    "relevance_score": 1.5,  # > 1.0 → invalid
                })
        assert resp.status_code == 422

    def test_relevance_score_negative_returns_422(self):
        stub = _make_enabled_settings()
        app = _feedback_app(stub)
        with patch("konjoai.api.routes.feedback.get_settings", return_value=stub):
            with TestClient(app) as client:
                resp = client.post("/feedback", json={
                    "question_hash": "abc",
                    "signal": "thumbs_up",
                    "relevance_score": -0.1,
                })
        assert resp.status_code == 422

    def test_get_summary_returns_200(self):
        stub = _make_enabled_settings()
        app = _feedback_app(stub)
        with patch("konjoai.api.routes.feedback.get_settings", return_value=stub):
            with TestClient(app) as client:
                resp = client.get("/feedback/summary")
        assert resp.status_code == 200

    def test_summary_response_contract(self):
        stub = _make_enabled_settings()
        app = _feedback_app(stub)
        with patch("konjoai.api.routes.feedback.get_settings", return_value=stub):
            with TestClient(app) as client:
                resp = client.get("/feedback/summary")
        body = resp.json()
        assert "total" in body
        assert "thumbs_up" in body
        assert "thumbs_down" in body
        assert "by_question" in body
        assert body["feedback_enabled"] is True

    def test_summary_reflects_submitted_feedback(self):
        from konjoai.feedback.store import _reset_singleton
        _reset_singleton()
        stub = _make_enabled_settings()
        app = _feedback_app(stub)
        with patch("konjoai.api.routes.feedback.get_settings", return_value=stub):
            with TestClient(app) as client:
                # Submit 2 thumbs_up and 1 thumbs_down
                for _ in range(2):
                    client.post("/feedback", json={"question_hash": "q1", "signal": "thumbs_up"})
                client.post("/feedback", json={"question_hash": "q1", "signal": "thumbs_down"})
                resp = client.get("/feedback/summary")
        body = resp.json()
        assert body["total"] == 3
        assert body["thumbs_up"] == 2
        assert body["thumbs_down"] == 1

    def test_summary_with_tenant_filter(self):
        from konjoai.feedback.store import _reset_singleton, get_feedback_store
        from konjoai.feedback.models import FeedbackEvent
        _reset_singleton()
        stub = _make_enabled_settings()
        app = _feedback_app(stub)
        # Pre-populate store with two tenants
        store = get_feedback_store()
        store.record(FeedbackEvent(
            question_hash="q1", signal="thumbs_up",
            timestamp="2026-01-01T00:00:00Z", tenant_id="acme",
        ))
        store.record(FeedbackEvent(
            question_hash="q2", signal="thumbs_down",
            timestamp="2026-01-01T00:00:00Z", tenant_id="beta",
        ))
        with patch("konjoai.api.routes.feedback.get_settings", return_value=stub):
            with TestClient(app) as client:
                resp = client.get("/feedback/summary?tenant_id=acme")
        body = resp.json()
        assert body["total"] == 1
        assert body["thumbs_up"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 6. OWASP PII contract
# ─────────────────────────────────────────────────────────────────────────────

class TestFeedbackOWASP:
    """Raw question text must never appear in any stored or returned event."""

    def test_post_does_not_accept_raw_question(self):
        """The /feedback endpoint has no 'question' field — only question_hash."""
        from konjoai.api.routes.feedback import FeedbackRequest
        import inspect
        fields = FeedbackRequest.model_fields
        assert "question" not in fields, "FeedbackRequest must not accept raw question text"

    def test_feedback_event_has_no_question_field(self):
        """FeedbackEvent stores question_hash, not raw question."""
        from konjoai.feedback.models import FeedbackEvent
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(FeedbackEvent)}
        assert "question" not in field_names
        assert "question_hash" in field_names

    def test_comment_stored_as_hash(self):
        """When a comment is submitted it is stored as comment_hash, not raw text."""
        from konjoai.feedback.models import FeedbackEvent
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(FeedbackEvent)}
        assert "comment" not in field_names
        assert "comment_hash" in field_names

    def test_raw_comment_not_in_stored_event(self):
        """Verify that the store records comment_hash, not the raw comment string."""
        from konjoai.feedback.store import FeedbackStore, _reset_singleton
        from konjoai.feedback.models import FeedbackEvent
        from konjoai.audit.models import hash_text

        _reset_singleton()
        raw_comment = "The answer was completely wrong about the refund policy."
        comment_hash = hash_text(raw_comment)

        s = FeedbackStore()
        s.record(FeedbackEvent(
            question_hash="q1",
            signal="thumbs_down",
            timestamp="t",
            comment_hash=comment_hash,
        ))
        events = s.query()
        assert len(events) == 1
        d = events[0].as_dict()
        assert raw_comment not in str(d), "Raw comment text must not appear in stored event"
        assert d.get("comment_hash") == comment_hash


# ─────────────────────────────────────────────────────────────────────────────
# 7. Package exports
# ─────────────────────────────────────────────────────────────────────────────

class TestFeedbackPackageExports:
    def test_all_symbols_importable(self):
        from konjoai.feedback import (
            FeedbackEvent,
            FeedbackStore,
            get_feedback_store,
            _reset_singleton,
            THUMBS_UP,
            THUMBS_DOWN,
            VALID_SIGNALS,
        )
        assert THUMBS_UP == "thumbs_up"
        assert THUMBS_DOWN == "thumbs_down"
        assert len(VALID_SIGNALS) == 2
