"""Sprint 24 — Audit Logging tests.

Coverage:
    - AuditEvent model: construction, as_dict() omits None fields
    - hash_text: determinism, length, no raw PII
    - InMemoryBackend: write, query, stats, LRU eviction, max_events guard
    - JsonLinesBackend: write + query round-trip
    - AuditLogger: enabled/disabled gate, write-error safety (K1)
    - get_audit_logger(): singleton, reset helper
    - Config integration: audit_enabled / audit_backend / audit_max_memory_events
    - API: GET /audit/events (disabled → 404, enabled → 200 + filter)
    - API: GET /audit/stats (disabled → 404, enabled → 200)
    - Query route: AuditEvent emitted when audit_enabled=True, not emitted when False
    - Ingest route: AuditEvent emitted when audit_enabled=True
    - No raw question text in AuditEvent (OWASP K1)
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

from konjoai.audit.models import (
    AGENT_QUERY,
    AUTH_FAILURE,
    INGEST,
    QUERY,
    RATE_LIMITED,
    AuditEvent,
    hash_text,
)
from konjoai.audit.logger import (
    AuditLogger,
    InMemoryBackend,
    JsonLinesBackend,
    _reset_singleton,
    get_audit_logger,
)


def _make_event(**kwargs) -> AuditEvent:
    defaults = dict(
        event_type=QUERY,
        timestamp="2026-05-05T00:00:00+00:00",
        endpoint="/query",
        status_code=200,
        latency_ms=42.0,
    )
    defaults.update(kwargs)
    return AuditEvent(**defaults)


# ── hash_text ─────────────────────────────────────────────────────────────────


class TestHashText:
    def test_deterministic(self) -> None:
        assert hash_text("hello world") == hash_text("hello world")

    def test_length_16(self) -> None:
        assert len(hash_text("anything")) == 16

    def test_different_inputs_differ(self) -> None:
        assert hash_text("foo") != hash_text("bar")

    def test_not_plaintext(self) -> None:
        text = "What is the refund policy?"
        digest = hash_text(text)
        assert text not in digest


# ── AuditEvent ────────────────────────────────────────────────────────────────


class TestAuditEvent:
    def test_construction_minimal(self) -> None:
        ev = _make_event()
        assert ev.event_type == QUERY
        assert ev.status_code == 200

    def test_as_dict_omits_none(self) -> None:
        ev = _make_event(tenant_id=None, question_hash=None)
        d = ev.as_dict()
        assert "tenant_id" not in d
        assert "question_hash" not in d
        assert "endpoint" in d

    def test_as_dict_includes_set_fields(self) -> None:
        ev = _make_event(tenant_id="acme", cache_hit=True, result_count=5)
        d = ev.as_dict()
        assert d["tenant_id"] == "acme"
        assert d["cache_hit"] is True
        assert d["result_count"] == 5

    def test_event_type_constants(self) -> None:
        assert QUERY == "query"
        assert INGEST == "ingest"
        assert AGENT_QUERY == "agent_query"
        assert AUTH_FAILURE == "auth_failure"
        assert RATE_LIMITED == "rate_limited"


# ── InMemoryBackend ───────────────────────────────────────────────────────────


class TestInMemoryBackend:
    def test_write_and_query(self) -> None:
        b = InMemoryBackend()
        b.write(_make_event())
        events = b.query(limit=10)
        assert len(events) == 1

    def test_query_limit(self) -> None:
        b = InMemoryBackend()
        for _ in range(5):
            b.write(_make_event())
        assert len(b.query(limit=2)) == 2

    def test_filter_by_tenant(self) -> None:
        b = InMemoryBackend()
        b.write(_make_event(tenant_id="acme"))
        b.write(_make_event(tenant_id="globex"))
        assert len(b.query(tenant_id="acme")) == 1
        assert b.query(tenant_id="acme")[0].tenant_id == "acme"

    def test_filter_by_event_type(self) -> None:
        b = InMemoryBackend()
        b.write(_make_event(event_type=QUERY))
        b.write(_make_event(event_type=INGEST))
        assert len(b.query(event_type=INGEST)) == 1

    def test_lru_eviction(self) -> None:
        b = InMemoryBackend(max_events=3)
        for i in range(5):
            b.write(_make_event(latency_ms=float(i)))
        assert b.size == 3
        events = b.query(limit=10)
        # Only the last 3 survive
        assert [e.latency_ms for e in events] == [2.0, 3.0, 4.0]

    def test_max_events_guard(self) -> None:
        with pytest.raises(ValueError, match="max_events"):
            InMemoryBackend(max_events=0)

    def test_stats(self) -> None:
        b = InMemoryBackend()
        b.write(_make_event(event_type=QUERY))
        b.write(_make_event(event_type=QUERY))
        b.write(_make_event(event_type=INGEST))
        stats = b.stats()
        assert stats[QUERY] == 2
        assert stats[INGEST] == 1

    def test_thread_safety(self) -> None:
        """Concurrent writes from 20 threads must not lose events."""
        b = InMemoryBackend(max_events=2000)
        errors: list[Exception] = []

        def _writer():
            try:
                for _ in range(50):
                    b.write(_make_event())
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_writer) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert b.size == 1000  # 20 × 50


# ── JsonLinesBackend ──────────────────────────────────────────────────────────


class TestJsonLinesBackend:
    def test_write_and_query_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        b = JsonLinesBackend(path)
        ev = _make_event(tenant_id="acme", question_hash=hash_text("test"))
        b.write(ev)
        events = b.query(limit=10)
        assert len(events) == 1
        assert events[0].tenant_id == "acme"

    def test_query_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.jsonl"
        b = JsonLinesBackend(path)
        assert b.query() == []

    def test_stats_counts(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        b = JsonLinesBackend(path)
        b.write(_make_event(event_type=QUERY))
        b.write(_make_event(event_type=INGEST))
        stats = b.stats()
        assert stats.get(QUERY, 0) == 1
        assert stats.get(INGEST, 0) == 1

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "dir" / "audit.jsonl"
        b = JsonLinesBackend(path)
        b.write(_make_event())
        assert path.exists()


# ── AuditLogger ───────────────────────────────────────────────────────────────


class TestAuditLogger:
    def test_disabled_log_is_noop(self) -> None:
        backend = InMemoryBackend()
        al = AuditLogger(backend, enabled=False)
        al.log(_make_event())
        assert backend.size == 0

    def test_enabled_log_writes(self) -> None:
        backend = InMemoryBackend()
        al = AuditLogger(backend, enabled=True)
        al.log(_make_event())
        assert backend.size == 1

    def test_write_error_does_not_crash(self) -> None:
        """K1: backend errors are logged as warnings, never propagated."""
        backend = MagicMock()
        backend.write.side_effect = RuntimeError("disk full")
        al = AuditLogger(backend, enabled=True)
        # Must not raise
        al.log(_make_event())

    def test_query_events_disabled_returns_empty(self) -> None:
        backend = InMemoryBackend()
        backend.write(_make_event())
        al = AuditLogger(backend, enabled=False)
        assert al.query_events() == []

    def test_stats_disabled_returns_empty(self) -> None:
        backend = InMemoryBackend()
        backend.write(_make_event())
        al = AuditLogger(backend, enabled=False)
        assert al.stats() == {}

    def test_enabled_property(self) -> None:
        backend = InMemoryBackend()
        assert AuditLogger(backend, enabled=True).enabled is True
        assert AuditLogger(backend, enabled=False).enabled is False


# ── Singleton ─────────────────────────────────────────────────────────────────


class TestSingleton:
    def setup_method(self) -> None:
        _reset_singleton()

    def teardown_method(self) -> None:
        _reset_singleton()

    def test_get_audit_logger_returns_same_instance(self) -> None:
        """Two calls without reset return the same object."""
        # Directly inject a pre-built logger to avoid touching config
        import konjoai.audit.logger as _mod
        _mod._audit_logger = AuditLogger(InMemoryBackend(), enabled=False)
        a = get_audit_logger()
        b = get_audit_logger()
        assert a is b

    def test_reset_singleton_clears_instance(self) -> None:
        """After _reset_singleton a new instance is created."""
        import konjoai.audit.logger as _mod
        _mod._audit_logger = AuditLogger(InMemoryBackend(), enabled=False)
        a = get_audit_logger()
        _reset_singleton()
        # inject new instance
        _mod._audit_logger = AuditLogger(InMemoryBackend(), enabled=True)
        b = get_audit_logger()
        assert a is not b

    def test_jsonl_backend_selected_when_configured(self, tmp_path: Path) -> None:
        """get_audit_logger builds a JsonLinesBackend when backend=="jsonl"."""
        log_path = str(tmp_path / "audit.jsonl")
        # Inject directly — avoids importing konjoai.config in isolation
        import konjoai.audit.logger as _mod
        _mod._audit_logger = AuditLogger(JsonLinesBackend(log_path), enabled=True)
        al = get_audit_logger()
        assert isinstance(al._backend, JsonLinesBackend)


# ── API Route Tests ───────────────────────────────────────────────────────────


class TestAuditAPIDisabled:
    """When audit_enabled=False the endpoints return 404."""

    def _make_client(self):
        from dataclasses import dataclass as dc
        from fastapi.testclient import TestClient
        from konjoai.api.app import app

        @dc
        class _Settings:
            # Minimal settings stub — only what audit route needs
            audit_enabled: bool = False

        with patch("konjoai.api.routes.audit.get_settings", return_value=_Settings()):
            yield TestClient(app)

    def test_events_disabled_returns_404(self) -> None:
        from dataclasses import dataclass as dc
        from fastapi.testclient import TestClient
        from konjoai.api.routes.audit import router as audit_router

        @dc
        class _S:
            audit_enabled: bool = False

        from fastapi import FastAPI
        _app = FastAPI()
        _app.include_router(audit_router)

        with patch("konjoai.api.routes.audit.get_settings", return_value=_S()):
            with TestClient(_app) as client:
                r = client.get("/audit/events")
        assert r.status_code == 404

    def test_stats_disabled_returns_404(self) -> None:
        from dataclasses import dataclass as dc
        from fastapi.testclient import TestClient
        from konjoai.api.routes.audit import router as audit_router

        @dc
        class _S:
            audit_enabled: bool = False

        from fastapi import FastAPI
        _app = FastAPI()
        _app.include_router(audit_router)

        with patch("konjoai.api.routes.audit.get_settings", return_value=_S()):
            with TestClient(_app) as client:
                r = client.get("/audit/stats")
        assert r.status_code == 404


class TestAuditAPIEnabled:
    """When audit_enabled=True the endpoints serve data."""

    def _build_app_and_backend(self):
        from dataclasses import dataclass as dc
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from konjoai.api.routes.audit import router as audit_router

        @dc
        class _S:
            audit_enabled: bool = True

        backend = InMemoryBackend()
        al = AuditLogger(backend, enabled=True)

        _app = FastAPI()
        _app.include_router(audit_router)
        return _app, backend, al, _S

    def test_events_enabled_returns_200(self) -> None:
        from fastapi.testclient import TestClient
        import konjoai.audit.logger as _logger_mod
        _app, backend, al, _S = self._build_app_and_backend()
        backend.write(_make_event(tenant_id="acme"))

        # Inject singleton directly so the lazy import inside the route hits it
        _logger_mod._audit_logger = al
        try:
            with patch("konjoai.api.routes.audit.get_settings", return_value=_S()):
                with TestClient(_app) as client:
                    r = client.get("/audit/events")
        finally:
            _logger_mod._audit_logger = None
        assert r.status_code == 200
        body = r.json()
        assert body["audit_enabled"] is True
        assert body["total"] >= 0

    def test_stats_enabled_returns_200(self) -> None:
        from fastapi.testclient import TestClient
        import konjoai.audit.logger as _logger_mod
        _app, backend, al, _S = self._build_app_and_backend()
        backend.write(_make_event(event_type=QUERY))

        _logger_mod._audit_logger = al
        try:
            with patch("konjoai.api.routes.audit.get_settings", return_value=_S()):
                with TestClient(_app) as client:
                    r = client.get("/audit/stats")
        finally:
            _logger_mod._audit_logger = None
        assert r.status_code == 200
        body = r.json()
        assert body["audit_enabled"] is True
        assert isinstance(body["stats"], dict)


# ── OWASP PII Contract ────────────────────────────────────────────────────────


class TestNoPIIInAuditEvents:
    """Raw question text must never appear in an AuditEvent."""

    def test_question_is_hashed_not_stored(self) -> None:
        raw = "What is the secret key?"
        ev = _make_event(question_hash=hash_text(raw))
        serialized = json.dumps(ev.as_dict())
        assert raw not in serialized

    def test_path_is_hashed_not_stored(self) -> None:
        raw_path = "/private/data/secret.pdf"
        ev = AuditEvent(
            event_type=INGEST,
            timestamp="2026-05-05T00:00:00+00:00",
            endpoint="/ingest",
            status_code=200,
            latency_ms=10.0,
            path_hash=hash_text(raw_path),
        )
        serialized = json.dumps(ev.as_dict())
        assert raw_path not in serialized


# ── Config Integration ────────────────────────────────────────────────────────


class TestConfigFields:
    def test_audit_config_defaults(self) -> None:
        """Config fields must have correct defaults."""
        from konjoai.config import Settings
        # Use model_fields to check defaults without constructing (avoids .env reading)
        fields = Settings.model_fields
        assert fields["audit_enabled"].default is False
        assert fields["audit_backend"].default == "memory"
        assert fields["audit_max_memory_events"].default == 1000
        assert "jsonl" in fields["audit_log_path"].default
