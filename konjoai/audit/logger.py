"""Audit backends and AuditLogger singleton.

Two backends ship out of the box (K5 — stdlib only):

* ``InMemoryBackend``   — bounded ring buffer (deque); default for dev/test.
* ``JsonLinesBackend``  — one JSON object per line; rotate externally.

The ``AuditLogger`` wraps a backend and enforces K3: when
``audit_enabled=False`` every ``log()`` call is a pure no-op with zero
allocation overhead.
"""
from __future__ import annotations

import json
import logging
import threading
from collections import deque
from pathlib import Path
from typing import Protocol

from konjoai.audit.models import AuditEvent

logger = logging.getLogger(__name__)


# ── Backend protocol ──────────────────────────────────────────────────────────


class AuditBackend(Protocol):
    """Structural protocol for pluggable audit backends."""

    def write(self, event: AuditEvent) -> None: ...

    def query(
        self,
        *,
        limit: int = 100,
        tenant_id: str | None = None,
        event_type: str | None = None,
    ) -> list[AuditEvent]: ...

    def stats(self) -> dict[str, int]: ...


# ── InMemoryBackend ───────────────────────────────────────────────────────────


class InMemoryBackend:
    """Bounded in-memory ring buffer — wraps ``collections.deque(maxlen=N)``.

    Thread-safe via a single ``threading.Lock``.  When the buffer is full the
    oldest event is silently evicted (FIFO).
    """

    def __init__(self, max_events: int = 1000) -> None:
        if max_events < 1:
            raise ValueError("max_events must be ≥ 1")
        self._events: deque[AuditEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def write(self, event: AuditEvent) -> None:
        with self._lock:
            self._events.append(event)

    def query(
        self,
        *,
        limit: int = 100,
        tenant_id: str | None = None,
        event_type: str | None = None,
    ) -> list[AuditEvent]:
        with self._lock:
            events = list(self._events)

        if tenant_id is not None:
            events = [e for e in events if e.tenant_id == tenant_id]
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]

        return events[-limit:]

    def stats(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for e in self._events:
                counts[e.event_type] = counts.get(e.event_type, 0) + 1
        return counts

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._events)


# ── JsonLinesBackend ──────────────────────────────────────────────────────────


class JsonLinesBackend:
    """Append-only JSON Lines file backend.

    Each event is serialised as a single JSON object on its own line.
    Thread-safe via a ``threading.Lock`` around each file append.  Rotation
    is expected to be handled externally (logrotate / k8s volume).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: AuditEvent) -> None:
        line = json.dumps(event.as_dict(), default=str) + "\n"
        with self._lock:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line)

    def query(
        self,
        *,
        limit: int = 100,
        tenant_id: str | None = None,
        event_type: str | None = None,
    ) -> list[AuditEvent]:
        if not self._path.exists():
            return []
        events: list[AuditEvent] = []
        with self._lock:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            try:
                data = json.loads(line)
                event = AuditEvent(**{
                    k: v for k, v in data.items()
                    if k in AuditEvent.__dataclass_fields__
                })
                if tenant_id is not None and event.tenant_id != tenant_id:
                    continue
                if event_type is not None and event.event_type != event_type:
                    continue
                events.append(event)
            except (json.JSONDecodeError, TypeError):
                pass
        return events[-limit:]

    def stats(self) -> dict[str, int]:
        events = self.query(limit=10_000)
        counts: dict[str, int] = {}
        for e in events:
            counts[e.event_type] = counts.get(e.event_type, 0) + 1
        return counts


# ── AuditLogger ───────────────────────────────────────────────────────────────


class AuditLogger:
    """Thin enabled/disabled wrapper around an ``AuditBackend``.

    When ``enabled=False`` (K3 default) the ``log()`` method returns
    immediately — zero allocation, zero file I/O.
    """

    def __init__(self, backend: AuditBackend, *, enabled: bool = False) -> None:
        self._backend = backend
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def log(self, event: AuditEvent) -> None:
        if not self._enabled:
            return
        try:
            self._backend.write(event)
        except Exception as exc:  # K1: log but never crash the request
            logger.warning("audit write failed: %s", exc)

    def query_events(
        self,
        *,
        limit: int = 100,
        tenant_id: str | None = None,
        event_type: str | None = None,
    ) -> list[AuditEvent]:
        if not self._enabled:
            return []
        return self._backend.query(limit=limit, tenant_id=tenant_id, event_type=event_type)

    def stats(self) -> dict[str, int]:
        if not self._enabled:
            return {}
        return self._backend.stats()


# ── Singleton ─────────────────────────────────────────────────────────────────

_audit_logger: AuditLogger | None = None
_singleton_lock = threading.Lock()


def get_audit_logger() -> AuditLogger:
    """Return the module-level ``AuditLogger`` singleton (lazy-init)."""
    global _audit_logger
    if _audit_logger is None:
        with _singleton_lock:
            if _audit_logger is None:
                from konjoai.config import get_settings
                settings = get_settings()
                enabled = getattr(settings, "audit_enabled", False)
                backend_name = getattr(settings, "audit_backend", "memory")
                max_events = getattr(settings, "audit_max_memory_events", 1000)
                log_path = getattr(settings, "audit_log_path", "")

                if backend_name == "jsonl" and log_path:
                    backend: AuditBackend = JsonLinesBackend(log_path)
                else:
                    backend = InMemoryBackend(max_events=max_events)

                _audit_logger = AuditLogger(backend, enabled=enabled)
    return _audit_logger


def _reset_singleton() -> None:
    """Reset the singleton for test isolation."""
    global _audit_logger
    with _singleton_lock:
        _audit_logger = None
