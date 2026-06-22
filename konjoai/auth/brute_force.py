"""IP-based brute-force protection for JWT/auth endpoints (Sprint 18).

Tracks failed authentication attempts by IP address. After N failures within
M seconds the IP is locked out for a configurable cool-down period. All state
is in-memory; no external dependency (K5).

Invariants:
- K3: when ``brute_force_enabled=False`` every call is a no-op.
- K5: pure Python stdlib — ``threading``, ``time``, ``collections``.
- K1: ``check_ip`` either returns ``None`` (OK) or raises ``IPLockedOut``.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from konjoai.config import get_settings

__all__ = [
    "BruteForceGuard",
    "IPLockedOut",
    "get_brute_force_guard",
]


class IPLockedOut(Exception):
    """Raised when an IP address is temporarily locked out."""

    def __init__(self, ip: str, lockout_seconds: int, retry_after: float) -> None:
        self.ip = ip
        self.lockout_seconds = lockout_seconds
        self.retry_after = retry_after  # monotonic timestamp when lockout expires
        super().__init__(
            f"IP {ip!r} is locked out for {lockout_seconds}s due to repeated failed authentication attempts."
        )


@dataclass
class _IPRecord:
    """Per-IP failure history and optional lockout expiry."""

    window_seconds: int
    max_attempts: int
    lockout_seconds: int
    _failures: deque[float] = field(default_factory=deque)
    _locked_until: float = 0.0  # monotonic timestamp
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_failure(self, now: float | None = None) -> None:
        """Record a failed attempt; does NOT raise — caller decides on lockout."""
        if now is None:
            now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            while self._failures and self._failures[0] <= cutoff:
                self._failures.popleft()
            self._failures.append(now)
            if len(self._failures) >= self.max_attempts:
                self._locked_until = now + self.lockout_seconds

    def is_locked(self, now: float | None = None) -> bool:
        """Return True if the lockout window has not yet expired."""
        if now is None:
            now = time.monotonic()
        with self._lock:
            return now < self._locked_until

    def locked_until(self) -> float:
        """Return the monotonic timestamp at which the lockout expires."""
        with self._lock:
            return self._locked_until

    def reset(self) -> None:
        """Clear all recorded failures and any active lockout."""
        with self._lock:
            self._failures.clear()
            self._locked_until = 0.0

    @property
    def failure_count(self) -> int:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            while self._failures and self._failures[0] <= cutoff:
                self._failures.popleft()
            return len(self._failures)


class BruteForceGuard:
    """Track failed auth attempts per IP and enforce temporary lockouts.

    Args:
        max_attempts:     Max failures in *window_seconds* before lockout.
        window_seconds:   Sliding window for counting failures.
        lockout_seconds:  How long a locked-out IP must wait.
        enabled:          When False all calls are no-ops (K3).
    """

    def __init__(
        self,
        max_attempts: int = 5,
        window_seconds: int = 60,
        lockout_seconds: int = 300,
        enabled: bool = True,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError(f"max_attempts must be > 0, got {max_attempts}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {window_seconds}")
        if lockout_seconds <= 0:
            raise ValueError(f"lockout_seconds must be > 0, got {lockout_seconds}")
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self.lockout_seconds = lockout_seconds
        self.enabled = enabled
        self._records: dict[str, _IPRecord] = {}
        self._registry_lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def check_ip(self, ip: str, now: float | None = None) -> None:
        """Assert *ip* is not locked out.

        Raises:
            IPLockedOut: if the IP is in lockout.
        """
        if not self.enabled:
            return
        record = self._get_record(ip)
        if now is None:
            now = time.monotonic()
        if record.is_locked(now):
            raise IPLockedOut(
                ip=ip,
                lockout_seconds=self.lockout_seconds,
                retry_after=record.locked_until(),
            )

    def record_failure(self, ip: str, now: float | None = None) -> None:
        """Record a failed auth attempt for *ip*.

        If the failure count within the window reaches ``max_attempts`` the IP
        is locked out immediately.
        """
        if not self.enabled:
            return
        record = self._get_record(ip)
        record.record_failure(now)

    def record_success(self, ip: str) -> None:
        """Clear the failure record for *ip* on a successful authentication."""
        if not self.enabled:
            return
        with self._registry_lock:
            if ip in self._records:
                self._records[ip].reset()

    def failure_count(self, ip: str) -> int:
        """Return the number of failures in the current window for *ip*."""
        with self._registry_lock:
            rec = self._records.get(ip)
        if rec is None:
            return 0
        return rec.failure_count

    def is_locked(self, ip: str, now: float | None = None) -> bool:
        """Return True if *ip* is currently locked out."""
        with self._registry_lock:
            rec = self._records.get(ip)
        if rec is None:
            return False
        return rec.is_locked(now)

    def reset(self, ip: str | None = None) -> None:
        """Clear all records, or only the record for *ip*."""
        with self._registry_lock:
            if ip is None:
                self._records.clear()
            elif ip in self._records:
                self._records[ip].reset()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_record(self, ip: str) -> _IPRecord:
        """Return the :class:`_IPRecord` for *ip*, creating it on first access."""
        with self._registry_lock:
            if ip not in self._records:
                self._records[ip] = _IPRecord(
                    window_seconds=self.window_seconds,
                    max_attempts=self.max_attempts,
                    lockout_seconds=self.lockout_seconds,
                )
            return self._records[ip]


# ── Module-level singleton ────────────────────────────────────────────────────

_guard_instance: BruteForceGuard | None = None
_guard_lock = threading.Lock()


def get_brute_force_guard() -> BruteForceGuard:
    """Return the module-level BruteForceGuard singleton (lazy, reads settings)."""
    global _guard_instance
    with _guard_lock:
        if _guard_instance is None:
            s = get_settings()
            _guard_instance = BruteForceGuard(
                max_attempts=getattr(s, "brute_force_max_attempts", 5),
                window_seconds=getattr(s, "brute_force_window_seconds", 60),
                lockout_seconds=getattr(s, "brute_force_lockout_seconds", 300),
                enabled=getattr(s, "brute_force_enabled", False),
            )
        return _guard_instance


def _reset_singleton() -> None:
    """Force re-creation of the singleton on next call (test helper)."""
    global _guard_instance
    with _guard_lock:
        _guard_instance = None
