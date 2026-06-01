"""Semantic cache federation — peer-to-peer cache sharing (Sprint 29).

Architecture
------------
A ``PeerRegistry`` stores registered peer nodes (URL + optional auth token).
The ``FederatedLookup`` helper queries all healthy peers before the caller
falls through to the LLM.  Results seed the local cache so subsequent
identical queries are served locally.

Peer health is checked lazily (on each federated lookup) and tracked with an
exponential-decay availability score so flapping peers are deprioritised
without being removed entirely.

K3: disabled by default (``cache_federation_enabled=False``).  When enabled
but all peers are unreachable, the lookup degrades gracefully and the caller
proceeds to compute normally — no exception is raised.
K5: uses ``httpx`` which is already a declared project dependency.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class PeerNode:
    """A registered federation peer."""

    peer_id: str
    url: str                         # base URL, no trailing slash
    name: str
    auth_token: str | None
    registered_at: float = field(default_factory=time.monotonic)
    last_checked: float | None = None
    last_healthy: bool | None = None  # None = never checked
    availability: float = 1.0         # exponential-decay availability score [0, 1]
    hits_contributed: int = 0


@dataclass
class PeerLookupResult:
    """Result returned by a federated peer lookup."""

    peer_id: str
    peer_name: str
    question: str
    answer: str
    similarity: float
    latency_ms: float


# ── Registry ──────────────────────────────────────────────────────────────────


class PeerRegistry:
    """Thread-safe in-memory registry of federation peers.

    Availability scores decay toward 0 on failure and recover toward 1 on
    success using an exponential moving average with ``alpha=0.3``.
    """

    AVAILABILITY_ALPHA = 0.3   # EMA coefficient
    MIN_AVAILABILITY   = 0.15  # threshold below which a peer is skipped

    def __init__(self) -> None:
        self._peers: dict[str, PeerNode] = {}
        self._lock = threading.Lock()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def register(self, url: str, *, name: str = "", auth_token: str | None = None) -> PeerNode:
        """Register a new peer.  Returns the created node."""
        url = url.rstrip("/")
        peer_id = uuid.uuid4().hex[:12]
        node = PeerNode(peer_id=peer_id, url=url, name=name or url, auth_token=auth_token)
        with self._lock:
            self._peers[peer_id] = node
        logger.info("federation: registered peer %s url=%s", peer_id, url)
        return node

    def remove(self, peer_id: str) -> bool:
        with self._lock:
            if peer_id not in self._peers:
                return False
            del self._peers[peer_id]
        return True

    def get(self, peer_id: str) -> PeerNode | None:
        with self._lock:
            return self._peers.get(peer_id)

    def all_peers(self) -> list[PeerNode]:
        with self._lock:
            return list(self._peers.values())

    def healthy_peers(self) -> list[PeerNode]:
        with self._lock:
            return [p for p in self._peers.values()
                    if p.last_healthy is not False or p.availability >= self.MIN_AVAILABILITY]

    # ── Health tracking ───────────────────────────────────────────────────────

    def _update_availability(self, peer_id: str, success: bool) -> None:
        with self._lock:
            p = self._peers.get(peer_id)
            if p is None:
                return
            p.last_checked = time.monotonic()
            p.last_healthy = success
            p.availability = (
                self.AVAILABILITY_ALPHA * (1.0 if success else 0.0)
                + (1.0 - self.AVAILABILITY_ALPHA) * p.availability
            )

    def record_hit(self, peer_id: str) -> None:
        with self._lock:
            p = self._peers.get(peer_id)
            if p:
                p.hits_contributed += 1

    # ── Health check ──────────────────────────────────────────────────────────

    def check_health(self, peer_id: str, timeout: float = 3.0) -> bool:
        """Synchronous health check against the peer's ``GET /health`` endpoint."""
        p = self.get(peer_id)
        if p is None:
            return False
        try:
            import httpx  # noqa: PLC0415

            headers = {}
            if p.auth_token:
                headers["Authorization"] = f"Bearer {p.auth_token}"
            resp = httpx.get(f"{p.url}/health", headers=headers, timeout=timeout)
            healthy = resp.status_code == 200
        except Exception as exc:  # noqa: BLE001
            logger.debug("federation health check failed for %s: %s", p.url, exc)
            healthy = False
        self._update_availability(peer_id, healthy)
        return healthy

    def check_all_health(self, timeout: float = 3.0) -> dict[str, bool]:
        """Check health for every registered peer.  Returns {peer_id: healthy}."""
        peer_ids = [p.peer_id for p in self.all_peers()]
        return {pid: self.check_health(pid, timeout=timeout) for pid in peer_ids}


# ── Federated lookup ─────────────────────────────────────────────────────────


class FederatedLookup:
    """Query healthy peers for a cache hit before falling through to the LLM.

    Peer lookup uses ``POST /cache/search`` (the Sprint-28 batch similarity
    search endpoint) so it works with any kyro instance running Sprint 28+.
    """

    def __init__(self, registry: PeerRegistry, timeout: float = 2.0) -> None:
        self._registry = registry
        self._timeout = timeout

    def lookup(
        self,
        question: str,
        top_k: int = 1,
        min_similarity: float = 0.95,
    ) -> PeerLookupResult | None:
        """Query all healthy peers; return the first hit above *min_similarity*.

        Returns ``None`` when no peer has a high-confidence answer.  Never
        raises — peer failures are logged and skipped.
        """
        peers = self._registry.healthy_peers()
        if not peers:
            return None

        import httpx  # noqa: PLC0415

        for peer in peers:
            try:
                t0 = time.monotonic()
                headers: dict[str, str] = {}
                if peer.auth_token:
                    headers["Authorization"] = f"Bearer {peer.auth_token}"
                resp = httpx.post(
                    f"{peer.url}/cache/search",
                    json={"queries": [question], "top_k": top_k},
                    headers=headers,
                    timeout=self._timeout,
                )
                latency_ms = (time.monotonic() - t0) * 1000.0
                if resp.status_code != 200:
                    self._registry._update_availability(peer.peer_id, False)
                    continue
                self._registry._update_availability(peer.peer_id, True)
                data = resp.json()
                results = (data.get("results") or [{}])[0].get("matches") or []
                if not results:
                    continue
                best = results[0]
                if best.get("similarity", 0.0) >= min_similarity:
                    self._registry.record_hit(peer.peer_id)
                    return PeerLookupResult(
                        peer_id=peer.peer_id,
                        peer_name=peer.name,
                        question=best.get("question", question),
                        answer=best.get("answer", ""),
                        similarity=float(best.get("similarity", 0.0)),
                        latency_ms=round(latency_ms, 3),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug("federation lookup failed for peer %s: %s", peer.url, exc)
                self._registry._update_availability(peer.peer_id, False)
        return None

    def peer_status(self) -> list[dict[str, Any]]:
        """Snapshot of all peers with health + hit-contribution stats."""
        peers = self._registry.all_peers()
        total_hits = sum(p.hits_contributed for p in peers) or 1
        return [
            {
                "peer_id":          p.peer_id,
                "name":             p.name,
                "url":              p.url,
                "availability":     round(p.availability, 3),
                "last_healthy":     p.last_healthy,
                "hits_contributed": p.hits_contributed,
                "hit_share_pct":    round(p.hits_contributed / total_hits * 100, 1),
                "registered_at":    p.registered_at,
            }
            for p in peers
        ]


# ── Module-level singletons ───────────────────────────────────────────────────

_registry: PeerRegistry | None = None
_lookup: FederatedLookup | None = None
_singleton_lock = threading.Lock()


def get_peer_registry() -> PeerRegistry:
    """Return the process-level ``PeerRegistry`` singleton."""
    global _registry  # noqa: PLW0603
    if _registry is not None:
        return _registry
    with _singleton_lock:
        if _registry is None:
            _registry = PeerRegistry()
    return _registry


def get_federated_lookup(timeout: float = 2.0) -> FederatedLookup:
    """Return the process-level ``FederatedLookup`` singleton."""
    global _lookup  # noqa: PLW0603
    if _lookup is not None:
        return _lookup
    with _singleton_lock:
        if _lookup is None:
            _lookup = FederatedLookup(get_peer_registry(), timeout=timeout)
    return _lookup


def _reset_federation() -> None:
    """Test helper — reset both singletons."""
    global _registry, _lookup  # noqa: PLW0603
    with _singleton_lock:
        _registry = None
        _lookup = None
