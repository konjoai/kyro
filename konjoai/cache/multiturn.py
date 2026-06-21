"""Multi-turn conversation cache — Sprint 28.

Extends the semantic cache with conversation-aware key scoping.  The lookup and
store key is a composite of the *turn hash* (a rolling SHA-256 of recent question
hashes) and the current question text, so semantically similar questions asked in
different conversation contexts do NOT collide.

Design
------
- ``TurnHistory``      — per-conversation bounded window of question hashes (no raw
                         text — OWASP).
- ``ConversationStore`` — thread-safe map of ``(tenant_id, conversation_id) → TurnHistory``
                         with LRU eviction at ``max_conversations``.
- ``MultiTurnCache``   — wraps a ``SemanticCache`` configured with the multi-turn
                         threshold and uses a ``ConversationStore`` to compute turn-
                         scoped keys.  Has its OWN inner ``SemanticCache`` instance,
                         separate from the main single-turn cache.

Turn hash
---------
  ``turn_hash = sha256(":".join([hash(q[-N]), ..., hash(q[-1]), hash(current)]))[:16]``

K1: ConversationStore mutations are wrapped; failures are logged, not raised.
K3: Cache is only instantiated when ``cache_multiturn_enabled=True``.
K5: stdlib only — ``hashlib``, ``threading``, ``collections.OrderedDict``.
K7: Conversation histories are scoped to ``(tenant_id, conversation_id)``.
OWASP: Raw question text is never stored — only 16-hex SHA-256 prefixes.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "ConversationStore",
    "MultiTurnCache",
    "TurnHistory",
    "compute_turn_hash",
    "question_hash",
    "get_conversation_store",
    "get_multiturn_cache",
    "_reset_singletons",
]


# ── Hash helpers ───────────────────────────────────────────────────────────────


def question_hash(question: str) -> str:
    """Return the 16-hex SHA-256 prefix for a question string (OWASP — no raw text)."""
    return hashlib.sha256(question.encode()).hexdigest()[:16]


def compute_turn_hash(hashes: list[str]) -> str:
    """Return a 16-hex turn fingerprint from an ordered list of question hashes."""
    combined = ":".join(hashes) if hashes else "empty"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


# ── Turn history ───────────────────────────────────────────────────────────────


@dataclass
class TurnHistory:
    """Bounded, OWASP-compliant conversation history for one ``(tenant, conversation)``.

    Stores only question hashes — never raw question text.

    Args:
        max_turns: Number of prior turns to include in the turn-hash window.
    """

    max_turns: int

    def __post_init__(self) -> None:
        self._hashes: list[str] = []

    def add(self, question: str) -> None:
        """Hash the question and append to the window, evicting the oldest when full."""
        self._hashes.append(question_hash(question))
        if len(self._hashes) > self.max_turns:
            self._hashes = self._hashes[-self.max_turns :]

    def current_turn_hash(self, current_question: str) -> str:
        """Return the turn hash for ``current_question`` given the current history."""
        hashes = self._hashes + [question_hash(current_question)]
        return compute_turn_hash(hashes)

    def __len__(self) -> int:
        return len(self._hashes)


# ── Conversation store ─────────────────────────────────────────────────────────


class ConversationStore:
    """Thread-safe store of TurnHistory instances with LRU eviction.

    Each key is ``(tenant_id, conversation_id)``.  When ``max_conversations`` is
    reached the least-recently-written conversation is evicted.

    Args:
        max_conversations: Maximum concurrent conversation histories to track.
        max_turns: Turn-window size passed to each new TurnHistory.
    """

    def __init__(
        self,
        max_conversations: int = 1000,
        max_turns: int = 5,
    ) -> None:
        if max_conversations < 1:
            raise ValueError(f"max_conversations must be >= 1, got {max_conversations}")
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")
        self._max_conversations = max_conversations
        self._max_turns = max_turns
        self._store: OrderedDict[tuple[str, str], TurnHistory] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_create(self, tenant_id: str, conversation_id: str) -> TurnHistory:
        """Return the TurnHistory for this conversation, creating it if absent."""
        key = (tenant_id, conversation_id)
        with self._lock:
            if key in self._store:
                return self._store[key]
            if len(self._store) >= self._max_conversations:
                self._store.popitem(last=False)
            history = TurnHistory(max_turns=self._max_turns)
            self._store[key] = history
            return history

    def add_turn(
        self,
        tenant_id: str,
        conversation_id: str,
        question: str,
    ) -> None:
        """Append a question hash to the conversation history. Never raises (K1)."""
        try:
            key = (tenant_id, conversation_id)
            with self._lock:
                if key in self._store:
                    self._store.move_to_end(key)
                else:
                    if len(self._store) >= self._max_conversations:
                        self._store.popitem(last=False)
                    self._store[key] = TurnHistory(max_turns=self._max_turns)
                self._store[key].add(question)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "conversation store: add_turn failed tenant=%s conv=%s: %s",
                tenant_id,
                conversation_id,
                exc,
            )

    def get_turn_hash(
        self,
        tenant_id: str,
        conversation_id: str,
        current_question: str,
    ) -> str:
        """Return the turn hash for ``current_question`` in this conversation's context."""
        key = (tenant_id, conversation_id)
        with self._lock:
            history = self._store.get(key)
            if history is None:
                return compute_turn_hash([question_hash(current_question)])
            return history.current_turn_hash(current_question)

    def conversation_count(self) -> int:
        """Return the number of active conversation histories."""
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        """Remove all conversation histories. Intended for tests."""
        with self._lock:
            self._store.clear()


# ── Multi-turn cache ───────────────────────────────────────────────────────────


class MultiTurnCache:
    """Conversation-scoped semantic cache.

    Lookup and store use a ``[conv:<turn_hash>] {question}`` key prefix, giving
    each conversation context a separate address space in the underlying
    ``SemanticCache``.

    Args:
        inner_cache: A ``SemanticCache`` configured with the multi-turn threshold.
        conversation_store: Injected store; defaults to the module-level singleton.
    """

    def __init__(
        self,
        inner_cache: object,
        conversation_store: ConversationStore | None = None,
    ) -> None:
        self._cache = inner_cache
        self._conversations = conversation_store or get_conversation_store()

    # ── Public API ─────────────────────────────────────────────────────────────

    def lookup(
        self,
        question: str,
        q_vec: np.ndarray,
        tenant_id: str,
        conversation_id: str,
    ) -> object | None:
        """Look up the question in the conversation's scoped cache.

        Does not advance conversation history — call :meth:`advance_turn` on hit.
        """
        turn_hash = self._conversations.get_turn_hash(tenant_id, conversation_id, question)
        return self._cache.lookup(  # type: ignore[attr-defined]
            self._keyed(question, turn_hash), q_vec
        )

    def store(
        self,
        question: str,
        q_vec: np.ndarray,
        response: object,
        tenant_id: str,
        conversation_id: str,
    ) -> None:
        """Store the response and advance conversation history."""
        turn_hash = self._conversations.get_turn_hash(tenant_id, conversation_id, question)
        self._cache.store(  # type: ignore[attr-defined]
            self._keyed(question, turn_hash), q_vec, response
        )
        self._conversations.add_turn(tenant_id, conversation_id, question)

    def advance_turn(
        self,
        tenant_id: str,
        conversation_id: str,
        question: str,
    ) -> None:
        """Advance conversation history without storing a response.

        Call this when the answer comes from a cache hit so the conversation
        context still advances for the next turn.
        """
        self._conversations.add_turn(tenant_id, conversation_id, question)

    @property
    def conversation_store(self) -> ConversationStore:
        """The underlying ConversationStore."""
        return self._conversations

    def stats(self) -> dict:
        """Return combined cache and conversation-store statistics."""
        cache_stats: dict = {}
        if hasattr(self._cache, "stats"):
            cache_stats = self._cache.stats()  # type: ignore[attr-defined]
        return {
            **cache_stats,
            "active_conversations": self._conversations.conversation_count(),
        }

    # ── Internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _keyed(question: str, turn_hash: str) -> str:
        """Prefix a question with its conversation turn hash for cache keying."""
        return f"[conv:{turn_hash}] {question}"


# ── Singletons ─────────────────────────────────────────────────────────────────


_conversation_store: ConversationStore | None = None
_conversation_store_lock = threading.Lock()

_multiturn_cache: MultiTurnCache | None = None
_multiturn_cache_lock = threading.Lock()


def get_conversation_store() -> ConversationStore:
    """Return the module-level ConversationStore singleton."""
    global _conversation_store  # noqa: PLW0603
    if _conversation_store is not None:
        return _conversation_store
    with _conversation_store_lock:
        if _conversation_store is None:
            from konjoai.config import get_settings  # noqa: PLC0415

            settings = get_settings()
            _conversation_store = ConversationStore(
                max_conversations=getattr(settings, "cache_multiturn_max_conversations", 1000),
                max_turns=getattr(settings, "cache_multiturn_window", 5),
            )
    return _conversation_store  # type: ignore[return-value]


def get_multiturn_cache() -> MultiTurnCache:
    """Return the module-level MultiTurnCache singleton."""
    global _multiturn_cache  # noqa: PLW0603
    if _multiturn_cache is not None:
        return _multiturn_cache
    with _multiturn_cache_lock:
        if _multiturn_cache is None:
            from konjoai.cache.semantic_cache import SemanticCache  # noqa: PLC0415
            from konjoai.config import get_settings  # noqa: PLC0415

            settings = get_settings()
            inner = SemanticCache(
                max_size=getattr(settings, "cache_max_size", 500),
                threshold=getattr(settings, "cache_multiturn_threshold", 0.88),
                ttl_seconds=getattr(settings, "cache_ttl_seconds", 0),
            )
            _multiturn_cache = MultiTurnCache(
                inner_cache=inner,
                conversation_store=get_conversation_store(),
            )
    return _multiturn_cache  # type: ignore[return-value]


def _reset_singletons() -> None:
    """Reset module-level singletons. Test helper — never call in production."""
    global _conversation_store, _multiturn_cache  # noqa: PLW0603
    _conversation_store = None
    _multiturn_cache = None
