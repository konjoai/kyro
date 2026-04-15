"""Query intent router — O(1) heuristic classification, no model inference.

Classifies incoming queries into three intent buckets:
    - RETRIEVAL:    Factual lookups that benefit from Qdrant + BM25 search.
    - AGGREGATION:  Comparative or enumerative questions (may decompose into sub-queries).
    - CHAT:         Conversational utterances that do not need retrieval at all.

CHAT classification provides the largest efficiency gain: we short-circuit the
entire hybrid_search + rerank + generate pipeline and return a canned response
instantly, saving Qdrant latency and generator tokens.

Design: pure regex + heuristic — no model, no network call, < 1 ms on any hardware.
This satisfies K5 (zero new deps) and K2 (telemetry-able, sub-millisecond step).
"""
from __future__ import annotations

import logging
import re
from enum import Enum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent enum
# ---------------------------------------------------------------------------


class QueryIntent(str, Enum):
    """Possible intent classifications for an incoming query."""

    RETRIEVAL = "retrieval"
    AGGREGATION = "aggregation"
    CHAT = "chat"


# ---------------------------------------------------------------------------
# Compiled patterns (module-level — compiled once at import time)
# ---------------------------------------------------------------------------

# Matches conversational openers and closers that never benefit from retrieval.
_CHAT_RE = re.compile(
    r"^\s*("
    r"hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|"
    r"thanks|thank\s+you|thx|ty|"
    r"bye|goodbye|see\s+you|take\s+care|"
    r"ok|okay|sure|got\s+it|sounds\s+good|"
    r"yes|no|yep|nope|"
    r"who\s+are\s+you|what\s+are\s+you|"
    r"what\s+can\s+you\s+do|help\s*me"
    r")[?!.,\s]*$",
    re.IGNORECASE,
)

# Matches aggregation/comparison keywords that often decompose well.
_AGGREGATION_RE = re.compile(
    r"\b("
    r"compare|comparison|difference\s+between|versus|vs|"
    r"list\s+(all|every)|list\s+the|enumerate|"
    r"how\s+many|how\s+much|count\s+of|"
    r"summarize|summary\s+of|overview\s+of|"
    r"pros\s+and\s+cons|advantages\s+and\s+disadvantages"
    r")\b",
    re.IGNORECASE,
)

# Conjunctions used to split aggregation queries into sub-queries.
_CONJUNCTION_RE = re.compile(r"\s+(?:and|vs\.?|versus|compared\s+to|,)\s+", re.IGNORECASE)

# Minimum token length below which a query is almost certainly CHAT.
_MIN_RETRIEVAL_TOKENS: int = 4


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_intent(query: str) -> QueryIntent:
    """Classify the intent of a user query with O(1) heuristics.

    Classification priority (first match wins):
        1. CHAT  — short conversational utterance or greeting.
        2. AGGREGATION  — contains comparison / enumeration keywords.
        3. RETRIEVAL  — everything else.

    Args:
        query: Raw user query string (not stripped — we strip internally).

    Returns:
        QueryIntent enum value.
    """
    q = query.strip()

    # Very short queries are almost always chat (greetings, acknowledgements).
    if len(q.split()) < _MIN_RETRIEVAL_TOKENS and _CHAT_RE.match(q):
        logger.debug("router: CHAT matched for query=%r", q[:60])
        return QueryIntent.CHAT

    # Longer queries that match the greeting pattern.
    if _CHAT_RE.match(q):
        logger.debug("router: CHAT (full match) for query=%r", q[:60])
        return QueryIntent.CHAT

    if _AGGREGATION_RE.search(q):
        logger.debug("router: AGGREGATION matched for query=%r", q[:60])
        return QueryIntent.AGGREGATION

    logger.debug("router: RETRIEVAL (default) for query=%r", q[:60])
    return QueryIntent.RETRIEVAL


def decompose_query(query: str, max_parts: int = 3) -> list[str]:
    """Split an AGGREGATION query into constituent sub-questions.

    Uses conjunction splitting on "and", "vs", "versus", "compared to", and
    comma-separated lists. Returns at most `max_parts` sub-queries so a caller
    can run parallel retrieval without unbounded fan-out.

    Falls back to [query] (single-element list) if no split is found, so callers
    can always iterate over the result without a branch.

    Args:
        query: Raw user query, assumed AGGREGATION intent.
        max_parts: Maximum number of parts to return (default 3).

    Returns:
        List of 1–max_parts non-empty sub-query strings, stripped.

    Example::
        >>> decompose_query("compare Python and Rust for systems programming")
        ['compare Python', 'Rust for systems programming']
    """
    parts = [p.strip() for p in _CONJUNCTION_RE.split(query) if p.strip()]

    if len(parts) <= 1:
        # No conjunction found — return the full query as a single item.
        return [query.strip()]

    return parts[:max_parts]
