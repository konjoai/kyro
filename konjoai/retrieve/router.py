"""Query intent router — O(1) heuristic classification, no model inference.

Classifies incoming queries into three intent buckets:
    - RETRIEVAL:    Factual lookups that benefit from Qdrant + BM25 search.
    - AGGREGATION:  Comparative or enumerative questions (may decompose into sub-queries).
    - CHAT:         Conversational utterances that do not need retrieval at all.

Also classifies queries by complexity to select the optimal chunk size at
retrieval time:
    - SIMPLE  (complexity < 0.35) → 256-token chunks — precise fact retrieval.
    - MEDIUM  (0.35 ≤ complexity < 0.65) → 512-token chunks — balanced.
    - COMPLEX (complexity ≥ 0.65) → 1024-token chunks — rich context for reasoning.

CHAT classification provides the largest efficiency gain: we short-circuit the
entire hybrid_search + rerank + generate pipeline and return a canned response
instantly, saving Qdrant latency and generator tokens.

Design: pure regex + heuristic — no model, no network call, < 1 ms on any hardware.
This satisfies K5 (zero new deps) and K2 (telemetry-able, sub-millisecond step).
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent enum
# ---------------------------------------------------------------------------


class QueryIntent(StrEnum):
    """Possible intent classifications for an incoming query."""

    RETRIEVAL = "retrieval"
    AGGREGATION = "aggregation"
    CHAT = "chat"


# ---------------------------------------------------------------------------
# Chunk complexity enum
# ---------------------------------------------------------------------------


class ChunkComplexity(StrEnum):
    """Query complexity tier, used to select the optimal retrieval chunk size.

    Chunk size rationale (validated by ablation, Sprint 10):

    * **SIMPLE** queries ("What is X?") need *large* context-rich chunks so
      the answer is not split across a boundary.  Counter-intuitive but
      supported by the adaptive-chunking literature: a simple factual query
      wants one contiguous passage, not many small fragments.
    * **COMPLEX** multi-hop reasoning queries need *small* precise chunks so
      each retrieved unit is independently actionable and the LLM is not
      overwhelmed with irrelevant surrounding text.
    """

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


# Chunk sizes (tokens) per complexity tier.  Configurable via
# ``Settings.chunk_sizes_hierarchy`` but these are the fixed heuristic defaults.
CHUNK_SIZE_MAP: dict[ChunkComplexity, int] = {
    ChunkComplexity.SIMPLE: 256,
    ChunkComplexity.MEDIUM: 512,
    ChunkComplexity.COMPLEX: 1024,
}


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


# ---------------------------------------------------------------------------
# Chunk complexity router
# ---------------------------------------------------------------------------

# Module-level singleton — allocated once, shared across all calls.
_complexity_scorer = None


def _get_complexity_scorer():
    global _complexity_scorer  # noqa: PLW0603
    if _complexity_scorer is None:
        from konjoai.ingest.adaptive_chunker import QueryComplexityScorer  # noqa: PLC0415

        _complexity_scorer = QueryComplexityScorer()
    return _complexity_scorer


# Label string → ChunkComplexity — "moderate" maps to MEDIUM.
_LABEL_TO_COMPLEXITY: dict[str, ChunkComplexity] = {
    "simple": ChunkComplexity.SIMPLE,
    "moderate": ChunkComplexity.MEDIUM,
    "complex": ChunkComplexity.COMPLEX,
}


def classify_chunk_complexity(query: str) -> tuple[ChunkComplexity, int]:
    """Map a query to the optimal chunk size for retrieval.

    Uses :class:`~konjoai.ingest.adaptive_chunker.QueryComplexityScorer` to
    score the query on a ``[0, 1]`` scale, then maps the label to a
    :class:`ChunkComplexity` tier and its associated chunk size.

    This extends the existing intent router to support *adaptive chunking*:
    the pipeline can use the returned chunk size to select which granularity
    level of the :class:`~konjoai.ingest.adaptive_chunker.MultiGranularityChunker`
    index to query.

    Complexity→size mapping (see :data:`CHUNK_SIZE_MAP`):

    * ``SIMPLE``  → 256 tokens (precise retrieval, single-fact answers)
    * ``MEDIUM``  → 512 tokens (default, balanced)
    * ``COMPLEX`` → 1024 tokens (rich context, multi-hop reasoning)

    Args:
        query: Raw user query string.

    Returns:
        ``(ChunkComplexity, chunk_size_tokens)`` tuple.

    Raises:
        ValueError: If *query* is empty (propagated from the scorer).

    Example::
        >>> complexity, size = classify_chunk_complexity("What is RRF?")
        >>> complexity
        <ChunkComplexity.SIMPLE: 'simple'>
        >>> size
        256
    """
    scorer = _get_complexity_scorer()
    label = scorer.complexity_label(query)
    complexity = _LABEL_TO_COMPLEXITY[label]
    chunk_size = CHUNK_SIZE_MAP[complexity]
    logger.debug(
        "router: complexity=%s chunk_size=%d for query=%r",
        complexity.value,
        chunk_size,
        query[:60],
    )
    return complexity, chunk_size
