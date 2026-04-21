"""Adaptive chunking: query-aware granularity selection for Kyro pipelines.

Problem
-------
Standard RAG uses a single fixed chunk size (e.g. 512 tokens) for every
document regardless of query complexity.  This is the primary failure mode
of naive RAG:

* Simple factual queries ("What is the capital of France?") need large,
  context-rich chunks so the answer doesn't span a boundary.
* Complex reasoning queries ("Compare the competitive positioning of three
  vendors across five dimensions") need small, precise chunks so each
  retrieved unit is actionable.

Solution
--------
Two components work together:

1. **MultiGranularityChunker** — indexes each document at multiple chunk
   sizes simultaneously (parent: 1024, base: 512, child: 128 tokens).
   All granularities share the same ``source`` identifier; the granularity
   level is stored in ``metadata["granularity"]``.

2. **QueryComplexityScorer** — scores an incoming query from 0.0 (very
   simple) to 1.0 (very complex) using lightweight heuristics:
   * Token count (more tokens → more complex)
   * Entity-like noun phrase density (simple regex proxy)
   * Multi-part question marker (presence of "and", "vs", "compare", …)
   * Aggregation signal (presence of "list all", "summarize", "overview", …)

   The scorer is intentionally heuristic-only (no LLM call, no network
   round-trip) so it adds < 1 ms overhead on the hot path.

3. **adaptive_chunk_size** — maps a complexity score to a chunk size from
   the configured ``chunk_sizes_hierarchy``.

K4: No float arrays cross this module's boundary.
K5: Zero new dependencies (stdlib + existing ``regex``/``re`` only).
K1: All public functions raise on invalid input rather than returning None.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from konjoai.ingest.chunkers import Chunk, RecursiveChunker
from konjoai.ingest.loaders import Document


# ── Complexity scoring ────────────────────────────────────────────────────────

_MULTI_PART_RE = re.compile(
    r"\b(and|vs\.?|versus|compare|contrast|difference between|both|all)\b",
    re.IGNORECASE,
)
_AGGREGATION_RE = re.compile(
    r"\b(list|summarize|overview|enumerate|what are|how many|all of)\b",
    re.IGNORECASE,
)
_ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


class QueryComplexityScorer:
    """Score query complexity from 0.0 (trivial) to 1.0 (highly complex).

    The score is the arithmetic mean of four sub-scores, each in [0, 1]:

    * **length_score** — normalised token count (≥ 10 tokens → 1.0)
    * **multi_part_score** — 1.0 if multi-part markers are present
    * **aggregation_score** — 1.0 if aggregation markers are present
    * **entity_score** — normalised named-entity count (≥ 3 entities → 1.0)
    """

    def score(self, query: str) -> float:
        """Return a complexity score in ``[0.0, 1.0]``."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        tokens = query.split()
        length_score = min(len(tokens) / 10.0, 1.0)
        multi_part_score = 1.0 if _MULTI_PART_RE.search(query) else 0.0
        aggregation_score = 1.0 if _AGGREGATION_RE.search(query) else 0.0
        entities = _ENTITY_PATTERN.findall(query)
        entity_score = min(len(entities) / 3.0, 1.0)

        return (length_score + multi_part_score + aggregation_score + entity_score) / 4.0

    def complexity_label(self, query: str) -> str:
        """Return ``"simple"``, ``"moderate"``, or ``"complex"``."""
        s = self.score(query)
        if s < 0.35:
            return "simple"
        if s < 0.65:
            return "moderate"
        return "complex"


def adaptive_chunk_size(complexity: float, hierarchy: list[int]) -> int:
    """Map a complexity score to a chunk size from *hierarchy*.

    The hierarchy should be ordered from largest to smallest, e.g.
    ``[1024, 512, 128]``.

    * Low complexity (< 0.35) → largest chunk (more context).
    * High complexity (> 0.65) → smallest chunk (more precision).
    * Moderate → middle tier.

    Args:
        complexity: Score in ``[0.0, 1.0]`` from :class:`QueryComplexityScorer`.
        hierarchy:  List of chunk sizes ordered largest → smallest (min 2 entries).

    Returns:
        The selected chunk size.

    Raises:
        ValueError: If *hierarchy* is empty or *complexity* is out of range.
    """
    if not hierarchy:
        raise ValueError("hierarchy must contain at least one chunk size")
    if not 0.0 <= complexity <= 1.0:
        raise ValueError(f"complexity must be in [0, 1], got {complexity}")
    if len(hierarchy) == 1:
        return hierarchy[0]

    n = len(hierarchy)
    # Map complexity [0, 1] → index [0, n-1] (0 = largest chunk)
    idx = round(complexity * (n - 1))
    return hierarchy[idx]


# ── Multi-granularity chunker ─────────────────────────────────────────────────

@dataclass
class GranularChunk:
    """A chunk annotated with its granularity level."""

    content: str
    source: str
    chunk_index: int
    granularity: str            # "parent" | "base" | "child"
    chunk_size: int             # actual configured size for this granularity
    metadata: dict = field(default_factory=dict)


class MultiGranularityChunker:
    """Index documents at multiple chunk sizes simultaneously.

    Each document is split into three granularity levels so that the
    adaptive retriever can select the appropriate level per query without
    re-chunking at query time.

    Args:
        sizes: Ordered list ``[parent_size, base_size, child_size]``.
               Defaults to ``[1024, 512, 128]``.
        overlap: Overlap tokens applied uniformly across all granularities.
    """

    _LABELS = ["parent", "base", "child"]

    def __init__(
        self,
        sizes: list[int] | None = None,
        overlap: int = 64,
    ) -> None:
        self.sizes = sizes or [1024, 512, 128]
        self.overlap = overlap
        if len(self.sizes) < 2:
            raise ValueError("sizes must contain at least 2 entries")
        self._chunkers = [
            RecursiveChunker(chunk_size=s, overlap=min(overlap, s // 4))
            for s in self.sizes
        ]

    def chunk(self, doc: Document) -> list[GranularChunk]:
        """Return chunks at all granularity levels for *doc*."""
        result: list[GranularChunk] = []
        labels = self._LABELS + [f"level_{i}" for i in range(3, len(self.sizes))]
        for i, chunker in enumerate(self._chunkers):
            label = labels[i] if i < len(labels) else f"level_{i}"
            for chunk in chunker.chunk(doc):
                result.append(
                    GranularChunk(
                        content=chunk.content,
                        source=chunk.source,
                        chunk_index=chunk.chunk_index,
                        granularity=label,
                        chunk_size=self.sizes[i],
                        metadata={**chunk.metadata, "granularity": label},
                    )
                )
        return result

    def chunk_at_level(self, doc: Document, granularity: str) -> list[Chunk]:
        """Return chunks only at the requested *granularity* label.

        Useful when you want to ingest a single level on demand.

        Raises:
            ValueError: If *granularity* is not in the configured labels.
        """
        labels = self._LABELS[: len(self.sizes)]
        if granularity not in labels:
            raise ValueError(
                f"Unknown granularity {granularity!r}; valid: {labels}"
            )
        idx = labels.index(granularity)
        return self._chunkers[idx].chunk(doc)
