"""Auto-strategy router: maps CRAG output → retrieval strategy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RouteStrategy(StrEnum):
    """Retrieval strategy selected for a query."""

    DIRECT = "direct"
    SELF_RAG = "self_rag"
    DECOMPOSE = "decompose"


@dataclass(frozen=True)
class RouteDecision:
    """Chosen strategy with its rationale and the originating CRAG verdict."""

    strategy: RouteStrategy
    rationale: str
    crag_classification: str
    crag_score: float | None


class AutoRouter:
    """Route queries based on CRAG classification and confidence score.

    Rules (case-insensitive classification match):
    - "correct"   → DIRECT   (retrieval is good; answer directly)
    - "ambiguous" → SELF_RAG (uncertain; refine via self-RAG loop)
    - anything else (e.g. "incorrect") → DECOMPOSE (poor retrieval; decompose query)
    """

    def decide(
        self,
        crag_classification: str,
        crag_score: float | None = None,
    ) -> RouteDecision:
        """Map a CRAG classification to a retrieval strategy and rationale."""
        classification = crag_classification.strip().lower()

        if classification == "correct":
            strategy = RouteStrategy.DIRECT
            rationale = "CRAG classified retrieval as correct; answering directly."
        elif classification == "ambiguous":
            strategy = RouteStrategy.SELF_RAG
            rationale = "CRAG classification is ambiguous; applying self-RAG refinement."
        else:
            strategy = RouteStrategy.DECOMPOSE
            rationale = f"CRAG classified retrieval as '{crag_classification}'; decomposing query to improve retrieval."

        return RouteDecision(
            strategy=strategy,
            rationale=rationale,
            crag_classification=crag_classification,
            crag_score=crag_score,
        )
