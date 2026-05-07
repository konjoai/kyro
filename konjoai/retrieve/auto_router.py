"""Auto-strategy router: maps CRAG output → retrieval strategy."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RouteStrategy(str, Enum):
    DIRECT = "direct"
    SELF_RAG = "self_rag"
    DECOMPOSE = "decompose"


@dataclass(frozen=True)
class RouteDecision:
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
        classification = crag_classification.strip().lower()

        if classification == "correct":
            strategy = RouteStrategy.DIRECT
            rationale = "CRAG classified retrieval as correct; answering directly."
        elif classification == "ambiguous":
            strategy = RouteStrategy.SELF_RAG
            rationale = "CRAG classification is ambiguous; applying self-RAG refinement."
        else:
            strategy = RouteStrategy.DECOMPOSE
            rationale = (
                f"CRAG classified retrieval as '{crag_classification}'; "
                "decomposing query to improve retrieval."
            )

        return RouteDecision(
            strategy=strategy,
            rationale=rationale,
            crag_classification=crag_classification,
            crag_score=crag_score,
        )
