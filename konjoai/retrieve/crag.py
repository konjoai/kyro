"""CRAG (Corrective RAG) evaluator.

Sprint 11 contract:
1. Score every retrieved chunk with the same cross-encoder used by reranking.
2. Classify each chunk using normalized score bands:
   - CORRECT:   score > 0.7
   - AMBIGUOUS: 0.3 <= score <= 0.7
   - INCORRECT: score < 0.3
3. If all chunks are INCORRECT, trigger ``web_fallback()``.
4. If there is a CORRECT/AMBIGUOUS mix, keep CORRECT chunks and refine
   AMBIGUOUS chunks via decomposed sub-queries.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class CRAGClassification(StrEnum):
    """Per-chunk CRAG quality class."""

    CORRECT = "CORRECT"
    AMBIGUOUS = "AMBIGUOUS"
    INCORRECT = "INCORRECT"


@dataclass
class CRAGChunk:
    """Retrieved chunk plus CRAG score and class."""

    content: str
    source: str
    score: float
    metadata: dict = field(default_factory=dict)
    crag_score: float = 0.0
    classification: CRAGClassification = CRAGClassification.AMBIGUOUS


@dataclass
class CRAGResult:
    """Output of ``CRAGEvaluator.run``."""

    selected_chunks: list[CRAGChunk]
    scored_chunks: list[CRAGChunk]
    fallback_chunks: list[CRAGChunk]
    crag_scores: list[float]
    crag_classification: list[str]
    refinement_triggered: bool
    fallback_triggered: bool
    mean_selected_score: float

    # Backward-compatible aliases used by the existing query route.
    @property
    def documents(self) -> list[CRAGChunk]:
        return self.selected_chunks

    @property
    def needs_fallback(self) -> bool:
        return self.fallback_triggered

    @property
    def overall_confidence(self) -> float:
        return self.mean_selected_score

    @property
    def discarded_count(self) -> int:
        return max(len(self.scored_chunks) - len(self.selected_chunks), 0)


class CRAGEvaluator:
    """Evaluate retrieval quality and apply corrective filtering.

    The evaluator reuses the existing cross-encoder model already loaded by the
    reranker for score consistency across critique and rank stages.
    """

    def __init__(
        self,
        correct_threshold: float = 0.7,
        ambiguous_threshold: float = 0.3,
        max_sub_queries: int = 4,
    ) -> None:
        if not (0.0 <= ambiguous_threshold < correct_threshold <= 1.0):
            raise ValueError("CRAG thresholds must satisfy 0.0 <= ambiguous < correct <= 1.0")
        self.correct_threshold = correct_threshold
        self.ambiguous_threshold = ambiguous_threshold
        self.max_sub_queries = max_sub_queries

    def run(self, query: str, chunks: list[Any]) -> CRAGResult:
        """Score, classify, and filter retrieved chunks."""
        if not query.strip():
            raise ValueError("query must be non-empty")

        scored_chunks = self._score_chunks(query, chunks)
        if not scored_chunks:
            return self._empty_result()

        correct_chunks, ambiguous_chunks, incorrect_chunks = self._partition(scored_chunks)

        if len(incorrect_chunks) == len(scored_chunks):
            fallback_chunks = self.web_fallback(query)
            return self._build_result(
                scored_chunks,
                fallback_chunks,
                fallback_chunks,
                refinement_triggered=False,
                fallback_triggered=True,
            )

        refinement_triggered = bool(ambiguous_chunks and correct_chunks)
        refined_chunks = self._refine_ambiguous(query, ambiguous_chunks) if ambiguous_chunks else []

        selected = correct_chunks + refined_chunks
        fallback_chunks = []
        fallback_triggered = False
        if not selected:
            fallback_chunks = self.web_fallback(query)
            selected = fallback_chunks
            fallback_triggered = True

        return self._build_result(
            scored_chunks,
            selected,
            fallback_chunks,
            refinement_triggered=refinement_triggered,
            fallback_triggered=fallback_triggered,
        )

    @staticmethod
    def _partition(
        scored_chunks: list[CRAGChunk],
    ) -> tuple[list[CRAGChunk], list[CRAGChunk], list[CRAGChunk]]:
        """Split scored chunks into (correct, ambiguous, incorrect) by classification."""
        correct: list[CRAGChunk] = []
        ambiguous: list[CRAGChunk] = []
        incorrect: list[CRAGChunk] = []
        for c in scored_chunks:
            if c.classification == CRAGClassification.CORRECT:
                correct.append(c)
            elif c.classification == CRAGClassification.AMBIGUOUS:
                ambiguous.append(c)
            else:
                incorrect.append(c)
        return correct, ambiguous, incorrect

    @staticmethod
    def _empty_result() -> CRAGResult:
        """Return the CRAGResult used when no chunks survive scoring."""
        return CRAGResult(
            selected_chunks=[],
            scored_chunks=[],
            fallback_chunks=[],
            crag_scores=[],
            crag_classification=[],
            refinement_triggered=False,
            fallback_triggered=False,
            mean_selected_score=0.0,
        )

    def _build_result(
        self,
        scored_chunks: list[CRAGChunk],
        selected: list[CRAGChunk],
        fallback_chunks: list[CRAGChunk],
        *,
        refinement_triggered: bool,
        fallback_triggered: bool,
    ) -> CRAGResult:
        """Assemble a CRAGResult, deriving scores/classification from scored_chunks."""
        return CRAGResult(
            selected_chunks=selected,
            scored_chunks=scored_chunks,
            fallback_chunks=fallback_chunks,
            crag_scores=[c.crag_score for c in scored_chunks],
            crag_classification=[c.classification.value for c in scored_chunks],
            refinement_triggered=refinement_triggered,
            fallback_triggered=fallback_triggered,
            mean_selected_score=self._mean_score(selected),
        )

    def web_fallback(self, query: str) -> list[CRAGChunk]:
        """Fallback retrieval path stub.

        Default behavior is intentionally conservative: return an empty list and
        log a warning. Integrations like Tavily/SearXNG can override this method
        or wrap the evaluator.
        """
        logger.warning(
            "CRAG web_fallback triggered for query=%r; returning empty fallback set",
            query[:120],
        )
        return []

    def _score_chunks(self, query: str, chunks: list[Any]) -> list[CRAGChunk]:
        if not chunks:
            return []
        pairs = [(query, str(getattr(c, "content", ""))) for c in chunks]
        scores = self._score_pairs(pairs)

        out: list[CRAGChunk] = []
        for chunk, score in zip(chunks, scores):
            out.append(
                CRAGChunk(
                    content=str(getattr(chunk, "content", "")),
                    source=str(getattr(chunk, "source", "unknown")),
                    score=float(getattr(chunk, "rrf_score", getattr(chunk, "score", 0.0))),
                    metadata=dict(getattr(chunk, "metadata", {}) or {}),
                    crag_score=score,
                    classification=self._classify(score),
                )
            )
        return out

    def _refine_ambiguous(self, query: str, ambiguous_chunks: list[CRAGChunk]) -> list[CRAGChunk]:
        sub_queries = self._decompose_query(query)
        self._reembed_subqueries(sub_queries)

        refined: list[CRAGChunk] = []
        for chunk in ambiguous_chunks:
            pairs = [(sq, chunk.content) for sq in sub_queries]
            sub_scores = self._score_pairs(pairs)
            best = max([chunk.crag_score] + sub_scores)
            if best > self.correct_threshold:
                refined.append(
                    CRAGChunk(
                        content=chunk.content,
                        source=chunk.source,
                        score=chunk.score,
                        metadata=chunk.metadata,
                        crag_score=best,
                        classification=CRAGClassification.CORRECT,
                    )
                )
        return refined

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        model = self._get_cross_encoder_model()
        if model is None:
            return [self._jaccard(a, b) for a, b in pairs]

        try:
            raw_scores = model.predict(pairs, show_progress_bar=False)
            if hasattr(raw_scores, "tolist"):
                raw_scores = raw_scores.tolist()
            return [self._sigmoid(float(s)) for s in raw_scores]
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("CRAG cross-encoder scoring failed (%s); using Jaccard fallback", exc)
            return [self._jaccard(a, b) for a, b in pairs]

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        ta = set(re.findall(r"\w+", a.lower()))
        tb = set(re.findall(r"\w+", b.lower()))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def _classify(self, score: float) -> CRAGClassification:
        if score > self.correct_threshold:
            return CRAGClassification.CORRECT
        if score < self.ambiguous_threshold:
            return CRAGClassification.INCORRECT
        return CRAGClassification.AMBIGUOUS

    def _decompose_query(self, query: str) -> list[str]:
        try:
            from konjoai.retrieve.router import decompose_query

            parts = [p for p in decompose_query(query, max_parts=self.max_sub_queries) if p.strip()]
            return parts or [query]
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("CRAG query decomposition failed (%s); using original query", exc)
            return [query]

    def _reembed_subqueries(self, sub_queries: list[str]) -> None:
        """Force sub-query embeddings to support refinement telemetry/debugging.

        The vectors are intentionally not returned; this step exists to satisfy
        the CRAG refinement contract and to keep behavior consistent with the
        pipeline's embedding boundary checks.
        """
        try:
            from konjoai.embed.encoder import get_encoder

            enc = get_encoder()
            for sq in sub_queries:
                enc.encode_query(sq)
        except Exception as exc:  # pragma: no cover - optional path
            logger.debug("CRAG sub-query re-embed skipped (%s)", exc)

    @staticmethod
    def _mean_score(chunks: list[CRAGChunk]) -> float:
        if not chunks:
            return 0.0
        return sum(c.crag_score for c in chunks) / len(chunks)

    @staticmethod
    def _get_cross_encoder_model() -> Any | None:
        try:
            from konjoai.retrieve.reranker import get_reranker

            reranker = get_reranker()
            model = getattr(reranker, "_model", None)
            if model is None or not hasattr(model, "predict"):
                return None
            return model
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("CRAG could not access reranker cross-encoder (%s)", exc)
            return None


# Backward-compatible names retained for existing imports.
RelevanceGrade = CRAGClassification
CRAGPipeline = CRAGEvaluator


class DocumentGrader:
    """Compatibility wrapper retained for old CRAG unit tests/imports."""

    def __init__(
        self,
        threshold: float = 0.7,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        _ = model_name  # model source is fixed to reranker singleton for consistency
        self._eval = CRAGEvaluator(correct_threshold=threshold, ambiguous_threshold=0.3)

    def grade(self, query: str, documents: list[Any]) -> list[CRAGChunk]:
        return self._eval._score_chunks(query, documents)


_crag_pipeline: CRAGEvaluator | None = None


def get_crag_pipeline() -> CRAGEvaluator:
    """Return the module-level CRAG evaluator singleton (lazy init)."""
    global _crag_pipeline
    if _crag_pipeline is None:
        from konjoai.config import get_settings

        s = get_settings()
        _crag_pipeline = CRAGEvaluator(
            correct_threshold=s.crag_correct_threshold,
            ambiguous_threshold=s.crag_ambiguous_threshold,
        )
    return _crag_pipeline


def _reset_crag() -> None:
    """Reset the singleton — test helper."""
    global _crag_pipeline
    _crag_pipeline = None
