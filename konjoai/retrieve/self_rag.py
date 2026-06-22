"""Self-RAG orchestrator for reflective generation.

Sprint 12 contract (v0.7.0):
1. Generate a partial answer.
2. Critique with reflection scores: ISREL / ISSUP / ISUSE.
3. If ISSUP < 0.5, perform retrieval with a refined query.
4. If ISREL > 0.8 and ISSUP is sufficient, continue without retrieval.
5. Stop after a bounded number of iterations.

This module keeps a pragmatic runtime design that does not require
fine-tuning the base model. It combines:
- existing cross-encoder support scoring,
- lightweight usefulness overlap heuristics,
- optional LLM rubric scoring callback for ISREL/ISSUP/ISUSE.
"""

from __future__ import annotations

import inspect
import logging
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

from konjoai.retrieve.router import QueryIntent, classify_intent

logger = logging.getLogger(__name__)


class RetrieveDecision(StrEnum):
    """[Retrieve] — should this query trigger retrieval?"""

    YES = "yes"
    NO = "no"


class RelevanceToken(StrEnum):
    """[IsRel] — document relevance token."""

    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"


class SupportToken(StrEnum):
    """[IsSup] — support token."""

    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"


class UsefulnessToken(IntEnum):
    """[IsUse] token with the 1-5 Self-RAG scale."""

    VERY_HIGH = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    VERY_LOW = 1


@dataclass
class DocumentCritique:
    """Per-document critique output."""

    content: str
    source: str
    relevance: RelevanceToken
    support: SupportToken
    support_score: float


@dataclass
class SelfRAGTokens:
    """Continuous reflection signals in [0, 1]."""

    isrel: float
    issup: float
    isuse: float


@dataclass
class SelfRAGResult:
    """Self-RAG orchestration result."""

    answer: str
    retrieve_decision: RetrieveDecision
    document_critiques: list[DocumentCritique]
    usefulness: UsefulnessToken
    usefulness_score: float
    support_score: float
    iterations: int
    iteration_scores: list[dict[str, float]] = field(default_factory=list)
    total_tokens: int = 0
    metadata: dict = field(default_factory=dict)


class SupportScorer:
    """Document-answer support scorer with graceful fallback."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model: object | None = None
        self._use_fallback = False

    def _load_model(self) -> None:
        """Lazily load the cross-encoder, switching to token-overlap fallback on failure."""
        if self._model is not None or self._use_fallback:
            return
        try:
            from sentence_transformers import CrossEncoder as CE

            self._model = CE(self._model_name)
            logger.debug("SupportScorer loaded cross-encoder %s", self._model_name)
        except Exception as exc:  # pragma: no cover - depends on runtime env
            logger.warning(
                "SupportScorer falling back to token-overlap mode (%s)",
                exc,
            )
            self._use_fallback = True

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        """Return word-level Jaccard overlap of two strings in [0, 1]."""
        ta = set(re.findall(r"\w+", a.lower()))
        tb = set(re.findall(r"\w+", b.lower()))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable logistic sigmoid."""
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def score(self, document: str, answer: str) -> float:
        """Return raw support score.

        Cross-encoder mode returns logits.
        Fallback mode returns Jaccard overlap in [0, 1].
        """
        self._load_model()
        if self._use_fallback:
            return self._jaccard(document, answer)
        scores = self._model.predict([(document, answer)], show_progress_bar=False)
        return float(scores[0])

    def normalize(self, score: float) -> float:
        """Normalize score to [0, 1] regardless of scoring backend."""
        if self._use_fallback:
            return max(0.0, min(score, 1.0))
        return self._sigmoid(score)

    def support_token(self, score: float, high: float = 2.0, low: float = -0.5) -> SupportToken:
        """Map raw score to support token."""
        if self._use_fallback:
            high, low = 0.20, 0.05
        if score >= high:
            return SupportToken.FULLY_SUPPORTED
        if score >= low:
            return SupportToken.PARTIALLY_SUPPORTED
        return SupportToken.NOT_SUPPORTED


class UsefulnessScorer:
    """Answer usefulness scorer with overlap proxy."""

    _STOPS = {
        "what",
        "who",
        "where",
        "when",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "for",
        "to",
        "and",
        "or",
        "but",
        "i",
        "you",
        "it",
        "this",
        "that",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "can",
        "could",
        "would",
        "should",
        "will",
    }

    def score(self, question: str, answer: str) -> tuple[UsefulnessToken, float]:
        """Return (token, normalized usefulness score in [0, 1])."""
        if not question.strip() or not answer.strip():
            return UsefulnessToken.VERY_LOW, 0.0

        q_tokens = set(re.findall(r"\w+", question.lower()))
        a_tokens = set(re.findall(r"\w+", answer.lower()))

        content_tokens = q_tokens - self._STOPS
        if not content_tokens:
            content_tokens = q_tokens

        overlap = len(content_tokens & a_tokens) / len(content_tokens)
        raw = min(int(overlap * 4) + 1, 5)
        return UsefulnessToken(raw), overlap


def decide_retrieve(question: str) -> RetrieveDecision:
    """Use router intent as [Retrieve] proxy decision."""
    try:
        if classify_intent(question) == QueryIntent.CHAT:
            return RetrieveDecision.NO
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("decide_retrieve failed (%s); defaulting to YES", exc)
    return RetrieveDecision.YES


class SelfRAGCritic:
    """Compute reflection scores for one partial answer."""

    def __init__(
        self,
        support_scorer: SupportScorer | None = None,
        usefulness_scorer: UsefulnessScorer | None = None,
        llm_score_fn: Callable[[str], float] | None = None,
    ) -> None:
        self._support = support_scorer or SupportScorer()
        self._usefulness = usefulness_scorer or UsefulnessScorer()
        self._llm_score_fn = llm_score_fn

    @staticmethod
    def _clamp01(value: float) -> float:
        """Clamp a value to the [0, 1] range."""
        return max(0.0, min(value, 1.0))

    @staticmethod
    def _docs_preview(documents: Sequence[Any], max_docs: int = 3, max_chars: int = 240) -> str:
        """Build a numbered, truncated text preview of the leading documents."""
        items: list[str] = []
        for idx, doc in enumerate(documents[:max_docs], start=1):
            text = str(getattr(doc, "content", ""))[:max_chars]
            items.append(f"[{idx}] {text}")
        return "\n".join(items)

    def _score_with_llm(self, prompt: str) -> float | None:
        """Score the prompt via the optional LLM callback, returning None if unset or failing."""
        if self._llm_score_fn is None:
            return None
        try:
            return self._clamp01(float(self._llm_score_fn(prompt)))
        except Exception as exc:  # pragma: no cover - defensive path
            logger.debug("SelfRAGCritic llm_score_fn failed (%s)", exc)
            return None

    def evaluate(
        self,
        question: str,
        partial_answer: str,
        documents: Sequence[Any],
    ) -> tuple[SelfRAGTokens, list[DocumentCritique]]:
        """Return reflection scores plus per-document critique metadata."""
        critiques: list[DocumentCritique] = []
        support_norm_scores: list[float] = []

        for doc in documents:
            content = str(getattr(doc, "content", ""))
            source = str(getattr(doc, "source", "unknown"))
            raw_support = self._support.score(content, partial_answer)
            norm_support = self._support.normalize(raw_support)
            support_norm_scores.append(norm_support)

            relevance = RelevanceToken.RELEVANT if norm_support >= 0.5 else RelevanceToken.IRRELEVANT
            support = self._support.support_token(raw_support)
            critiques.append(
                DocumentCritique(
                    content=content,
                    source=source,
                    relevance=relevance,
                    support=support,
                    support_score=norm_support,
                )
            )

        heuristic_isrel = max(support_norm_scores) if support_norm_scores else 0.0
        heuristic_issup = sum(support_norm_scores) / len(support_norm_scores) if support_norm_scores else 0.0

        _, usefulness_overlap = self._usefulness.score(question, partial_answer)
        heuristic_isuse = usefulness_overlap

        docs_preview = self._docs_preview(documents)
        llm_isrel = self._score_with_llm(
            "Rate retrieval relevance in [0,1]. Return only a float.\n"
            f"Question: {question}\n"
            f"Partial answer: {partial_answer}\n"
            f"Retrieved context:\n{docs_preview}\n"
        )
        llm_issup = self._score_with_llm(
            "Rate factual support of the partial answer by context in [0,1]. Return only a float.\n"
            f"Question: {question}\n"
            f"Partial answer: {partial_answer}\n"
            f"Retrieved context:\n{docs_preview}\n"
        )
        llm_isuse = self._score_with_llm(
            "Rate usefulness of the partial answer for the question in [0,1]. Return only a float.\n"
            f"Question: {question}\n"
            f"Partial answer: {partial_answer}\n"
        )

        tokens = SelfRAGTokens(
            isrel=heuristic_isrel if llm_isrel is None else llm_isrel,
            issup=heuristic_issup if llm_issup is None else llm_issup,
            isuse=heuristic_isuse if llm_isuse is None else llm_isuse,
        )
        return tokens, critiques


class SelfRAGOrchestrator:
    """Iterative reflective generation with optional retrieval refinement."""

    def __init__(
        self,
        critic: SelfRAGCritic | None = None,
        support_scorer: SupportScorer | None = None,
        usefulness_scorer: UsefulnessScorer | None = None,
        llm_score_fn: Callable[[str], float] | None = None,
        max_iterations: int = 3,
        issup_threshold: float = 0.5,
        isrel_no_retrieve_threshold: float = 0.8,
        max_partial_tokens: int = 100,
    ) -> None:
        self._critic = critic or SelfRAGCritic(
            support_scorer=support_scorer,
            usefulness_scorer=usefulness_scorer,
            llm_score_fn=llm_score_fn,
        )
        # Compatibility aliases kept for older tests/importers.
        self._support = self._critic._support
        self._usefulness = self._critic._usefulness
        self._max_iterations = max(1, max_iterations)
        self._issup_threshold = issup_threshold
        self._isrel_no_retrieve_threshold = isrel_no_retrieve_threshold
        self._max_partial_tokens = max_partial_tokens

    @staticmethod
    def _token_count(text: str) -> int:
        """Count whitespace-delimited tokens in the text."""
        return len(re.findall(r"\S+", text))

    def _partial_answer(self, full_answer: str) -> str:
        """Truncate an answer to the first ``max_partial_tokens`` words."""
        tokens = full_answer.split()
        return " ".join(tokens[: self._max_partial_tokens])

    @staticmethod
    def _map_usefulness(score: float) -> UsefulnessToken:
        """Bucket a [0, 1] usefulness score into a discrete usefulness token."""
        if score >= 0.875:
            return UsefulnessToken.VERY_HIGH
        if score >= 0.625:
            return UsefulnessToken.HIGH
        if score >= 0.375:
            return UsefulnessToken.MEDIUM
        if score >= 0.125:
            return UsefulnessToken.LOW
        return UsefulnessToken.VERY_LOW

    @staticmethod
    def _refined_query(question: str, partial_answer: str) -> str:
        """Augment the question with the longest sentence of the draft answer as a retrieval clue."""
        focus = partial_answer.strip()
        if "." in focus:
            sentences = [s.strip() for s in focus.split(".") if s.strip()]
            if sentences:
                focus = max(sentences, key=len)
        return f"{question}\n\nRefine retrieval using this draft answer clue: {focus[:300]}"

    @staticmethod
    def _call_generate(generate_fn: Callable[..., str], documents: Sequence[Any]) -> str:
        """Invoke ``generate_fn`` with documents if it accepts an argument, else with none."""
        try:
            sig = inspect.signature(generate_fn)
            if len(sig.parameters) >= 1:
                return str(generate_fn(documents))
        except (TypeError, ValueError):  # pragma: no cover - builtin/partial callables
            pass
        return str(generate_fn())

    def run(
        self,
        question: str,
        documents: Sequence[Any],
        generate_fn: Callable[..., str],
        retrieve_fn: Callable[[str], Sequence[Any]] | None = None,
        max_iterations: int | None = None,
    ) -> SelfRAGResult:
        """Run iterative Self-RAG critique/generation loop.

        Args:
            question: User question.
            documents: Initial retrieved docs.
            generate_fn: Callable returning answer text. Accepts optional docs arg.
            retrieve_fn: Optional callable for refined retrieval.
            max_iterations: Optional override.
        """
        if not question.strip():
            raise ValueError("question must be non-empty")

        retrieve_decision = decide_retrieve(question)
        docs = list(documents)
        n_iter = max_iterations if max_iterations is not None else self._max_iterations
        n_iter = max(1, n_iter)

        best_answer = ""
        best_tokens = SelfRAGTokens(isrel=0.0, issup=0.0, isuse=0.0)
        best_critiques: list[DocumentCritique] = []
        iteration_scores: list[dict[str, float]] = []
        total_tokens = 0

        for idx in range(n_iter):
            candidate = self._call_generate(generate_fn, docs)
            total_tokens += self._token_count(candidate)
            partial = self._partial_answer(candidate)

            tokens, critiques = self._critic.evaluate(
                question=question,
                partial_answer=partial,
                documents=docs,
            )

            iteration_scores.append(
                {
                    "iteration": float(idx + 1),
                    "isrel": float(tokens.isrel),
                    "issup": float(tokens.issup),
                    "isuse": float(tokens.isuse),
                }
            )

            is_better = tokens.issup > best_tokens.issup or (
                abs(tokens.issup - best_tokens.issup) < 1e-9 and tokens.isuse > best_tokens.isuse
            )
            if is_better:
                best_answer = candidate
                best_tokens = tokens
                best_critiques = critiques

            logger.debug(
                "Self-RAG iter=%d isrel=%.3f issup=%.3f isuse=%.3f",
                idx + 1,
                tokens.isrel,
                tokens.issup,
                tokens.isuse,
            )

            if tokens.issup >= self._issup_threshold:
                if retrieve_decision == RetrieveDecision.NO or tokens.isrel >= self._isrel_no_retrieve_threshold:
                    break
                break

            if idx >= n_iter - 1:
                continue

            if retrieve_decision == RetrieveDecision.YES and retrieve_fn is not None:
                refined_query = self._refined_query(question, partial)
                try:
                    refreshed_docs = retrieve_fn(refined_query)
                    if refreshed_docs:
                        docs = list(refreshed_docs)
                except Exception as exc:  # pragma: no cover - defensive fallback
                    logger.warning("Self-RAG refined retrieval failed (%s)", exc)

        usefulness = self._map_usefulness(best_tokens.isuse)
        return SelfRAGResult(
            answer=best_answer,
            retrieve_decision=retrieve_decision,
            document_critiques=best_critiques,
            usefulness=usefulness,
            usefulness_score=best_tokens.isuse,
            support_score=best_tokens.issup,
            iterations=len(iteration_scores),
            iteration_scores=iteration_scores,
            total_tokens=total_tokens,
            metadata={
                "issup_threshold": self._issup_threshold,
                "isrel_no_retrieve_threshold": self._isrel_no_retrieve_threshold,
            },
        )


# Backward-compatible alias for existing imports.
SelfRAGPipeline = SelfRAGOrchestrator


_self_rag_pipeline: SelfRAGOrchestrator | None = None


def get_self_rag_pipeline() -> SelfRAGOrchestrator:
    """Return module-level Self-RAG orchestrator singleton."""
    global _self_rag_pipeline
    if _self_rag_pipeline is None:
        from konjoai.config import get_settings

        s = get_settings()
        _self_rag_pipeline = SelfRAGOrchestrator(
            max_iterations=s.self_rag_max_iterations,
        )
    return _self_rag_pipeline


def _reset_self_rag() -> None:
    """Reset singleton (test helper)."""
    global _self_rag_pipeline
    _self_rag_pipeline = None
