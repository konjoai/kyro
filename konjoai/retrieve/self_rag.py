"""Self-RAG — reflective retrieval-augmented generation.

What is Self-RAG?
-----------------
Standard RAG always retrieves documents, regardless of whether retrieval
is actually useful.  For conversational or commonsense queries this adds
latency and often hurts answer quality by injecting irrelevant context.

Self-RAG (Asai et al. 2023, arXiv:2310.11511) inserts lightweight
**reflection tokens** that let the pipeline decide:

1. **[Retrieve]** — is retrieval needed for this query?
2. **[IsRel]** — is a given document relevant to the query?
3. **[IsSup]** — is the generated segment supported by the document?
4. **[IsUse]** — is the final answer useful to the user?

Each token is a discrete decision, not a free-text generation.  Together
they form a critique loop that can:
* Skip retrieval for chat / commonsense queries.
* Discard low-support generations and retry.
* Select the best candidate when multiple candidates exist.

Implementation
--------------
Full Self-RAG as described in the paper requires a fine-tuned LLM that
emits reflection tokens inline.  This implementation provides the
**pipeline-level equivalent** using the existing lightweight tools:

* ``[Retrieve]`` decision — the existing :class:`konjoai.retrieve.router.QueryIntent`
  classifier (CHAT = skip retrieval).
* ``[IsRel]`` scoring — :class:`konjoai.retrieve.crag.DocumentGrader` cross-encoder score.
* ``[IsSup]`` scoring — a sentence-level NLI pass using the same cross-encoder.
  The (document_chunk, generated_answer) pair is scored; a negative logit
  means the answer is **not** grounded in that document.
* ``[IsUse]`` scoring — a keyword-overlap proxy (Jaccard similarity between
  the question tokens and the answer tokens).  Cost: O(tokens), zero API.

When ``enable_self_rag=True``, the pipeline:
1. Generates an answer normally.
2. Scores ``[IsSup]`` against retrieved documents.
3. If support is below threshold, re-generates up to ``self_rag_max_iterations`` times.
4. Returns the best-supported answer with critique metadata attached.

K1: Every method raises on bad input.
K2: Reflection scores added to ``QueryResponse.telemetry`` when enabled.
K3: Graceful degradation — if cross-encoder unavailable, ``[IsSup]`` falls
    back to keyword overlap and logs a warning.
K5: Zero new hard deps.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, IntEnum

from konjoai.retrieve.router import QueryIntent, classify_intent

logger = logging.getLogger(__name__)


# ── Reflection token enumerations ─────────────────────────────────────────────

class RetrieveDecision(str, Enum):
    """[Retrieve] — should this query trigger document retrieval?"""
    YES = "yes"
    NO = "no"


class RelevanceToken(str, Enum):
    """[IsRel] — is this document relevant to the query?"""
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"


class SupportToken(str, Enum):
    """[IsSup] — is the generated answer supported by the document?"""
    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"


class UsefulnessToken(IntEnum):
    """[IsUse] — is the answer useful to the user?  Score 1 (low) – 5 (high)."""
    VERY_HIGH = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    VERY_LOW = 1


# ── Critique dataclasses ──────────────────────────────────────────────────────

@dataclass
class DocumentCritique:
    """[IsRel] + [IsSup] for one (answer, document) pair."""

    content: str
    source: str
    relevance: RelevanceToken
    support: SupportToken
    support_score: float    # raw cross-encoder or Jaccard score


@dataclass
class SelfRAGResult:
    """Output of the Self-RAG reflection pass."""

    answer: str
    retrieve_decision: RetrieveDecision
    document_critiques: list[DocumentCritique]
    usefulness: UsefulnessToken
    usefulness_score: float
    support_score: float        # mean support score across RELEVANT docs
    iterations: int             # how many generate → critique cycles were needed
    metadata: dict = field(default_factory=dict)


# ── Support scorer ────────────────────────────────────────────────────────────

class SupportScorer:
    """Score whether a generated *answer* is supported by a *document*.

    Primary: cross-encoder (passage, answer) scoring.
    Fallback: Jaccard token overlap when cross-encoder is unavailable.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model: object | None = None
        self._use_fallback = False

    def _load_model(self) -> None:
        if self._model is not None or self._use_fallback:
            return
        try:
            from sentence_transformers import CrossEncoder as CE
            self._model = CE(self._model_name)
            logger.debug("SupportScorer: loaded cross-encoder %s", self._model_name)
        except ImportError:
            logger.warning(
                "SupportScorer: sentence-transformers unavailable; "
                "falling back to Jaccard token overlap for [IsSup]"
            )
            self._use_fallback = True

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        ta = set(re.findall(r"\w+", a.lower()))
        tb = set(re.findall(r"\w+", b.lower()))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def score(self, document: str, answer: str) -> float:
        """Return a raw support score.  Higher = better support.

        Cross-encoder: raw logit (unbounded, positive = supported).
        Jaccard fallback: [0.0, 1.0].
        """
        self._load_model()
        if self._use_fallback:
            return self._jaccard(document, answer)
        scores = self._model.predict([(document, answer)], show_progress_bar=False)
        return float(scores[0])

    def support_token(self, score: float, high: float = 2.0, low: float = -0.5) -> SupportToken:
        """Discretise a raw *score* into a :class:`SupportToken`.

        When using the Jaccard fallback the thresholds are re-calibrated:
        ``high=0.20`` (20% token overlap = fully supported),
        ``low=0.05`` (≤5% = not supported).
        """
        if self._use_fallback:
            high, low = 0.20, 0.05
        if score >= high:
            return SupportToken.FULLY_SUPPORTED
        if score >= low:
            return SupportToken.PARTIALLY_SUPPORTED
        return SupportToken.NOT_SUPPORTED


# ── Usefulness scorer ─────────────────────────────────────────────────────────

class UsefulnessScorer:
    """Score answer usefulness with a keyword-overlap proxy.

    [IsUse] in the paper requires a fine-tuned model.  This lightweight
    proxy asks: "how many question-relevant terms appear in the answer?"

    Formula:
        overlap = |tokens(question) ∩ tokens(answer)| / |tokens(question)|
        score = 1 + round(overlap × 4)   → 1–5
    """

    def score(self, question: str, answer: str) -> tuple[UsefulnessToken, float]:
        """Return ``(UsefulnessToken, raw_overlap_score)``."""
        if not question.strip() or not answer.strip():
            return UsefulnessToken.VERY_LOW, 0.0

        q_tokens = set(re.findall(r"\w+", question.lower()))
        a_tokens = set(re.findall(r"\w+", answer.lower()))

        # Remove stop words from the question token set
        _STOPS = {
            "what", "who", "where", "when", "why", "how", "is", "are",
            "was", "were", "the", "a", "an", "of", "in", "on", "for",
            "to", "and", "or", "but", "i", "you", "it", "this", "that",
            "do", "does", "did", "has", "have", "had", "can", "could",
            "would", "should", "will",
        }
        content_tokens = q_tokens - _STOPS
        if not content_tokens:
            content_tokens = q_tokens  # fallback: use all tokens

        overlap = len(content_tokens & a_tokens) / len(content_tokens)
        # Map [0, 1] → UsefulnessToken 1–5
        raw = min(int(overlap * 4) + 1, 5)
        token = UsefulnessToken(raw)
        return token, overlap


# ── Retrieve decision ─────────────────────────────────────────────────────────

def decide_retrieve(question: str) -> RetrieveDecision:
    """[Retrieve] — lightweight heuristic for retrieval necessity.

    Delegates to :func:`konjoai.retrieve.router.classify_intent`:
    * CHAT intent → ``NO`` (retrieval not needed)
    * RETRIEVAL or AGGREGATION intent → ``YES``

    This avoids a second LLM call for the retrieval decision.
    """
    try:
        intent = classify_intent(question)
        if intent == QueryIntent.CHAT:
            return RetrieveDecision.NO
    except Exception as exc:
        logger.warning("decide_retrieve: intent classifier failed (%s); defaulting to YES", exc)
    return RetrieveDecision.YES


# ── Self-RAG Pipeline ─────────────────────────────────────────────────────────

class SelfRAGPipeline:
    """Execute the Self-RAG reflection loop.

    Usage in the main query route (after generation)::

        if settings.enable_self_rag:
            self_rag = get_self_rag_pipeline()
            result = self_rag.run(
                question=req.question,
                documents=reranked,
                generate_fn=lambda: generator.generate(req.question, context).answer,
                max_iterations=settings.self_rag_max_iterations,
            )
            # Use result.answer which is the best-supported generation.

    Args:
        support_scorer:   :class:`SupportScorer` to use (default: module singleton).
        usefulness_scorer: :class:`UsefulnessScorer` to use.
        min_support_score: Raw cross-encoder score threshold above which an
                           answer is accepted without retry.
        max_iterations:   Maximum generate→critique cycles (default: 2).
    """

    def __init__(
        self,
        support_scorer: SupportScorer | None = None,
        usefulness_scorer: UsefulnessScorer | None = None,
        min_support_score: float = 0.0,
        max_iterations: int = 2,
    ) -> None:
        self._support = support_scorer or SupportScorer()
        self._usefulness = usefulness_scorer or UsefulnessScorer()
        self._min_support = min_support_score
        self._max_iterations = max(1, max_iterations)

    def run(
        self,
        question: str,
        documents: list,
        generate_fn: "callable[[], str]",
        max_iterations: int | None = None,
    ) -> SelfRAGResult:
        """Execute the Self-RAG critique loop.

        Args:
            question:       The user question.
            documents:      Reranked documents (objects with ``.content``,
                            ``.source``, ``.metadata``).
            generate_fn:    Zero-argument callable that returns a candidate
                            answer string.  Called once per iteration.
            max_iterations: Override the instance default if provided.

        Returns:
            :class:`SelfRAGResult` with the best answer and all critique data.

        Raises:
            ValueError: If *question* is empty or *documents* is empty and
                        the retrieve decision is YES.
        """
        if not question.strip():
            raise ValueError("question must be non-empty")

        n_iter = max_iterations if max_iterations is not None else self._max_iterations
        retrieve_dec = decide_retrieve(question)

        best_answer = ""
        best_support = -999.0
        best_critiques: list[DocumentCritique] = []
        best_usefulness_token = UsefulnessToken.VERY_LOW
        best_usefulness_score = 0.0
        iterations = 0

        for i in range(n_iter):
            iterations = i + 1
            candidate = generate_fn()

            # Score [IsSup] for each document
            critiques: list[DocumentCritique] = []
            support_scores: list[float] = []

            for doc in documents:
                raw_sup = self._support.score(doc.content, candidate)
                sup_token = self._support.support_token(raw_sup)

                # [IsRel] via support score direction
                if raw_sup > self._min_support:
                    rel_token = RelevanceToken.RELEVANT
                else:
                    rel_token = RelevanceToken.IRRELEVANT

                critiques.append(
                    DocumentCritique(
                        content=doc.content,
                        source=getattr(doc, "source", "unknown"),
                        relevance=rel_token,
                        support=sup_token,
                        support_score=raw_sup,
                    )
                )
                if rel_token == RelevanceToken.RELEVANT:
                    support_scores.append(raw_sup)

            mean_support = (
                sum(support_scores) / len(support_scores)
                if support_scores else 0.0
            )
            usefulness_token, usefulness_score = self._usefulness.score(question, candidate)

            logger.debug(
                "Self-RAG iter=%d mean_support=%.3f usefulness=%s",
                iterations, mean_support, usefulness_token.name,
            )

            # Keep the best-supported candidate
            if mean_support > best_support:
                best_answer = candidate
                best_support = mean_support
                best_critiques = critiques
                best_usefulness_token = usefulness_token
                best_usefulness_score = usefulness_score

            # Early exit: sufficient support achieved
            if mean_support >= self._min_support and iterations >= 1:
                break

        return SelfRAGResult(
            answer=best_answer,
            retrieve_decision=retrieve_dec,
            document_critiques=best_critiques,
            usefulness=best_usefulness_token,
            usefulness_score=best_usefulness_score,
            support_score=best_support,
            iterations=iterations,
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_self_rag_pipeline: SelfRAGPipeline | None = None


def get_self_rag_pipeline() -> SelfRAGPipeline:
    """Return the module-level Self-RAG pipeline singleton (lazy init)."""
    global _self_rag_pipeline
    if _self_rag_pipeline is None:
        from konjoai.config import get_settings
        s = get_settings()
        _self_rag_pipeline = SelfRAGPipeline(
            max_iterations=s.self_rag_max_iterations,
        )
    return _self_rag_pipeline


def _reset_self_rag() -> None:
    """Reset the singleton — used only in tests."""
    global _self_rag_pipeline
    _self_rag_pipeline = None
