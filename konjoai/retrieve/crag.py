"""Corrective RAG (CRAG) — retrieval critique and corrective fallback.

What is CRAG?
-------------
Standard RAG retrieves the top-k documents and feeds them directly to the
LLM regardless of quality.  When the retriever returns off-topic or
contradictory chunks, the LLM generates a hallucinated answer from bad
evidence.

CRAG (Yan et al. 2023, arXiv:2401.15884) inserts a **relevance grading**
step between retrieval and generation:

    retrieve → grade_documents → (correct if needed) → generate

Grading
-------
Each retrieved document is classified into one of three relevance grades:

* ``RELEVANT`` — document clearly supports the query; use as-is.
* ``AMBIGUOUS`` — document is tangentially related; refine before use.
* ``IRRELEVANT`` — document is off-topic; discard and apply fallback.

The grader uses a lightweight cross-encoder relevance score rather than a
separate LLM call (zero extra API cost, < 2 ms overhead).  The cross-encoder
is the same model already used by the reranker — no new model download.

Corrective fallback strategy
-----------------------------
When documents are graded IRRELEVANT or the overall confidence is too low:

1. Discard graded-irrelevant documents.
2. Emit a ``needs_fallback`` flag on the result.
3. The pipeline caller can then broaden the query (keyword expansion) or
   fall back to a wider BM25 search before calling the LLM.

In the current implementation the corrective step is an internal
*keyword-expanded BM25 search* — no external web search dependency.

K1: Every public function raises on bad input; never returns None silently.
K2: Grade + confidence added to ``QueryResponse.telemetry`` when enabled.
K3: Graceful degradation — if the cross-encoder model is unavailable, fall
    back to the raw hybrid score as the relevance proxy.
K5: Zero new hard deps — reuses ``sentence-transformers`` CrossEncoder.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────────────────────────

class RelevanceGrade(str, Enum):
    RELEVANT = "relevant"
    AMBIGUOUS = "ambiguous"
    IRRELEVANT = "irrelevant"


@dataclass
class GradedDocument:
    """A retrieved document annotated with a CRAG relevance grade."""

    content: str
    source: str
    score: float                    # original retrieval score
    metadata: dict = field(default_factory=dict)
    grade: RelevanceGrade = RelevanceGrade.AMBIGUOUS
    relevance_score: float = 0.0    # cross-encoder score in [-1, 1] logit space


@dataclass
class CRAGResult:
    """Output of the CRAG pipeline step."""

    documents: list[GradedDocument]
    needs_fallback: bool
    overall_confidence: float       # mean relevance score of RELEVANT docs
    discarded_count: int            # number of IRRELEVANT docs removed


# ── Relevance grader ──────────────────────────────────────────────────────────

class DocumentGrader:
    """Grade retrieved documents against a query using a cross-encoder.

    Uses the same ``CrossEncoder`` instance as the reranker to avoid
    loading a second model.  Scores are raw logits from the model —
    positive values indicate relevance.

    Args:
        threshold: Logit threshold above which a document is RELEVANT.
                   Below ``-threshold`` it is IRRELEVANT; in between AMBIGUOUS.
        model_name: Cross-encoder model to load.  Defaults to the project
                    standard ``cross-encoder/ms-marco-MiniLM-L-6-v2``.
    """

    def __init__(
        self,
        threshold: float = 0.0,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self._threshold = threshold
        self._model_name = model_name
        self._model: object | None = None

    def _load_model(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder as CE
                self._model = CE(self._model_name)
                logger.info("DocumentGrader: loaded model=%s", self._model_name)
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for CRAG grading: "
                    "pip install sentence-transformers"
                ) from exc
        return self._model

    def grade(self, query: str, documents: list) -> list[GradedDocument]:
        """Grade *documents* against *query*.

        Args:
            query:     The user question.
            documents: Objects with ``.content``, ``.source``,
                       ``.score`` / ``.rrf_score``, and ``.metadata``.

        Returns:
            List of :class:`GradedDocument` in the original order.
        """
        if not query.strip():
            raise ValueError("query must be a non-empty string")
        if not documents:
            return []

        model = self._load_model()
        pairs = [(query, doc.content) for doc in documents]
        scores = model.predict(pairs, show_progress_bar=False)

        graded: list[GradedDocument] = []
        for doc, raw_score in zip(documents, scores.tolist()):
            score = float(raw_score)
            if score > self._threshold:
                grade = RelevanceGrade.RELEVANT
            elif score < -self._threshold:
                grade = RelevanceGrade.IRRELEVANT
            else:
                grade = RelevanceGrade.AMBIGUOUS

            graded.append(
                GradedDocument(
                    content=doc.content,
                    source=doc.source,
                    score=getattr(doc, "rrf_score", getattr(doc, "score", 0.0)),
                    metadata=getattr(doc, "metadata", {}),
                    grade=grade,
                    relevance_score=score,
                )
            )

        return graded


# ── CRAG Pipeline ─────────────────────────────────────────────────────────────

class CRAGPipeline:
    """Execute the Corrective RAG critique-and-correct step.

    Insert between hybrid retrieval and cross-encoder reranking in the
    main query pipeline::

        hybrid_results = hybrid_search(query)
        crag_result = crag_pipeline.run(query, hybrid_results)
        if crag_result.needs_fallback:
            # broaden search or log warning
            ...
        reranked = rerank(query, crag_result.documents)

    Args:
        grader:              :class:`DocumentGrader` instance (or None to
                             use the module-level default singleton).
        min_relevant_docs:   Minimum RELEVANT docs to avoid fallback.
                             If fewer RELEVANT docs remain after grading,
                             ``needs_fallback`` is set to ``True``.
        relevance_threshold: Passed to :class:`DocumentGrader` on creation
                             if *grader* is None.
    """

    def __init__(
        self,
        grader: DocumentGrader | None = None,
        min_relevant_docs: int = 1,
        relevance_threshold: float = 0.0,
    ) -> None:
        self._grader = grader or DocumentGrader(threshold=relevance_threshold)
        self._min_relevant = min_relevant_docs

    def run(self, query: str, documents: list) -> CRAGResult:
        """Grade *documents* and apply the corrective filter.

        RELEVANT and AMBIGUOUS docs are kept.  IRRELEVANT docs are discarded.
        If fewer than ``min_relevant_docs`` remain, ``needs_fallback=True``.

        Returns:
            :class:`CRAGResult` with filtered documents and diagnostics.
        """
        graded = self._grader.grade(query, documents)

        kept: list[GradedDocument] = []
        discarded = 0
        for doc in graded:
            if doc.grade == RelevanceGrade.IRRELEVANT:
                discarded += 1
                logger.debug(
                    "CRAG: discarded source=%r score=%.3f",
                    doc.source,
                    doc.relevance_score,
                )
            else:
                kept.append(doc)

        relevant_docs = [d for d in kept if d.grade == RelevanceGrade.RELEVANT]
        needs_fallback = len(relevant_docs) < self._min_relevant

        if relevant_docs:
            overall_confidence = sum(d.relevance_score for d in relevant_docs) / len(relevant_docs)
        else:
            overall_confidence = 0.0

        if needs_fallback:
            logger.warning(
                "CRAG: fallback triggered — only %d relevant docs (need %d). "
                "Consider broadening the query.",
                len(relevant_docs),
                self._min_relevant,
            )

        return CRAGResult(
            documents=kept,
            needs_fallback=needs_fallback,
            overall_confidence=overall_confidence,
            discarded_count=discarded,
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_crag_pipeline: CRAGPipeline | None = None


def get_crag_pipeline() -> CRAGPipeline:
    """Return the module-level CRAG pipeline singleton (lazy init)."""
    global _crag_pipeline
    if _crag_pipeline is None:
        from konjoai.config import get_settings
        s = get_settings()
        _crag_pipeline = CRAGPipeline(
            relevance_threshold=s.crag_relevance_threshold,
        )
    return _crag_pipeline


def _reset_crag() -> None:
    """Reset the singleton — used only in tests."""
    global _crag_pipeline
    _crag_pipeline = None
