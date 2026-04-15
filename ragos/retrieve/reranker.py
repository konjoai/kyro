from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_reranker: "CrossEncoderReranker | None" = None


@dataclass
class RerankResult:
    score: float
    content: str
    source: str
    metadata: dict


class CrossEncoderReranker:
    """Re-score (query, passage) pairs with a cross-encoder model."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for reranking: pip install sentence-transformers"
            ) from e

        self._model = CrossEncoder(model_name)
        logger.info("CrossEncoderReranker: loaded model=%s", model_name)

    def rerank(self, query: str, passages: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        """Return ``(original_index, score)`` sorted best-first, truncated to *top_k*."""
        pairs = [(query, p) for p in passages]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(enumerate(scores.tolist()), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


def get_reranker() -> CrossEncoderReranker:
    """Return the module-level singleton reranker (lazy init)."""
    global _reranker
    if _reranker is None:
        from ragos.config import get_settings

        s = get_settings()
        _reranker = CrossEncoderReranker(model_name=s.reranker_model)
    return _reranker


def rerank(query: str, candidates: list, top_k: int | None = None) -> list[RerankResult]:
    """Rerank *candidates* (HybridResult or SearchResult list) for *query*.

    Parameters
    ----------
    query:
        The user question.
    candidates:
        Any list of objects with ``.content``, ``.source``, and ``.metadata``.
    top_k:
        Number of results to return (default: ``settings.top_k_rerank``).
    """
    from ragos.config import get_settings

    k = top_k if top_k is not None else get_settings().top_k_rerank
    reranker = get_reranker()
    passages = [c.content for c in candidates]
    ranked_pairs = reranker.rerank(query, passages, top_k=k)
    return [
        RerankResult(
            score=score,
            content=candidates[idx].content,
            source=candidates[idx].source,
            metadata=candidates[idx].metadata,
        )
        for idx, score in ranked_pairs
    ]
