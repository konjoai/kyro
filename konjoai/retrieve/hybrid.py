from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from konjoai.store.qdrant import SearchResult
from konjoai.retrieve.sparse import BM25Result


@dataclass
class HybridResult:
    rrf_score: float
    content: str
    source: str
    metadata: dict


def reciprocal_rank_fusion(
    dense: list[SearchResult],
    sparse: list[BM25Result],
    alpha: float = 0.7,
    k: int = 60,
) -> list[HybridResult]:
    """Merge dense and sparse result lists using RRF.

    Score formula: ``alpha * 1/(k + rank_dense) + (1 - alpha) * 1/(k + rank_sparse)``

    Parameters
    ----------
    dense:
        Ordered dense results (rank 0 = best).
    sparse:
        Ordered sparse results (rank 0 = best).
    alpha:
        Weight for the dense signal (0 … 1). Sparse weight = 1 - alpha.
    k:
        RRF smoothing constant (default 60).
    """
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for rank, result in enumerate(dense):
        key = result.content
        scores[key] = scores.get(key, 0.0) + alpha * (1.0 / (k + rank))
        payloads[key] = {"source": result.source, "metadata": result.metadata}

    for rank, result in enumerate(sparse):
        key = result.content
        scores[key] = scores.get(key, 0.0) + (1.0 - alpha) * (1.0 / (k + rank))
        if key not in payloads:
            payloads[key] = {"source": result.source, "metadata": result.metadata}

    return [
        HybridResult(
            rrf_score=score,
            content=content,
            source=payloads[content]["source"],
            metadata=payloads[content]["metadata"],
        )
        for content, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]


def hybrid_search(
    query: str,
    top_k_dense: int | None = None,
    top_k_sparse: int | None = None,
    alpha: float | None = None,
    q_vec: "np.ndarray | None" = None,
) -> list[HybridResult]:
    """Run dense + sparse searches then fuse with RRF.

    Args:
        q_vec:  Pre-computed query embedding.  Forwarded to ``dense_search``
                to avoid re-embedding when the cache layer already embedded it.

    If ``use_vectro_retriever`` is enabled in :class:`konjoai.config.Settings`,
    delegates to :class:`konjoai.retrieve.vectro_retriever.VectroRetrieverAdapter`
    which uses Vectro's SIMD-accelerated in-memory hybrid search when
    ``vectro_py`` Rust bindings are available, or a numpy fallback otherwise.
    """
    from konjoai.config import get_settings
    from konjoai.retrieve.dense import dense_search
    from konjoai.retrieve.sparse import get_sparse_index

    s = get_settings()
    kd = top_k_dense if top_k_dense is not None else s.top_k_dense
    ks = top_k_sparse if top_k_sparse is not None else s.top_k_sparse
    a = alpha if alpha is not None else s.hybrid_alpha

    if getattr(s, "use_vectro_retriever", False):
        from konjoai.retrieve.vectro_retriever import get_vectro_retriever
        return get_vectro_retriever().search(query, top_k=max(kd, ks))

    dense_results = dense_search(query, top_k=kd, q_vec=q_vec)

    bm25 = get_sparse_index()
    sparse_results = bm25.search(query, top_k=ks) if bm25.built else []

    return reciprocal_rank_fusion(dense_results, sparse_results, alpha=a)
