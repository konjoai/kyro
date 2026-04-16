from __future__ import annotations

import numpy as np

from konjoai.store.qdrant import SearchResult, get_store
from konjoai.embed.encoder import get_encoder


def dense_search(
    query: str,
    top_k: int | None = None,
    q_vec: np.ndarray | None = None,
) -> list[SearchResult]:
    """Embed *query* and return the top-*k* closest chunks from Qdrant.

    Args:
        query:  Raw question string (used only when *q_vec* is None).
        top_k:  Number of results. Defaults to ``settings.top_k_dense``.
        q_vec:  Pre-computed query embedding (float32, shape (1, dim)).  When
                supplied the encoder call is skipped — avoids duplicate work
                when the cache layer has already embedded the question.
    """
    from konjoai.config import get_settings

    k = top_k if top_k is not None else get_settings().top_k_dense
    store = get_store()
    if q_vec is None:
        encoder = get_encoder()
        q_vec = encoder.encode_query(query)
    assert q_vec.dtype == np.float32, f"dense_search: q_vec must be float32, got {q_vec.dtype}"
    return store.search(q_vec, top_k=k)
