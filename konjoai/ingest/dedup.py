"""Near-duplicate chunk filtering for the ingest pipeline.

Applies greedy cosine similarity filtering to remove near-duplicate text
chunks before they are upserted into the vector store.  Uses only NumPy
— no additional runtime dependencies (K5).

Algorithm
---------
A greedy O(N²) forward pass:

1. Compute L2-normalised embeddings once.
2. For each candidate chunk (in order), compute its cosine similarity
   against every already-accepted chunk.
3. If the maximum similarity is >= *threshold*, the candidate is a
   near-duplicate and is discarded.  Otherwise it is accepted.

Complexity: O(N × K × D) where K is the number of accepted chunks and
D is the embedding dimension.  For typical ingest batches (N < 10 000,
D = 768) this is negligible compared to the encoder forward pass.

Notes
-----
- Cosine similarity is computed in float32 (K4).
- The function is pure — it does not mutate the inputs.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def filter_near_duplicates(
    embeddings: np.ndarray,
    contents: list[str],
    sources: list[str],
    metadatas: list[dict[str, Any]],
    threshold: float = 0.98,
) -> tuple[np.ndarray, list[str], list[str], list[dict[str, Any]], int]:
    """Remove near-duplicate chunks based on cosine similarity.

    Parameters
    ----------
    embeddings:
        Float32 array of shape ``(N, D)``.  Must be 2-D with at least
        one row.
    contents:
        Corresponding text chunks; ``len(contents) == N``.
    sources:
        Corresponding source identifiers; ``len(sources) == N``.
    metadatas:
        Corresponding metadata dicts; ``len(metadatas) == N``.
    threshold:
        Cosine similarity gate in ``[0, 1]``.  A candidate chunk is
        dropped when its maximum similarity to any already-accepted chunk
        is **greater than or equal to** *threshold*.  Set to ``1.0`` to
        disable dedup (only exact duplicates are removed).

    Returns
    -------
    (kept_embeddings, kept_contents, kept_sources, kept_metadatas, n_removed)
        where ``n_removed`` is the number of chunks that were filtered out.

    Raises
    ------
    ValueError
        If ``embeddings`` is not 2-D or lengths are inconsistent.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
    n = embeddings.shape[0]
    if not (len(contents) == len(sources) == len(metadatas) == n):
        raise ValueError(
            f"Length mismatch: embeddings={n}, contents={len(contents)}, "
            f"sources={len(sources)}, metadatas={len(metadatas)}"
        )

    if n == 0:
        return embeddings, contents, sources, metadatas, 0

    # L2-normalise once; cosine similarity = dot product of normalised vectors.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero for zero-vectors (treat as distinct).
    norms = np.where(norms == 0, 1.0, norms)
    normed = (embeddings / norms).astype(np.float32)  # (N, D)

    kept_idx: list[int] = []
    # accepted_normed accumulates the normalised embeddings of accepted chunks.
    # Initialise empty; shape (0, D).
    accepted_normed = np.empty((0, normed.shape[1]), dtype=np.float32)

    for i in range(n):
        if accepted_normed.shape[0] == 0:
            # First chunk is always accepted.
            kept_idx.append(i)
            accepted_normed = normed[i : i + 1]
        else:
            # Cosine similarity of chunk i vs all accepted chunks: (K,)
            sims = accepted_normed @ normed[i]  # dot product = cosine sim (normalised)
            if float(np.max(sims)) >= threshold:
                logger.debug(
                    "Dedup: dropping chunk %d (max_sim=%.4f >= threshold=%.4f) source=%r",
                    i,
                    float(np.max(sims)),
                    threshold,
                    sources[i],
                )
            else:
                kept_idx.append(i)
                accepted_normed = np.vstack([accepted_normed, normed[i : i + 1]])

    n_removed = n - len(kept_idx)

    if not kept_idx:
        # Edge-case: all chunks are near-duplicates (shouldn't happen in practice).
        logger.warning("filter_near_duplicates: all %d chunks were filtered; keeping first.", n)
        kept_idx = [0]
        n_removed = n - 1

    kept_embeddings = embeddings[kept_idx]
    kept_contents = [contents[i] for i in kept_idx]
    kept_sources = [sources[i] for i in kept_idx]
    kept_metadatas = [metadatas[i] for i in kept_idx]

    return kept_embeddings, kept_contents, kept_sources, kept_metadatas, n_removed
