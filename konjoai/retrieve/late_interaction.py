"""Late-interaction scoring (ColBERT-style MaxSim).

Implements the MaxSim operation from Khattab & Zaharia, 2020 (arXiv:2004.12832).

Rather than comparing single-vector query/document representations, late
interaction encodes both as sets of token-level vectors.  Relevance is scored
by summing, over each query token, the *maximum* cosine similarity between
that query token and any document token.

This module is a re-scoring layer only — it operates on existing embeddings
from the sentence-transformer encoder and does NOT require a separate ColBERT
checkpoint.  The query sentence embedding is treated as a single token
(Q=1, D=dim), while each retrieved document chunk is encoded as a single
token (S=1, D=dim).  This degenerates to standard cosine similarity for
single-token inputs and therefore introduces zero regression against the
existing pipeline when used as an optional re-ranker.

Shape contract (multi-token generalization):
    query_vecs:    (Q, D)       — Q query token embeddings, dimension D
    doc_vecs_batch: (K, S, D)  — K candidate documents, each with S token
                                  embeddings
    returns:        (K,)        — one MaxSim scalar per candidate

K-invariants:
    K3 — Never raises; returns empty array for empty inputs.
    K4 — float32 throughout; no implicit upcasting.
"""
from __future__ import annotations

import numpy as np


def maxsim_score(
    query_vecs: np.ndarray,
    doc_vecs_batch: np.ndarray,
) -> np.ndarray:
    """Compute ColBERT-style MaxSim relevance scores.

    For each candidate document *k*, computes::

        score_k = sum_{q} max_{s} cosine_sim(query_vecs[q], doc_vecs_batch[k, s])

    Parameters
    ----------
    query_vecs:
        Shape ``(Q, D)`` — Q query token embeddings.  A single-vector query
        embedding should be reshaped to ``(1, D)`` before calling.
        Must be float32.
    doc_vecs_batch:
        Shape ``(K, S, D)`` — K candidate documents, each represented by S
        token embeddings.  A single-vector document should be reshaped to
        ``(K, 1, D)`` before calling.
        Must be float32.

    Returns
    -------
    np.ndarray
        Shape ``(K,)`` float32 — MaxSim score for each candidate.  Higher
        is more relevant.  Returns an empty float32 array when *K == 0*.

    Raises
    ------
    ValueError
        If shapes are incompatible (dimension mismatch between query and docs).
    """
    query_vecs = np.asarray(query_vecs, dtype=np.float32)
    doc_vecs_batch = np.asarray(doc_vecs_batch, dtype=np.float32)

    if query_vecs.ndim != 2:
        raise ValueError(
            f"query_vecs must be 2-D (Q, D), got shape {query_vecs.shape}"
        )
    if doc_vecs_batch.ndim != 3:
        raise ValueError(
            f"doc_vecs_batch must be 3-D (K, S, D), got shape {doc_vecs_batch.shape}"
        )

    Q, D_q = query_vecs.shape
    K, S, D_d = doc_vecs_batch.shape

    if D_q != D_d:
        raise ValueError(
            f"Dimension mismatch: query_vecs dim={D_q}, doc_vecs_batch dim={D_d}"
        )

    # Empty candidate list — K-invariant K3: never raises.
    if K == 0:
        return np.empty(0, dtype=np.float32)

    # ── L2-normalise query token vectors ─────────────────────────────────────
    # Shape: (Q, D)
    q_norms = np.linalg.norm(query_vecs, axis=-1, keepdims=True)  # (Q, 1)
    q_norms = np.maximum(q_norms, 1e-12)  # avoid div-by-zero for zero vectors
    query_norm = query_vecs / q_norms  # (Q, D)

    # ── L2-normalise document token vectors ──────────────────────────────────
    # Shape: (K, S, D)
    d_norms = np.linalg.norm(doc_vecs_batch, axis=-1, keepdims=True)  # (K, S, 1)
    d_norms = np.maximum(d_norms, 1e-12)
    doc_norm = doc_vecs_batch / d_norms  # (K, S, D)

    # ── Cosine similarity matrix ──────────────────────────────────────────────
    # query_norm: (Q, D) → (1, Q, D) broadcast over K
    # doc_norm:   (K, S, D)
    # sim[k, q, s] = cosine_sim(query_token_q, doc_token_s_in_doc_k)
    # Using einsum for clarity: "qd,ksd->kqs"
    sim = np.einsum("qd,ksd->kqs", query_norm, doc_norm)  # (K, Q, S)

    # ── MaxSim: for each (k, q), take max over doc tokens s ──────────────────
    max_sim = sim.max(axis=-1)   # (K, Q)

    # ── Sum over query tokens q ───────────────────────────────────────────────
    scores = max_sim.sum(axis=-1)  # (K,)

    return scores.astype(np.float32)


def rerank_with_maxsim(
    query_embedding: np.ndarray,
    results: list,
    get_embedding: "callable | None" = None,
) -> list:
    """Re-rank a list of retrieval results using MaxSim scores.

    Convenience wrapper used by the query route.  Each result is expected to
    have a ``.content`` attribute (text) and a ``.score`` attribute (float).
    The MaxSim score replaces ``.score`` in-place on a copy.

    Parameters
    ----------
    query_embedding:
        Shape ``(D,)`` or ``(1, D)`` float32 — query vector from the encoder.
    results:
        List of retrieval result objects with ``.content`` and ``.score``.
    get_embedding:
        Optional callable ``(text: str) -> np.ndarray of shape (D,)`` that
        returns the embedding for a document.  When *None*, the function
        falls back to the module-level encoder singleton via
        ``konjoai.embed.encoder.get_encoder().encode()``.

    Returns
    -------
    list
        New list sorted descending by MaxSim score.  Original list is not
        mutated.  Returns the original order on any error (K3: graceful
        degradation).
    """
    if not results:
        return results

    query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    D = query_embedding.shape[-1]

    if get_embedding is None:
        try:
            from konjoai.embed.encoder import get_encoder
            _enc = get_encoder()
            get_embedding = lambda text: _enc.encode(text)
        except Exception:
            return results  # K3: graceful degradation

    doc_embeddings: list[np.ndarray] = []
    for r in results:
        try:
            emb = np.asarray(get_embedding(r.content), dtype=np.float32).reshape(-1)
            if emb.shape[0] != D:
                return results  # dimension mismatch — degrade
            doc_embeddings.append(emb)
        except Exception:
            return results

    # Stack into (K, 1, D) — single-token representation per document
    doc_batch = np.stack(doc_embeddings, axis=0)[:, np.newaxis, :]  # (K, 1, D)

    try:
        scores = maxsim_score(query_embedding, doc_batch)  # (K,)
    except Exception:
        return results

    # Attach MaxSim scores and re-sort descending
    import copy
    reranked = []
    for r, s in zip(results, scores):
        rc = copy.copy(r)
        rc.score = float(s)
        reranked.append(rc)

    reranked.sort(key=lambda x: x.score, reverse=True)
    return reranked
