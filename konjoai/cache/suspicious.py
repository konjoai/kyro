"""Automatic suspicious-entry detection for the semantic cache (Sprint 29).

Detection heuristics
--------------------
Three signal types, each computable from the live cache state without any
external service:

1. **Embedding outlier**: a cached question whose L2 distance from its k-means
   cluster centroid exceeds ``mean + z_threshold * stddev`` for that cluster.
   Indicates the entry belongs to a different topic than its neighbours — a
   potential injection.

2. **Hit-count anomaly**: an entry whose ``hit_count`` is more than
   ``z_threshold`` standard deviations above the cluster mean.  An adversary
   could inflate hit counts to keep a poisoned answer "warm" in the LRU.

3. **Answer length anomaly**: an entry whose answer character length falls
   outside ``[mean - z_threshold*σ, mean + z_threshold*σ]`` for the whole
   cache.  Truncated or excessively long answers are suspicious.

The detector is intentionally **read-only** — it never modifies the cache.
Callers decide whether to approve (keep) or reject (evict) flagged entries via
:class:`SuspiciousFlagStore`.

K3: ``GET /cache/suspicious`` returns an empty list when the cache has fewer
than 2×k entries (not enough data for reliable clustering).
K5: pure stdlib + numpy.
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class SuspiciousFlag:
    """A single flagged cache entry awaiting review."""

    entry_hash: str          # SHA-256 prefix of the normalised question
    question: str            # the (normalised) question text
    reason: str              # human-readable detection reason
    score: float             # outlier score (higher = more suspicious)
    signal: str              # "embedding_outlier" | "hit_count_anomaly" | "answer_length_anomaly"
    status: Literal["pending", "approved", "rejected"] = "pending"
    created_at: float = field(default_factory=time.monotonic)


# ── Flag store ────────────────────────────────────────────────────────────────


class SuspiciousFlagStore:
    """In-memory ring buffer of suspicious-entry flags.

    Thread-safe.  Approved/rejected flags are kept for audit purposes.
    """

    MAX_FLAGS = 500

    def __init__(self) -> None:
        self._flags: dict[str, SuspiciousFlag] = {}
        self._lock = threading.Lock()

    def flag(self, entry_hash: str, question: str, reason: str, score: float, signal: str) -> None:
        """Add or refresh a flag for *entry_hash*.  Existing flags are replaced."""
        with self._lock:
            if len(self._flags) >= self.MAX_FLAGS and entry_hash not in self._flags:
                # Evict the oldest resolved flag to make room
                oldest_resolved = next(
                    (k for k, f in self._flags.items() if f.status != "pending"), None
                )
                if oldest_resolved:
                    del self._flags[oldest_resolved]
            self._flags[entry_hash] = SuspiciousFlag(
                entry_hash=entry_hash,
                question=question,
                reason=reason,
                score=score,
                signal=signal,
            )

    def resolve(self, entry_hash: str, action: Literal["approve", "reject"]) -> bool:
        """Mark a flag as approved or rejected.  Returns False if hash not found."""
        with self._lock:
            flag = self._flags.get(entry_hash)
            if flag is None:
                return False
            flag.status = "approved" if action == "approve" else "rejected"
            return True

    def pending(self) -> list[SuspiciousFlag]:
        """Return all flags currently in *pending* state."""
        with self._lock:
            return [f for f in self._flags.values() if f.status == "pending"]

    def all_flags(self) -> list[SuspiciousFlag]:
        with self._lock:
            return list(self._flags.values())

    def get(self, entry_hash: str) -> SuspiciousFlag | None:
        with self._lock:
            return self._flags.get(entry_hash)

    def clear(self) -> None:
        with self._lock:
            self._flags.clear()


# ── Detection logic ───────────────────────────────────────────────────────────


def _entry_hash(normalised_question: str) -> str:
    return hashlib.sha256(normalised_question.encode()).hexdigest()[:16]


def scan_for_suspicious(
    cache: object,
    k: int = 5,
    z_threshold: float = 2.0,
) -> list[dict]:
    """Scan the live ``SemanticCache`` and return a list of suspicious entries.

    Args:
        cache:       A :class:`~konjoai.cache.semantic_cache.SemanticCache` instance.
        k:           Number of clusters for the embedding-outlier check.
        z_threshold: Number of standard deviations above which a metric is flagged.

    Returns:
        List of dicts ready for the API response:
        ``{entry_hash, question, reason, score, signal}``.
    """
    from konjoai.cache.semantic_cache import SemanticCache  # noqa: PLC0415

    if not isinstance(cache, SemanticCache):
        return []

    with cache._lock:  # type: ignore[attr-defined]
        entries = [
            (key, e.question, e.question_vec, e.hit_count, e.response)
            for key, e in cache._lru.items()  # type: ignore[attr-defined]
            if not e.is_expired()
        ]

    n = len(entries)
    if n < max(4, k * 2):
        return []

    questions   = [e[1] for e in entries]
    vecs        = np.vstack([e[2].ravel() for e in entries]).astype(np.float32)
    hit_counts  = np.array([e[3] for e in entries], dtype=np.float32)
    answers     = [e[4].answer if hasattr(e[4], "answer") else str(e[4]) for e in entries]
    ans_lengths = np.array([len(a) for a in answers], dtype=np.float32)

    # L2-normalise vecs for cosine distance
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    unit_vecs = vecs / norms

    suspicious: list[dict] = []

    # ── Signal 1: embedding outlier (per-cluster) ─────────────────────────
    actual_k = min(k, n // 2)
    centroids, labels = _mini_kmeans(unit_vecs, actual_k, iters=15)
    for ci in range(actual_k):
        mask = labels == ci
        if mask.sum() < 2:
            continue
        cluster_vecs = unit_vecs[mask]
        centroid = centroids[ci]
        dists = 1.0 - cluster_vecs @ centroid   # cosine distance
        mean_d = float(dists.mean())
        std_d  = float(dists.std()) + 1e-9
        for local_i, global_i in enumerate(np.where(mask)[0]):
            score = (dists[local_i] - mean_d) / std_d
            if score > z_threshold:
                q = questions[global_i]
                suspicious.append({
                    "entry_hash": _entry_hash(q.lower().strip()),
                    "question": q,
                    "reason": f"embedding distance {score:.2f}σ above cluster mean (cluster {ci})",
                    "score": round(float(score), 3),
                    "signal": "embedding_outlier",
                })

    # ── Signal 2: hit-count anomaly ────────────────────────────────────────
    hc_mean = float(hit_counts.mean())
    hc_std  = float(hit_counts.std()) + 1e-9
    for i, hc in enumerate(hit_counts):
        score = (hc - hc_mean) / hc_std
        if score > z_threshold:
            q = questions[i]
            suspicious.append({
                "entry_hash": _entry_hash(q.lower().strip()),
                "question": q,
                "reason": f"hit count {hc:.0f} is {score:.2f}σ above mean",
                "score": round(float(score), 3),
                "signal": "hit_count_anomaly",
            })

    # ── Signal 3: answer length anomaly ───────────────────────────────────
    al_mean = float(ans_lengths.mean())
    al_std  = float(ans_lengths.std()) + 1e-9
    for i, al in enumerate(ans_lengths):
        score = abs(al - al_mean) / al_std
        if score > z_threshold:
            q = questions[i]
            suspicious.append({
                "entry_hash": _entry_hash(q.lower().strip()),
                "question": q,
                "reason": f"answer length {al:.0f} chars is {score:.2f}σ from mean",
                "score": round(float(score), 3),
                "signal": "answer_length_anomaly",
            })

    # Deduplicate by entry_hash, keep highest score
    by_hash: dict[str, dict] = {}
    for item in suspicious:
        h = item["entry_hash"]
        if h not in by_hash or item["score"] > by_hash[h]["score"]:
            by_hash[h] = item
    return sorted(by_hash.values(), key=lambda x: x["score"], reverse=True)


def _mini_kmeans(
    vecs: np.ndarray, k: int, iters: int = 15
) -> tuple[np.ndarray, np.ndarray]:
    """Lightweight k-means++ on L2-normalised unit vectors.  Returns (centroids, labels)."""
    rng = np.random.default_rng(seed=42)
    n = len(vecs)
    # k-means++ seeding
    centroids = [vecs[rng.integers(n)].copy()]
    for _ in range(k - 1):
        dists = np.array([min(1.0 - float(c @ v) for c in centroids) for v in vecs])
        dists = np.clip(dists, 0.0, None)
        total = dists.sum()
        probs = dists / total if total > 0 else np.ones(n) / n
        centroids.append(vecs[rng.choice(n, p=probs)].copy())
    centroids_arr = np.array(centroids)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        sims = vecs @ centroids_arr.T
        new_labels = np.argmax(sims, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for ci in range(k):
            m = labels == ci
            if m.any():
                c = vecs[m].mean(axis=0)
                nm = np.linalg.norm(c)
                centroids_arr[ci] = c / nm if nm > 1e-10 else c
    return centroids_arr, labels


# ── Module-level singleton ────────────────────────────────────────────────────

_flag_store: SuspiciousFlagStore | None = None
_flag_store_lock = threading.Lock()


def get_flag_store() -> SuspiciousFlagStore:
    """Return the process-level flag store singleton."""
    global _flag_store  # noqa: PLW0603
    if _flag_store is not None:
        return _flag_store
    with _flag_store_lock:
        if _flag_store is None:
            _flag_store = SuspiciousFlagStore()
    return _flag_store


def _reset_flag_store() -> None:
    """Test helper."""
    global _flag_store  # noqa: PLW0603
    with _flag_store_lock:
        _flag_store = None
