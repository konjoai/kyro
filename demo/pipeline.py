"""Kyro Pipeline Theater — runs the *real* retrieval pipeline, stage by stage.

Where ``server.py`` exposes the live :class:`~konjoai.cache.SemanticCache`, this
module lights up the rest of the kyro retrieval stack so the front-end can show
*every* feature working against real production code rather than a mock:

============  ===================================================================
Stage         Real kyro code path
============  ===================================================================
route         :func:`konjoai.retrieve.router.classify_intent`
              :func:`konjoai.retrieve.router.classify_chunk_complexity`
              :func:`konjoai.cache.threshold.classify_query`
decompose     :func:`konjoai.retrieve.router.decompose_query`
embed         the demo encoder (float32, L2-unit — same K4 dtype contract)
dense         cosine over the embedded corpus → ``SearchResult``
sparse        :class:`konjoai.retrieve.sparse.BM25Index` ``.build/.search``
fuse          :func:`konjoai.retrieve.hybrid.reciprocal_rank_fusion`
threshold     :class:`konjoai.cache.threshold.AdaptiveThresholdEngine`
route-decide  :class:`konjoai.retrieve.auto_router.AutoRouter`
============  ===================================================================

Every number the UI renders is computed by the functions above. The only thing
the demo supplies is the tiny deterministic embedder (so the page is
interactive on first paint) and a score→CRAG band heuristic for the router
input — both are labelled as such in the payload.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from konjoai.cache.threshold import AdaptiveThresholdEngine, classify_query
from konjoai.retrieve.auto_router import AutoRouter
from konjoai.retrieve.hybrid import (
    BM25Result,
    SearchResult,
    reciprocal_rank_fusion,
)
from konjoai.retrieve.router import (
    classify_chunk_complexity,
    classify_intent,
    decompose_query,
)
from konjoai.retrieve.sparse import BM25Index

__all__ = ["PipelineEngine", "RRF_K"]

# RRF smoothing constant — mirror of the production default so the math the UI
# renders matches konjoai.retrieve.hybrid.reciprocal_rank_fusion exactly.
RRF_K = 60

_DOMAINS = {"legal", "medical", "technical"}


def _classify_domain(filename: str) -> str:
    head = filename.split("_", 1)[0]
    return head if head in _DOMAINS else "other"


def _short_title(filename: str) -> str:
    parts = Path(filename).stem.split("_", 2)
    slug = parts[2] if len(parts) >= 3 else Path(filename).stem
    return slug.replace("_", " ")


class PipelineEngine:
    """Lazily-built hybrid retriever wired to the real kyro retrieval stack.

    Parameters
    ----------
    corpus_root:
        Directory of ``*.txt`` documents (the demo corpus).
    embed_fn:
        Encoder returning an L2-unit ``float32`` vector. The demo passes its
        deterministic char-trigram embedder; production would pass
        ``sentence-transformers``.
    """

    def __init__(self, corpus_root: Path, embed_fn: Callable[[str], np.ndarray]) -> None:
        self._root = corpus_root
        self._embed = embed_fn
        self._lock = threading.Lock()
        self._loaded = False
        self._docs: list[dict[str, Any]] = []
        self._mat: np.ndarray | None = None  # (N, dim) float32
        self._bm25 = BM25Index()
        self._threshold = AdaptiveThresholdEngine()
        self._router = AutoRouter()

    # ── Loading ─────────────────────────────────────────────────────────────

    def load(self) -> dict[str, Any]:
        """Embed + BM25-index every corpus document. Idempotent."""
        with self._lock:
            if self._loaded:
                return {"loaded": True, "count": len(self._docs)}
            if not self._root.exists():
                return {"loaded": False, "error": f"corpus root missing: {self._root}"}

            paths = sorted(self._root.glob("*.txt"))
            contents: list[str] = []
            sources: list[str] = []
            metadatas: list[dict[str, Any]] = []
            vecs: list[np.ndarray] = []
            for i, path in enumerate(paths):
                text = path.read_text(encoding="utf-8").strip()
                if not text:
                    continue
                domain = _classify_domain(path.name)
                vec = self._embed(text)
                if vec.ndim == 2:
                    vec = vec.reshape(-1)
                vecs.append(vec.astype(np.float32))
                contents.append(text)
                sources.append(path.name)
                metadatas.append({"domain": domain})
                self._docs.append(
                    {
                        "id": i,
                        "content": text,
                        "source": path.name,
                        "title": _short_title(path.name),
                        "domain": domain,
                        "preview": text[:160],
                    }
                )

            if not self._docs:
                return {"loaded": False, "error": "corpus is empty"}

            self._mat = np.vstack(vecs).astype(np.float32)
            self._bm25.build(contents, sources, metadatas)
            self._loaded = True
            return {"loaded": True, "count": len(self._docs)}

    # ── Per-stage helpers ─────────────────────────────────────────────────────

    def _dense(self, query: str, top_k: int) -> list[SearchResult]:
        """Cosine retrieval → ``SearchResult`` list (rank 0 = best)."""
        assert self._mat is not None  # load() guarantees this
        q = self._embed(query)
        if q.ndim == 2:
            q = q.reshape(-1)
        scores = self._mat @ q.astype(np.float32)
        order = np.argsort(-scores)[:top_k]
        return [
            SearchResult(
                id=str(self._docs[int(i)]["id"]),
                score=float(scores[int(i)]),
                content=self._docs[int(i)]["content"],
                source=self._docs[int(i)]["source"],
                metadata={"domain": self._docs[int(i)]["domain"]},
            )
            for i in order
        ]

    def _sparse(self, query: str, top_k: int) -> list[BM25Result]:
        return self._bm25.search(query, top_k=top_k)

    def _meta_for(self, source: str) -> dict[str, Any]:
        for d in self._docs:
            if d["source"] == source:
                return {"title": d["title"], "domain": d["domain"], "preview": d["preview"]}
        return {"title": source, "domain": "other", "preview": ""}

    @staticmethod
    def _crag_band(top_score: float) -> str:
        """Map the best fused score onto a CRAG-style band for the router.

        Heuristic stand-in for the LLM ``DocumentGrader`` — labelled
        ``score-derived`` in the payload so the UI never claims an LLM ran.
        """
        if top_score >= 0.02:
            return "correct"
        if top_score >= 0.012:
            return "ambiguous"
        return "incorrect"

    # ── Public entrypoint ─────────────────────────────────────────────────────

    def analyze(self, query: str, top_k: int = 4, alpha: float = 0.7) -> dict[str, Any]:
        """Run one query through every stage and return a structured trace.

        The returned dict is a wire contract consumed by ``pipeline.html``;
        :mod:`tests.unit.test_demo_pipeline` pins its shape.
        """
        query = (query or "").strip()
        if not query:
            return {"error": "query must be non-empty"}
        if not self._loaded:
            self.load()
        top_k = max(1, min(int(top_k), len(self._docs)))
        alpha = float(min(1.0, max(0.0, alpha)))

        timings: dict[str, float] = {}

        # 1 — Route classification (all real, no LLM).
        t = time.perf_counter()
        intent = classify_intent(query)
        complexity, est_tokens = classify_chunk_complexity(query)
        cache_qtype = classify_query(query)
        timings["route_ms"] = (time.perf_counter() - t) * 1000.0

        # 2 — Decomposition.
        t = time.perf_counter()
        sub_queries = decompose_query(query, max_parts=3)
        timings["decompose_ms"] = (time.perf_counter() - t) * 1000.0

        # 3 — Embed (float32 / L2-unit — the K4 dtype contract).
        t = time.perf_counter()
        q_vec = self._embed(query)
        if q_vec.ndim == 2:
            q_vec = q_vec.reshape(-1)
        timings["embed_ms"] = (time.perf_counter() - t) * 1000.0

        # 4 — Dense ∥ Sparse retrieval.
        t = time.perf_counter()
        dense = self._dense(query, top_k)
        timings["dense_ms"] = (time.perf_counter() - t) * 1000.0
        t = time.perf_counter()
        sparse = self._sparse(query, top_k)
        timings["sparse_ms"] = (time.perf_counter() - t) * 1000.0

        # 5 — RRF fusion (real production code).
        t = time.perf_counter()
        fused = reciprocal_rank_fusion(dense, sparse, alpha=alpha, k=RRF_K)
        timings["fuse_ms"] = (time.perf_counter() - t) * 1000.0

        # 6 — Adaptive cache threshold for this query type.
        t = time.perf_counter()
        resolved_type, threshold = self._threshold.resolve(query)
        timings["threshold_ms"] = (time.perf_counter() - t) * 1000.0

        # 7 — Route decision (real AutoRouter on a score-derived CRAG band).
        top_score = fused[0].rrf_score if fused else 0.0
        band = self._crag_band(top_score)
        decision = self._router.decide(band, crag_score=round(float(top_score), 6))

        return {
            "query": query,
            "alpha": alpha,
            "rrf_k": RRF_K,
            "top_k": top_k,
            "route": {
                "intent": intent.value,
                "complexity": complexity.value,
                "estimated_tokens": int(est_tokens),
                "cache_query_type": cache_qtype.value,
            },
            "decomposition": {
                "decomposed": len(sub_queries) > 1,
                "sub_queries": sub_queries,
            },
            "embedding": {
                "dim": int(q_vec.shape[0]),
                "dtype": str(q_vec.dtype),
                "norm": round(float(np.linalg.norm(q_vec)), 6),
                "nonzero": int(np.count_nonzero(q_vec)),
                "preview": [round(float(x), 4) for x in q_vec[:16].tolist()],
            },
            "dense": [self._fmt_dense(r, rank) for rank, r in enumerate(dense)],
            "sparse": [self._fmt_sparse(r, rank) for rank, r in enumerate(sparse)],
            "fused": self._fmt_fused(fused, dense, sparse, alpha),
            "threshold": {
                "query_type": resolved_type.value,
                "value": round(float(threshold), 4),
            },
            "decision": {
                "strategy": decision.strategy.value,
                "rationale": decision.rationale,
                "crag_band": band,
                "crag_score": decision.crag_score,
                "crag_source": "score-derived (no LLM grader in demo)",
            },
            "timings_ms": {k: round(v, 4) for k, v in timings.items()},
            "source": "konjoai.retrieve.{router,sparse,hybrid} + cache.threshold",
        }

    # ── Formatting (UI never receives full document bodies) ───────────────────

    def _fmt_dense(self, r: SearchResult, rank: int) -> dict[str, Any]:
        meta = self._meta_for(r.source)
        return {
            "rank": rank,
            "source": r.source,
            "score": round(float(r.score), 4),
            **meta,
        }

    def _fmt_sparse(self, r: BM25Result, rank: int) -> dict[str, Any]:
        meta = self._meta_for(r.source)
        return {
            "rank": rank,
            "source": r.source,
            "score": round(float(r.score), 4),
            **meta,
        }

    def _fmt_fused(
        self,
        fused: list[Any],
        dense: list[SearchResult],
        sparse: list[BM25Result],
        alpha: float,
    ) -> list[dict[str, Any]]:
        """Attach the per-document RRF contribution breakdown.

        The ordering + ``rrf_score`` come straight from the real
        ``reciprocal_rank_fusion``; we recompute the two additive terms only to
        let the UI show *why* a document fused where it did. The reconstruction
        uses the identical formula, so it equals the production score.
        """
        dense_rank = {r.content: i for i, r in enumerate(dense)}
        sparse_rank = {r.content: i for i, r in enumerate(sparse)}
        out: list[dict[str, Any]] = []
        for fr in fused:
            content = getattr(fr, "content", "")
            rd = dense_rank.get(content)
            rs = sparse_rank.get(content)
            dense_term = alpha * (1.0 / (RRF_K + rd)) if rd is not None else 0.0
            sparse_term = (1.0 - alpha) * (1.0 / (RRF_K + rs)) if rs is not None else 0.0
            meta = self._meta_for(fr.source)
            out.append(
                {
                    "source": fr.source,
                    "rrf_score": round(float(fr.rrf_score), 6),
                    "dense_rank": rd,
                    "sparse_rank": rs,
                    "dense_term": round(dense_term, 6),
                    "sparse_term": round(sparse_term, 6),
                    "in_both": rd is not None and rs is not None,
                    **meta,
                }
            )
        return out
