"""Kyro Demo Server — runs the *real* `konjoai.cache.SemanticCache`.

This server backs the interactive widgets in ``demo/index.html`` with live calls
into the actual kyro library. Cosine similarity, LRU eviction, exact-match
fast path, the float32 dtype contract, hit/miss accounting — every one of
those is the same code path the production unit suite exercises.

Embedding strategy
------------------
Kyro production uses ``sentence-transformers`` via
``konjoai.embed.encoder.get_encoder()``. Loading that on first call costs
several seconds and downloads model weights, which makes for a slow demo.
This server ships a tiny deterministic embedder so the page is interactive
the moment the process starts. The embedder vector is still float32 and
unit-normalised, so it satisfies the same K4 dtype contract that the real
encoder does. To exercise the real encoder instead, set
``DEMO_USE_REAL_ENCODER=1`` before launching this server.

Run::

    python3 demo/server.py            # http://localhost:8766
    python3 demo/server.py --port 9000

Then open ``demo/index.html`` (the page auto-detects the server on load).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

# ── Wire kyro into the demo ─────────────────────────────────────────────
# The server lives in `demo/`; the package lives at the repo root. Make sure
# we can `import konjoai` whether the user runs us from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agent import AgentEngine  # noqa: E402  (sibling module, demo/ on sys.path)
from hyde import HyDEEngine  # noqa: E402  (sibling module, demo/ on sys.path)
from pipeline import PipelineEngine  # noqa: E402  (sibling module, demo/ on sys.path)
from rewriting import RewriteEngine  # noqa: E402  (sibling module, demo/ on sys.path)
from security import SecurityEngine  # noqa: E402  (sibling module, demo/ on sys.path)

from konjoai.cache.semantic_cache import SemanticCache, SemanticCacheEntry  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  demo-server  %(message)s")
log = logging.getLogger("kyro-demo-server")


# ── Tiny deterministic embedder ──────────────────────────────────────────
# Char-trigram + word-token hashing into a 256-dim L2-unit float32 vector.
# Paraphrases on the same topic share enough trigrams + content words to
# clear ~0.78–0.92 cosine. We configure SemanticCache(threshold=0.78) so the
# user's eyes can see paraphrases collapse to the same answer.

_DIM = 256
_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "of", "to", "for",
    "in", "on", "at", "what", "which", "how", "where", "when", "why", "who",
    "do", "does", "did", "you", "your", "my", "i", "it", "that", "this",
    "with", "and", "or", "but", "as", "from", "by", "so", "if",
    "can", "could", "would", "should", "have", "has", "had", "will",
    "about", "tell", "me", "us", "any", "some", "s",
    "into", "onto", "over", "out", "up", "down", "very", "more", "less",
}


def _stable_hash(s: str, mod: int) -> int:
    """FNV-1a — deterministic across processes (unlike Python's hash())."""
    h = 0x811C9DC5
    for c in s:
        h ^= ord(c)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h % mod


def _stem(w: str) -> str:
    """Cheap suffix stripper so paraphrases collapse onto the same token.

    Real production uses dense embeddings — this only exists so a 100-line
    HTTP server can cluster ``refund / refunds / refunded`` for the demo.
    """
    w = w.replace("'s", "").replace("'", "")
    for suf in ("ing", "tion", "ions", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def fast_embed(text: str, dim: int = _DIM) -> np.ndarray:
    """Tiny demo embedder. Kyro's real encoder is sentence-transformers.

    Recipe: tokenise → strip apostrophes → cheap stemmer → drop stop-words
    → hash tokens (weight 1.0) and char-trigrams (weight 0.4) into a 256-dim
    bucket vector → L2-normalise to float32. Lossy but deterministic and
    fast enough that the demo stays interactive on first paint.
    """
    text = (text or "").lower().strip()
    v = np.zeros(dim, dtype=np.float32)
    if not text:
        v[0] = 1.0
        return v

    raw = re.findall(r"[a-z']+", text)
    content: list[str] = []
    for tok in raw:
        if len(tok) <= 1:
            continue
        if tok in _STOP:
            continue
        s = _stem(tok)
        if s and s not in _STOP and len(s) > 1:
            content.append(s)

    # Word features (carry the topic signal).
    for w in content:
        v[_stable_hash("w:" + w, dim)] += 1.0

    # Char-trigram features over content (extra paraphrase tolerance).
    s = "  " + " ".join(content) + "  "
    for i in range(len(s) - 2):
        tri = s[i : i + 3]
        if tri.strip():
            v[_stable_hash("t:" + tri, dim)] += 0.4

    n = float(np.linalg.norm(v))
    if n < 1e-12:
        v[0] = 1.0
        return v
    return (v / n).astype(np.float32)


_REAL_ENCODER = None


def encode(text: str) -> np.ndarray:
    """Embed a query. Honours DEMO_USE_REAL_ENCODER=1 if set + available."""
    global _REAL_ENCODER
    if os.environ.get("DEMO_USE_REAL_ENCODER") == "1":
        if _REAL_ENCODER is None:
            try:
                from konjoai.embed.encoder import get_encoder

                _REAL_ENCODER = get_encoder()
                log.info("DEMO_USE_REAL_ENCODER=1 → loaded sentence-transformers encoder")
            except Exception as exc:  # noqa: BLE001
                log.warning("real encoder unavailable (%s) — falling back to fast embedder", exc)
                _REAL_ENCODER = False
        if _REAL_ENCODER and _REAL_ENCODER is not False:
            v = _REAL_ENCODER.encode_query(text)
            return v.astype(np.float32) if v.dtype != np.float32 else v
    return fast_embed(text)


# ── Demo state — wraps the real kyro cache ──────────────────────────────


class DemoState:
    """Real kyro SemanticCache + accounting for the demo UI.

    Threading: BaseHTTPRequestHandler is one thread per request. SemanticCache
    is internally locked, but our hit/miss/savings counters need their own.
    """

    SIMULATED_LLM_MS = 800.0
    COST_PER_LLM_CALL_USD = 0.002
    LATENCY_HISTORY_LEN = 60

    def __init__(self) -> None:
        # Configured for the demo: smaller cache + a permissive threshold so
        # the tiny char-trigram embedder can show paraphrase hits live. With
        # the real ``sentence-transformers`` encoder (DEMO_USE_REAL_ENCODER=1)
        # paraphrases routinely clear 0.95, which is the production default.
        self.cache = SemanticCache(max_size=64, threshold=0.65)
        self._lock = threading.Lock()
        self._calls_avoided = 0
        self._time_saved_ms = 0.0
        self._dollars_saved = 0.0
        self._seed_pairs: list[tuple[str, str]] = []
        self._seeded = False
        # Observatory accounting (real counters wired to the live cache):
        self._total_queries = 0
        self._singleflight_collapsed = 0
        self._latency_history: deque[float] = deque(maxlen=self.LATENCY_HISTORY_LEN)
        self._top_terms: Counter[str] = Counter()
        # Thread-level singleflight gate: concurrent identical misses collapse
        # onto one LLM synth instead of firing N separate calls.
        self._inflight: dict[str, threading.Event] = {}
        self._inflight_lock = threading.Lock()

    # ── Embedded queries inspected against real cache state ─────────────

    def best_match(self, q_vec: np.ndarray) -> tuple[str | None, float]:
        """Walk the cache's *real* internal state to compute the best similarity.

        Same algorithm SemanticCache.lookup() uses internally (l2-norm + dot).
        We expose the score in the API; the cache's lookup() returns only the
        cached value. Reading `cache._lru` is a deliberate inspection — we
        want the demo to render the *actual* state of the kyro object.
        """
        with self.cache._lock:  # type: ignore[attr-defined]
            entries = list(self.cache._lru.items())  # type: ignore[attr-defined]
        if not entries:
            return None, -1.0
        q_norm = SemanticCache._l2_norm(q_vec)
        best_key: str | None = None
        best_sim: float = -1.0
        for k, e in entries:
            sim = float(np.dot(q_norm, SemanticCache._l2_norm(e.question_vec)))
            if sim > best_sim:
                best_sim = sim
                best_key = k
        return best_key, best_sim

    def best_match_question(self, key: str) -> str:
        with self.cache._lock:  # type: ignore[attr-defined]
            entry: SemanticCacheEntry | None = self.cache._lru.get(key)  # type: ignore[attr-defined]
        return entry.question if entry else key

    # ── Endpoints ──────────────────────────────────────────────────────

    def probe(self, question: str) -> dict[str, Any]:
        """Read-only score: real cosine vs current cache state, no mutation.

        Powers the "Try it yourself" widget — lets a user type freely and
        see exactly what kyro *would* do, without polluting the cache.
        """
        question = (question or "").strip()
        if not question:
            return {"error": "question must be non-empty"}
        embed_t0 = time.perf_counter()
        q_vec = encode(question)
        embed_ms = (time.perf_counter() - embed_t0) * 1000.0
        best_key, best_sim = self.best_match(q_vec)
        best_question = self.best_match_question(best_key) if best_key else None
        threshold = self.cache._threshold  # type: ignore[attr-defined]
        would_hit = bool(best_key and best_sim >= threshold)
        return {
            "probe": True,
            "would_hit": would_hit,
            "score": round(float(best_sim), 4) if best_sim >= 0 else None,
            "score_pct": round(float(best_sim) * 100.0, 2) if best_sim >= 0 else None,
            "threshold": threshold,
            "matched_question": best_question,
            "embed_ms": round(embed_ms, 3),
            "size": len(self.cache._lru),  # type: ignore[attr-defined]
            "source": "kyro.SemanticCache (read-only)",
        }

    def ask(self, question: str) -> dict[str, Any]:
        """One full demo lookup with singleflight collapse for concurrent misses.

        Concurrent callers with the same normalised question collapse onto a
        single synthesise call; followers wait on a threading.Event and read
        from the populated cache rather than firing their own LLM synth.
        """
        question = (question or "").strip()
        if not question:
            return {"error": "question must be non-empty"}

        ask_t0 = time.perf_counter()
        embed_t0 = time.perf_counter()
        q_vec = encode(question)
        embed_ms = (time.perf_counter() - embed_t0) * 1000.0

        best_key, best_sim = self.best_match(q_vec)
        best_question = self.best_match_question(best_key) if best_key else None

        lookup_t0 = time.perf_counter()
        cached = self.cache.lookup(question, q_vec)
        lookup_ms = (time.perf_counter() - lookup_t0) * 1000.0

        if cached is not None:
            answer = cached.answer if hasattr(cached, "answer") else str(cached)
            latency_ms = (time.perf_counter() - ask_t0) * 1000.0
            with self._lock:
                self._calls_avoided += 1
                self._time_saved_ms += self.SIMULATED_LLM_MS
                self._dollars_saved += self.COST_PER_LLM_CALL_USD
                self._record_query(question, latency_ms)
            return {
                "hit": True,
                "score": round(float(best_sim), 4),
                "score_pct": round(float(best_sim) * 100.0, 2),
                "threshold": self.cache._threshold,  # type: ignore[attr-defined]
                "cached_answer": answer,
                "matched_question": best_question,
                "embed_ms": round(embed_ms, 3),
                "cache_lookup_ms": round(lookup_ms, 3),
                "latency_ms": round(latency_ms, 3),
                "would_have_been_ms": self.SIMULATED_LLM_MS,
                "source": "kyro.SemanticCache.lookup()",
            }

        # Miss — singleflight gate.
        key = self.cache._normalise(question)  # type: ignore[attr-defined]
        with self._inflight_lock:
            event = self._inflight.get(key)
            is_leader = event is None
            if is_leader:
                event = threading.Event()
                self._inflight[key] = event

        if not is_leader:
            # Follower: block until the leader finishes, then serve from cache.
            event.wait(timeout=10.0)
            cached = self.cache.lookup(question, q_vec)
            latency_ms = (time.perf_counter() - ask_t0) * 1000.0
            with self._lock:
                self._singleflight_collapsed += 1
                if cached is not None:
                    self._calls_avoided += 1
                    self._time_saved_ms += self.SIMULATED_LLM_MS
                    self._dollars_saved += self.COST_PER_LLM_CALL_USD
                self._record_query(question, latency_ms)
            answer = (
                cached.answer if cached is not None and hasattr(cached, "answer")
                else str(cached) if cached is not None
                else f'I would ask the LLM about "{question}".'
            )
            return {
                "hit": True,
                "collapsed": True,
                "score": round(float(best_sim), 4) if best_sim >= 0 else None,
                "score_pct": round(float(best_sim) * 100.0, 2) if best_sim >= 0 else None,
                "threshold": self.cache._threshold,  # type: ignore[attr-defined]
                "cached_answer": answer,
                "matched_question": best_question,
                "embed_ms": round(embed_ms, 3),
                "cache_lookup_ms": round(lookup_ms, 3),
                "latency_ms": round(latency_ms, 3),
                "would_have_been_ms": self.SIMULATED_LLM_MS,
                "source": "demo singleflight (collapsed onto leader)",
            }

        # Leader — synthesise + store + signal followers.
        try:
            synth_t0 = time.perf_counter()
            time.sleep(0.18)
            answer = synthesise_answer(question)
            synth_ms = (time.perf_counter() - synth_t0) * 1000.0
            store_t0 = time.perf_counter()
            self.cache.store(question, q_vec, _CachedAnswer(answer))
            store_ms = (time.perf_counter() - store_t0) * 1000.0
        finally:
            event.set()
            with self._inflight_lock:
                self._inflight.pop(key, None)

        latency_ms = (time.perf_counter() - ask_t0) * 1000.0
        with self._lock:
            self._record_query(question, latency_ms)
        return {
            "hit": False,
            "score": round(float(best_sim), 4) if best_sim >= 0 else None,
            "score_pct": round(float(best_sim) * 100.0, 2) if best_sim >= 0 else None,
            "threshold": self.cache._threshold,  # type: ignore[attr-defined]
            "cached_answer": answer,
            "matched_question": best_question,
            "embed_ms": round(embed_ms, 3),
            "cache_lookup_ms": round(lookup_ms, 3),
            "synth_ms": round(synth_ms, 3),
            "store_ms": round(store_ms, 3),
            "latency_ms": round(latency_ms, 3),
            "would_have_been_ms": self.SIMULATED_LLM_MS,
            "source": "kyro.SemanticCache.store()",
        }

    def seed(self) -> dict[str, Any]:
        """Populate the cache so first-time demo visitors see hits immediately."""
        with self._lock:
            if self._seeded:
                return {"seeded": False, "size": len(self.cache._lru), "reason": "already seeded"}  # type: ignore[attr-defined]
            self._seeded = True

        pairs = [
            ("What is the capital of France?",          "Paris."),
            ("What is your refund policy?",             "Refunds are accepted within 30 days with a receipt."),
            ("How fast does shipping arrive?",          "Free shipping over $50; typically 2–3 business days."),
            ("What is your SLA?",                       "Our SLA is 99.95% uptime measured per quarter."),
            ("How do I install Kyro?",                  "pip install konjoai"),
            ("What language is Kyro written in?",       "Python 3.11+."),
            ("Is Kyro open source?",                    "Yes — Kyro is MIT licensed."),
            ("What does Kyro actually do?",             "It's a production RAG agent with semantic caching and live answer streaming."),
        ]
        for q, a in pairs:
            v = encode(q)
            self.cache.store(q, v, _CachedAnswer(a))
        self._seed_pairs = pairs
        return {
            "seeded": True,
            "size": len(self.cache._lru),  # type: ignore[attr-defined]
            "pairs": [{"question": q, "answer": a} for q, a in pairs],
        }

    def stats(self) -> dict[str, Any]:
        s = self.cache.stats()  # real kyro stats
        with self._lock:
            s.update(
                {
                    "calls_avoided": self._calls_avoided,
                    "time_saved_ms": round(self._time_saved_ms, 1),
                    "time_saved_seconds": round(self._time_saved_ms / 1000.0, 2),
                    "dollars_saved": round(self._dollars_saved, 4),
                    "seeded": self._seeded,
                }
            )
        s["backend"] = "konjoai.cache.SemanticCache (in-memory)"
        s["embedder"] = (
            "sentence-transformers (DEMO_USE_REAL_ENCODER=1)"
            if os.environ.get("DEMO_USE_REAL_ENCODER") == "1"
            else "demo char-trigram (256-d)"
        )
        # List the seeded entries so the UI can show "what's already known".
        with self.cache._lock:  # type: ignore[attr-defined]
            entries = list(self.cache._lru.items())  # type: ignore[attr-defined]
        s["entries"] = [
            {"question": e.question, "hit_count": e.hit_count}
            for _, e in entries
        ]
        return s

    def _record_query(self, question: str, latency_ms: float) -> None:
        """Called inside ``self._lock`` — updates observatory counters."""
        self._total_queries += 1
        self._latency_history.append(round(float(latency_ms), 3))
        for tok in re.findall(r"[a-z']+", question.lower()):
            if len(tok) <= 1 or tok in _STOP:
                continue
            stem = _stem(tok)
            if stem and stem not in _STOP and len(stem) > 1:
                self._top_terms[stem] += 1

    def metrics(self) -> dict[str, Any]:
        """Live observatory feed. Real counters wired to SemanticCache stats.

        Contract (exercised by tests/unit/test_demo_metrics.py):
            cache_hit_rate     float in [0, 1]
            avg_latency_ms     float
            singleflight_ratio float in [0, 1]
            total_queries      int
            top_terms          list[{term, count}]  length <= 10
            latency_history    list[float]          length <= 60
        """
        cache_stats = self.cache.stats()
        with self._lock:
            history = list(self._latency_history)
            avg_latency = sum(history) / len(history) if history else 0.0
            ratio = (
                self._singleflight_collapsed / self._total_queries
                if self._total_queries > 0 else 0.0
            )
            top = [
                {"term": term, "count": count}
                for term, count in self._top_terms.most_common(10)
            ]
            return {
                "cache_hit_rate": float(cache_stats["hit_rate"]),
                "avg_latency_ms": round(avg_latency, 3),
                "singleflight_ratio": round(ratio, 4),
                "total_queries": self._total_queries,
                "singleflight_collapsed": self._singleflight_collapsed,
                "top_terms": top,
                "latency_history": history,
                "cache_size": cache_stats["size"],
                "cache_max_size": cache_stats["max_size"],
                "threshold": cache_stats["threshold"],
                "total_hits": cache_stats["total_hits"],
                "total_misses": cache_stats["total_misses"],
                "calls_avoided": self._calls_avoided,
                "time_saved_ms": round(self._time_saved_ms, 1),
                "dollars_saved": round(self._dollars_saved, 4),
                "history_capacity": self.LATENCY_HISTORY_LEN,
            }

    def reset(self) -> dict[str, Any]:
        self.cache.invalidate()
        with self._lock:
            self._calls_avoided = 0
            self._time_saved_ms = 0.0
            self._dollars_saved = 0.0
            self._seeded = False
            self._seed_pairs = []
            self._total_queries = 0
            self._singleflight_collapsed = 0
            self._latency_history.clear()
            self._top_terms.clear()
        return {"reset": True}


class _CachedAnswer:
    """Minimal stand-in for the production QueryResponse pydantic model."""

    __slots__ = ("answer",)

    def __init__(self, answer: str) -> None:
        self.answer = answer


def synthesise_answer(question: str) -> str:
    """Cheap rule-based "LLM" for the miss path — keeps the demo offline.

    Real deployments wire this slot to ``konjoai.generate.get_generator()``.
    """
    q = question.lower()
    if "capital" in q and ("france" in q or "french" in q):
        return "Paris."
    if "capital" in q and "germany" in q:
        return "Berlin."
    if "capital" in q and ("uk" in q or "england" in q or "britain" in q):
        return "London."
    if "refund" in q or "return" in q or "money" in q:
        return "Refunds are accepted within 30 days with a receipt."
    if "ship" in q or "delivery" in q or "deliver" in q:
        return "Free shipping over $50; typically 2–3 business days."
    if "sla" in q or "uptime" in q:
        return "Our SLA is 99.95% uptime measured per quarter."
    if "install" in q and ("kyro" in q or "konjo" in q):
        return "pip install konjoai"
    if "open source" in q or "license" in q or "licence" in q:
        return "Yes — Kyro is MIT licensed."
    if "language" in q and ("kyro" in q or "konjo" in q):
        return "Python 3.11+."
    if "meaning of life" in q:
        return "42. (Source: Hitchhiker's Guide to the Galaxy.)"
    if "kyro" in q or "konjo" in q:
        return "Kyro is a production RAG agent with semantic caching and live answer streaming."
    return f'I would ask the LLM about "{question}" and remember the answer for next time.'


# ── Corpus index (Sprint K4) ─────────────────────────────────────────────


class CorpusIndex:
    """Tiny in-memory cosine retriever over the demo corpus.

    Files live in ``demo/corpus/<domain>_NN_<slug>.txt``. The first
    underscore-delimited segment of the filename is parsed as the domain
    label so the UI can render documents grouped by topic without a
    separate manifest. Each chunk is embedded with :func:`fast_embed` so
    the demo runs offline; with ``DEMO_USE_REAL_ENCODER=1`` we honour the
    same encoder swap used by the cache demo.
    """

    DEFAULT_DEMO_QUERIES: list[str] = [
        "How do I authenticate API requests?",
        "What is the rate limit policy?",
        "How are warfarin and aspirin interactions managed?",
        "How do I dose acetaminophen for a child?",
        "What are my GDPR rights?",
        "How do I deploy kyro to Kubernetes?",
    ]

    def __init__(self, root: Path) -> None:
        self._root = root
        self._lock = threading.Lock()
        self._loaded = False
        self._docs: list[dict[str, Any]] = []
        self._mat: np.ndarray | None = None  # shape (N, dim), float32

    # ── Loading ─────────────────────────────────────────────────────────

    def _classify_domain(self, filename: str) -> str:
        head = filename.split("_", 1)[0]
        return head if head in {"legal", "medical", "technical"} else "other"

    def _short_title(self, filename: str) -> str:
        stem = Path(filename).stem
        # legal_01_privacy_policy → privacy policy
        parts = stem.split("_", 2)
        slug = parts[2] if len(parts) >= 3 else stem
        return slug.replace("_", " ")

    def load(self) -> dict[str, Any]:
        """Embed every .txt file under the corpus root. Idempotent."""
        with self._lock:
            if self._loaded:
                return self._manifest_locked(reload=False)

            if not self._root.exists():
                return {"loaded": False, "error": f"corpus root missing: {self._root}"}

            paths = sorted(self._root.glob("*.txt"))
            if not paths:
                return {"loaded": False, "error": "corpus is empty"}

            docs: list[dict[str, Any]] = []
            vecs: list[np.ndarray] = []
            for i, path in enumerate(paths):
                text = path.read_text(encoding="utf-8").strip()
                if not text:
                    continue
                vec = encode(text)
                if vec.ndim == 2:
                    vec = vec.reshape(-1)
                vecs.append(vec.astype(np.float32))
                domain = self._classify_domain(path.name)
                docs.append(
                    {
                        "id": i,
                        "filename": path.name,
                        "title": self._short_title(path.name),
                        "domain": domain,
                        "preview": text[:200],
                        "char_count": len(text),
                    }
                )

            self._docs = docs
            self._mat = np.vstack(vecs).astype(np.float32) if vecs else None
            self._loaded = True
            return self._manifest_locked(reload=True)

    def _manifest_locked(self, *, reload: bool) -> dict[str, Any]:
        domains: dict[str, int] = {}
        for d in self._docs:
            domains[d["domain"]] = domains.get(d["domain"], 0) + 1
        return {
            "loaded": True,
            "reloaded": reload,
            "count": len(self._docs),
            "domains": domains,
            "documents": [
                {
                    "id": d["id"],
                    "filename": d["filename"],
                    "title": d["title"],
                    "domain": d["domain"],
                    "preview": d["preview"],
                }
                for d in self._docs
            ],
            "source": "demo/corpus/",
        }

    # ── Search ──────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return top_k ranked hits by cosine similarity."""
        with self._lock:
            if not self._loaded or self._mat is None or not self._docs:
                return []
            mat = self._mat
            docs = self._docs

        top_k = max(1, min(int(top_k or 3), len(docs)))
        q = encode(query)
        if q.ndim == 2:
            q = q.reshape(-1)
        # Vectors are already L2-normalised (encode/fast_embed), so dot = cosine.
        scores = mat @ q.astype(np.float32)
        order = np.argsort(-scores)[:top_k]
        return [
            {
                "rank": rank,
                "doc_id": int(idx),
                "filename": docs[int(idx)]["filename"],
                "title": docs[int(idx)]["title"],
                "domain": docs[int(idx)]["domain"],
                "score": round(float(scores[int(idx)]), 4),
                "score_pct": round(float(scores[int(idx)]) * 100.0, 2),
                "preview": docs[int(idx)]["preview"],
            }
            for rank, idx in enumerate(order)
        ]


# ── HTTP layer ──────────────────────────────────────────────────────────


_state = DemoState()
_corpus = CorpusIndex(Path(__file__).parent / "corpus")
_pipeline = PipelineEngine(Path(__file__).parent / "corpus", encode)
_agent = AgentEngine(_pipeline)
_security = SecurityEngine(encode)
_hyde = HyDEEngine(_pipeline, encode)
_rewrite = RewriteEngine()
_HTML_PATH = Path(__file__).parent / "index.html"
_OBSERVATORY_PATH = Path(__file__).parent / "observatory.html"
_PIPELINE_PATH = Path(__file__).parent / "pipeline.html"
_AGENT_PATH = Path(__file__).parent / "agent.html"
_SECURITY_PATH = Path(__file__).parent / "security.html"
_HYDE_PATH = Path(__file__).parent / "hyde.html"
_REWRITE_PATH = Path(__file__).parent / "rewrite.html"


class Handler(BaseHTTPRequestHandler):
    server_version = "kyro-demo/1.0"

    # CORS + JSON helpers ────────────────────────────────────────────────

    def _send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type")

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, events: Any, pace: float = 0.3) -> None:
        """Stream an iterable of dict events as Server-Sent Events.

        Each event is written as ``data: {json}\\n\\n`` and flushed immediately
        so the browser renders the ReAct loop live. A small pace between events
        keeps the reveal cinematic even on instant localhost retrieval.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")  # disable proxy buffering
        self._send_cors()
        self.end_headers()
        try:
            for i, ev in enumerate(events):
                if i:
                    time.sleep(pace)
                payload = json.dumps(ev, ensure_ascii=False)
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return  # client navigated away mid-stream

    def _read_json(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length", "0") or "0")
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    # Quiet down the per-request access log (one line is plenty).
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        log.info("%s - %s", self.address_string(), fmt % args)

    # Routing ──────────────────────────────────────────────────────────

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        from urllib.parse import parse_qs, urlsplit  # noqa: PLC0415

        split = urlsplit(self.path)
        path = split.path
        params = parse_qs(split.query)
        if path in ("/", "/index.html"):
            return self._serve_html(_HTML_PATH)
        if path in ("/observatory", "/observatory.html"):
            return self._serve_html(_OBSERVATORY_PATH)
        if path in ("/pipeline", "/pipeline.html"):
            return self._serve_html(_PIPELINE_PATH)
        if path in ("/agent", "/agent.html"):
            return self._serve_html(_AGENT_PATH)
        if path in ("/security", "/security.html"):
            return self._serve_html(_SECURITY_PATH)
        if path in ("/hyde", "/hyde.html"):
            return self._serve_html(_HYDE_PATH)
        if path in ("/rewrite", "/rewrite.html"):
            return self._serve_html(_REWRITE_PATH)
        if path == "/api/health":
            return self._send_json(
                {
                    "status": "ok",
                    "service": "kyro-demo-server",
                    "version": "1.1",
                    "real_kyro_cache": True,
                    "real_encoder": os.environ.get("DEMO_USE_REAL_ENCODER") == "1",
                    "corpus_loaded": _corpus._loaded,
                }
            )
        if path == "/api/cache/stats":
            return self._send_json(_state.stats())
        if path == "/api/corpus/load":
            return self._send_json(_corpus.load())
        if path == "/api/corpus/demo":
            # Bound the user-controlled query: max length, ascii-only fallback.
            raw = (params.get("query", [""])[0] or "").strip()
            if len(raw) > 256:
                raw = raw[:256]
            top_k_raw = params.get("top_k", ["3"])[0]
            try:
                top_k = max(1, min(int(top_k_raw), 5))
            except ValueError:
                top_k = 3
            if not _corpus._loaded:
                _corpus.load()
            if not raw:
                # Rotate through a small preset list so a no-arg hit
                # demonstrates retrieval against multiple domains.
                idx = int(time.time()) % len(CorpusIndex.DEFAULT_DEMO_QUERIES)
                raw = CorpusIndex.DEFAULT_DEMO_QUERIES[idx]
            t0 = time.perf_counter()
            hits = _corpus.search(raw, top_k=top_k)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return self._send_json(
                {
                    "query": raw,
                    "top_k": top_k,
                    "results": hits,
                    "latency_ms": round(elapsed_ms, 3),
                    "source": "CorpusIndex.search() — cosine over fast_embed",
                }
            )
        if path == "/api/pipeline/analyze":
            # User-controlled query — bound length; clamp top_k + alpha.
            raw = (params.get("query", [""])[0] or "").strip()
            if len(raw) > 256:
                raw = raw[:256]
            if not raw:
                idx = int(time.time()) % len(CorpusIndex.DEFAULT_DEMO_QUERIES)
                raw = CorpusIndex.DEFAULT_DEMO_QUERIES[idx]
            try:
                top_k = max(1, min(int(params.get("top_k", ["4"])[0]), 6))
            except ValueError:
                top_k = 4
            try:
                alpha = min(1.0, max(0.0, float(params.get("alpha", ["0.7"])[0])))
            except ValueError:
                alpha = 0.7
            return self._send_json(_pipeline.analyze(raw, top_k=top_k, alpha=alpha))
        if path in ("/api/agent/stream", "/api/agent/analyze"):
            question = (params.get("question", [""])[0] or "").strip()[:256]
            if not question:
                idx = int(time.time()) % len(CorpusIndex.DEFAULT_DEMO_QUERIES)
                question = CorpusIndex.DEFAULT_DEMO_QUERIES[idx]
            try:
                max_steps = max(1, min(int(params.get("max_steps", ["4"])[0]), 6))
            except ValueError:
                max_steps = 4
            try:
                top_k = max(1, min(int(params.get("top_k", ["4"])[0]), 6))
            except ValueError:
                top_k = 4
            if path == "/api/agent/stream":
                return self._send_sse(_agent.stream(question, max_steps=max_steps, top_k=top_k))
            return self._send_json(_agent.analyze(question, max_steps=max_steps, top_k=top_k))
        if path == "/api/hyde/analyze":
            raw = (params.get("query", [""])[0] or "").strip()[:256]
            if not raw:
                idx = int(time.time()) % len(CorpusIndex.DEFAULT_DEMO_QUERIES)
                raw = CorpusIndex.DEFAULT_DEMO_QUERIES[idx]
            try:
                top_k = max(1, min(int(params.get("top_k", ["4"])[0]), 6))
            except ValueError:
                top_k = 4
            return self._send_json(_hyde.analyze(raw, top_k=top_k))
        if path == "/api/rewrite/trace":
            raw = (params.get("query", [""])[0] or "").strip()[:256]
            if not raw:
                raw = "Hey, what's the REFUND policy, please??"
            return self._send_json(_rewrite.trace(raw))
        if path == "/api/rewrite/scenario":
            return self._send_json(_rewrite.scenario())
        if path == "/api/security/scenario":
            return self._send_json(_security.scenario())
        if path == "/api/security/stats":
            return self._send_json(_security.stats())
        return self._send_json({"error": f"no route for GET {path}"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/cache/ask":
            body = self._read_json()
            q = str(body.get("question", "")).strip()
            if not q:
                return self._send_json({"error": "question is required"}, status=400)
            return self._send_json(_state.ask(q))
        if path == "/api/cache/probe":
            body = self._read_json()
            q = str(body.get("question", "")).strip()
            if not q:
                return self._send_json({"error": "question is required"}, status=400)
            return self._send_json(_state.probe(q))
        if path == "/api/cache/seed":
            return self._send_json(_state.seed())
        if path == "/api/cache/reset":
            return self._send_json(_state.reset())
        if path == "/api/security/check":
            body = self._read_json()
            question = str(body.get("question", "")).strip()[:512]
            answer = str(body.get("answer", "")).strip()[:4096]
            tenant = (str(body.get("tenant", "playground")).strip() or "playground")[:64]
            return self._send_json(_security.check(question, answer, tenant))
        if path == "/api/security/reset":
            return self._send_json(_security.reset())
        return self._send_json({"error": f"no route for POST {path}"}, status=404)

    # Static HTML ──────────────────────────────────────────────────────

    def _serve_html(self, html_path: Path) -> None:
        self._serve_static(html_path, "text/html; charset=utf-8")

    def _serve_static(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_response(404)
            self._send_cors()
            self.end_headers()
            self.wfile.write(f"{path.name} not found next to server.py".encode())
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._send_cors()
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kyro demo server")
    parser.add_argument("--port", type=int, default=8766, help="port (default 8766)")
    parser.add_argument("--host", default="127.0.0.1", help="host (default 127.0.0.1)")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"

    log.info("──────────────────────────────────────────────────────────────────")
    log.info("Kyro demo server running on %s", url)
    log.info("Real kyro module: %s", SemanticCache.__module__)
    log.info(
        "Embedder: %s",
        "sentence-transformers" if os.environ.get("DEMO_USE_REAL_ENCODER") == "1" else "demo char-trigram (256-d)",
    )
    log.info("Endpoints:")
    log.info("  GET  /                  → demo/index.html")
    log.info("  GET  /observatory       → demo/observatory.html")
    log.info("  GET  /pipeline          → demo/pipeline.html (hybrid retrieval theater)")
    log.info("  GET  /agent             → demo/agent.html (ReAct agent theater)")
    log.info("  GET  /security          → demo/security.html (cache-poisoning guard theater)")
    log.info("  GET  /hyde              → demo/hyde.html (HyDE retrieval theater)")
    log.info("  GET  /rewrite           → demo/rewrite.html (query-rewrite theater)")
    log.info("  GET  /api/health        → liveness")
    log.info("  GET  /api/cache/stats   → real SemanticCache.stats()")
    log.info("  POST /api/cache/ask     → real cosine + lookup, JSON {question}")
    log.info("  POST /api/cache/probe   → read-only score, no mutation")
    log.info("  POST /api/cache/seed    → seed 8 example Q/A pairs")
    log.info("  POST /api/cache/reset   → invalidate the cache")
    log.info("  GET  /api/corpus/load   → load demo/corpus/*.txt into the index")
    log.info("  GET  /api/corpus/demo   → run a preset demo query (?query=...&top_k=N)")
    log.info("  GET  /api/pipeline/analyze → real hybrid pipeline trace (?query=...&top_k=N&alpha=F)")
    log.info("  GET  /api/agent/stream  → real ReAct loop as SSE (?question=...&max_steps=N&top_k=K)")
    log.info("──────────────────────────────────────────────────────────────────")
    log.info("Open %s in your browser. Ctrl-C to stop.", url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
        server.server_close()


if __name__ == "__main__":
    main()
