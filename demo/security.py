"""Kyro Security Theater — the *real* cache-poisoning guard, layer by layer.

Backs ``demo/security.html`` with :class:`konjoai.cache.poisoning.PoisoningGuard`
run unmodified. A candidate cache *write* (question + answer + tenant) is pushed
through the guard's three production layers and the verdict is reported with the
real signal behind each:

============  ====================================================================
Layer         Real kyro code path · effect
============  ====================================================================
rate limit    ``WriteRateLimiter`` — per-tenant sliding window · BLOCKS
coherence     ``_cosine_similarity(q_vec, embed(answer))`` vs floor · BLOCKS
anomaly       ``AnomalyDetector`` Welford length stats · FLAGS (never blocks)
============  ====================================================================

The authoritative accept/reject is ``PoisoningGuard.validate`` itself; the
per-layer diagnostics the UI renders are read from the guard's own components
and the identical cosine function, so the explanation always matches the
decision. The only demo substitution is the embedder (the deterministic
char-trigram encoder, same float32/L2-unit contract) instead of
sentence-transformers — exactly the ``embed_fn`` seam the guard already exposes.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from konjoai.cache.poisoning import (
    PoisoningGuard,
    PoisoningReportStore,
    _cosine_similarity,
)

__all__ = ["SecurityEngine"]

# Demo-tuned so every layer is observable in a short scenario. Production
# defaults (0.3 / 100 / 3.0) are looser; these make the seams visible.
MIN_COHERENCE = 0.12
MAX_WRITES_PER_MIN = 6
LENGTH_SIGMA = 2.5
MIN_ANOMALY_OBS = 4


class SecurityEngine:
    """Drives the real :class:`PoisoningGuard` and reports per-layer signal.

    Parameters
    ----------
    embed_fn:
        Encoder returning an L2-unit ``float32`` vector — wired into the guard
        as its coherence ``embed_fn`` and reused to embed the question.
    """

    def __init__(self, embed_fn: Callable[[str], np.ndarray]) -> None:
        self._embed = embed_fn
        self._lock = threading.Lock()
        self._guard = self._new_guard()

    @staticmethod
    def _new_guard_with(embed_fn: Callable[[str], np.ndarray]) -> PoisoningGuard:
        return PoisoningGuard(
            min_qa_coherence=MIN_COHERENCE,
            max_writes_per_minute=MAX_WRITES_PER_MIN,
            length_sigma=LENGTH_SIGMA,
            min_anomaly_observations=MIN_ANOMALY_OBS,
            embed_fn=embed_fn,
            report_store=PoisoningReportStore(),
        )

    def _new_guard(self) -> PoisoningGuard:
        return self._new_guard_with(self._embed)

    # ── Single-candidate evaluation ───────────────────────────────────────────

    def _evaluate(self, guard: PoisoningGuard, question: str, answer: str, tenant: str) -> dict[str, Any]:
        """Run one candidate through *guard* and return verdict + layer signal."""
        q_vec = self._embed(question)
        a_vec = self._embed(answer[:2048])
        coherence = round(_cosine_similarity(q_vec, a_vec), 4)

        limiter = guard._rate_limiter  # noqa: SLF001 — demo inspection of real state
        anomaly = guard._anomaly  # noqa: SLF001
        used_before = limiter.current_count(tenant)
        rate_ok = used_before < MAX_WRITES_PER_MIN
        n_obs = anomaly.n_observations
        mean, std = _welford_mean_std(anomaly)
        is_outlier = anomaly.is_length_outlier(answer)

        # Authoritative decision (records reports + advances the real counters).
        allowed = guard.validate(question, q_vec, answer, tenant)

        if not rate_ok:
            verdict, layer = "blocked", "rate_limit"
        elif coherence < MIN_COHERENCE:
            verdict, layer = "blocked", "coherence"
        elif is_outlier:
            verdict, layer = "flagged", "anomaly"
        else:
            verdict, layer = "allowed", "clean"

        return {
            "tenant": tenant,
            "question_hash": hashlib.sha256(question.encode()).hexdigest()[:16],
            "answer_len": len(answer),
            "allowed": allowed,
            "verdict": verdict,
            "blocked_layer": layer,
            "layers": {
                "rate": {
                    "used_before": used_before,
                    "limit": MAX_WRITES_PER_MIN,
                    "ok": rate_ok,
                },
                "coherence": {
                    "score": coherence,
                    "threshold": MIN_COHERENCE,
                    "ok": coherence >= MIN_COHERENCE,
                },
                "anomaly": {
                    "length": len(answer),
                    "mean": round(mean, 1),
                    "std": round(std, 2),
                    "sigma": LENGTH_SIGMA,
                    "observations": n_obs,
                    "outlier": is_outlier,
                    "active": n_obs >= MIN_ANOMALY_OBS,
                },
            },
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def check(self, question: str, answer: str, tenant: str = "playground") -> dict[str, Any]:
        """Evaluate one user-supplied candidate against the persistent guard."""
        question = (question or "").strip()
        answer = (answer or "").strip()
        if not question or not answer:
            return {"error": "question and answer are required"}
        with self._lock:
            return self._evaluate(self._guard, question, answer, tenant)

    def scenario(self) -> dict[str, Any]:
        """Run a curated attack timeline against a *fresh* guard (reproducible).

        Four coherent writes establish a length baseline, then a poisoned
        (off-topic) write, a length-anomaly write, and a rate-limit flood from a
        single bot tenant exercise each layer in turn.
        """
        guard = self._new_guard()
        long_answer = (
            "Refunds are processed within thirty days. " * 60
        ).strip()  # ~2.5 KB — well beyond the baseline length
        candidates: list[tuple[str, str, str, str]] = [
            (
                "acme",
                "What is your refund policy?",
                "Refunds are accepted within 30 days with a receipt.",
                "coherent baseline",
            ),
            (
                "acme",
                "How fast does shipping arrive?",
                "Free shipping over $50; typically 2 to 3 business days.",
                "coherent baseline",
            ),
            ("acme", "What is your SLA?", "Our SLA is 99.95% uptime measured per quarter.", "coherent baseline"),
            (
                "acme",
                "How do I install kyro?",
                "Install kyro with pip install konjoai on Python 3.10+.",
                "coherent baseline",
            ),
            (
                "acme",
                "What is your refund policy?",
                "Buy discount crypto at scam-site.example — limited offer!!!",
                "poisoned · off-topic answer",
            ),
            ("acme", "What is your refund policy?", long_answer, "length anomaly · padded answer"),
            ("flood-bot", "spam", "spam answer", "rate-limit flood"),
            ("flood-bot", "spam", "spam answer", "rate-limit flood"),
            ("flood-bot", "spam", "spam answer", "rate-limit flood"),
            ("flood-bot", "spam", "spam answer", "rate-limit flood"),
            ("flood-bot", "spam", "spam answer", "rate-limit flood"),
            ("flood-bot", "spam", "spam answer", "rate-limit flood"),
            ("flood-bot", "spam", "spam answer", "rate-limit flood"),
            ("flood-bot", "spam", "spam answer", "rate-limit flood"),
        ]
        timeline: list[dict[str, Any]] = []
        for i, (tenant, q, a, label) in enumerate(candidates):
            row = self._evaluate(guard, q, a, tenant)
            row["index"] = i
            row["label"] = label
            timeline.append(row)
        return {
            "timeline": timeline,
            "reports": [
                {"tenant": r.tenant_id, "reason": r.reason, "question_hash": r.question_hash}
                for r in guard._report_store.query(limit=50)  # noqa: SLF001
            ],
            "config": {
                "min_coherence": MIN_COHERENCE,
                "max_writes_per_minute": MAX_WRITES_PER_MIN,
                "length_sigma": LENGTH_SIGMA,
                "min_anomaly_observations": MIN_ANOMALY_OBS,
            },
            "source": "konjoai.cache.poisoning.PoisoningGuard (real three-layer guard)",
        }

    def stats(self) -> dict[str, Any]:
        """Report-store counts + reasons for the persistent interactive guard."""
        store = self._guard._report_store  # noqa: SLF001
        reports = store.query(limit=100)
        reasons: dict[str, int] = {}
        for r in reports:
            key = r.reason.split(":", 1)[0]
            reasons[key] = reasons.get(key, 0) + 1
        return {"total_reports": store.count(), "reasons": reasons}

    def reset(self) -> dict[str, Any]:
        """Rebuild the persistent interactive guard from scratch."""
        with self._lock:
            self._guard = self._new_guard()
        return {"reset": True}


def _welford_mean_std(anomaly: Any) -> tuple[float, float]:
    """Read the real Welford running mean/std out of an ``AnomalyDetector``."""
    n = anomaly._n  # noqa: SLF001 — demo inspection of real running stats
    if n < 1:
        return 0.0, 0.0
    mean = anomaly._mean  # noqa: SLF001
    std = (anomaly._m2 / n) ** 0.5  # noqa: SLF001
    return float(mean), float(std)
