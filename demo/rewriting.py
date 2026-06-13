"""Kyro Rewrite Theater — the *real* query normalizer, step by step.

Before a semantic-cache lookup, kyro canonicalizes the query with
:class:`konjoai.cache.rewriter.QueryRewriter` — a deterministic, LLM-free
pipeline (lowercase → expand contractions → strip fillers → strip punctuation →
normalize whitespace → strip trailing '?'). Two differently-phrased questions
that canonicalize to the *same* string share a cache key, so the second one is
a hit instead of a miss — a direct lift to the cache hit-rate.

Everything here is the real production rewriter: :meth:`QueryRewriter.explain`
gives the per-step before/after trace the UI renders, and :meth:`rewrite` is the
exact transform used on the live lookup path. No stubs.
"""
from __future__ import annotations

from typing import Any

from konjoai.cache.rewriter import QueryRewriter

__all__ = ["RewriteEngine"]

# Curated paraphrase clusters — each is a set of phrasings a real user might
# type for the same intent. The scenario shows how many collapse to one key.
_CLUSTERS: list[dict[str, Any]] = [
    {
        "intent": "refund policy",
        "variants": [
            "What is the refund policy?",
            "what's the REFUND policy",
            "What is the refund   policy??",
            "what is the refund policy",
        ],
    },
    {
        "intent": "shipping speed",
        "variants": [
            "How fast is shipping?",
            "how fast is shipping",
            "How fast is SHIPPING??",
        ],
    },
    {
        "intent": "SLA uptime",
        "variants": [
            "What's your SLA?",
            "what is your sla",
            "What is your SLA??",
        ],
    },
]


class RewriteEngine:
    """Traces the real query-rewrite pipeline and the cache-key collapse it buys."""

    def __init__(self) -> None:
        self._rewriter = QueryRewriter()

    @property
    def steps(self) -> list[str]:
        """The ordered names of the real rewrite steps."""
        return list(self._rewriter.step_names)

    def trace(self, question: str) -> dict[str, Any]:
        """Per-step before/after trace for one query (real ``explain``)."""
        question = (question or "").strip()
        if not question:
            return {"error": "question must be non-empty"}
        result = self._rewriter.explain(question)
        steps = [
            {
                "name": s.name,
                "before": s.before,
                "after": s.after,
                "changed": s.before != s.after,
            }
            for s in result.steps
        ]
        return {
            "original": result.original,
            "rewritten": result.rewritten,
            "steps": steps,
            "changed_count": sum(1 for s in steps if s["changed"]),
            "source": "konjoai.cache.rewriter.QueryRewriter.explain (real)",
        }

    def _collapse_cluster(self, intent: str, variants: list[str]) -> dict[str, Any]:
        rows = [{"original": v, "rewritten": self._rewriter.rewrite(v)} for v in variants]
        canonical = sorted({r["rewritten"] for r in rows})
        # Each variant after the first that maps to an already-seen key is a hit.
        seen: set[str] = set()
        hits = 0
        for r in rows:
            if r["rewritten"] in seen:
                hits += 1
            seen.add(r["rewritten"])
        return {
            "intent": intent,
            "rows": rows,
            "variants": len(rows),
            "unique_keys": len(canonical),
            "collapsed": len(canonical) < len(rows),
            "cache_hits_gained": hits,
        }

    def scenario(self) -> dict[str, Any]:
        """Run the curated paraphrase clusters and report the cache-key collapse."""
        clusters = [self._collapse_cluster(c["intent"], c["variants"]) for c in _CLUSTERS]
        total_variants = sum(c["variants"] for c in clusters)
        total_keys = sum(c["unique_keys"] for c in clusters)
        return {
            "steps": self.steps,
            "clusters": clusters,
            "totals": {
                "variants": total_variants,
                "unique_keys": total_keys,
                "keys_saved": total_variants - total_keys,
                "hits_gained": sum(c["cache_hits_gained"] for c in clusters),
            },
            "source": "konjoai.cache.rewriter.QueryRewriter (real normalizer)",
        }
