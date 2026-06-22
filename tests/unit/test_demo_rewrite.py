"""Rewrite Theater contract — ``demo/rewriting.py`` wraps the real normalizer.

``demo/rewrite.html`` renders every field these tests pin down. The guarantee
is *realness*: the per-step trace is :meth:`QueryRewriter.explain` and the
canonical key is :meth:`QueryRewriter.rewrite` — the exact transform on kyro's
cache lookup path, no stubs.

Konjo gates exercised:
  K3 — the rewrite pipeline is real konjoai code, not a re-implementation.
  K6 — adding the demo trace is purely additive over the production rewriter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from rewriting import RewriteEngine  # noqa: E402


@pytest.fixture
def engine() -> RewriteEngine:
    return RewriteEngine()


# ── 1. Per-step trace ───────────────────────────────────────────────────────


def test_trace_has_full_shape(engine: RewriteEngine) -> None:
    out = engine.trace("Hey, what's the REFUND  policy, please??")
    assert out.keys() >= {"original", "rewritten", "steps", "changed_count", "source"}
    assert "QueryRewriter" in out["source"]
    assert out["steps"], "expected per-step trace"
    for s in out["steps"]:
        assert s.keys() >= {"name", "before", "after", "changed"}
        assert s["changed"] is (s["before"] != s["after"])


def test_trace_matches_real_rewrite(engine: RewriteEngine) -> None:
    """The traced final string must equal the real rewrite() on the lookup path."""
    from konjoai.cache.rewriter import QueryRewriter

    q = "Could you EXPLAIN the refund   policy??"
    assert engine.trace(q)["rewritten"] == QueryRewriter().rewrite(q)


def test_changed_count_is_consistent(engine: RewriteEngine) -> None:
    out = engine.trace("WHAT IS your SLA??")
    assert out["changed_count"] == sum(1 for s in out["steps"] if s["changed"])
    assert 0 <= out["changed_count"] <= len(out["steps"])


def test_empty_query_is_rejected(engine: RewriteEngine) -> None:
    assert engine.trace("   ")["error"]


def test_steps_are_the_real_pipeline(engine: RewriteEngine) -> None:
    assert engine.steps == [
        "lowercase",
        "expand_contractions",
        "strip_fillers",
        "strip_punctuation",
        "normalize_whitespace",
        "strip_trailing_question_mark",
    ]


# ── 2. Paraphrase collapse scenario ─────────────────────────────────────────


def test_scenario_collapses_paraphrases(engine: RewriteEngine) -> None:
    out = engine.scenario()
    assert out.keys() >= {"steps", "clusters", "totals", "source"}
    t = out["totals"]
    # The whole point: fewer canonical keys than raw phrasings → hits gained.
    assert t["unique_keys"] < t["variants"]
    assert t["keys_saved"] == t["variants"] - t["unique_keys"]
    assert t["hits_gained"] == t["keys_saved"]


def test_each_cluster_is_internally_consistent(engine: RewriteEngine) -> None:
    for c in engine.scenario()["clusters"]:
        assert c["variants"] == len(c["rows"])
        assert c["unique_keys"] == len({r["rewritten"] for r in c["rows"]})
        assert c["collapsed"] is (c["unique_keys"] < c["variants"])
        assert c["cache_hits_gained"] == c["variants"] - c["unique_keys"]
        for r in c["rows"]:
            assert r.keys() >= {"original", "rewritten"}


def test_collapsed_variants_share_one_key(engine: RewriteEngine) -> None:
    """A cluster of case/whitespace/punctuation variants must canonicalize to 1 key."""
    refund = next(c for c in engine.scenario()["clusters"] if c["intent"] == "refund policy")
    assert refund["unique_keys"] == 1
    assert refund["collapsed"] is True
