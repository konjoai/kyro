"""HyDE Theater contract — ``demo/hyde.py`` real before/after retrieval.

``demo/hyde.html`` renders every field these tests pin down. The headline
guarantee is *honesty*: only the hypothesis text is rule-based (labelled); both
the raw query and the hypothesis are embedded and run through the *same* real
dense cosine retrieval, so the before/after scores and rankings are measured.

Konjo gates exercised:
  K3 — retrieval is real konjoai dense cosine; the hypothesis is disclosed.
  K4 — vector stats report the float32 embedding contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from hyde import HyDEEngine, _synthesize_hypothesis  # noqa: E402
from pipeline import PipelineEngine  # noqa: E402

_DIM = 64


def _embed(text: str) -> np.ndarray:
    """Deterministic bag-of-words embedder — denser text → more non-zero dims,
    which is exactly the query→document shift HyDE exploits."""
    v = np.zeros(_DIM, dtype=np.float32)
    for tok in text.lower().split():
        v[hash(tok) % _DIM] += 1.0
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        v[0] = 1.0
        return v
    return (v / n).astype(np.float32)


@pytest.fixture
def engine(tmp_path: Path) -> HyDEEngine:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    docs = {
        "technical_01_api_authentication.txt": "API authentication uses bearer tokens and OAuth scopes for requests.",
        "technical_02_rate_limiting.txt": "Rate limiting throttles requests per minute using a token bucket policy.",
        "legal_01_privacy_policy.txt": "The privacy policy describes GDPR data subject rights, consent, and retention.",
        "medical_01_acetaminophen.txt": "Acetaminophen pediatric dosing depends on the child weight in kilograms.",
    }
    for name, body in docs.items():
        (corpus / name).write_text(body, encoding="utf-8")
    pipe = PipelineEngine(corpus, _embed)
    pipe.load()
    return HyDEEngine(pipe, _embed)


# ── 1. Shape + honesty labels ───────────────────────────────────────────────


def test_analyze_has_full_shape(engine: HyDEEngine) -> None:
    out = engine.analyze("What are my GDPR rights?", top_k=3)
    assert out.keys() >= {
        "question", "hypothesis", "hypothesis_source", "baseline", "hyde",
        "query_vec", "hyde_vec", "comparison", "source",
    }
    assert "no llm" in out["hypothesis_source"].lower()
    assert "hyde" in out["source"].lower()
    assert out["comparison"].keys() >= {
        "baseline_top_score", "hyde_top_score", "delta", "winner", "winner_changed",
    }


def test_empty_question_is_rejected(engine: HyDEEngine) -> None:
    assert engine.analyze("   ")["error"]


# ── 2. Hypothesis is document-shaped (denser than the query) ────────────────


def test_hypothesis_is_denser_than_query(engine: HyDEEngine) -> None:
    out = engine.analyze("What are my GDPR rights?", top_k=3)
    assert out["hyde_vec"]["tokens"] > out["query_vec"]["tokens"]
    assert out["query_vec"]["dtype"] == "float32"
    assert out["hyde_vec"]["dtype"] == "float32"


def test_hypothesis_contains_query_terms(engine: HyDEEngine) -> None:
    hyp = _synthesize_hypothesis("How do I deploy kyro to Kubernetes?")
    assert "deploy" in hyp.lower()
    assert "kubernetes" in hyp.lower()
    assert len(hyp) > 80  # a paragraph, not a query


# ── 3. Retrieval is real and bounded ────────────────────────────────────────


def test_both_runs_are_ranked_and_bounded(engine: HyDEEngine) -> None:
    out = engine.analyze("GDPR data subject rights", top_k=2)
    for rows in (out["baseline"], out["hyde"]):
        assert 1 <= len(rows) <= 2
        assert [r["rank"] for r in rows] == sorted(r["rank"] for r in rows)
        scores = [r["score"] for r in rows]
        assert scores == sorted(scores, reverse=True)


def test_hyde_closes_the_gap(engine: HyDEEngine) -> None:
    """The document-shaped hypothesis should raise the top-document cosine —
    the measured HyDE effect, not a staged one."""
    out = engine.analyze("What are my GDPR rights?", top_k=4)
    c = out["comparison"]
    assert c["hyde_top_score"] >= c["baseline_top_score"]
    assert c["delta"] == pytest.approx(c["hyde_top_score"] - c["baseline_top_score"], abs=1e-6)


def test_comparison_winner_is_a_real_source(engine: HyDEEngine) -> None:
    out = engine.analyze("rate limiting policy", top_k=3)
    assert out["comparison"]["winner"] == out["hyde"][0]["source"]
    assert out["comparison"]["winner"].endswith(".txt")
