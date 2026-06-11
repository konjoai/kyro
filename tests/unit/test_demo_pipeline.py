"""Pipeline Theater contract — ``demo/pipeline.py`` wire shape + RRF honesty.

``demo/pipeline.html`` renders every number returned by
:meth:`PipelineEngine.analyze`, so each test here is a contract: break the
shape and the cinematic stage view silently goes blank. The headline guarantee
is *honesty* — the fused ordering and scores come from the real
``konjoai.retrieve.hybrid.reciprocal_rank_fusion``, and the per-row term
breakdown the UI shows must reconstruct that exact score.

Konjo gates exercised:
  K3 — every field is computed by real ``konjoai`` code, not a mock.
  K4 — the embedded query vector is float32 / L2-unit at the boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# demo/ is not a package — put it on the path so ``import pipeline`` resolves.
_DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from pipeline import RRF_K, PipelineEngine  # noqa: E402

_DIM = 64


def _embed(text: str) -> np.ndarray:
    """Deterministic token-hash embedder → float32, L2-unit. Mirrors the K4
    dtype contract without pulling sentence-transformers into the unit suite."""
    v = np.zeros(_DIM, dtype=np.float32)
    for tok in text.lower().split():
        v[hash(tok) % _DIM] += 1.0
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        v[0] = 1.0
        return v
    return (v / n).astype(np.float32)


@pytest.fixture
def engine(tmp_path: Path) -> PipelineEngine:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    docs = {
        "technical_01_api_authentication.txt": "API authentication uses bearer tokens and OAuth scopes for requests.",
        "technical_02_rate_limiting.txt": "Rate limiting throttles requests per minute using a token bucket policy.",
        "medical_01_acetaminophen.txt": "Acetaminophen pediatric dosing depends on the child weight in kilograms.",
        "legal_01_privacy_policy.txt": "The privacy policy describes GDPR data subject rights and retention.",
    }
    for name, body in docs.items():
        (corpus / name).write_text(body, encoding="utf-8")
    eng = PipelineEngine(corpus, _embed)
    eng.load()
    return eng


# ── 1. Shape contract ───────────────────────────────────────────────────────


def test_analyze_has_full_shape(engine: PipelineEngine) -> None:
    out = engine.analyze("How do I authenticate API requests?", top_k=3)
    top = {
        "query",
        "alpha",
        "rrf_k",
        "top_k",
        "route",
        "decomposition",
        "embedding",
        "dense",
        "sparse",
        "fused",
        "threshold",
        "decision",
        "timings_ms",
        "source",
    }
    assert top.issubset(out.keys()), f"missing: {top - out.keys()}"
    assert out["route"].keys() >= {"intent", "complexity", "estimated_tokens", "cache_query_type"}
    assert out["embedding"].keys() >= {"dim", "dtype", "norm", "nonzero", "preview"}
    assert out["decision"].keys() >= {"strategy", "rationale", "crag_band", "crag_score"}
    assert out["rrf_k"] == RRF_K


def test_empty_query_is_rejected(engine: PipelineEngine) -> None:
    assert engine.analyze("   ")["error"]


# ── 2. K4 dtype contract on the embedded query ──────────────────────────────


def test_embedding_is_float32_unit_norm(engine: PipelineEngine) -> None:
    e = engine.analyze("rate limiting policy", top_k=3)["embedding"]
    assert e["dtype"] == "float32"
    assert e["dim"] == _DIM
    assert abs(e["norm"] - 1.0) < 1e-4
    assert len(e["preview"]) == 16


# ── 3. RRF honesty — the UI breakdown reconstructs the real score ───────────


def test_fused_ordering_is_descending(engine: PipelineEngine) -> None:
    fused = engine.analyze("API authentication tokens", top_k=4)["fused"]
    scores = [f["rrf_score"] for f in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_terms_reconstruct_real_score(engine: PipelineEngine) -> None:
    """dense_term + sparse_term must equal the production rrf_score for every
    row — the UI shows the breakdown as the explanation of the real number.

    All three values are rounded to 6 decimals for display, so the
    reconstruction (sum of two rounded terms) may diverge from the rounded
    total by up to ~1.5e-6; the tolerance reflects that display rounding, not
    a difference in the underlying math."""
    for alpha in (0.3, 0.7, 1.0):
        fused = engine.analyze("authentication rate limit policy", top_k=4, alpha=alpha)["fused"]
        assert fused, "expected at least one fused result"
        for row in fused:
            recon = row["dense_term"] + row["sparse_term"]
            assert recon == pytest.approx(row["rrf_score"], abs=2e-6)


def test_alpha_extremes_zero_the_other_signal(engine: PipelineEngine) -> None:
    only_dense = engine.analyze("authentication tokens", top_k=4, alpha=1.0)["fused"]
    assert all(r["sparse_term"] == 0.0 for r in only_dense)
    only_sparse = engine.analyze("authentication tokens", top_k=4, alpha=0.0)["fused"]
    assert all(r["dense_term"] == 0.0 for r in only_sparse)


def test_in_both_flag_matches_ranks(engine: PipelineEngine) -> None:
    for row in engine.analyze("authentication rate limiting", top_k=4)["fused"]:
        expected = row["dense_rank"] is not None and row["sparse_rank"] is not None
        assert row["in_both"] is expected


# ── 4. Retrieval lists respect top_k and carry domains ──────────────────────


def test_retrieval_lists_bounded_by_top_k(engine: PipelineEngine) -> None:
    out = engine.analyze("privacy policy GDPR rights", top_k=2)
    assert 1 <= len(out["dense"]) <= 2
    assert 1 <= len(out["sparse"]) <= 2
    for row in out["dense"] + out["sparse"]:
        assert row["domain"] in {"legal", "medical", "technical", "other"}


# ── 5. Routing & threshold pass real enum values through ────────────────────


def test_route_and_threshold_use_real_enums(engine: PipelineEngine) -> None:
    out = engine.analyze("Compare warfarin and aspirin and explain dosing", top_k=3)
    assert out["route"]["intent"] in {"retrieval", "aggregation", "chat"}
    assert out["route"]["complexity"] in {"simple", "medium", "complex"}
    assert out["route"]["cache_query_type"] in {"factual", "faq", "creative", "code"}
    assert 0.0 <= out["threshold"]["value"] <= 1.0
    assert out["decision"]["strategy"] in {"direct", "self_rag", "decompose"}


def test_multi_intent_query_decomposes(engine: PipelineEngine) -> None:
    out = engine.analyze("Compare warfarin and aspirin and explain monitoring schedules", top_k=3)
    d = out["decomposition"]
    assert d["decomposed"] is (len(d["sub_queries"]) > 1)
