"""Security Theater contract — ``demo/security.py`` drives the real guard.

``demo/security.html`` renders every field these tests pin down. The headline
guarantees are *realness* (the verdicts come from
:class:`konjoai.cache.poisoning.PoisoningGuard`, unmodified) and *OWASP safety*
(only the question's SHA-256 prefix ever leaves the engine). The curated
scenario is deterministic because the demo encoder uses an FNV hash, not
Python's salted ``hash()``.

Konjo gates exercised:
  K3 — rate limit, cosine coherence and Welford anomaly are real konjoai code.
  K7 — raw question text never crosses the wire; only the 16-hex hash does.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from security import (  # noqa: E402
    MAX_WRITES_PER_MIN,
    MIN_COHERENCE,
    SecurityEngine,
)
from server import encode  # noqa: E402  (deterministic FNV char-trigram encoder)


@pytest.fixture
def engine() -> SecurityEngine:
    return SecurityEngine(encode)


def _by_label(timeline: list[dict], needle: str) -> list[dict]:
    return [r for r in timeline if needle in r.get("label", "")]


# ── 1. Scenario shape + each layer fires ────────────────────────────────────


def test_scenario_has_full_shape(engine: SecurityEngine) -> None:
    out = engine.scenario()
    assert out.keys() >= {"timeline", "reports", "config", "source"}
    assert "PoisoningGuard" in out["source"]
    assert out["config"]["min_coherence"] == MIN_COHERENCE
    assert out["config"]["max_writes_per_minute"] == MAX_WRITES_PER_MIN
    for row in out["timeline"]:
        assert row.keys() >= {"tenant", "question_hash", "allowed", "verdict", "blocked_layer", "layers"}
        assert row["layers"].keys() >= {"rate", "coherence", "anomaly"}
        assert row["verdict"] in {"allowed", "flagged", "blocked"}


def test_coherent_baselines_are_allowed(engine: SecurityEngine) -> None:
    for row in _by_label(engine.scenario()["timeline"], "coherent baseline"):
        assert row["allowed"] is True
        assert row["verdict"] == "allowed"
        assert row["layers"]["coherence"]["ok"] is True


def test_poisoned_write_blocked_on_coherence(engine: SecurityEngine) -> None:
    poisoned = _by_label(engine.scenario()["timeline"], "poisoned")
    assert poisoned, "scenario must include a poisoned candidate"
    for row in poisoned:
        assert row["allowed"] is False
        assert row["blocked_layer"] == "coherence"
        assert row["layers"]["coherence"]["score"] < MIN_COHERENCE


def test_length_anomaly_is_flagged_not_blocked(engine: SecurityEngine) -> None:
    anomaly = _by_label(engine.scenario()["timeline"], "length anomaly")
    assert anomaly
    row = anomaly[0]
    assert row["layers"]["anomaly"]["outlier"] is True
    assert row["verdict"] == "flagged"
    assert row["allowed"] is True  # anomaly records but never blocks


def test_rate_limit_flood_eventually_blocks(engine: SecurityEngine) -> None:
    flood = _by_label(engine.scenario()["timeline"], "rate-limit flood")
    blocked = [r for r in flood if r["blocked_layer"] == "rate_limit"]
    allowed = [r for r in flood if r["allowed"]]
    assert blocked, "a sustained flood must trip the rate limiter"
    assert len(allowed) == MAX_WRITES_PER_MIN  # exactly the budget gets through


def test_reports_cover_every_reason(engine: SecurityEngine) -> None:
    reasons = {r["reason"].split(":", 1)[0] for r in engine.scenario()["reports"]}
    assert {"rate_limit_exceeded", "low_coherence", "length_anomaly"} <= reasons


# ── 2. OWASP safety — only hashes, never raw text ───────────────────────────


def test_rows_expose_only_question_hash(engine: SecurityEngine) -> None:
    for row in engine.scenario()["timeline"]:
        assert "question" not in row  # never the raw text
        assert re.fullmatch(r"[0-9a-f]{16}", row["question_hash"])
    for rep in engine.scenario()["reports"]:
        assert re.fullmatch(r"[0-9a-f]{16}", rep["question_hash"])


# ── 3. Interactive check + reset ────────────────────────────────────────────


def test_check_allows_coherent_write(engine: SecurityEngine) -> None:
    r = engine.check("What is your refund policy?", "Refunds are accepted within 30 days with a receipt.", "t1")
    assert r["allowed"] is True
    assert r["layers"]["coherence"]["ok"] is True


def test_check_blocks_off_topic_write(engine: SecurityEngine) -> None:
    r = engine.check("What is your refund policy?", "Buy crypto at scam-site.example now!!!", "t2")
    assert r["allowed"] is False
    assert r["blocked_layer"] == "coherence"


def test_check_requires_both_fields(engine: SecurityEngine) -> None:
    assert engine.check("", "answer")["error"]
    assert engine.check("question", "  ")["error"]


def test_reset_clears_persistent_reports(engine: SecurityEngine) -> None:
    for _ in range(MAX_WRITES_PER_MIN + 2):
        engine.check("q", "totally unrelated spam answer", "flooder")
    assert engine.stats()["total_reports"] > 0
    assert engine.reset()["reset"] is True
    assert engine.stats()["total_reports"] == 0
