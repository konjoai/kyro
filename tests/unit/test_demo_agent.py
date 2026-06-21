"""Agent Theater contract — ``demo/agent.py`` drives the real ReAct loop.

``demo/agent.html`` streams every field these tests pin down. The headline
guarantees are *realness* and *containment*: the loop is the production
:class:`konjoai.agent.react.RAGAgent`, the retrieve tool runs real BM25 + dense
+ RRF over the corpus, and the offline retrieve-tool swap is restored after
every run so importing this module never disturbs the rest of the suite.

Konjo gates exercised:
  K3 — the loop, decomposition and retrieval are real ``konjoai`` code.
  K6 — the global retrieve-tool patch is fully reverted (no test bleed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from agent import AgentEngine, DemoPlanner  # noqa: E402
from pipeline import PipelineEngine  # noqa: E402

from konjoai.agent import react as react_mod  # noqa: E402
from konjoai.agent.react import _parse_action_payload  # noqa: E402

_DIM = 64


def _embed(text: str) -> np.ndarray:
    v = np.zeros(_DIM, dtype=np.float32)
    for tok in text.lower().split():
        v[hash(tok) % _DIM] += 1.0
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        v[0] = 1.0
        return v
    return (v / n).astype(np.float32)


@pytest.fixture
def engine(tmp_path: Path) -> AgentEngine:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    docs = {
        "technical_01_api_authentication.txt": "API authentication uses bearer tokens and OAuth scopes for requests.",
        "technical_02_rate_limiting.txt": "Rate limiting throttles requests per minute using a token bucket policy.",
        "medical_01_acetaminophen.txt": "Acetaminophen pediatric dosing depends on the child weight in kilograms.",
        "legal_01_privacy_policy.txt": "The privacy policy describes GDPR data subject rights and retention windows.",
    }
    for name, body in docs.items():
        (corpus / name).write_text(body, encoding="utf-8")
    pipe = PipelineEngine(corpus, _embed)
    pipe.load()
    return AgentEngine(pipe)


# ── 1. Full-run shape + real loop ───────────────────────────────────────────


def test_analyze_runs_real_loop_to_an_answer(engine: AgentEngine) -> None:
    out = engine.analyze("How do I authenticate API requests?", max_steps=4, top_k=3)
    assert out.keys() >= {"question", "plan", "steps", "result", "source"}
    assert "RAGAgent" in out["source"]
    assert out["result"] is not None
    assert out["result"]["answer"].strip()
    # No-LLM planner is disclosed in the result model string.
    assert "no llm" in out["result"]["model"].lower()


def test_loop_finishes_within_step_budget(engine: AgentEngine) -> None:
    out = engine.analyze("What are GDPR data subject rights?", max_steps=3, top_k=3)
    assert 1 <= len(out["steps"]) <= 3
    assert out["steps"][-1]["action"] == "finish"
    assert out["result"]["step_count"] == len(out["steps"])


def test_step_events_have_full_shape(engine: AgentEngine) -> None:
    out = engine.analyze("rate limiting policy", max_steps=4, top_k=3)
    for step in out["steps"]:
        assert step.keys() >= {
            "index",
            "thought",
            "action",
            "action_input",
            "observation",
            "docs",
            "elapsed_ms",
        }
        assert step["action"] in {"retrieve", "finish"}
        if step["action"] == "retrieve":
            assert step["docs"], "retrieve step should surface real documents"
            for d in step["docs"]:
                assert d["domain"] in {"legal", "medical", "technical", "other"}


def test_sources_are_real_and_typed(engine: AgentEngine) -> None:
    sources = engine.analyze("pediatric acetaminophen dosing", max_steps=3, top_k=3)["result"]["sources"]
    assert sources
    for s in sources:
        assert s["source"].endswith(".txt")
        assert isinstance(s["score"], float)
        assert s["domain"] in {"legal", "medical", "technical", "other"}


def test_empty_question_yields_error(engine: AgentEngine) -> None:
    assert engine.analyze("   ")["error"]


# ── 2. Planner emits JSON the real parser accepts ───────────────────────────


def test_planner_actions_parse_with_real_react_parser() -> None:
    planner = DemoPlanner(["sub one", "sub two"])
    actions = [_parse_action_payload(planner.generate("q", "ctx").answer) for _ in range(3)]
    assert [a.action for a in actions] == ["retrieve", "retrieve", "finish"]
    assert actions[0].action_input == "sub one"
    assert actions[-1].final_answer  # grounded answer present on finish


def test_planner_emits_valid_json() -> None:
    payload = DemoPlanner(["x"]).generate("q", "ctx").answer
    obj = json.loads(payload)  # must be parseable
    assert obj["action"] == "retrieve"


# ── 3. Containment — the global retrieve-tool swap is restored ──────────────


def test_offline_patch_does_not_leak(engine: AgentEngine) -> None:
    before_hs, before_rr = react_mod.hybrid_search, react_mod.rerank
    engine.analyze("rate limiting", max_steps=2, top_k=2)
    assert react_mod.hybrid_search is before_hs
    assert react_mod.rerank is before_rr
