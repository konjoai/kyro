"""Kyro Agent Theater — drives the *real* bounded ReAct loop, offline.

The reasoning loop here is the production class
:class:`konjoai.agent.react.RAGAgent` — its Thought/Action/Observation control
flow, JSON action parsing, action normalization, max-step guard, tool dispatch,
and step/result event shaping all run unmodified. Two seams make it run offline
in the demo, and both are labelled in the payload so the UI never overclaims:

* **Planner (no LLM).** :class:`DemoPlanner` implements the ``Generator``
  protocol the loop expects and emits valid ReAct JSON actions. Its *plan* is
  real, though — sub-queries come from
  :func:`konjoai.retrieve.router.decompose_query`.
* **Retrieve tool (offline).** ``RAGAgent`` resolves ``hybrid_search`` and
  ``rerank`` from its module globals; for the duration of each run we point
  those at the demo's :class:`~pipeline.PipelineEngine`, so the agent retrieves
  with real BM25 + dense + ``reciprocal_rank_fusion`` over the demo corpus
  instead of a live Qdrant/cross-encoder stack. The originals are restored
  afterwards, so importing this module never disturbs the rest of the suite.

Net result: the agent's *behaviour* — when it retrieves, which documents it
pulls, when it stops — is real code. Only token generation and the storage
backend are stubbed, exactly as a production swap would isolate them.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from pipeline import PipelineEngine

from konjoai.agent import react as _react_mod
from konjoai.agent.react import RAGAgent
from konjoai.generate.generator import GenerationResult
from konjoai.retrieve.hybrid import HybridResult
from konjoai.retrieve.reranker import RerankResult
from konjoai.retrieve.router import decompose_query

__all__ = ["AgentEngine", "DemoPlanner"]

# Serialize the global retrieve-tool swap so concurrent SSE requests on the
# threaded demo server never observe each other's patch window.
_PATCH_LOCK = threading.Lock()


class DemoPlanner:
    """Deterministic ReAct planner (no LLM) implementing the ``Generator`` API.

    Issues one ``retrieve`` per planned sub-query, then ``finish`` with an
    answer grounded in the retrieved context. The plan is real — it comes from
    :func:`konjoai.retrieve.router.decompose_query`.
    """

    MODEL = "demo-planner (rule-based · no LLM)"

    def __init__(self, sub_queries: list[str]) -> None:
        self._plan = sub_queries or [""]
        self._calls = 0

    def generate(self, question: str, context: str) -> GenerationResult:
        """Return the next ReAct action as the JSON the loop parses."""
        idx = self._calls
        self._calls += 1
        if idx < len(self._plan):
            target = self._plan[idx].strip() or question
            obj = {
                "thought": f"Sub-query {idx + 1} of {len(self._plan)}: gather evidence for “{target}”.",
                "action": "retrieve",
                "action_input": target,
                "final_answer": "",
            }
        else:
            obj = {
                "thought": "Enough evidence gathered; composing a grounded answer.",
                "action": "finish",
                "action_input": "",
                "final_answer": _ground_answer(question, context),
            }
        return GenerationResult(
            answer=json.dumps(obj, ensure_ascii=False),
            model=self.MODEL,
            usage={"prompt_tokens": 0, "completion_tokens": 0},
        )


def _ground_answer(question: str, context: str) -> str:
    """Compose a short answer anchored to the top retrieved passage."""
    if not context or context.startswith("No retrieved"):
        return f"I could not retrieve supporting documents for “{question}”."
    first_block = context.split("\n\n---\n\n", 1)[0]
    # Blocks look like ``[source.txt] body…`` — drop the source tag.
    body = first_block.split("] ", 1)[-1].strip()
    if len(body) <= 200:
        return body
    return body[:200].rsplit(" ", 1)[0] + "…"


class AgentEngine:
    """Runs the real ``RAGAgent`` loop over the demo corpus and streams events.

    Parameters
    ----------
    pipeline:
        Loaded :class:`~pipeline.PipelineEngine` providing offline hybrid
        retrieval for the agent's retrieve tool.
    """

    def __init__(self, pipeline: PipelineEngine) -> None:
        self.pipeline = pipeline

    # ── Offline retrieve-tool seam ────────────────────────────────────────────

    def _offline_hybrid_search(
        self,
        query: str,
        top_k_dense: int | None = None,
        top_k_sparse: int | None = None,
        alpha: float | None = None,
        q_vec: Any = None,
    ) -> list[HybridResult]:
        k = top_k_dense or top_k_sparse or 5
        return self.pipeline.hybrid(query, top_k=k, alpha=0.7 if alpha is None else alpha)

    @staticmethod
    def _offline_rerank(query: str, candidates: list, top_k: int | None = None) -> list[RerankResult]:
        # No cross-encoder offline: keep the RRF-fused order as the ranking.
        k = top_k or len(candidates)
        return [
            RerankResult(
                score=float(getattr(c, "rrf_score", getattr(c, "score", 0.0))),
                content=c.content,
                source=c.source,
                metadata=getattr(c, "metadata", {}) or {},
            )
            for c in candidates[:k]
        ]

    @contextmanager
    def _offline_retrieval(self) -> Iterator[None]:
        """Point ``RAGAgent``'s retrieve tool at this engine's pipeline, then
        restore the production functions so the swap never leaks."""
        with _PATCH_LOCK:
            orig_hs = _react_mod.hybrid_search
            orig_rr = _react_mod.rerank
            _react_mod.hybrid_search = self._offline_hybrid_search
            _react_mod.rerank = self._offline_rerank
            try:
                yield
            finally:
                _react_mod.hybrid_search = orig_hs
                _react_mod.rerank = orig_rr

    # ── Public API ────────────────────────────────────────────────────────────

    def stream(self, question: str, max_steps: int = 4, top_k: int = 4) -> Iterator[dict[str, Any]]:
        """Yield UI events for one agent run, live as the loop produces them.

        Event shapes (all consumed by ``agent.html``)::

            {"type": "plan",   "sub_queries": [...], "max_steps": int, "top_k": int}
            {"type": "step",   "index": int, "thought": str, "action": str,
             "action_input": str, "observation": str, "docs": [...], "elapsed_ms": float}
            {"type": "result", "answer": str, "model": str, "step_count": int,
             "sources": [...], "elapsed_ms": float}
        """
        question = (question or "").strip()
        if not question:
            yield {"type": "error", "error": "question must be non-empty"}
            return

        max_steps = max(1, min(int(max_steps), 6))
        top_k = max(1, min(int(top_k), 6))
        plan = decompose_query(question, max_parts=min(3, max(1, max_steps - 1)))
        yield {"type": "plan", "sub_queries": plan, "max_steps": max_steps, "top_k": top_k}

        agent = RAGAgent(max_steps=max_steps, top_k=top_k)
        planner = DemoPlanner(plan)
        t0 = time.perf_counter()
        with self._offline_retrieval():
            for event in agent.run_stream(question, generator=planner):
                elapsed = round((time.perf_counter() - t0) * 1000.0, 2)
                if event["type"] == "step":
                    yield {
                        "type": "step",
                        "index": event["index"],
                        "thought": event["thought"],
                        "action": event["action"],
                        "action_input": event["action_input"],
                        "observation": event["observation"],
                        "docs": self._parse_observation(event["observation"]),
                        "elapsed_ms": elapsed,
                    }
                else:  # result
                    yield {
                        "type": "result",
                        "answer": event["answer"],
                        "model": event["model"],
                        "step_count": len(event["steps"]),
                        "sources": [self._fmt_source(s) for s in event["sources"]],
                        "elapsed_ms": elapsed,
                    }

    def analyze(self, question: str, max_steps: int = 4, top_k: int = 4) -> dict[str, Any]:
        """Collect a full run into one dict (non-streaming clients + tests)."""
        plan: list[str] = []
        steps: list[dict[str, Any]] = []
        result: dict[str, Any] | None = None
        for event in self.stream(question, max_steps=max_steps, top_k=top_k):
            kind = event["type"]
            if kind == "error":
                return event
            if kind == "plan":
                plan = event["sub_queries"]
            elif kind == "step":
                steps.append(event)
            else:
                result = event
        return {
            "question": question,
            "plan": plan,
            "steps": steps,
            "result": result,
            "source": "konjoai.agent.react.RAGAgent (real loop) + offline retrieve tool",
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _parse_observation(self, observation: str) -> list[dict[str, Any]]:
        """Turn a retrieve tool's JSON observation into enriched doc cards."""
        try:
            raw = json.loads(observation)
        except (json.JSONDecodeError, TypeError):
            return []
        if not isinstance(raw, list):
            return []
        cards: list[dict[str, Any]] = []
        for d in raw:
            source = str(d.get("source", ""))
            meta = self.pipeline.describe_source(source)
            cards.append(
                {
                    "source": source,
                    "score": d.get("score"),
                    "title": meta["title"],
                    "domain": meta["domain"],
                    "preview": d.get("preview", meta["preview"]),
                }
            )
        return cards

    def _fmt_source(self, source: RerankResult) -> dict[str, Any]:
        meta = self.pipeline.describe_source(source.source)
        return {
            "source": source.source,
            "score": round(float(source.score), 4),
            "title": meta["title"],
            "domain": meta["domain"],
        }
