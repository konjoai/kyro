from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Iterator

from konjoai.generate.generator import GenerationResult, Generator, get_generator
from konjoai.retrieve.hybrid import HybridResult, hybrid_search
from konjoai.retrieve.reranker import RerankResult, rerank

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    """Single Thought/Action/Observation turn."""

    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class AgentResult:
    """Final result returned by the agent loop."""

    answer: str
    model: str
    usage: dict
    steps: list[AgentStep] = field(default_factory=list)
    sources: list[RerankResult] = field(default_factory=list)


@dataclass
class _ActionPayload:
    thought: str
    action: str
    action_input: str
    final_answer: str


class ToolRegistry:
    """Action registry for the agent loop."""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[[str], tuple[list[RerankResult], str]]] = {}

    def register(self, name: str, fn: Callable[[str], tuple[list[RerankResult], str]]) -> None:
        self._tools[name] = fn

    def actions(self) -> list[str]:
        return sorted(self._tools.keys())

    def invoke(self, name: str, action_input: str) -> tuple[list[RerankResult], str]:
        return self._tools[name](action_input)


class RAGAgent:
    """Minimal ReAct loop for Sprint 14 foundation.

    The model returns a JSON action on each step:

    {"thought":"...","action":"retrieve|finish","action_input":"...","final_answer":"..."}
    """

    def __init__(self, *, max_steps: int = 5, top_k: int = 5) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.max_steps = max_steps
        self.top_k = top_k

    def run(self, question: str, *, generator: Generator | None = None) -> AgentResult:
        """Drive the agent loop and return the final assembled result."""
        result: AgentResult | None = None
        for event in self.run_stream(question, generator=generator):
            if event.get("type") == "result":
                result = AgentResult(
                    answer=event["answer"],
                    model=event["model"],
                    usage=event["usage"],
                    steps=event["steps"],
                    sources=event["sources"],
                )
        if result is None:  # pragma: no cover — run_stream always yields a result
            raise RuntimeError("agent stream terminated without a result event")
        return result

    def run_stream(
        self,
        question: str,
        *,
        generator: Generator | None = None,
    ) -> Iterator[dict]:
        """Drive the ReAct loop, yielding one event dict per step plus a final result.

        Event shapes:
          {"type": "step", "index": int, "thought": str, "action": str,
           "action_input": str, "observation": str}
          {"type": "result", "answer": str, "model": str, "usage": dict,
           "steps": list[AgentStep], "sources": list[RerankResult]}
        """
        if not question.strip():
            raise ValueError("question must be non-empty")

        llm = generator or get_generator()
        registry = self._build_registry(question)
        steps: list[AgentStep] = []
        working_docs: list[RerankResult] = []
        last_generation: GenerationResult | None = None

        def _emit_step(step: AgentStep) -> dict:
            steps.append(step)
            return {
                "type": "step",
                "index": len(steps),
                "thought": step.thought,
                "action": step.action,
                "action_input": step.action_input,
                "observation": step.observation,
            }

        for _ in range(self.max_steps):
            prompt = self._build_prompt(question, steps, registry.actions())
            context = self._format_context(working_docs)
            generation = llm.generate(question=prompt, context=context)
            last_generation = generation

            payload = _parse_action_payload(generation.answer)
            if payload is None:
                yield _emit_step(
                    AgentStep(
                        thought="parser_fallback",
                        action="finish",
                        action_input="",
                        observation="model output was not valid JSON; returning raw answer",
                    )
                )
                yield {
                    "type": "result",
                    "answer": generation.answer.strip(),
                    "model": generation.model,
                    "usage": generation.usage,
                    "steps": steps,
                    "sources": working_docs,
                }
                return

            action = _normalize_action(payload.action)
            if action == "finish":
                answer = payload.final_answer.strip() if payload.final_answer.strip() else generation.answer.strip()
                yield _emit_step(
                    AgentStep(
                        thought=payload.thought,
                        action="finish",
                        action_input=payload.action_input,
                        observation="completed",
                    )
                )
                yield {
                    "type": "result",
                    "answer": answer,
                    "model": generation.model,
                    "usage": generation.usage,
                    "steps": steps,
                    "sources": working_docs,
                }
                return

            if action not in registry.actions():
                yield _emit_step(
                    AgentStep(
                        thought=payload.thought,
                        action=action,
                        action_input=payload.action_input,
                        observation=(
                            f"unknown action '{action}' — expected one of {', '.join(registry.actions())}"
                        ),
                    )
                )
                continue

            docs, observation = registry.invoke(action, payload.action_input)
            if docs:
                working_docs = docs

            yield _emit_step(
                AgentStep(
                    thought=payload.thought,
                    action=action,
                    action_input=payload.action_input,
                    observation=observation,
                )
            )

        # Max-step fallback: perform one final direct generation using the latest docs.
        if not working_docs:
            working_docs, _ = registry.invoke("retrieve", question)

        fallback_context = self._format_context(working_docs)
        fallback = llm.generate(question=question, context=fallback_context)
        yield _emit_step(
            AgentStep(
                thought="max_steps_guard",
                action="finish",
                action_input="",
                observation="max step limit reached; returned direct generation",
            )
        )
        yield {
            "type": "result",
            "answer": fallback.answer.strip(),
            "model": fallback.model,
            "usage": fallback.usage,
            "steps": steps,
            "sources": working_docs,
        }
        _ = last_generation  # retained for parity with prior closure

    def _build_registry(self, question: str) -> ToolRegistry:
        registry = ToolRegistry()

        def _retrieve(action_input: str) -> tuple[list[RerankResult], str]:
            q = action_input.strip() or question
            hybrid: list[HybridResult] = hybrid_search(
                q,
                top_k_dense=self.top_k,
                top_k_sparse=self.top_k,
            )
            reranked: list[RerankResult] = rerank(q, hybrid, top_k=self.top_k)
            if not reranked:
                return [], "retrieve returned 0 documents"

            preview = [
                {
                    "source": d.source,
                    "score": round(float(d.score), 4),
                    "preview": d.content[:120],
                }
                for d in reranked[:3]
            ]
            return reranked, json.dumps(preview, ensure_ascii=True)

        registry.register("retrieve", _retrieve)
        return registry

    def _build_prompt(self, question: str, steps: list[AgentStep], actions: list[str]) -> str:
        if steps:
            trace = "\n".join(
                (
                    f"Step {i}:\n"
                    f"Thought: {s.thought}\n"
                    f"Action: {s.action}\n"
                    f"Input: {s.action_input}\n"
                    f"Observation: {s.observation}"
                )
                for i, s in enumerate(steps, start=1)
            )
        else:
            trace = "None"

        return (
            "You are Kyro Agent. Use a ReAct loop with JSON output only.\n"
            f"Allowed actions: {', '.join(actions)}, finish\n"
            "Return compact JSON with keys: thought, action, action_input, final_answer.\n"
            "If you need documents, choose action='retrieve'.\n"
            "When done, choose action='finish' and set final_answer.\n\n"
            f"Question:\n{question}\n\n"
            f"Previous Trace:\n{trace}"
        )

    def _format_context(self, docs: list[RerankResult]) -> str:
        if not docs:
            return "No retrieved documents yet."
        return "\n\n---\n\n".join(f"[{d.source}] {d.content}" for d in docs)


def _normalize_action(action: str) -> str:
    value = action.strip().lower()
    aliases = {
        "search": "retrieve",
        "lookup": "retrieve",
        "find": "retrieve",
        "answer": "finish",
        "final": "finish",
        "done": "finish",
    }
    return aliases.get(value, value)


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_action_payload(raw: str) -> _ActionPayload | None:
    text = _strip_code_fence(raw)
    candidates = [text]

    # Common case: model wraps JSON in prose; extract the first object.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is not None:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        thought = str(obj.get("thought", "")).strip()
        action = str(obj.get("action", "")).strip()
        action_input = str(obj.get("action_input", "")).strip()
        final_answer = str(obj.get("final_answer", "")).strip()
        if not action:
            continue
        return _ActionPayload(
            thought=thought,
            action=action,
            action_input=action_input,
            final_answer=final_answer,
        )

    logger.debug("RAGAgent: failed to parse action payload from model output: %r", raw[:240])
    return None
