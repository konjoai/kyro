"""Query decomposition: split a question into sub-queries, retrieve in parallel, and synthesize."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Protocol

from konjoai.generate.generator import Generator
from konjoai.retrieve.router import decompose_query


@dataclass
class DecompositionPlan:
    """Sub-queries and synthesis guidance produced by decomposing a question."""

    sub_queries: list[str]
    synthesis_hint: str
    used_fallback: bool = False


@dataclass
class SubQueryAnswer:
    """A single sub-query paired with its answer."""

    sub_query: str
    answer: str


class RetrievalFn(Protocol):
    """Callable that retrieves results for a query, sync or async."""

    def __call__(self, query: str) -> list[Any] | Awaitable[list[Any]]: ...


class QueryDecomposer:
    """LLM-backed decomposer with deterministic fallback for Sprint 13.

    The model is asked for strict JSON:
    {
      "sub_queries": ["..."],
      "synthesis_hint": "..."
    }

    If parsing fails, we fall back to regex decomposition from the intent router.
    """

    def __init__(self, generator: Generator, max_sub_queries: int = 4) -> None:
        self._generator = generator
        self._max_sub_queries = max(1, max_sub_queries)

    def decompose(self, question: str) -> DecompositionPlan:
        """Produce a decomposition plan via the LLM, falling back to regex on parse failure."""
        if not question.strip():
            raise ValueError("question must be non-empty")

        prompt = (
            "Return STRICT JSON with keys sub_queries and synthesis_hint. "
            "Do not include prose, markdown, or code fences. "
            "Rules: produce 2-4 atomic sub-queries that together answer the user question; "
            "sub_queries must be specific and retrieval-friendly."
        )
        result = self._generator.generate(question=prompt, context=question)
        parsed = self._parse(result.answer)
        if parsed is not None:
            return parsed

        fallback_sub_queries = decompose_query(question, max_parts=self._max_sub_queries)
        return DecompositionPlan(
            sub_queries=fallback_sub_queries,
            synthesis_hint="Combine the sub-answers into one coherent response and preserve key constraints.",
            used_fallback=True,
        )

    def _parse(self, raw: str) -> DecompositionPlan | None:
        """Parse and validate the model's JSON into a plan, or None if invalid."""
        payload = self._extract_json(raw)
        if payload is None:
            return None

        sub_queries_raw = payload.get("sub_queries")
        synthesis_hint_raw = payload.get("synthesis_hint")
        if not isinstance(sub_queries_raw, list):
            return None
        if not isinstance(synthesis_hint_raw, str):
            return None

        cleaned: list[str] = []
        seen: set[str] = set()
        for item in sub_queries_raw:
            if not isinstance(item, str):
                continue
            candidate = item.strip()
            if not candidate:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            cleaned.append(candidate)
            if len(cleaned) >= self._max_sub_queries:
                break

        if not cleaned:
            return None

        synthesis_hint = synthesis_hint_raw.strip()
        if not synthesis_hint:
            synthesis_hint = "Combine the sub-answers into one coherent response."

        return DecompositionPlan(sub_queries=cleaned, synthesis_hint=synthesis_hint, used_fallback=False)

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any] | None:
        """Extract a JSON object from raw model output, stripping code fences."""
        text = raw.strip()

        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        candidates: list[str] = [text]

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start : end + 1])

        for candidate in candidates:
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
        return None


class ParallelRetriever:
    """Execute sub-query retrieval concurrently with asyncio.gather()."""

    async def retrieve(self, sub_queries: list[str], retrieve_fn: RetrievalFn) -> list[list[Any]]:
        """Retrieve results for all sub-queries concurrently, preserving order."""
        tasks: list[Awaitable[list[Any]]] = []
        for query in sub_queries:
            result = retrieve_fn(query)
            if asyncio.iscoroutine(result):
                tasks.append(result)
            else:
                tasks.append(asyncio.to_thread(lambda r=result: r))
        return await asyncio.gather(*tasks)


class AnswerSynthesizer:
    """Compose final answer from decomposed sub-answers."""

    def __init__(self, generator: Generator) -> None:
        self._generator = generator

    def synthesize(
        self,
        question: str,
        sub_answers: list[SubQueryAnswer],
        synthesis_hint: str,
    ) -> str:
        """Synthesize sub-answers into one final answer, or empty string if none."""
        if not sub_answers:
            return ""

        context_parts = []
        for idx, item in enumerate(sub_answers, start=1):
            context_parts.append(f"[{idx}] Sub-query: {item.sub_query}\n[{idx}] Sub-answer: {item.answer}")
        context = "\n\n".join(context_parts)

        synthesis_question = (
            f"Original question: {question}\n"
            f"Synthesis hint: {synthesis_hint}\n"
            "Produce one final answer that integrates all sub-answers. "
            "Resolve overlaps, avoid contradictions, and keep factual grounding."
        )

        return self._generator.generate(question=synthesis_question, context=context).answer
