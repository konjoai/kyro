from __future__ import annotations

import asyncio
from dataclasses import dataclass

from konjoai.generate.generator import GenerationResult
from konjoai.retrieve.decomposition import (
    AnswerSynthesizer,
    ParallelRetriever,
    QueryDecomposer,
    SubQueryAnswer,
)


@dataclass
class _GeneratorStub:
    answer: str

    def generate(self, question: str, context: str) -> GenerationResult:
        _ = (question, context)
        return GenerationResult(answer=self.answer, model="stub", usage={})


def test_query_decomposer_parses_strict_json():
    gen = _GeneratorStub(
        answer='{"sub_queries": ["Find policy date", "Find policy owner"], "synthesis_hint": "Merge chronologically."}'
    )
    out = QueryDecomposer(gen, max_sub_queries=4).decompose("When and by whom was the policy updated?")

    assert out.used_fallback is False
    assert out.sub_queries == ["Find policy date", "Find policy owner"]
    assert out.synthesis_hint == "Merge chronologically."


def test_query_decomposer_parses_markdown_json_block():
    gen = _GeneratorStub(answer='```json\n{"sub_queries": ["A", "B"], "synthesis_hint": "Use both."}\n```')
    out = QueryDecomposer(gen, max_sub_queries=4).decompose("Compare A and B")

    assert out.used_fallback is False
    assert out.sub_queries == ["A", "B"]


def test_query_decomposer_fallback_when_json_invalid():
    gen = _GeneratorStub(answer="this is not json")
    out = QueryDecomposer(gen, max_sub_queries=3).decompose("compare Python and Rust and Go")

    assert out.used_fallback is True
    assert len(out.sub_queries) >= 1
    assert len(out.sub_queries) <= 3
    assert out.synthesis_hint


def test_query_decomposer_dedupes_and_applies_max_sub_queries():
    gen = _GeneratorStub(answer='{"sub_queries": ["A", "A", "B", "C", "D"], "synthesis_hint": "hint"}')
    out = QueryDecomposer(gen, max_sub_queries=3).decompose("question")

    assert out.sub_queries == ["A", "B", "C"]


def test_parallel_retriever_runs_sync_retrieve_fn():
    async def _run():
        ret = ParallelRetriever()

        def _retrieve(q: str):
            return [q.upper()]

        result = await ret.retrieve(["a", "b"], _retrieve)
        assert result == [["A"], ["B"]]

    asyncio.run(_run())


def test_parallel_retriever_runs_async_retrieve_fn():
    async def _run():
        ret = ParallelRetriever()

        async def _retrieve(q: str):
            return [q + "!"]

        result = await ret.retrieve(["x", "y"], _retrieve)
        assert result == [["x!"], ["y!"]]

    asyncio.run(_run())


def test_answer_synthesizer_builds_final_answer():
    gen = _GeneratorStub(answer="final synthesized")
    synth = AnswerSynthesizer(gen)

    out = synth.synthesize(
        question="What changed?",
        sub_answers=[
            SubQueryAnswer(sub_query="When?", answer="Yesterday"),
            SubQueryAnswer(sub_query="Who?", answer="Platform Team"),
        ],
        synthesis_hint="Prioritize timeline then owner.",
    )

    assert out == "final synthesized"


def test_answer_synthesizer_empty_input_returns_empty_string():
    gen = _GeneratorStub(answer="unused")
    synth = AnswerSynthesizer(gen)

    assert synth.synthesize("Q", [], "hint") == ""
