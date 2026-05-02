from __future__ import annotations

from dataclasses import dataclass

from konjoai.agent.react import RAGAgent
from konjoai.generate.generator import GenerationResult
from konjoai.retrieve.hybrid import HybridResult
from konjoai.retrieve.reranker import RerankResult


@dataclass
class _SeqGenerator:
    responses: list[str]

    def __post_init__(self) -> None:
        self._i = 0

    def generate(self, question: str, context: str) -> GenerationResult:
        _ = (question, context)
        idx = min(self._i, len(self.responses) - 1)
        self._i += 1
        return GenerationResult(
            answer=self.responses[idx],
            model="stub-agent-model",
            usage={"prompt_tokens": 7, "completion_tokens": 3},
        )


def test_agent_retrieve_then_finish(monkeypatch):
    def _hybrid(query: str, top_k_dense: int, top_k_sparse: int):
        _ = (query, top_k_dense, top_k_sparse)
        return [
            HybridResult(
                rrf_score=0.88,
                content="Refund policy: 30 days with receipt.",
                source="policy.md",
                metadata={},
            )
        ]

    def _rerank(query: str, candidates: list[HybridResult], top_k: int):
        _ = query
        c = candidates[0]
        return [
            RerankResult(
                score=0.95,
                content=c.content,
                source=c.source,
                metadata=c.metadata,
            )
        ][:top_k]

    monkeypatch.setattr("konjoai.agent.react.hybrid_search", _hybrid)
    monkeypatch.setattr("konjoai.agent.react.rerank", _rerank)

    gen = _SeqGenerator(
        responses=[
            '{"thought":"Need docs first","action":"retrieve","action_input":"refund policy","final_answer":""}',
            '{"thought":"Enough evidence","action":"finish","action_input":"","final_answer":"Refunds are accepted within 30 days with receipt."}',
        ]
    )

    result = RAGAgent(max_steps=3, top_k=3).run("What is the refund policy?", generator=gen)

    assert result.answer.startswith("Refunds are accepted")
    assert [s.action for s in result.steps] == ["retrieve", "finish"]
    assert result.sources
    assert result.sources[0].source == "policy.md"


def test_agent_invalid_json_falls_back_to_raw_answer(monkeypatch):
    monkeypatch.setattr("konjoai.agent.react.hybrid_search", lambda *args, **kwargs: [])
    monkeypatch.setattr("konjoai.agent.react.rerank", lambda *args, **kwargs: [])

    gen = _SeqGenerator(responses=["plain text answer without JSON"])
    result = RAGAgent(max_steps=2, top_k=2).run("hello", generator=gen)

    assert result.answer == "plain text answer without JSON"
    assert result.steps
    assert result.steps[0].thought == "parser_fallback"
    assert result.steps[0].action == "finish"


def test_agent_max_steps_guard_triggers_direct_generation(monkeypatch):
    def _hybrid(query: str, top_k_dense: int, top_k_sparse: int):
        _ = (query, top_k_dense, top_k_sparse)
        return [
            HybridResult(
                rrf_score=0.5,
                content="Document content",
                source="doc.txt",
                metadata={},
            )
        ]

    def _rerank(query: str, candidates: list[HybridResult], top_k: int):
        _ = query
        c = candidates[0]
        return [RerankResult(score=0.4, content=c.content, source=c.source, metadata=c.metadata)][:top_k]

    monkeypatch.setattr("konjoai.agent.react.hybrid_search", _hybrid)
    monkeypatch.setattr("konjoai.agent.react.rerank", _rerank)

    gen = _SeqGenerator(
        responses=[
            '{"thought":"retrieve","action":"retrieve","action_input":"doc","final_answer":""}',
            "fallback final answer",
        ]
    )
    result = RAGAgent(max_steps=1, top_k=1).run("Question", generator=gen)

    assert result.answer == "fallback final answer"
    assert result.steps[-1].thought == "max_steps_guard"
    assert result.steps[-1].action == "finish"


def test_agent_run_stream_yields_step_then_result(monkeypatch):
    def _hybrid(query, top_k_dense, top_k_sparse):
        _ = (query, top_k_dense, top_k_sparse)
        return [
            HybridResult(
                rrf_score=0.7,
                content="A doc about apples.",
                source="apples.md",
                metadata={},
            )
        ]

    def _rerank(query, candidates, top_k):
        _ = query
        c = candidates[0]
        return [RerankResult(score=0.9, content=c.content, source=c.source, metadata=c.metadata)][:top_k]

    monkeypatch.setattr("konjoai.agent.react.hybrid_search", _hybrid)
    monkeypatch.setattr("konjoai.agent.react.rerank", _rerank)

    gen = _SeqGenerator(
        responses=[
            '{"thought":"need docs","action":"retrieve","action_input":"apples","final_answer":""}',
            '{"thought":"have enough","action":"finish","action_input":"","final_answer":"Apples are red."}',
        ]
    )

    events = list(RAGAgent(max_steps=3, top_k=2).run_stream("Tell me about apples", generator=gen))

    types = [e["type"] for e in events]
    assert types[-1] == "result"
    assert types[:-1] == ["step", "step"]
    # Step events carry the agent trace
    assert events[0]["action"] == "retrieve"
    assert events[0]["index"] == 1
    assert events[1]["action"] == "finish"
    assert events[1]["index"] == 2
    # Result event carries assembled answer + sources
    result = events[-1]
    assert result["answer"] == "Apples are red."
    assert result["model"] == "stub-agent-model"
    assert len(result["sources"]) == 1
    assert result["sources"][0].source == "apples.md"
    assert len(result["steps"]) == 2


def test_agent_run_stream_parser_fallback_emits_result(monkeypatch):
    monkeypatch.setattr("konjoai.agent.react.hybrid_search", lambda *a, **kw: [])
    monkeypatch.setattr("konjoai.agent.react.rerank", lambda *a, **kw: [])

    gen = _SeqGenerator(responses=["not json at all"])
    events = list(RAGAgent(max_steps=2, top_k=2).run_stream("hi", generator=gen))

    assert events[0]["type"] == "step"
    assert events[0]["thought"] == "parser_fallback"
    assert events[-1]["type"] == "result"
    assert events[-1]["answer"] == "not json at all"


def test_agent_run_stream_rejects_empty_question():
    import pytest

    with pytest.raises(ValueError):
        list(RAGAgent(max_steps=1, top_k=1).run_stream("   "))


def test_agent_run_consumes_stream_into_result(monkeypatch):
    """run() must remain semantically identical after refactor."""
    monkeypatch.setattr("konjoai.agent.react.hybrid_search", lambda *a, **kw: [])
    monkeypatch.setattr("konjoai.agent.react.rerank", lambda *a, **kw: [])

    gen = _SeqGenerator(
        responses=[
            '{"thought":"answer directly","action":"finish","action_input":"","final_answer":"42"}',
        ]
    )
    result = RAGAgent(max_steps=2, top_k=2).run("ultimate question", generator=gen)
    assert result.answer == "42"
    assert [s.action for s in result.steps] == ["finish"]
