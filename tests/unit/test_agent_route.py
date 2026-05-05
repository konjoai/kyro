from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from konjoai.agent.react import AgentResult, AgentStep
from konjoai.api.routes.agent import router
from konjoai.retrieve.reranker import RerankResult


@dataclass
class _SettingsTelemetryOn:
    enable_telemetry: bool = True
    request_timeout_seconds: float = 30.0
    audit_enabled: bool = False


@dataclass
class _SettingsTelemetryOff:
    enable_telemetry: bool = False
    request_timeout_seconds: float = 30.0
    audit_enabled: bool = False


@dataclass
class _SettingsTimeout:
    enable_telemetry: bool = True
    request_timeout_seconds: float = 0.01
    audit_enabled: bool = False


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def _sample_result() -> AgentResult:
    return AgentResult(
        answer="Final answer",
        model="stub-agent-model",
        usage={"prompt_tokens": 11, "completion_tokens": 5},
        steps=[
            AgentStep(
                thought="Need documents",
                action="retrieve",
                action_input="refund policy",
                observation="[]",
            ),
            AgentStep(
                thought="Done",
                action="finish",
                action_input="",
                observation="completed",
            ),
        ],
        sources=[
            RerankResult(
                score=0.91,
                content="Refunds are allowed for 30 days.",
                source="policy.md",
                metadata={},
            )
        ],
    )


def test_agent_query_route_returns_agent_result(monkeypatch):
    for var in (
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "GRPC_PROXY",
        "grpc_proxy",
    ):
        monkeypatch.delenv(var, raising=False)

    app = _make_app()
    client = TestClient(app)

    with (
        patch("konjoai.api.routes.agent.get_settings", return_value=_SettingsTelemetryOn()),
        patch("konjoai.api.routes.agent.RAGAgent.run", return_value=_sample_result()) as run_mock,
    ):
        resp = client.post(
            "/agent/query",
            json={"question": "What is refund policy?", "top_k": 3, "max_steps": 4},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert run_mock.called
    assert body["answer"] == "Final answer"
    assert body["sources"][0]["source"] == "policy.md"
    assert body["steps"][0]["action"] == "retrieve"
    assert body["telemetry"] is not None
    assert body["telemetry"]["steps"]["agent"]["max_steps"] == 4


def test_agent_query_route_disables_telemetry_when_off(monkeypatch):
    for var in (
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "GRPC_PROXY",
        "grpc_proxy",
    ):
        monkeypatch.delenv(var, raising=False)

    app = _make_app()
    client = TestClient(app)

    with (
        patch("konjoai.api.routes.agent.get_settings", return_value=_SettingsTelemetryOff()),
        patch("konjoai.api.routes.agent.RAGAgent.run", return_value=_sample_result()),
    ):
        resp = client.post("/agent/query", json={"question": "Q"})

    assert resp.status_code == 200
    assert resp.json()["telemetry"] is None


def test_agent_query_route_returns_504_on_timeout(monkeypatch):
    for var in (
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "GRPC_PROXY",
        "grpc_proxy",
    ):
        monkeypatch.delenv(var, raising=False)

    app = _make_app()
    client = TestClient(app)

    async def _slow_to_thread(_fn, *_args, **_kwargs):
        import asyncio

        await asyncio.sleep(0.05)
        return _sample_result()

    with (
        patch("konjoai.api.routes.agent.get_settings", return_value=_SettingsTimeout()),
        patch("konjoai.api.routes.agent.asyncio.to_thread", _slow_to_thread),
    ):
        resp = client.post("/agent/query", json={"question": "Q", "max_steps": 2})

    assert resp.status_code == 504
    assert "timed out" in resp.json()["detail"]


def _stream_events():
    """Synthetic event sequence the patched RAGAgent.run_stream returns."""
    yield {
        "type": "step",
        "index": 1,
        "thought": "need docs",
        "action": "retrieve",
        "action_input": "policy",
        "observation": "[]",
    }
    yield {
        "type": "step",
        "index": 2,
        "thought": "done",
        "action": "finish",
        "action_input": "",
        "observation": "completed",
    }
    yield {
        "type": "result",
        "answer": "Final streamed answer",
        "model": "stub-agent-model",
        "usage": {"prompt_tokens": 11, "completion_tokens": 5},
        "steps": [
            AgentStep(
                thought="need docs",
                action="retrieve",
                action_input="policy",
                observation="[]",
            ),
            AgentStep(
                thought="done",
                action="finish",
                action_input="",
                observation="completed",
            ),
        ],
        "sources": [
            RerankResult(
                score=0.91,
                content="Refunds are allowed for 30 days.",
                source="policy.md",
                metadata={},
            )
        ],
    }


def _parse_sse_frames(body: bytes) -> list:
    import json as _json

    frames = []
    for line in body.decode("utf-8").splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload.strip() == "[DONE]":
            frames.append("[DONE]")
            continue
        frames.append(_json.loads(payload))
    return frames


def _drop_proxy_env(monkeypatch):
    for var in (
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "GRPC_PROXY",
        "grpc_proxy",
    ):
        monkeypatch.delenv(var, raising=False)


def test_agent_query_stream_emits_step_result_and_done(monkeypatch):
    _drop_proxy_env(monkeypatch)

    app = _make_app()
    client = TestClient(app)

    with (
        patch("konjoai.api.routes.agent.get_settings", return_value=_SettingsTelemetryOn()),
        patch(
            "konjoai.api.routes.agent.RAGAgent.run_stream",
            lambda self, question, generator=None: _stream_events(),
        ),
    ):
        resp = client.post(
            "/agent/query/stream",
            json={"question": "What is refund policy?", "top_k": 3, "max_steps": 4},
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    frames = _parse_sse_frames(resp.content)

    assert frames[-1] == "[DONE]"
    types = [f["type"] for f in frames if isinstance(f, dict)]
    assert types == ["step", "step", "result", "telemetry"]
    # Step frames preserve the agent trace
    assert frames[0]["action"] == "retrieve"
    assert frames[1]["action"] == "finish"
    # Result frame carries serialized sources + steps
    result_frame = frames[2]
    assert result_frame["answer"] == "Final streamed answer"
    assert result_frame["sources"][0]["source"] == "policy.md"
    assert result_frame["sources"][0]["score"] == 0.91
    assert result_frame["steps"][0]["action"] == "retrieve"
    # Telemetry frame is populated when enabled
    telemetry_frame = frames[3]
    assert telemetry_frame["telemetry"] is not None
    assert telemetry_frame["telemetry"]["steps"]["agent_stream"]["max_steps"] == 4


def test_agent_query_stream_omits_telemetry_when_disabled(monkeypatch):
    _drop_proxy_env(monkeypatch)

    app = _make_app()
    client = TestClient(app)

    with (
        patch("konjoai.api.routes.agent.get_settings", return_value=_SettingsTelemetryOff()),
        patch(
            "konjoai.api.routes.agent.RAGAgent.run_stream",
            lambda self, question, generator=None: _stream_events(),
        ),
    ):
        resp = client.post(
            "/agent/query/stream",
            json={"question": "Q"},
        )

    assert resp.status_code == 200
    frames = _parse_sse_frames(resp.content)
    telemetry_frame = next(f for f in frames if isinstance(f, dict) and f["type"] == "telemetry")
    assert telemetry_frame["telemetry"] is None


def test_agent_query_stream_returns_504_on_timeout(monkeypatch):
    _drop_proxy_env(monkeypatch)

    app = _make_app()
    client = TestClient(app)

    async def _slow_wait_for(coro, timeout):
        import asyncio as _aio

        # Cancel the coroutine and surface a TimeoutError unconditionally.
        coro_task = _aio.ensure_future(coro)
        coro_task.cancel()
        try:
            await coro_task
        except (_aio.CancelledError, BaseException):
            pass
        raise _aio.TimeoutError()

    with (
        patch("konjoai.api.routes.agent.get_settings", return_value=_SettingsTimeout()),
        patch("konjoai.api.routes.agent.asyncio.wait_for", _slow_wait_for),
    ):
        resp = client.post("/agent/query/stream", json={"question": "Q"})

    assert resp.status_code == 504
    assert "timed out" in resp.json()["detail"]


def test_agent_query_stream_rejects_empty_question(monkeypatch):
    _drop_proxy_env(monkeypatch)

    app = _make_app()
    client = TestClient(app)

    resp = client.post("/agent/query/stream", json={"question": ""})
    assert resp.status_code == 422  # pydantic min_length validation
