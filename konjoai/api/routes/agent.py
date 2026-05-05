from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from konjoai.agent.react import RAGAgent
from konjoai.api.schemas import SourceDoc
from konjoai.audit.models import AGENT_QUERY, AuditEvent, hash_text
from konjoai.config import get_settings
from konjoai.telemetry import PipelineTelemetry, timed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentStepResponse(BaseModel):
    thought: str
    action: str
    action_input: str
    observation: str


class AgentQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    max_steps: int = Field(5, ge=1, le=20)


class AgentQueryResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]
    model: str
    usage: dict
    steps: list[AgentStepResponse]
    telemetry: dict | None = None


@router.post("/query", response_model=AgentQueryResponse)
async def agent_query(req: AgentQueryRequest) -> AgentQueryResponse:
    """Run a bounded ReAct loop over Kyro retrieval tools."""
    settings = get_settings()
    tel = PipelineTelemetry()
    timeout_seconds = float(settings.request_timeout_seconds)

    try:
        with timed(tel, "agent", top_k=req.top_k, max_steps=req.max_steps):
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    RAGAgent(top_k=req.top_k, max_steps=req.max_steps).run,
                    req.question,
                ),
                timeout=timeout_seconds,
            )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"agent request timed out after {timeout_seconds:.2f}s",
        ) from exc

    sources = [
        SourceDoc(
            source=s.source,
            content_preview=s.content[:200],
            score=float(s.score),
        )
        for s in result.sources
    ]

    response = AgentQueryResponse(
        answer=result.answer,
        sources=sources,
        model=result.model,
        usage=result.usage,
        steps=[
            AgentStepResponse(
                thought=step.thought,
                action=step.action,
                action_input=step.action_input,
                observation=step.observation,
            )
            for step in result.steps
        ],
        telemetry=tel.as_dict() if settings.enable_telemetry else None,
    )

    # ── Audit log (Sprint 24; K3: no-op when audit_enabled=False) ────────────
    if settings.audit_enabled:
        from datetime import datetime, timezone
        from konjoai.audit import get_audit_logger
        _latency = sum(t.elapsed_ms for t in tel.steps) if tel.steps else 0.0
        get_audit_logger().log(AuditEvent(
            event_type=AGENT_QUERY,
            timestamp=datetime.now(timezone.utc).isoformat(),
            endpoint="/agent/query",
            status_code=200,
            latency_ms=_latency,
            question_hash=hash_text(req.question),
            result_count=len(sources),
        ))

    return response


@router.post("/query/stream")
async def agent_query_stream(req: AgentQueryRequest) -> StreamingResponse:
    """SSE streaming version of the bounded ReAct loop.

    Emits one ``data:`` frame per ReAct step plus a final ``result`` frame and
    a terminal ``[DONE]`` sentinel. Each frame is a JSON object with a ``type``
    discriminator (``"step"`` or ``"result"``) so clients can branch cleanly.
    Telemetry is attached to the ``result`` frame when enabled.
    """
    settings = get_settings()
    tel = PipelineTelemetry()
    timeout_seconds = float(settings.request_timeout_seconds)

    async def _produce_events() -> AsyncIterator[dict]:
        loop = asyncio.get_running_loop()
        agent = RAGAgent(top_k=req.top_k, max_steps=req.max_steps)
        queue: asyncio.Queue[object] = asyncio.Queue()
        sentinel = object()
        error_holder: dict[str, BaseException] = {}

        def _drive() -> None:
            try:
                for event in agent.run_stream(req.question):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except BaseException as exc:  # noqa: BLE001 — propagate to async side
                error_holder["error"] = exc
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        worker = asyncio.create_task(asyncio.to_thread(_drive))
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                yield item  # type: ignore[misc]
            if "error" in error_holder:
                raise error_holder["error"]
        finally:
            await worker

    async def _stream_body() -> AsyncIterator[bytes]:
        try:
            with timed(tel, "agent_stream", top_k=req.top_k, max_steps=req.max_steps):
                async for event in _produce_events():
                    if event.get("type") == "result":
                        sources_payload = [
                            {
                                "source": s.source,
                                "content_preview": s.content[:200],
                                "score": float(s.score),
                            }
                            for s in event["sources"]
                        ]
                        steps_payload = [
                            {
                                "thought": s.thought,
                                "action": s.action,
                                "action_input": s.action_input,
                                "observation": s.observation,
                            }
                            for s in event["steps"]
                        ]
                        frame = {
                            "type": "result",
                            "answer": event["answer"],
                            "model": event["model"],
                            "usage": event["usage"],
                            "steps": steps_payload,
                            "sources": sources_payload,
                        }
                    else:
                        frame = event
                    yield f"data: {json.dumps(frame, ensure_ascii=False)}\n\n".encode("utf-8")

            telemetry_frame = {
                "type": "telemetry",
                "telemetry": tel.as_dict() if settings.enable_telemetry else None,
            }
            yield f"data: {json.dumps(telemetry_frame, ensure_ascii=False)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"
        except asyncio.CancelledError:  # pragma: no cover — client disconnect
            raise

    async def _execute_with_timeout() -> StreamingResponse:
        # Materialize the generator inside a timeout-bounded coroutine so
        # producer-side bugs surface as 504 rather than mid-stream errors.
        return StreamingResponse(_stream_body(), media_type="text/event-stream")

    try:
        return await asyncio.wait_for(_execute_with_timeout(), timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        logger.warning("agent/query/stream timed out after %.2fs", timeout_seconds)
        raise HTTPException(
            status_code=504,
            detail=f"agent stream request timed out after {timeout_seconds:.2f}s",
        ) from exc
