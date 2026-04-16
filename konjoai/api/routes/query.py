from __future__ import annotations

import json
import logging
from typing import Generator as IterGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from konjoai.api.schemas import QueryRequest, QueryResponse, SourceDoc
from konjoai.config import get_settings
from konjoai.telemetry import PipelineTelemetry, timed

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:  # noqa: C901
    """Run the full RAG pipeline: route → (HyDE) → hybrid_search → rerank → generate.

    Pipeline steps (each wrapped in timed() when telemetry is enabled):
        1. route       — classify intent; CHAT short-circuits immediately.
        2. hyde        — (optional) replace query embedding with hypothesis embedding.
        3. hybrid_search — dense + BM25 retrieval with RRF fusion.
        4. rerank      — cross-encoder reranking.
        5. generate    — LLM answer synthesis.
    """
    from konjoai.generate.generator import get_generator
    from konjoai.retrieve.hybrid import HybridResult, hybrid_search
    from konjoai.retrieve.hyde import hyde_encode
    from konjoai.retrieve.reranker import rerank
    from konjoai.retrieve.router import QueryIntent, classify_intent, decompose_query

    settings = get_settings()
    tel = PipelineTelemetry()

    # ── Step 1: Intent routing ────────────────────────────────────────────────
    intent = QueryIntent.RETRIEVAL
    if settings.enable_query_router:
        with timed(tel, "route"):
            intent = classify_intent(req.question)

    if intent == QueryIntent.CHAT:
        return QueryResponse(
            answer="I'm KonjoOS, a retrieval-augmented generation assistant. "
                   "Ask me a question about your documents!",
            sources=[],
            model="router",
            usage={},
            telemetry=tel.as_dict() if settings.enable_telemetry else None,
            intent=intent.value,
        )

    # ── Step 2: Query embedding (HyDE or raw) ────────────────────────────────
    effective_question = req.question
    hypothesis: str | None = None

    if req.use_hyde or settings.enable_hyde:
        with timed(tel, "hyde"):
            _, hypothesis = hyde_encode(req.question)
        effective_question = hypothesis or req.question
        logger.debug("HyDE hypothesis: %r", effective_question[:120])

    # ── Step 3: Hybrid retrieval ──────────────────────────────────────────────
    if intent == QueryIntent.AGGREGATION:
        with timed(tel, "decompose"):
            sub_questions = decompose_query(effective_question)
        logger.debug("AGGREGATION fan-out: %d sub-questions", len(sub_questions))
        all_results: list[HybridResult] = []
        for i, sq in enumerate(sub_questions):
            with timed(tel, "hybrid_search",
                       sub=i, top_k_dense=settings.top_k_dense,
                       top_k_sparse=settings.top_k_sparse):
                all_results.extend(hybrid_search(sq))
        seen: dict[str, HybridResult] = {}
        for r in all_results:
            if r.content not in seen or r.rrf_score > seen[r.content].rrf_score:
                seen[r.content] = r
        hybrid_results: list[HybridResult] = sorted(
            seen.values(), key=lambda x: x.rrf_score, reverse=True
        )
    else:
        with timed(
            tel,
            "hybrid_search",
            top_k_dense=settings.top_k_dense,
            top_k_sparse=settings.top_k_sparse,
        ):
            hybrid_results = hybrid_search(effective_question)

    # ── Step 4: Cross-encoder reranking ──────────────────────────────────────
    with timed(tel, "rerank", top_k=req.top_k):
        reranked = rerank(req.question, hybrid_results, top_k=req.top_k)

    # ── Step 5: Generation ───────────────────────────────────────────────────
    context = "\n\n---\n\n".join(r.content for r in reranked)
    generator = get_generator()

    with timed(tel, "generate", model=settings.openai_model):
        result = generator.generate(question=req.question, context=context)

    sources = [
        SourceDoc(
            source=r.source,
            content_preview=r.content[:200],
            score=float(r.score),
        )
        for r in reranked
    ]

    return QueryResponse(
        answer=result.answer,
        sources=sources,
        model=result.model,
        usage=result.usage,
        telemetry=tel.as_dict() if settings.enable_telemetry else None,
        intent=intent.value,
    )


@router.post("/stream")
def query_stream(req: QueryRequest) -> StreamingResponse:  # noqa: C901
    """SSE streaming version of the RAG pipeline.

    Emits Server-Sent Events in the format::

        data: {"token": "...", "done": false}\n\n

    Followed by a final frame::

        data: {"token": "", "done": true, "model": "...",
               "sources": [...], "intent": "..."}\n\n

    The pipeline steps (intent → HyDE → hybrid_search → rerank) run
    synchronously before streaming starts so that source metadata is
    available in the final frame.
    """
    from konjoai.generate.generator import get_generator
    from konjoai.retrieve.hybrid import HybridResult, hybrid_search
    from konjoai.retrieve.hyde import hyde_encode
    from konjoai.retrieve.reranker import rerank
    from konjoai.retrieve.router import QueryIntent, classify_intent

    settings = get_settings()

    # ── Intent routing ────────────────────────────────────────────────────
    intent = QueryIntent.RETRIEVAL
    if settings.enable_query_router:
        intent = classify_intent(req.question)

    if intent == QueryIntent.CHAT:

        def _chat_stream() -> IterGenerator[str, None, None]:
            chat_answer = (
                "I'm KonjoOS, a retrieval-augmented generation assistant. "
                "Ask me a question about your documents!"
            )
            for token in chat_answer.split():
                yield f"data: {json.dumps({'token': token + ' ', 'done': False})}\n\n"
            yield f"data: {json.dumps({'token': '', 'done': True, 'model': 'router', 'sources': [], 'intent': intent.value})}\n\n"

        return StreamingResponse(_chat_stream(), media_type="text/event-stream")

    # ── HyDE (optional) ───────────────────────────────────────────────────
    effective_question = req.question
    if req.use_hyde or settings.enable_hyde:
        _, hypothesis = hyde_encode(req.question)
        effective_question = hypothesis or req.question

    # ── Hybrid retrieval + rerank ─────────────────────────────────────────
    hybrid_results = hybrid_search(effective_question)
    reranked = rerank(req.question, hybrid_results, top_k=req.top_k)
    context = "\n\n---\n\n".join(r.content for r in reranked)

    sources = [
        SourceDoc(
            source=r.source,
            content_preview=r.content[:200],
            score=float(r.score),
        ).model_dump()
        for r in reranked
    ]

    generator = get_generator()

    def _stream_tokens() -> IterGenerator[str, None, None]:
        model_name = "unknown"
        if hasattr(generator, "generate_stream"):
            for token in generator.generate_stream(question=req.question, context=context):
                if token:
                    yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
            # Try to get model name for final frame
            model_name = getattr(generator, "_model", "unknown")
        else:
            # Fallback: call generate() synchronously and emit all at once
            result = generator.generate(question=req.question, context=context)
            if result.answer:
                yield f"data: {json.dumps({'token': result.answer, 'done': False})}\n\n"
            model_name = result.model

        yield f"data: {json.dumps({'token': '', 'done': True, 'model': model_name, 'sources': sources, 'intent': intent.value})}\n\n"

    return StreamingResponse(_stream_tokens(), media_type="text/event-stream")
