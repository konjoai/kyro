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
    """Run the full RAG pipeline: route → (HyDE) → hybrid_search → (CRAG) → rerank → generate → (Self-RAG).

    Pipeline steps (each wrapped in timed() when telemetry is enabled):
        1. route         — classify intent; CHAT short-circuits immediately.
        2. hyde          — (optional) replace query embedding with hypothesis embedding.
        2b. embed+cache  — embed once; check semantic cache before hitting Qdrant.
        3. hybrid_search — dense + BM25 retrieval with RRF fusion.
        3b. crag         — (optional) relevance grading + corrective filter.
        4. rerank        — cross-encoder reranking.
        4.5 colbert      — (optional) MaxSim late-interaction re-scoring.
        5. generate      — LLM answer synthesis.
        5b. self_rag     — (optional) reflection critique; retries if unsupported.
    """
    from konjoai.generate.generator import get_generator
    from konjoai.retrieve.hybrid import HybridResult, hybrid_search
    from konjoai.retrieve.hyde import hyde_encode
    from konjoai.retrieve.reranker import rerank
    from konjoai.retrieve.router import QueryIntent, classify_intent, decompose_query

    settings = get_settings()
    tel = PipelineTelemetry()

    # Optional pipeline component flags
    if settings.use_vectro_retriever:
        from konjoai.retrieve.vectro_retriever import get_vectro_retriever
    if settings.use_colbert:
        from konjoai.retrieve.late_interaction import rerank_with_maxsim

    # CRAG + Self-RAG metadata (populated only when those features are enabled)
    crag_confidence: float | None = None
    crag_fallback: bool | None = None
    self_rag_support: float | None = None
    self_rag_iterations: int | None = None

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
    # ── Step 2b: Early embed + cache lookup (RETRIEVAL only) ─────────────────
    q_vec = None
    if intent != QueryIntent.AGGREGATION:
        from konjoai.cache import get_semantic_cache
        _cache_chk = get_semantic_cache()
        if _cache_chk is not None:
            from konjoai.embed.encoder import get_encoder
            with timed(tel, "embed"):
                q_vec = get_encoder().encode_query(effective_question)
            _cached = _cache_chk.lookup(effective_question, q_vec)
            if _cached is not None:
                logger.debug("semantic cache hit — skipping Qdrant")
                return _cached.model_copy(update={
                    "cache_hit": True,
                    "telemetry": tel.as_dict() if settings.enable_telemetry else None,
                })
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
            if settings.use_vectro_retriever:
                hybrid_results = get_vectro_retriever().search(
                    effective_question,
                    top_k=settings.top_k_dense,
                )
            else:
                hybrid_results = hybrid_search(effective_question, q_vec=q_vec)

    # ── Step 3b: CRAG — Corrective RAG relevance grading ─────────────────────
    if settings.enable_crag and hybrid_results:
        from konjoai.retrieve.crag import get_crag_pipeline
        with timed(tel, "crag", n_docs=len(hybrid_results)):
            crag_result = get_crag_pipeline().run(req.question, hybrid_results)
        crag_confidence = crag_result.overall_confidence
        crag_fallback = crag_result.needs_fallback
        # Use the CRAG-filtered document list for downstream steps
        if crag_result.documents:
            hybrid_results = crag_result.documents  # type: ignore[assignment]
        logger.debug(
            "CRAG: kept=%d discarded=%d confidence=%.3f fallback=%s",
            len(crag_result.documents),
            crag_result.discarded_count,
            crag_result.overall_confidence,
            crag_result.needs_fallback,
        )

    # ── Step 4: Cross-encoder reranking ──────────────────────────────────────
    with timed(tel, "rerank", top_k=req.top_k):
        reranked = rerank(req.question, hybrid_results, top_k=req.top_k)

    # ── Step 4.5: MaxSim late-interaction re-scoring (ColBERT-style) ─────────
    if settings.use_colbert:
        with timed(tel, "colbert_maxsim", top_k=req.top_k):
            from konjoai.embed.encoder import get_encoder
            _enc = get_encoder()
            query_emb = _enc.encode(req.question)
            reranked = rerank_with_maxsim(query_emb, reranked)
            reranked = reranked[: req.top_k]

    # ── Step 5: Generation ───────────────────────────────────────────────────
    context = "\n\n---\n\n".join(r.content for r in reranked)
    generator = get_generator()

    with timed(tel, "generate", model=settings.openai_model):
        result = generator.generate(question=req.question, context=context)

    answer = result.answer

    # ── Step 5b: Self-RAG reflection critique ────────────────────────────────
    if settings.enable_self_rag and reranked:
        from konjoai.retrieve.self_rag import get_self_rag_pipeline
        with timed(tel, "self_rag"):
            def _gen() -> str:
                return generator.generate(question=req.question, context=context).answer
            sr = get_self_rag_pipeline().run(
                question=req.question,
                documents=reranked,
                generate_fn=_gen,
            )
        answer = sr.answer
        self_rag_support = sr.support_score
        self_rag_iterations = sr.iterations
        logger.debug(
            "Self-RAG: support=%.3f usefulness=%s iterations=%d",
            sr.support_score,
            sr.usefulness.name,
            sr.iterations,
        )

    sources = [
        SourceDoc(
            source=r.source,
            content_preview=r.content[:200],
            score=float(r.score),
        )
        for r in reranked
    ]

    response = QueryResponse(
        answer=answer,
        sources=sources,
        model=result.model,
        usage=result.usage,
        telemetry=tel.as_dict() if settings.enable_telemetry else None,
        intent=intent.value,
        crag_confidence=crag_confidence,
        crag_fallback=crag_fallback,
        self_rag_support=self_rag_support,
        self_rag_iterations=self_rag_iterations,
    )
    # Cache store (after full pipeline; K3: no-op when cache_enabled=False)
    if q_vec is not None:
        from konjoai.cache import get_semantic_cache
        _cache_store = get_semantic_cache()
        if _cache_store is not None:
            _cache_store.store(effective_question, q_vec, response)
    return response


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
    if settings.use_vectro_retriever:
        from konjoai.retrieve.vectro_retriever import get_vectro_retriever
        hybrid_results = get_vectro_retriever().search(
            effective_question, top_k=settings.top_k_dense
        )
    else:
        hybrid_results = hybrid_search(effective_question)
    reranked = rerank(req.question, hybrid_results, top_k=req.top_k)

    # ── MaxSim late-interaction re-scoring ────────────────────────────────
    if settings.use_colbert:
        from konjoai.embed.encoder import get_encoder
        from konjoai.retrieve.late_interaction import rerank_with_maxsim
        _query_emb = get_encoder().encode(req.question)
        reranked = rerank_with_maxsim(_query_emb, reranked)[: req.top_k]

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
