from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ragos.api.schemas import EvalRequest, EvalResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/eval", tags=["eval"])


@router.post("", response_model=EvalResponse)
def run_eval(req: EvalRequest) -> EvalResponse:
    """Run RAGAS evaluation over provided QA pairs."""
    if len(req.questions) == 0:
        raise HTTPException(status_code=422, detail="questions list must not be empty.")

    try:
        from ragos.eval.ragas_eval import evaluate
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    scores = evaluate(
        questions=req.questions,
        answers=req.answers,
        contexts=req.contexts,
        ground_truths=req.ground_truths,
    )
    return EvalResponse(scores=scores)
