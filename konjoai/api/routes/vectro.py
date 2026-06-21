"""POST /vectro/pipeline — invoke the vectro pipeline CLI from KonjoOS.

K1: All errors surfaced as typed HTTP responses (503 binary missing, 501 stub format).
K2: Telemetry delegated to timed() inside vectro_pipeline_service.run_pipeline.
K3: VectroBinaryNotFoundError → 503 with build instructions; no silent pass.
K6: Request schema validated via Pydantic; optional fields have safe defaults.
"""

from __future__ import annotations

import tempfile

from fastapi import APIRouter, HTTPException

from konjoai.api.schemas import VectroPipelineRequest, VectroPipelineResponse

router = APIRouter(prefix="/vectro", tags=["vectro"])


@router.post("/pipeline", response_model=VectroPipelineResponse, status_code=200)
def vectro_pipeline(req: VectroPipelineRequest) -> VectroPipelineResponse:
    """Run the vectro pipeline CLI: quantize a JSONL embedding file and build an HNSW index.

    The vectro binary must be built locally::

        cd ~/vectro && cargo build --release -p vectro_cli

    Returns 503 when the binary is not found, 501 for stub formats (rq / auto — vectro_lib v5.0).
    """
    from konjoai.services.vectro_pipeline_service import (
        VectroBinaryNotFoundError,
        VectroStubFormatError,
        run_pipeline,
    )

    out_dir = req.out_dir
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="vectro_pipeline_")

    try:
        result = run_pipeline(
            input_jsonl=req.input_jsonl,
            out_dir=out_dir,
            format=req.format,
            m=req.m,
            ef_construction=req.ef_construction,
            ef_search=req.ef_search,
            query_file=req.query_file,
            top_k=req.top_k,
            archive=req.archive,
        )
    except VectroBinaryNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except VectroStubFormatError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    return VectroPipelineResponse(
        n_vectors=result.n_vectors,
        dims=result.dims,
        format=result.format,
        out_dir=result.out_dir,
        index_size_bytes=result.index_size_bytes,
        duration_ms=result.duration_ms,
        query_results=result.query_results,
        binary_path=result.binary_path,
    )
