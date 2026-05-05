from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import logging

from fastapi import FastAPI

from konjoai.api.routes import (
    agent as agent_route,
    audit as audit_route,
    eval as eval_route,
    health as health_route,
    ingest,
    query,
    vectro as vectro_route,
)
from konjoai.api.schemas import HealthResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Warm up the encoder and vector store on startup."""
    logger.info("KonjoOS API starting — warming up encoder and store…")
    from konjoai.embed.encoder import get_encoder
    from konjoai.store.qdrant import get_store

    get_encoder()
    get_store()
    logger.info("KonjoOS API ready.")
    yield
    logger.info("KonjoOS API shutting down.")


app = FastAPI(
    title="KonjoOS",
    description="Production RAG pipeline — ingest, query, evaluate.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(eval_route.router)
app.include_router(vectro_route.router)
app.include_router(agent_route.router)
app.include_router(health_route.router)
app.include_router(audit_route.router)


def create_app() -> FastAPI:
    """Return the module-level FastAPI application.

    Tests should call ``create_app()`` rather than importing ``app`` directly
    so this factory can be swapped for a test-only variant if needed.
    """
    return app


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    from konjoai.store.qdrant import get_store
    from konjoai.retrieve.sparse import get_sparse_index

    return HealthResponse(
        status="ok",
        vector_count=get_store().count(),
        bm25_built=get_sparse_index().built,
    )
