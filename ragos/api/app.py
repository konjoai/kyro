from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import logging

from fastapi import FastAPI

from ragos.api.routes import ingest, query, eval as eval_route
from ragos.api.schemas import HealthResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Warm up the encoder and vector store on startup."""
    logger.info("RagOS API starting — warming up encoder and store…")
    from ragos.embed.encoder import get_encoder
    from ragos.store.qdrant import get_store

    get_encoder()
    get_store()
    logger.info("RagOS API ready.")
    yield
    logger.info("RagOS API shutting down.")


app = FastAPI(
    title="RagOS",
    description="Production RAG pipeline — ingest, query, evaluate.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(eval_route.router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    from ragos.store.qdrant import get_store
    from ragos.retrieve.sparse import get_sparse_index

    return HealthResponse(
        status="ok",
        vector_count=get_store().count(),
        bm25_built=get_sparse_index().built,
    )
