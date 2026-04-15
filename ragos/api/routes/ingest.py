from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ragos.api.schemas import IngestRequest, IngestResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    """Load files from *path*, chunk, embed, and upsert into Qdrant + BM25."""
    from pathlib import Path

    from ragos.ingest.loaders import load_path
    from ragos.ingest.chunkers import get_chunker
    from ragos.embed.encoder import get_encoder
    from ragos.store.qdrant import get_store
    from ragos.retrieve.sparse import get_sparse_index

    path = Path(req.path)
    if not path.exists():
        raise HTTPException(status_code=422, detail=f"Path does not exist: {req.path}")

    chunker = get_chunker(req.strategy, req.chunk_size, req.overlap)
    encoder = get_encoder()
    store = get_store()

    all_contents: list[str] = []
    all_sources: list[str] = []
    all_metadatas: list[dict] = []
    sources_seen: set[str] = set()

    for doc in load_path(path):
        sources_seen.add(doc.source)
        for chunk in chunker.chunk(doc):
            all_contents.append(chunk.content)
            all_sources.append(chunk.source)
            all_metadatas.append(chunk.metadata)

    if not all_contents:
        raise HTTPException(status_code=422, detail="No content found at the given path.")

    embeddings = encoder.encode(all_contents)
    store.upsert(embeddings, all_contents, all_sources, all_metadatas)

    bm25 = get_sparse_index()
    bm25.build(all_contents, all_sources, all_metadatas)

    logger.info("Ingested %d chunks from %d sources", len(all_contents), len(sources_seen))
    return IngestResponse(chunks_indexed=len(all_contents), sources_processed=len(sources_seen))
