from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from konjoai.api.schemas import IngestRequest, IngestResponse, ManifestResponse, VerifyResponse
from konjoai.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    """Load files from *path*, chunk, embed, and upsert into Qdrant + BM25."""
    from pathlib import Path

    from konjoai.ingest.loaders import load_path
    from konjoai.ingest.chunkers import get_chunker
    from konjoai.embed.encoder import get_encoder
    from konjoai.store.qdrant import get_store
    from konjoai.retrieve.sparse import get_sparse_index

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

    settings = get_settings()
    embeddings = encoder.encode(all_contents)

    # ── Optional: near-duplicate chunk filtering ──────────────────────────
    n_deduped = 0
    if settings.dedup_threshold is not None:
        from konjoai.ingest.dedup import filter_near_duplicates
        embeddings, all_contents, all_sources, all_metadatas, n_deduped = filter_near_duplicates(
            embeddings, all_contents, all_sources, all_metadatas, settings.dedup_threshold
        )
        if n_deduped:
            logger.info("Dedup: removed %d near-duplicate chunks (threshold=%.3f)", n_deduped, settings.dedup_threshold)

    vectro_metrics = store.upsert(embeddings, all_contents, all_sources, all_metadatas)

    bm25 = get_sparse_index()
    bm25.build(all_contents, all_sources, all_metadatas)

    # ── Semantic cache invalidation (K3: stale data protection) ──────────────
    from konjoai.cache import get_semantic_cache
    _cache = get_semantic_cache()
    if _cache is not None:
        _cache.invalidate()
        logger.info("Semantic cache invalidated after ingest")

    # ── Optional: auto-verify corpus after ingest ─────────────────────────
    drift_count: int | None = None
    if settings.rag_auto_verify and settings.rag_corpus_dir:
        from konjoai.ingest.rag_bridge import verify_corpus
        verify_result = verify_corpus(settings.rag_corpus_dir)
        if verify_result["available"]:
            drift_count = verify_result["drift_count"]
            if drift_count:
                logger.warning("Corpus drift detected: %d changed files in %s", drift_count, settings.rag_corpus_dir)

    logger.info("Ingested %d chunks from %d sources", len(all_contents), len(sources_seen))
    return IngestResponse(
        chunks_indexed=len(all_contents),
        sources_processed=len(sources_seen),
        vectro_metrics=vectro_metrics,
        chunks_deduplicated=n_deduped,
        drift_count=drift_count,
    )


class ManifestBody(BaseModel):
    corpus_dir: str


@router.post("/manifest", response_model=ManifestResponse)
def ingest_manifest(body: ManifestBody) -> ManifestResponse:
    """Index *corpus_dir* with RagScanner and write a .rag_manifest.json.

    Returns file count, manifest hash, and ISO-8601 timestamp.
    When Squish is unavailable, returns ``available=False`` (K3).
    """
    from konjoai.ingest.rag_bridge import index_corpus
    result = index_corpus(body.corpus_dir)
    return ManifestResponse(**result)


@router.get("/verify", response_model=VerifyResponse)
def ingest_verify(
    corpus_dir: str = Query(..., description="Path to the corpus directory to verify against its manifest."),
) -> VerifyResponse:
    """Compare *corpus_dir* against the stored .rag_manifest.json.

    Returns ok, total_files, drift_count, and a list of changed files.
    When Squish is unavailable, returns ``available=False`` (K3).
    """
    from konjoai.ingest.rag_bridge import verify_corpus
    result = verify_corpus(corpus_dir)
    return VerifyResponse(**result)
