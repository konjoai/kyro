"""Qdrant vector-store wrappers: sync/async upsert and tenant-scoped search."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_store: QdrantStore | None = None


@dataclass
class SearchResult:
    """A single dense-search hit with its score, content, and payload metadata."""

    id: str
    score: float
    content: str
    source: str
    metadata: dict


class QdrantStore:
    """Qdrant wrapper: upsert chunks, search by dense vector."""

    def __init__(self, url: str, api_key: str | None, collection: str, dim: int) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as e:
            raise ImportError("qdrant-client is required: pip install qdrant-client") from e

        self._client = QdrantClient(url=url, api_key=api_key, prefer_grpc=False)
        self._collection = collection
        self._dim = dim

        # Create collection if it doesn't already exist
        existing = {c.name for c in self._client.get_collections().collections}
        if collection not in existing:
            from qdrant_client.models import Distance, VectorParams

            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info("QdrantStore: created collection '%s' dim=%d", collection, dim)
        else:
            logger.info("QdrantStore: using existing collection '%s'", collection)

    def upsert(self, embeddings: np.ndarray, contents: list[str], sources: list[str], metadatas: list[dict]) -> dict | None:
        """Upsert a batch of chunks into the collection.

        Parameters
        ----------
        embeddings:
            Shape ``(N, dim)``, dtype ``float32``.
        contents:
            Raw text for each embedding.
        sources:
            File-path or URL for each embedding.
        metadatas:
            Per-chunk metadata dicts.
        """
        assert embeddings.dtype == np.float32, "embeddings must be float32"
        assert embeddings.ndim == 2, "embeddings must be 2-D"
        n = len(contents)
        assert embeddings.shape[0] == n == len(sources) == len(metadatas)

        from konjoai.config import get_settings  # noqa: PLC0415
        _s = get_settings()
        _vectro_metrics: dict | None = None
        if getattr(_s, "vectro_quantize", False):
            from konjoai.embed.vectro_bridge import quantize_for_storage  # noqa: PLC0415
            embeddings, _vectro_metrics = quantize_for_storage(embeddings)
            logger.info("Vectro quantization metrics: %s", _vectro_metrics)

        from qdrant_client.models import PointStruct

        # ── Sprint 17: attach tenant_id to every point payload ────────────────
        from konjoai.auth.tenant import get_current_tenant_id  # noqa: PLC0415
        tenant_id = get_current_tenant_id()

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i].tolist(),
                payload={
                    "content": contents[i],
                    "source": sources[i],
                    **metadatas[i],
                    **({"tenant_id": tenant_id} if tenant_id is not None else {}),
                },
            )
            for i in range(n)
        ]
        self._client.upsert(collection_name=self._collection, points=points, wait=True)
        logger.debug(
            "QdrantStore: upserted %d points tenant_id=%s", n, tenant_id
        )
        return _vectro_metrics

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> list[SearchResult]:
        """Dense cosine search.

        Parameters
        ----------
        query_vector:
            Shape ``(1, dim)`` or ``(dim,)``, dtype ``float32``.

        Tenant scoping (Sprint 17): when a tenant_id is active in the current
        context (set by get_tenant_id FastAPI dep), results are filtered to
        points whose ``tenant_id`` payload field matches. When no tenant_id is
        set (multi_tenancy_enabled=False), all points are returned unchanged
        (K6 backward-compatible).
        """
        from konjoai.auth.tenant import get_current_tenant_id  # noqa: PLC0415

        vec = query_vector.flatten().tolist()
        tenant_id = get_current_tenant_id()
        query_filter = None
        if tenant_id is not None:
            try:
                from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: PLC0415
                query_filter = Filter(
                    must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
                )
            except ImportError:
                logger.warning("qdrant-client models unavailable — tenant filter skipped")

        hits = self._client.query_points(
            collection_name=self._collection,
            query=vec,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        ).points

        return [
            SearchResult(
                id=str(hit.id),
                score=float(hit.score),
                content=hit.payload.get("content", ""),
                source=hit.payload.get("source", ""),
                metadata={k: v for k, v in hit.payload.items() if k not in ("content", "source")},
            )
            for hit in hits
        ]

    def count(self) -> int:
        """Return the number of points in the collection."""
        return self._client.count(collection_name=self._collection).count

    def scroll_all(
        self, batch_size: int = 256
    ) -> tuple[np.ndarray, list[str], list[str], list[str]]:
        """Scroll through every point in the collection.

        Returns
        -------
        tuple of (vectors float32 (N, dim), texts, sources, ids)

        Used by :class:`konjoai.retrieve.vectro_retriever.VectroRetrieverAdapter`
        to build an in-memory corpus for Vectro SIMD hybrid search.
        """
        vecs, texts, sources, ids = [], [], [], []
        offset = None
        while True:
            result, next_offset = self._client.scroll(
                collection_name=self._collection,
                limit=batch_size,
                with_vectors=True,
                with_payload=True,
                offset=offset,
            )
            for pt in result:
                vecs.append(pt.vector)
                texts.append(pt.payload.get("content", ""))
                sources.append(pt.payload.get("source", ""))
                ids.append(str(pt.id))
            if next_offset is None:
                break
            offset = next_offset

        if vecs:
            return np.array(vecs, dtype=np.float32), texts, sources, ids
        return np.empty((0, self._dim), dtype=np.float32), [], [], []


def get_store() -> QdrantStore:
    """Return the module-level singleton store (lazy init)."""
    global _store
    if _store is None:
        from konjoai.config import get_settings
        from konjoai.embed.encoder import get_encoder

        s = get_settings()
        _store = QdrantStore(
            url=s.qdrant_url,
            api_key=s.qdrant_api_key,
            collection=s.qdrant_collection,
            dim=get_encoder().dim,
        )
    return _store


class AsyncQdrantStore:
    """Async Qdrant wrapper with httpx connection pooling (Sprint 8)."""

    def __init__(self) -> None:
        try:
            import httpx  # noqa: PLC0415, F401
            from qdrant_client import AsyncQdrantClient  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("qdrant-client>=1.7 and httpx are required") from exc

        from konjoai.config import get_settings  # noqa: PLC0415

        settings = get_settings()

        self._client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
        self._collection = settings.qdrant_collection

    async def search(self, query_vector: np.ndarray, top_k: int = 20) -> list[SearchResult]:
        """Async dense cosine search returning the top-*k* results."""
        vec = query_vector.flatten().tolist()
        result = await self._client.query_points(
            collection_name=self._collection,
            query=vec,
            limit=top_k,
            with_payload=True,
        )
        return [
            SearchResult(
                id=str(h.id),
                score=float(h.score),
                content=h.payload.get("content", ""),
                source=h.payload.get("source", ""),
                metadata={k: v for k, v in h.payload.items() if k not in {"content", "source"}},
            )
            for h in result.points
        ]


_async_store: AsyncQdrantStore | None = None


def get_async_store() -> AsyncQdrantStore:
    """Return the module-level singleton async store (lazy init)."""
    global _async_store
    if _async_store is None:
        _async_store = AsyncQdrantStore()
    return _async_store
