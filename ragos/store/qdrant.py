from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_store: "QdrantStore | None" = None


@dataclass
class SearchResult:
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

    def upsert(self, embeddings: np.ndarray, contents: list[str], sources: list[str], metadatas: list[dict]) -> None:
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

        from ragos.config import get_settings  # noqa: PLC0415
        _s = get_settings()
        if getattr(_s, "vectro_quantize", False):
            from ragos.embed.vectro_bridge import quantize_for_storage  # noqa: PLC0415
            embeddings, _vectro_metrics = quantize_for_storage(embeddings)
            logger.info("Vectro quantization metrics: %s", _vectro_metrics)

        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i].tolist(),
                payload={"content": contents[i], "source": sources[i], **metadatas[i]},
            )
            for i in range(n)
        ]
        self._client.upsert(collection_name=self._collection, points=points, wait=True)
        logger.debug("QdrantStore: upserted %d points", n)

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> list[SearchResult]:
        """Dense cosine search.

        Parameters
        ----------
        query_vector:
            Shape ``(1, dim)`` or ``(dim,)``, dtype ``float32``.
        """
        vec = query_vector.flatten().tolist()
        hits = self._client.query_points(
            collection_name=self._collection,
            query=vec,
            limit=top_k,
            with_payload=True,
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


def get_store() -> QdrantStore:
    """Return the module-level singleton store (lazy init)."""
    global _store
    if _store is None:
        from ragos.config import get_settings
        from ragos.embed.encoder import get_encoder

        s = get_settings()
        _store = QdrantStore(
            url=s.qdrant_url,
            api_key=s.qdrant_api_key,
            collection=s.qdrant_collection,
            dim=get_encoder().dim,
        )
    return _store
