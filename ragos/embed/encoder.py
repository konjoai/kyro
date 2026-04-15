from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_encoder: "SentenceEncoder | None" = None


class SentenceEncoder:
    """Thin wrapper around sentence-transformers with a float32 dtype contract.

    All returned arrays are ``np.float32`` and L2-normalised.
    """

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 64) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            ) from e

        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size
        self._dim: int = self._model.get_sentence_embedding_dimension()  # type: ignore[assignment]
        logger.info("SentenceEncoder: model=%s dim=%d device=%s", model_name, self._dim, device)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts.

        Returns
        -------
        np.ndarray
            Shape ``(N, dim)``, dtype ``float32``, L2-normalised.
        """
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        vecs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        arr = np.array(vecs, dtype=np.float32)
        assert arr.ndim == 2 and arr.shape[1] == self._dim, (
            f"Unexpected embedding shape: {arr.shape}"
        )
        return arr

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query string.

        Returns
        -------
        np.ndarray
            Shape ``(1, dim)``, dtype ``float32``, L2-normalised.
        """
        arr = self.encode([text])
        assert arr.shape == (1, self._dim), f"Unexpected query shape: {arr.shape}"
        return arr


def get_encoder() -> SentenceEncoder:
    """Return the module-level singleton encoder (lazy init)."""
    global _encoder
    if _encoder is None:
        from ragos.config import get_settings

        s = get_settings()
        _encoder = SentenceEncoder(
            model_name=s.embed_model,
            device=s.embed_device,
            batch_size=s.embed_batch_size,
        )
    return _encoder
