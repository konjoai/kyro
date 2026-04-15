from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import ragos.embed.encoder as enc_module
from ragos.embed.encoder import SentenceEncoder


# ---------------------------------------------------------------------------
# SentenceEncoder via mocked SentenceTransformer
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_st(monkeypatch) -> MagicMock:
    """Patch SentenceTransformer so no model is downloaded."""
    dim = 384
    mock = MagicMock()
    # encode returns float32 unit vectors
    def fake_encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=False):
        n = len(texts)
        vecs = np.random.default_rng(0).standard_normal((n, dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    mock.get_sentence_embedding_dimension.return_value = dim
    mock.encode = fake_encode
    monkeypatch.setattr("ragos.embed.encoder.SentenceTransformer", lambda *a, **kw: mock)
    # reset singleton between tests
    enc_module._encoder = None
    yield mock
    enc_module._encoder = None


class TestSentenceEncoder:
    def test_encode_returns_float32(self, mock_st) -> None:
        encoder = SentenceEncoder()
        result = encoder.encode(["hello", "world"])
        assert result.dtype == np.float32

    def test_encode_shape(self, mock_st) -> None:
        encoder = SentenceEncoder()
        texts = ["a", "b", "c"]
        result = encoder.encode(texts)
        assert result.shape == (3, 384)

    def test_encode_query_shape(self, mock_st) -> None:
        encoder = SentenceEncoder()
        result = encoder.encode_query("what is rag?")
        assert result.shape == (1, 384)

    def test_encode_query_dtype(self, mock_st) -> None:
        encoder = SentenceEncoder()
        result = encoder.encode_query("test")
        assert result.dtype == np.float32

    def test_encode_query_is_l2_normalised(self, mock_st) -> None:
        encoder = SentenceEncoder()
        result = encoder.encode_query("test")
        norm = float(np.linalg.norm(result))
        assert abs(norm - 1.0) < 1e-5

    def test_encode_empty_raises(self, mock_st) -> None:
        encoder = SentenceEncoder()
        with pytest.raises((ValueError, IndexError)):
            encoder.encode([])

    def test_dim_property(self, mock_st) -> None:
        encoder = SentenceEncoder()
        assert encoder.dim == 384
