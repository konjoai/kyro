"""Unit tests for konjoai.adapters.base — Protocol conformance.

Tests verify that:
1. All four protocols are runtime_checkable.
2. Duck-typed stub classes satisfy each protocol.
3. Objects that are missing required methods do NOT satisfy the protocol.
4. The protocol contracts (return types, signatures) are correct.
"""
from __future__ import annotations

import numpy as np
import pytest

from konjoai.adapters import (
    EmbedderAdapter,
    GeneratorAdapter,
    RetrieverAdapter,
    VectorStoreAdapter,
)


# ── Stub implementations ─────────────────────────────────────────────────────

class _StubVectorStore:
    def upsert(self, vectors, payloads, ids=None):
        return len(vectors)

    def search(self, query_vector, top_k=10, filter=None):
        return []

    def delete_collection(self):
        pass

    def count(self):
        return 0


class _StubEmbedder:
    def encode(self, texts):
        return np.zeros((len(texts), 384), dtype=np.float32)

    def encode_query(self, text):
        return np.zeros((1, 384), dtype=np.float32)

    @property
    def dim(self):
        return 384


class _StubGenerator:
    def generate(self, question, context):
        class _R:
            answer = "stub"
            model = "stub"
            usage = {}
        return _R()

    def generate_stream(self, question, context):
        yield "stub"


class _StubRetriever:
    def search(self, query, top_k=10, q_vec=None):
        return []


# ── Incomplete stubs ──────────────────────────────────────────────────────────

class _IncompleteStore:
    """Missing `search` and `count`."""
    def upsert(self, vectors, payloads, ids=None):
        return 0
    def delete_collection(self):
        pass


class _IncompleteEmbedder:
    """Missing `encode_query`."""
    def encode(self, texts):
        return np.zeros((len(texts), 10), dtype=np.float32)
    @property
    def dim(self):
        return 10


class _IncompleteGenerator:
    """Missing `generate_stream`."""
    def generate(self, question, context):
        class _R:
            answer = "x"
            model = "x"
            usage = {}
        return _R()


# ── VectorStoreAdapter ───────────────────────────────────────────────────────

def test_vector_store_isinstance():
    assert isinstance(_StubVectorStore(), VectorStoreAdapter)


def test_vector_store_incomplete_fails():
    # _IncompleteStore is missing required methods
    assert not isinstance(_IncompleteStore(), VectorStoreAdapter)


def test_vector_store_upsert_returns_int():
    store = _StubVectorStore()
    vecs = [np.zeros((384,), dtype=np.float32)]
    result = store.upsert(vecs, [{"source": "x"}])
    assert isinstance(result, int)


def test_vector_store_search_returns_list():
    store = _StubVectorStore()
    q = np.zeros((1, 384), dtype=np.float32)
    result = store.search(q, top_k=5)
    assert isinstance(result, list)


def test_vector_store_count_returns_int():
    store = _StubVectorStore()
    assert isinstance(store.count(), int)


# ── EmbedderAdapter ──────────────────────────────────────────────────────────

def test_embedder_isinstance():
    assert isinstance(_StubEmbedder(), EmbedderAdapter)


def test_embedder_incomplete_fails():
    assert not isinstance(_IncompleteEmbedder(), EmbedderAdapter)


def test_embedder_encode_shape_and_dtype():
    emb = _StubEmbedder()
    result = emb.encode(["hello", "world"])
    assert result.shape == (2, 384)
    assert result.dtype == np.float32


def test_embedder_encode_query_shape_and_dtype():
    emb = _StubEmbedder()
    result = emb.encode_query("hello")
    assert result.shape == (1, 384)
    assert result.dtype == np.float32


def test_embedder_dim_property():
    assert _StubEmbedder().dim == 384


# ── GeneratorAdapter ─────────────────────────────────────────────────────────

def test_generator_isinstance():
    assert isinstance(_StubGenerator(), GeneratorAdapter)


def test_generator_incomplete_fails():
    assert not isinstance(_IncompleteGenerator(), GeneratorAdapter)


def test_generator_generate_has_answer():
    gen = _StubGenerator()
    result = gen.generate("q", "ctx")
    assert hasattr(result, "answer")
    assert hasattr(result, "model")
    assert hasattr(result, "usage")


def test_generator_stream_yields_str():
    gen = _StubGenerator()
    tokens = list(gen.generate_stream("q", "ctx"))
    assert all(isinstance(t, str) for t in tokens)


# ── RetrieverAdapter ─────────────────────────────────────────────────────────

def test_retriever_isinstance():
    assert isinstance(_StubRetriever(), RetrieverAdapter)


def test_retriever_search_returns_list():
    ret = _StubRetriever()
    result = ret.search("what is X?", top_k=5)
    assert isinstance(result, list)


def test_retriever_search_accepts_q_vec():
    ret = _StubRetriever()
    q_vec = np.zeros((1, 384), dtype=np.float32)
    result = ret.search("what is X?", top_k=5, q_vec=q_vec)
    assert isinstance(result, list)


# ── Import smoke test ────────────────────────────────────────────────────────

def test_adapter_imports():
    from konjoai.adapters import (
        EmbedderAdapter,
        GeneratorAdapter,
        RetrieverAdapter,
        VectorStoreAdapter,
    )
    assert all(
        callable(p)
        for p in [EmbedderAdapter, GeneratorAdapter, RetrieverAdapter, VectorStoreAdapter]
    )
