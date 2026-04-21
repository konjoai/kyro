"""Adapter protocols for Kyro's swappable backend architecture.

Every component that crosses a system boundary (vector store, embedder,
generator, retriever) must satisfy one of the Protocol interfaces defined
in :mod:`konjoai.adapters.base`.  This lets any backend be swapped with
zero changes to pipeline logic.

Usage::

    from konjoai.adapters import VectorStoreAdapter, EmbedderAdapter
    from konjoai.adapters import GeneratorAdapter, RetrieverAdapter

    # isinstance() works because all protocols are @runtime_checkable
    assert isinstance(get_store(), VectorStoreAdapter)
"""
from konjoai.adapters.base import (
    EmbedderAdapter,
    GeneratorAdapter,
    RetrieverAdapter,
    VectorStoreAdapter,
)

__all__ = [
    "EmbedderAdapter",
    "GeneratorAdapter",
    "RetrieverAdapter",
    "VectorStoreAdapter",
]
