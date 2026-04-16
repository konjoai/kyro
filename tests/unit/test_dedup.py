"""Unit tests for konjoai.ingest.dedup.filter_near_duplicates.

Test classes:
    TestFilterNearDuplicates      — shape / dtype contracts, correctness
    TestFilterNearDuplicateEdgeCases — empty input, all-duplicate, single chunk
    TestFilterNearDuplicateErrors    — invalid input shapes / lengths
"""
from __future__ import annotations

import numpy as np
import pytest

from konjoai.ingest.dedup import filter_near_duplicates


def _unit(v: np.ndarray) -> np.ndarray:
    """Return L2-normalised copy of v."""
    return v / (np.linalg.norm(v) or 1.0)


# ---------------------------------------------------------------------------
# Shape / dtype contract tests (K4 — float32)
# ---------------------------------------------------------------------------

class TestFilterNearDuplicatesContracts:
    def _run(self, n: int = 4, d: int = 8, threshold: float = 0.98) -> tuple:
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((n, d)).astype(np.float32)
        contents = [f"chunk {i}" for i in range(n)]
        sources = ["src"] * n
        metadatas = [{}] * n
        return filter_near_duplicates(emb, contents, sources, metadatas, threshold)

    def test_output_is_tuple_of_5(self):
        result = self._run()
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_kept_embeddings_dtype_float32(self):
        kept_emb, *_ = self._run()
        assert kept_emb.dtype == np.float32

    def test_kept_embeddings_2d(self):
        kept_emb, *_ = self._run()
        assert kept_emb.ndim == 2

    def test_lengths_consistent(self):
        kept_emb, kept_contents, kept_sources, kept_metadatas, n_removed = self._run()
        assert len(kept_contents) == len(kept_sources) == len(kept_metadatas) == kept_emb.shape[0]

    def test_n_removed_plus_kept_equals_n(self):
        n = 6
        kept_emb, _, _, _, n_removed = self._run(n=n)
        assert kept_emb.shape[0] + n_removed == n


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

class TestFilterNearDuplicates:
    def test_exact_duplicate_removed(self):
        """Two identical vectors at threshold=0.99 → one is dropped."""
        d = 16
        rng = np.random.default_rng(0)
        base = rng.standard_normal(d).astype(np.float32)
        emb = np.stack([base, base])  # (2, 16) — perfectly identical
        contents = ["chunk A", "chunk B"]
        sources = ["s1", "s2"]
        metadatas = [{}, {}]
        _, kept_c, _, _, n_removed = filter_near_duplicates(emb, contents, sources, metadatas, threshold=0.99)
        assert n_removed == 1
        assert kept_c == ["chunk A"]

    def test_distinct_vectors_all_kept(self):
        """Four orthogonal-ish vectors should all be kept."""
        d = 128
        # Build 4 orthogonal unit vectors via QR decomposition
        rng = np.random.default_rng(7)
        A = rng.standard_normal((d, 4)).astype(np.float32)
        Q, _ = np.linalg.qr(A)
        emb = Q[:, :4].T  # (4, 128) — orthogonal rows
        contents = [f"c{i}" for i in range(4)]
        sources = ["s"] * 4
        metadatas = [{}] * 4
        _, _, _, _, n_removed = filter_near_duplicates(emb, contents, sources, metadatas, threshold=0.98)
        assert n_removed == 0

    def test_near_duplicate_removed(self):
        """Vector slightly perturbed beyond threshold should be removed."""
        d = 64
        rng = np.random.default_rng(3)
        base = _unit(rng.standard_normal(d).astype(np.float32))
        noise = _unit(rng.standard_normal(d).astype(np.float32)) * 0.005  # tiny perturbation
        near_dup = _unit(base + noise)
        emb = np.stack([base, near_dup])
        contents = ["original", "near-dup"]
        sources = ["s", "s"]
        metadatas = [{}, {}]
        _, kept_c, _, _, n_removed = filter_near_duplicates(emb, contents, sources, metadatas, threshold=0.98)
        assert n_removed == 1
        assert kept_c == ["original"]

    def test_first_chunk_always_kept(self):
        """First chunk is always accepted regardless of threshold."""
        emb = np.ones((1, 8), dtype=np.float32)
        contents = ["only chunk"]
        sources = ["s"]
        metadatas = [{}]
        _, kept_c, _, _, n_removed = filter_near_duplicates(emb, contents, sources, metadatas, threshold=0.0)
        assert kept_c == ["only chunk"]
        assert n_removed == 0

    def test_threshold_1_keeps_all_non_exact(self):
        """threshold=1.0 only removes exact duplicates (cosine sim == 1.0)."""
        d = 32
        rng = np.random.default_rng(11)
        emb = rng.standard_normal((5, d)).astype(np.float32)
        contents = [f"c{i}" for i in range(5)]
        sources = ["s"] * 5
        metadatas = [{}] * 5
        _, _, _, _, n_removed = filter_near_duplicates(emb, contents, sources, metadatas, threshold=1.0)
        assert n_removed == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestFilterNearDuplicateEdgeCases:
    def test_empty_input_returns_empty(self):
        emb = np.empty((0, 16), dtype=np.float32)
        kept_emb, kept_c, kept_s, kept_m, n_removed = filter_near_duplicates(emb, [], [], [], threshold=0.98)
        assert kept_emb.shape == (0, 16)
        assert kept_c == []
        assert n_removed == 0

    def test_single_chunk_always_kept(self):
        emb = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        _, kept_c, _, _, n_removed = filter_near_duplicates(emb, ["only"], ["s"], [{}], threshold=0.0)
        assert kept_c == ["only"]
        assert n_removed == 0

    def test_all_identical_keeps_one(self):
        """When all chunks are identical, at least the first must be kept."""
        d = 8
        row = np.ones((1, d), dtype=np.float32)
        emb = np.tile(row, (5, 1))
        contents = [f"c{i}" for i in range(5)]
        sources = ["s"] * 5
        metadatas = [{}] * 5
        _, kept_c, _, _, n_removed = filter_near_duplicates(emb, contents, sources, metadatas, threshold=0.99)
        assert len(kept_c) >= 1
        assert n_removed == 4

    def test_zero_vector_treated_as_distinct(self):
        """Zero vectors (norm=0) should not crash and should be accepted."""
        emb = np.zeros((3, 8), dtype=np.float32)
        contents = ["a", "b", "c"]
        sources = ["s"] * 3
        metadatas = [{}] * 3
        kept_emb, _, _, _, _ = filter_near_duplicates(emb, contents, sources, metadatas, threshold=0.98)
        # Should not raise; result length >= 1
        assert kept_emb.shape[0] >= 1


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestFilterNearDuplicateErrors:
    def test_1d_embeddings_raises(self):
        emb = np.ones(8, dtype=np.float32)
        with pytest.raises(ValueError, match="2-D"):
            filter_near_duplicates(emb, ["a"], ["s"], [{}])

    def test_length_mismatch_raises(self):
        emb = np.ones((3, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="Length mismatch"):
            filter_near_duplicates(emb, ["a", "b"], ["s", "s"], [{}, {}])  # 3 emb, 2 contents

    def test_source_mismatch_raises(self):
        emb = np.ones((2, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="Length mismatch"):
            filter_near_duplicates(emb, ["a", "b"], ["s"], [{}, {}])  # 2 contents, 1 source
