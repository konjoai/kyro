"""Unit tests for ragos.embed.vectro_bridge — _check_vectro(), compression_ratio(), quantize_for_storage()."""
from __future__ import annotations

import builtins

import numpy as np
import pytest


def _reset_cache() -> None:
    """Force re-evaluation of Vectro availability on next _check_vectro() call."""
    import ragos.embed.vectro_bridge as bridge

    bridge._VECTRO_AVAILABLE = None


# ---------------------------------------------------------------------------
# _check_vectro
# ---------------------------------------------------------------------------


class TestCheckVectro:
    def setup_method(self) -> None:
        _reset_cache()

    def test_returns_bool(self) -> None:
        from ragos.embed.vectro_bridge import _check_vectro

        result = _check_vectro()
        assert isinstance(result, bool)

    def test_caching_returns_same_value_on_second_call(self) -> None:
        from ragos.embed.vectro_bridge import _check_vectro

        first = _check_vectro()
        second = _check_vectro()
        assert first == second

    def test_false_when_vectro_import_blocked(self, monkeypatch) -> None:
        """Simulate Vectro being absent by blocking the import."""
        import ragos.embed.vectro_bridge as bridge

        _reset_cache()

        real_import = builtins.__import__

        def blocked_import(name: str, *args, **kwargs):
            if "vectro" in name:
                raise ImportError("Simulated: vectro not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)
        # Cache was already reset above — do NOT reset again here; _reset_cache
        # imports ragos.embed.vectro_bridge which contains "vectro" and would
        # itself be blocked by the monkeypatch.

        result = bridge._check_vectro()
        assert result is False


# ---------------------------------------------------------------------------
# compression_ratio
# ---------------------------------------------------------------------------


class TestCompressionRatio:
    def test_float32_to_int8_is_4x(self) -> None:
        from ragos.embed.vectro_bridge import compression_ratio

        original = np.zeros((10, 128), dtype=np.float32)  # 4 bytes/element
        quantized = np.zeros((10, 128), dtype=np.int8)    # 1 byte/element
        assert abs(compression_ratio(original, quantized) - 4.0) < 1e-6

    def test_same_dtype_is_1x(self) -> None:
        from ragos.embed.vectro_bridge import compression_ratio

        arr = np.zeros((10, 128), dtype=np.float32)
        assert abs(compression_ratio(arr, arr) - 1.0) < 1e-6

    def test_ratio_is_positive(self) -> None:
        from ragos.embed.vectro_bridge import compression_ratio

        original = np.zeros((5, 64), dtype=np.float32)
        quantized = np.zeros((5, 64), dtype=np.int8)
        assert compression_ratio(original, quantized) > 0.0

    def test_float32_to_float16_is_2x(self) -> None:
        from ragos.embed.vectro_bridge import compression_ratio

        original = np.zeros((8, 64), dtype=np.float32)   # 4 bytes/element
        half = np.zeros((8, 64), dtype=np.float16)       # 2 bytes/element
        assert abs(compression_ratio(original, half) - 2.0) < 1e-6


# ---------------------------------------------------------------------------
# quantize_for_storage (passthrough path — Vectro unavailable)
# ---------------------------------------------------------------------------


class TestQuantizeForStorage:
    def setup_method(self) -> None:
        _reset_cache()

    def _passthrough_emb(self, rows: int = 4, cols: int = 16) -> np.ndarray:
        return np.random.randn(rows, cols).astype(np.float32)

    def test_passthrough_output_dtype_is_float32(self, monkeypatch) -> None:
        import ragos.embed.vectro_bridge as bridge

        monkeypatch.setattr(bridge, "_VECTRO_AVAILABLE", False)
        emb = self._passthrough_emb()
        vectors, _ = bridge.quantize_for_storage(emb)
        assert vectors.dtype == np.float32

    def test_passthrough_output_shape_matches_input(self, monkeypatch) -> None:
        import ragos.embed.vectro_bridge as bridge

        monkeypatch.setattr(bridge, "_VECTRO_AVAILABLE", False)
        emb = self._passthrough_emb(rows=7, cols=32)
        vectors, _ = bridge.quantize_for_storage(emb)
        assert vectors.shape == emb.shape

    def test_passthrough_metrics_method_is_correct(self, monkeypatch) -> None:
        import ragos.embed.vectro_bridge as bridge

        monkeypatch.setattr(bridge, "_VECTRO_AVAILABLE", False)
        _, metrics = bridge.quantize_for_storage(self._passthrough_emb())
        assert metrics["method"] == "float32_passthrough"

    def test_passthrough_metrics_compression_ratio_is_1(self, monkeypatch) -> None:
        import ragos.embed.vectro_bridge as bridge

        monkeypatch.setattr(bridge, "_VECTRO_AVAILABLE", False)
        _, metrics = bridge.quantize_for_storage(self._passthrough_emb())
        assert metrics["compression_ratio"] == 1.0

    def test_passthrough_metrics_similarity_is_1(self, monkeypatch) -> None:
        import ragos.embed.vectro_bridge as bridge

        monkeypatch.setattr(bridge, "_VECTRO_AVAILABLE", False)
        _, metrics = bridge.quantize_for_storage(self._passthrough_emb())
        assert metrics["mean_cosine_similarity"] == 1.0

    def test_passthrough_preserves_values(self, monkeypatch) -> None:
        import ragos.embed.vectro_bridge as bridge

        monkeypatch.setattr(bridge, "_VECTRO_AVAILABLE", False)
        emb = self._passthrough_emb()
        vectors, _ = bridge.quantize_for_storage(emb)
        np.testing.assert_array_equal(vectors, emb)

    def test_nan_input_raises_value_error(self, monkeypatch) -> None:
        import ragos.embed.vectro_bridge as bridge

        monkeypatch.setattr(bridge, "_VECTRO_AVAILABLE", False)
        emb = self._passthrough_emb()
        emb[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN or Inf"):
            bridge.quantize_for_storage(emb)

    def test_inf_input_raises_value_error(self, monkeypatch) -> None:
        import ragos.embed.vectro_bridge as bridge

        monkeypatch.setattr(bridge, "_VECTRO_AVAILABLE", False)
        emb = self._passthrough_emb()
        emb[1, 2] = float("inf")
        with pytest.raises(ValueError, match="NaN or Inf"):
            bridge.quantize_for_storage(emb)

    def test_neg_inf_input_raises_value_error(self, monkeypatch) -> None:
        import ragos.embed.vectro_bridge as bridge

        monkeypatch.setattr(bridge, "_VECTRO_AVAILABLE", False)
        emb = self._passthrough_emb()
        emb[2, 3] = float("-inf")
        with pytest.raises(ValueError, match="NaN or Inf"):
            bridge.quantize_for_storage(emb)

    def test_returns_tuple_of_ndarray_and_dict(self, monkeypatch) -> None:
        import ragos.embed.vectro_bridge as bridge

        monkeypatch.setattr(bridge, "_VECTRO_AVAILABLE", False)
        result = bridge.quantize_for_storage(self._passthrough_emb())
        assert isinstance(result, tuple)
        assert len(result) == 2
        vectors, metrics = result
        assert isinstance(vectors, np.ndarray)
        assert isinstance(metrics, dict)
