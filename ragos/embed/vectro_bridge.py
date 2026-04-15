"""Vectro integration bridge for embedding quantization.

This module provides a graceful-degradation wrapper around the Vectro library
(Mojo-first INT8/INT4/NF4 embedding compression, 12.5M+ vec/s on M3 Silicon).

If Vectro is not installed the module works identically but returns float32
passthrough vectors instead of quantized reconstructions.  This satisfies K3
(graceful degradation) — the pipeline is fully functional without Vectro; Vectro
adds measurable compression telemetry that demonstrates the portfolio integration.

Primary use-cases:
    1. Pre-compute compression metrics (ratio, cosine similarity) before upsert.
    2. Benchmark Vectro throughput as a pipeline step timed with timed().
    3. Return dequantized float32 vectors for Qdrant (which requires float32).

Vectro API reference (from vectro.python.interface):
    quantize_embeddings(embeddings: np.ndarray) -> QuantizationResult
        QuantizationResult.quantized: np.int8 (n, d)
        QuantizationResult.scales:    np.float32 (n,)
    reconstruct_embeddings(result: QuantizationResult) -> np.ndarray (float32)
    mean_cosine_similarity(original: np.ndarray, reconstructed: np.ndarray) -> float
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path registration — vectro is a namespace package under ~/vectro/.
# Inserting ~ onto sys.path lets Python resolve vectro.python.interface
# without a separate pip install step.
# ---------------------------------------------------------------------------
_home = os.path.expanduser("~")
if _home not in sys.path:
    sys.path.insert(0, _home)

# ---------------------------------------------------------------------------
# Availability probe (cached after first call)
# ---------------------------------------------------------------------------

_VECTRO_AVAILABLE: bool | None = None


def _check_vectro() -> bool:
    """Return True if the Vectro library is importable.

    Caches the result on first call so subsequent calls are O(1) dictionary
    lookups rather than full import attempts — important for hot-path ingest.
    """
    global _VECTRO_AVAILABLE  # noqa: PLW0603

    if _VECTRO_AVAILABLE is not None:
        return _VECTRO_AVAILABLE

    try:
        # Probe import only — do not hold references here; the actual import
        # is performed inside quantize_for_storage() to keep module load fast.
        import vectro.python.interface  # noqa: F401

        _VECTRO_AVAILABLE = True
        logger.info("Vectro bridge: library available (INT8/INT4/NF4 quantization enabled)")
    except ImportError:
        _VECTRO_AVAILABLE = False
        logger.warning(
            "Vectro bridge: library not available — float32 passthrough active. "
            "Install vectro (see /Users/wscholl/vectro) to enable INT8 quantization."
        )

    return _VECTRO_AVAILABLE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def quantize_for_storage(
    embeddings: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Quantize embeddings with Vectro INT8 and return storage-ready vectors + metrics.

    The returned vectors are always float32 — Qdrant's dense vector collection
    requires float32 inputs.  Vectro quantizes to INT8 (4× compression), measures
    the compression ratio and mean cosine similarity, then reconstructs back to
    float32 for storage.  The quantization is therefore a measurement + benchmark
    step rather than a storage-space reduction step for Qdrant.

    To use native Qdrant int8 quantization for storage reduction, configure
    ``ScalarQuantizationConfig`` in QdrantStore — that is complementary to this
    bridge, which demonstrates the Vectro library in the portfolio pipeline.

    Args:
        embeddings: Float32 array of shape (n, d).  Must be finite (no NaN/Inf).

    Returns:
        (vectors, metrics) where:
            vectors: float32 ndarray (n, d), dequantized or passthrough.
            metrics: dict with keys "method", "compression_ratio",
                     "mean_cosine_similarity", "original_bytes", "quantized_bytes".

    Raises:
        ValueError: If embeddings contains NaN or Inf values.
        AssertionError: If dtype contracts are violated post-quantization.
    """
    # Validate input.
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("quantize_for_storage: input embeddings contain NaN or Inf values.")

    # Passthrough path — Vectro not available.
    if not _check_vectro():
        metrics: dict[str, Any] = {
            "method": "float32_passthrough",
            "compression_ratio": 1.0,
            "mean_cosine_similarity": 1.0,
            "original_bytes": embeddings.nbytes,
            "quantized_bytes": embeddings.nbytes,
        }
        return embeddings.astype(np.float32), metrics

    # Vectro quantization path.
    from vectro.python.interface import (  # type: ignore[import]
        mean_cosine_similarity,
        quantize_embeddings,
        reconstruct_embeddings,
    )

    original_f32 = embeddings.astype(np.float32)

    result = quantize_embeddings(original_f32)
    reconstructed = reconstruct_embeddings(result)

    # K4 — dtype contracts at boundaries.
    assert reconstructed.dtype == np.float32, (
        f"Vectro bridge: expected float32 reconstruction, got {reconstructed.dtype}"
    )
    assert not np.any(np.isnan(reconstructed)), "Vectro bridge: NaN in reconstructed vectors."
    assert not np.any(np.isinf(reconstructed)), "Vectro bridge: Inf in reconstructed vectors."

    ratio = compression_ratio(original_f32, result.quantized)
    sim = float(mean_cosine_similarity(original_f32, reconstructed))

    metrics = {
        "method": getattr(result, "precision_mode", "int8"),
        "compression_ratio": round(ratio, 3),
        "mean_cosine_similarity": round(sim, 6),
        "original_bytes": original_f32.nbytes,
        "quantized_bytes": int(result.quantized.nbytes),
        "n_vectors": int(original_f32.shape[0]),
        "dims": int(original_f32.shape[1]),
    }

    logger.info(
        "Vectro INT8: n=%d d=%d ratio=%.1fx mean_cos_sim=%.6f",
        metrics["n_vectors"],
        metrics["dims"],
        ratio,
        sim,
    )

    return reconstructed, metrics


def compression_ratio(original: np.ndarray, quantized: np.ndarray) -> float:
    """Compute the byte-level compression ratio.

    Args:
        original:  Original float32 array.
        quantized: Quantized (e.g. int8) array with the same number of elements.

    Returns:
        Ratio > 1.0 means compression.  Returns 1.0 if quantized has no bytes.
    """
    if quantized.nbytes == 0:
        return 1.0
    return round(original.nbytes / quantized.nbytes, 3)
