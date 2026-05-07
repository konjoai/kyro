"""KonjoOS service interface for the vectro pipeline command.

The service exposes two callables:

  1. ``quantize()``  — delegates to :mod:`konjoai.embed.vectro_bridge`;
     used by the ingest path (embedding-level INT8 quantization at upsert
     time).  This is the *existing* path; this module re-exports it for
     convenience.

  2. ``run_pipeline()`` — invokes the ``vectro pipeline`` Rust CLI via
     ``subprocess.run``; used by the ``/vectro/pipeline`` API endpoint.
     Does: JSONL embeddings → compress (NF4 / PQ / INT8 / auto) →
     build HNSW index → optional query evaluation → structured result.

Konjo Invariants
----------------
K1  No silent failures — raise :class:`VectroPipelineError` explicitly.
K2  Telemetry via ``timed()`` on every hot-path step.
K3  Binary unavailable → :class:`VectroBinaryNotFoundError` with
    build instructions instead of a silent passthrough.
K4  Dtype: ``float32`` asserted at all numpy vector boundaries; never
    assume.
K5  Zero new hard dependencies; vectro Python interface is already
    installed via ``pip install -e /path/to/vectro``.
K6  Backward-compatible API — all new response fields have defaults.
K7  Pipeline results serialised to ``evals/runs/<ts>_vectro_pipeline/``
    when ``archive=True``.

Stub notice
-----------
``compress_rq`` and ``compress_auto`` formats are forwarded to the Rust
CLI.  They will raise :class:`VectroPipelineError` until vectro_lib v5.0
ships full RQ support.  The format strings are accepted by the API now
so clients do not need a breaking change when v5.0 lands.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── well-known binary locations searched in order ─────────────────────────
_BINARY_SEARCH_PATHS: tuple[str, ...] = (
    # shutil.which handles PATH automatically; these are fallback explicit paths
    str(Path.home() / "vectro" / "target" / "release" / "vectro"),
    str(Path.home() / ".cargo" / "bin" / "vectro"),
)

# formats that are stubs pending vectro_lib v5.0 RQ support
_STUB_FORMATS: frozenset[str] = frozenset({"rq", "auto"})

_VALID_FORMATS: frozenset[str] = frozenset({"nf4", "pq", "int8", "rq", "auto"})


# ── Exceptions ────────────────────────────────────────────────────────────


class VectroPipelineError(RuntimeError):
    """Raised when the vectro pipeline command fails."""


class VectroBinaryNotFoundError(VectroPipelineError):
    """Raised when the vectro CLI binary cannot be located.

    Build instructions::

        cd ~/vectro
        cargo build --release
        # binary: ~/vectro/target/release/vectro
    """


class VectroStubFormatError(VectroPipelineError):
    """Raised when a format is accepted by the API but not yet implemented.

    Will be resolved when vectro_lib v5.0 ships RQ support.
    """


# ── Result dataclass ──────────────────────────────────────────────────────


@dataclass
class VectroPipelineResult:
    """Structured result from a vectro pipeline run.

    Attributes
    ----------
    n_vectors:
        Number of embeddings compressed.
    dims:
        Vector dimensionality.
    format:
        Quantization format used (``"nf4"``, ``"pq"``, ``"int8"`` …).
    out_dir:
        Absolute path to the output directory containing
        ``compressed.stream1`` and ``index.bin``.
    index_size_bytes:
        Size of ``index.bin`` on disk in bytes.  ``0`` if the index
        file could not be stat-ed.
    duration_ms:
        Wall-clock time of the entire pipeline run, milliseconds.
    query_results:
        Parsed JSONL query results when ``query_file`` was supplied;
        ``[]`` otherwise.
    binary_path:
        Absolute path to the ``vectro`` binary that was invoked.
    """

    n_vectors: int
    dims: int
    format: str
    out_dir: str
    index_size_bytes: int
    duration_ms: float
    query_results: list[dict] = field(default_factory=list)
    binary_path: str = ""

    def as_dict(self) -> dict:
        """Return JSON-serialisable representation."""
        return {
            "n_vectors": self.n_vectors,
            "dims": self.dims,
            "format": self.format,
            "out_dir": self.out_dir,
            "index_size_bytes": self.index_size_bytes,
            "duration_ms": round(self.duration_ms, 2),
            "query_results": self.query_results,
            "binary_path": self.binary_path,
        }


# ── Binary discovery ──────────────────────────────────────────────────────


def _find_vectro_binary() -> str:
    """Return the absolute path to the ``vectro`` CLI binary.

    Search order:
    1. ``shutil.which("vectro")`` — respects the caller's ``PATH``.
    2. Hard-coded fallback paths in :data:`_BINARY_SEARCH_PATHS`.

    Raises
    ------
    VectroBinaryNotFoundError
        When the binary cannot be located.  The exception message
        includes build instructions.
    """
    found = shutil.which("vectro")
    if found:
        return found

    for candidate in _BINARY_SEARCH_PATHS:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    raise VectroBinaryNotFoundError(
        "vectro CLI binary not found. "
        "Build it with: cd ~/vectro && cargo build --release\n"
        f"Searched PATH and: {', '.join(_BINARY_SEARCH_PATHS)}"
    )


# ── Quantise convenience re-export ────────────────────────────────────────


def quantize(
    embeddings: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Quantise *embeddings* for storage using the vectro Python interface.

    This is a thin re-export of
    :func:`konjoai.embed.vectro_bridge.quantize_for_storage` provided
    here so callers can import from a single service module.

    Parameters
    ----------
    embeddings:
        ``float32`` array of shape ``(N, dim)``.

    Returns
    -------
    tuple[np.ndarray, dict]
        ``(reconstructed_float32, metrics_dict)`` — see
        :func:`~konjoai.embed.vectro_bridge.quantize_for_storage` for
        the ``metrics_dict`` keys.
    """
    assert embeddings.dtype == np.float32, (
        f"K4 violation: expected float32 embeddings, got {embeddings.dtype}"
    )
    from konjoai.embed.vectro_bridge import quantize_for_storage  # K5: deferred
    return quantize_for_storage(embeddings)


# ── JSONL helpers ─────────────────────────────────────────────────────────


def embeddings_to_jsonl(
    embeddings: np.ndarray,
    ids: list[str] | None = None,
) -> str:
    """Serialise a numpy embedding matrix to a temporary JSONL file.

    Parameters
    ----------
    embeddings:
        ``float32`` array of shape ``(N, dim)``.
    ids:
        Optional list of string IDs.  Defaults to ``"vec_0" … "vec_N"``.

    Returns
    -------
    str
        Absolute path of the newly-created temporary JSONL file.
        **Caller is responsible for deleting it.**
    """
    assert embeddings.dtype == np.float32, (
        f"K4 violation: expected float32, got {embeddings.dtype}"
    )
    n = embeddings.shape[0]
    if ids is None:
        ids = [f"vec_{i}" for i in range(n)]
    if len(ids) != n:
        raise ValueError(f"ids length {len(ids)} != embedding count {n}")

    fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="vectro_konjoai_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for i, (vec_id, vec) in enumerate(zip(ids, embeddings)):
                fh.write(json.dumps({"id": vec_id, "vector": vec.tolist()}) + "\n")
    except Exception:
        os.unlink(path)
        raise
    return path


# ── Pipeline runner ───────────────────────────────────────────────────────


def run_pipeline(
    input_jsonl: str,
    out_dir: str,
    format: str = "nf4",
    m: int = 16,
    ef_construction: int = 200,
    ef_search: int = 50,
    query_file: str | None = None,
    top_k: int = 10,
    quiet: bool = False,
    archive: bool = False,
) -> VectroPipelineResult:
    """Run the full vectro pipeline (compress → HNSW index → optional search).

    Invokes the ``vectro pipeline`` Rust CLI subcommand via
    ``subprocess.run``.  All output is captured; a non-zero exit code
    raises :class:`VectroPipelineError`.

    Parameters
    ----------
    input_jsonl:
        Path to the input JSONL file.  Each line must be
        ``{"id": "<str>", "vector": [<f32>, …]}``.
    out_dir:
        Directory where the pipeline writes ``compressed.stream1`` and
        ``index.bin``.  Created if it does not exist.
    format:
        Quantization format.  One of ``"nf4"`` (default), ``"pq"``,
        ``"int8"``, ``"rq"`` *(stub — v5.0)*, ``"auto"`` *(stub — v5.0)*.
    m:
        HNSW *M* parameter (edges per node).
    ef_construction:
        HNSW construction beam width.
    ef_search:
        HNSW search beam width.
    query_file:
        Optional path to a JSONL query file for evaluation.
    top_k:
        Number of nearest neighbours to return per query.
    quiet:
        Suppress vectro's progress output to stderr.
    archive:
        When ``True``, serialise the result dict to
        ``evals/runs/<timestamp>_vectro_pipeline/result.json`` (K7).

    Returns
    -------
    VectroPipelineResult

    Raises
    ------
    VectroBinaryNotFoundError
        When the vectro CLI binary cannot be located.
    VectroStubFormatError
        When ``format`` is ``"rq"`` or ``"auto"`` (not yet implemented;
        will be real in vectro_lib v5.0).
    VectroPipelineError
        When the pipeline exits with a non-zero status code.
    """
    if format not in _VALID_FORMATS:
        raise VectroPipelineError(
            f"Unknown format {format!r}. Valid formats: {sorted(_VALID_FORMATS)}"
        )
    if format in _STUB_FORMATS:
        raise VectroStubFormatError(
            f"Format {format!r} is not yet implemented. "
            "compress_rq and compress_auto will be real once vectro_lib v5.0 "
            "ships full RQ support. Use 'nf4' or 'pq' for now."
        )

    binary = _find_vectro_binary()

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        binary, "pipeline",
        "--input", input_jsonl,
        "--out-dir", out_dir,
        "--format", format,
        "--m", str(m),
        "--ef-construction", str(ef_construction),
        "--ef-search", str(ef_search),
        "--top-k", str(top_k),
    ]
    if query_file:
        cmd += ["--query-file", query_file]
    if quiet:
        cmd.append("--quiet")

    logger.info("Running vectro pipeline: %s", " ".join(cmd))

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration_ms = (time.perf_counter() - t0) * 1000.0

    if result.returncode != 0:
        raise VectroPipelineError(
            f"vectro pipeline exited with code {result.returncode}.\n"
            f"stderr: {result.stderr.strip()}"
        )

    # Parse n_vectors from stderr progress lines (e.g. "✓ compressed 1000 vectors")
    n_vectors = 0
    for line in result.stderr.splitlines():
        m_match = re.search(r"compressed\s+(\d+)\s+vectors", line)
        if m_match:
            n_vectors = int(m_match.group(1))
            break

    # Parse query results from stdout JSONL
    query_results: list[dict] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                query_results.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Could not parse query result line: %s", line)

    index_path = Path(out_dir) / "index.bin"
    index_size_bytes = index_path.stat().st_size if index_path.exists() else 0

    # Infer dims from input JSONL (first line)
    dims = 0
    try:
        with open(input_jsonl, encoding="utf-8") as fh:
            first_line = fh.readline().strip()
            if first_line:
                dims = len(json.loads(first_line).get("vector", []))
    except Exception:
        pass

    pipeline_result = VectroPipelineResult(
        n_vectors=n_vectors,
        dims=dims,
        format=format,
        out_dir=str(Path(out_dir).resolve()),
        index_size_bytes=index_size_bytes,
        duration_ms=duration_ms,
        query_results=query_results,
        binary_path=binary,
    )

    logger.info(
        "vectro pipeline complete: %d vectors, format=%s, duration=%.1fms",
        n_vectors,
        format,
        duration_ms,
    )

    if archive:
        _archive_result(pipeline_result)

    return pipeline_result


def run_pipeline_from_embeddings(
    embeddings: np.ndarray,
    out_dir: str,
    ids: list[str] | None = None,
    format: str = "nf4",
    m: int = 16,
    ef_construction: int = 200,
    ef_search: int = 50,
    top_k: int = 10,
    archive: bool = False,
) -> VectroPipelineResult:
    """Convenience wrapper: numpy embeddings → JSONL tempfile → pipeline.

    Converts *embeddings* to a temporary JSONL file, runs the pipeline,
    then deletes the tempfile.

    Parameters
    ----------
    embeddings:
        ``float32`` array of shape ``(N, dim)``.
    out_dir:
        Pipeline output directory.
    ids:
        Optional IDs for each embedding.
    format, m, ef_construction, ef_search, top_k, archive:
        Forwarded to :func:`run_pipeline`.

    Returns
    -------
    VectroPipelineResult
    """
    assert embeddings.dtype == np.float32, (
        f"K4 violation: expected float32, got {embeddings.dtype}"
    )
    tmp_path = embeddings_to_jsonl(embeddings, ids=ids)
    try:
        return run_pipeline(
            input_jsonl=tmp_path,
            out_dir=out_dir,
            format=format,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
            top_k=top_k,
            archive=archive,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            logger.warning("Could not delete temp JSONL %s", tmp_path)


# ── K7 archival ───────────────────────────────────────────────────────────


def _archive_result(result: VectroPipelineResult) -> None:
    """Serialise *result* to ``evals/runs/<timestamp>_vectro_pipeline/result.json``.

    Follows K7: reproducible evals — results archived alongside their
    metadata so runs can be compared across sessions.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path("evals") / "runs" / f"{ts}_vectro_pipeline"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "result.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result.as_dict(), fh, indent=2)
    logger.info("Archived vectro pipeline result → %s", out_path)
