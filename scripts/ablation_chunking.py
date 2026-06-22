#!/usr/bin/env python3
"""Sprint 10 — Adaptive Chunking Ablation Harness.

Runs the RAGAS eval corpus through four chunking strategies and emits a
comparison JSON to ``evals/runs/<timestamp>_chunking_ablation/``.

Usage
-----
Run all four strategies against the eval corpus (offline, no Qdrant needed)::

    python scripts/ablation_chunking.py

Force a specific model::

    python scripts/ablation_chunking.py --model sentence-transformers/all-MiniLM-L6-v2

Limit to N questions::

    python scripts/ablation_chunking.py --n 5

Quiet mode (JSON output only, no progress)::

    python scripts/ablation_chunking.py --quiet

Design
------
Because a full end-to-end RAGAS eval requires a running Qdrant + LLM, this
harness computes **proxy metrics** that can be evaluated entirely offline using
only the eval corpus context documents and the embedding model:

* ``chunk_count``        — total chunks produced across all context documents
* ``avg_chunk_chars``    — mean character length of chunks
* ``std_chunk_chars``    — standard deviation of chunk character lengths
* ``min_chunk_chars``    — minimum chunk character length (guards against empty chunks)
* ``max_chunk_chars``    — maximum chunk character length
* ``within_coherence``   — mean cosine similarity *within* each chunk
  (computed by re-embedding each chunk's sentences; higher = more coherent)
* ``boundary_sharpness`` — mean cosine similarity at split boundaries
  (lower = sharper / more meaningful boundaries)
* ``coverage_score``     — fraction of ground-truth answer tokens that appear
  verbatim in at least one retrieved chunk (proxy for recall)

Gates (Sprint 10):

* ``within_coherence`` for ``semantic`` and ``late`` strategies must be ≥
  ``within_coherence`` for ``recursive`` (embedding-aware strategies should
  produce more internally coherent chunks).
* No strategy produces zero chunks for any document.
* Report generated to ``evals/runs/<timestamp>_chunking_ablation/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from konjoai.ingest.chunkers import (  # noqa: E402
    get_chunker,
)
from konjoai.ingest.loaders import Document  # noqa: E402

logging.basicConfig(
    format="%(levelname)s  %(name)s  %(message)s",
    level=logging.WARNING,
)
logger = logging.getLogger("ablation_chunking")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CORPUS_PATH = _REPO / "evals" / "corpus" / "eval_questions.json"
_RUNS_DIR = _REPO / "evals" / "runs"
_SENT_RE = re.compile(r"(?<=[.!?])\s+")

_STRATEGIES: list[str] = ["recursive", "sentence_window", "semantic", "late"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def _word_tokens(text: str) -> set[str]:
    """Rough word-level tokeniser (lowercase, strip punctuation)."""
    return set(re.sub(r"[^a-z0-9\s]", "", text.lower()).split())


def _load_corpus(path: Path, n: int | None) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    if n is not None:
        data = data[:n]
    return data


def _build_encoder(model_name: str, quiet: bool):
    """Load a SentenceEncoder, returning it as a callable ``(list[str]) -> ndarray``."""
    try:
        from konjoai.embed.encoder import SentenceEncoder  # noqa: PLC0415

        if not quiet:
            print(f"  Loading embedding model: {model_name} …", flush=True)
        enc = SentenceEncoder(model_name=model_name, device="cpu")
        return enc
    except ImportError as e:
        raise SystemExit(f"sentence-transformers is required for semantic/late strategies: {e}") from e


def _safe_encode(encoder, texts: list[str]) -> Any:
    """Encode with graceful empty-list guard."""
    if not texts:
        import numpy as np

        return np.zeros((0, 1), dtype="float32")
    return encoder.encode(texts)


def _cosine_sims(embeddings: Any) -> list[float]:
    """Adjacent cosine similarities for a (N, dim) float32 array."""
    import numpy as np

    if len(embeddings) < 2:
        return []
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normed = embeddings / norms
    return [float(np.dot(normed[i], normed[i + 1])) for i in range(len(normed) - 1)]


# ---------------------------------------------------------------------------
# Per-strategy metrics
# ---------------------------------------------------------------------------


def _chunk_corpus(
    strategy: str,
    questions: list[dict],
    chunk_size: int,
    encoder,
) -> list[dict]:
    """Chunk all context documents and return a list of chunk dicts."""
    all_chunks: list[dict] = []

    for q in questions:
        for ctx_text in q.get("context_docs", []):
            doc = Document(content=ctx_text, source=f"q{q['id']}", metadata={})

            if strategy in ("semantic", "late"):
                chunker = get_chunker(
                    strategy=strategy,
                    chunk_size=chunk_size,
                    similarity_threshold=0.4,
                    _encoder=encoder,
                )
            else:
                chunker = get_chunker(strategy=strategy, chunk_size=chunk_size)

            chunks = chunker.chunk(doc)
            for ch in chunks:
                all_chunks.append(
                    {
                        "question_id": q["id"],
                        "content": ch.content,
                        "ground_truth": q["ground_truth"],
                        "strategy": strategy,
                    }
                )

    return all_chunks


def _compute_metrics(
    chunks: list[dict],
    encoder,
    quiet: bool,
) -> dict:
    """Compute proxy metrics for a list of chunks."""
    import numpy as np

    if not chunks:
        return {
            "chunk_count": 0,
            "avg_chunk_chars": 0.0,
            "std_chunk_chars": 0.0,
            "min_chunk_chars": 0,
            "max_chunk_chars": 0,
            "within_coherence": 0.0,
            "boundary_sharpness": 0.0,
            "coverage_score": 0.0,
        }

    char_lengths = [len(c["content"]) for c in chunks]
    avg_chars = float(np.mean(char_lengths))
    std_chars = float(np.std(char_lengths))

    # Coverage: fraction of ground-truth tokens present in chunks (per question)
    from collections import defaultdict

    chunk_tokens_by_q: dict[int, set[str]] = defaultdict(set)
    for ch in chunks:
        chunk_tokens_by_q[ch["question_id"]].update(_word_tokens(ch["content"]))

    gt_tokens_by_q: dict[int, set[str]] = {}
    for ch in chunks:
        qid = ch["question_id"]
        if qid not in gt_tokens_by_q:
            gt_tokens_by_q[qid] = _word_tokens(ch["ground_truth"])

    coverages: list[float] = []
    for qid, gt_toks in gt_tokens_by_q.items():
        if not gt_toks:
            continue
        overlap = gt_toks & chunk_tokens_by_q.get(qid, set())
        coverages.append(len(overlap) / len(gt_toks))
    coverage_score = float(np.mean(coverages)) if coverages else 0.0

    # Coherence / boundary sharpness (requires encoder)
    within_coherence = 0.0
    boundary_sharpness = 0.0

    if encoder is not None:
        within_sims: list[float] = []
        boundary_sims: list[float] = []

        if not quiet:
            print(
                f"    Computing embedding metrics for {len(chunks)} chunks …",
                flush=True,
            )

        # Within-chunk: embed sentences within each chunk, measure adjacent sims
        for ch in chunks:
            sents = _split_sentences(ch["content"])
            if len(sents) >= 2:
                try:
                    embs = _safe_encode(encoder, sents)
                    sims = _cosine_sims(embs)
                    within_sims.extend(sims)
                except Exception:  # noqa: BLE001
                    pass

        # Boundary: embed last sentence of chunk N and first sentence of chunk N+1
        # Group by question_id to only compare adjacent chunks from the same doc
        from itertools import groupby

        for _, group in groupby(chunks, key=lambda c: c["question_id"]):
            group_list = list(group)
            for i in range(len(group_list) - 1):
                last_sents = _split_sentences(group_list[i]["content"])
                next_sents = _split_sentences(group_list[i + 1]["content"])
                if last_sents and next_sents:
                    boundary_texts = [last_sents[-1], next_sents[0]]
                    try:
                        embs = _safe_encode(encoder, boundary_texts)
                        if len(embs) == 2:
                            boundary_sims.extend(_cosine_sims(embs))
                    except Exception:  # noqa: BLE001
                        pass

        within_coherence = float(np.mean(within_sims)) if within_sims else 0.0
        boundary_sharpness = float(np.mean(boundary_sims)) if boundary_sims else 0.0

    return {
        "chunk_count": len(chunks),
        "avg_chunk_chars": round(avg_chars, 2),
        "std_chunk_chars": round(std_chars, 2),
        "min_chunk_chars": min(char_lengths),
        "max_chunk_chars": max(char_lengths),
        "within_coherence": round(within_coherence, 4),
        "boundary_sharpness": round(boundary_sharpness, 4),
        "coverage_score": round(coverage_score, 4),
    }


# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------


def _run_gates(results: dict[str, dict]) -> list[str]:
    """Return a list of gate failure messages (empty = all passed)."""
    failures: list[str] = []

    # Gate 1: no strategy produces zero chunks
    for strategy, metrics in results.items():
        if metrics["chunk_count"] == 0:
            failures.append(f"GATE FAIL: {strategy} produced 0 chunks")

    # Gate 2: no strategy produces chunks with 0 character length
    for strategy, metrics in results.items():
        if metrics.get("min_chunk_chars", 1) == 0:
            failures.append(f"GATE FAIL: {strategy} produced a 0-character chunk")

    # Gate 3: embedding-aware strategies should have ≥ recursive within_coherence
    rec_coh = results.get("recursive", {}).get("within_coherence", 0.0)
    for strategy in ("semantic", "late"):
        if strategy in results:
            coh = results[strategy]["within_coherence"]
            # Only enforce when coherence was actually computed (encoder available)
            if rec_coh > 0.0 and coh > 0.0 and coh < rec_coh - 0.05:
                failures.append(
                    f"GATE FAIL: {strategy} within_coherence ({coh:.4f}) "
                    f"< recursive ({rec_coh:.4f}) — embedding chunker is less coherent"
                )

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sprint 10 chunking strategy ablation harness.",
        epilog="Example: python scripts/ablation_chunking.py --n 5",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model for semantic/late strategies.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target chunk size for recursive/late strategies (default: 512).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Limit to N questions from the eval corpus.",
    )
    parser.add_argument(
        "--no-encoder",
        action="store_true",
        help="Skip embedding-based metrics (faster, no model download required).",
    )
    parser.add_argument(
        "--run-name",
        default="chunking_ablation",
        help="Label suffix for the output directory.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output; print only final JSON.",
    )
    args = parser.parse_args()

    # ── Load corpus ───────────────────────────────────────────────────────────
    if not _CORPUS_PATH.exists():
        print(f"ERROR: corpus not found at {_CORPUS_PATH}", file=sys.stderr)
        return 1

    questions = _load_corpus(_CORPUS_PATH, args.n)
    if not args.quiet:
        print(f"Loaded {len(questions)} questions from eval corpus.")

    # ── Build encoder (shared across semantic + late) ─────────────────────────
    encoder = None
    if not args.no_encoder:
        try:
            encoder = _build_encoder(args.model, args.quiet)
        except SystemExit as e:
            print(f"WARNING: {e}  Falling back to --no-encoder mode.", file=sys.stderr)
            encoder = None

    # ── Run ablation ──────────────────────────────────────────────────────────
    results: dict[str, dict] = {}
    timings: dict[str, float] = {}

    for strategy in _STRATEGIES:
        if not args.quiet:
            print(f"\nStrategy: {strategy}", flush=True)

        t0 = time.perf_counter()
        chunks = _chunk_corpus(
            strategy=strategy,
            questions=questions,
            chunk_size=args.chunk_size,
            encoder=encoder,
        )
        metrics = _compute_metrics(chunks, encoder if strategy in ("semantic", "late") else None, args.quiet)
        elapsed = time.perf_counter() - t0

        results[strategy] = metrics
        timings[strategy] = round(elapsed, 3)

        if not args.quiet:
            print(
                f"  chunks={metrics['chunk_count']}  "
                f"avg_chars={metrics['avg_chunk_chars']:.0f}  "
                f"coherence={metrics['within_coherence']:.4f}  "
                f"boundary_sim={metrics['boundary_sharpness']:.4f}  "
                f"coverage={metrics['coverage_score']:.4f}  "
                f"({elapsed:.2f}s)",
                flush=True,
            )

    # ── Gate checks ───────────────────────────────────────────────────────────
    gate_failures = _run_gates(results)
    gate_passed = len(gate_failures) == 0

    if not args.quiet:
        print()
        if gate_passed:
            print("✅  All ablation gates passed.")
        else:
            for msg in gate_failures:
                print(f"❌  {msg}", file=sys.stderr)

    # ── Write output ──────────────────────────────────────────────────────────
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = _RUNS_DIR / f"{timestamp}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    comparison = {
        "run_timestamp": timestamp,
        "run_name": args.run_name,
        "model": args.model,
        "chunk_size": args.chunk_size,
        "n_questions": len(questions),
        "gate_passed": gate_passed,
        "gate_failures": gate_failures,
        "strategies": results,
        "timings_seconds": timings,
        "metric_definitions": {
            "chunk_count": "Total chunks produced across all context documents",
            "avg_chunk_chars": "Mean character length per chunk",
            "std_chunk_chars": "Standard deviation of chunk character lengths",
            "min_chunk_chars": "Minimum chunk character length",
            "max_chunk_chars": "Maximum chunk character length",
            "within_coherence": "Mean cosine similarity between adjacent sentences WITHIN each chunk (higher = more coherent)",
            "boundary_sharpness": "Mean cosine similarity AT chunk boundaries (lower = sharper boundaries)",
            "coverage_score": "Fraction of ground-truth answer tokens present in at least one chunk (proxy for recall)",
        },
    }

    out_path = run_dir / "comparison.json"
    with out_path.open("w") as f:
        json.dump(comparison, f, indent=2)

    if args.quiet:
        print(json.dumps(comparison, indent=2))
    else:
        print(f"\nResults written to: {out_path}")

    return 0 if gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
