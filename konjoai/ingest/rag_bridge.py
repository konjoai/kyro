"""RagScanner bridge for corpus file-integrity checking.

Graceful-degradation wrapper around the Squish RagScanner
(squish/squish/squash/rag.py) which tracks per-file SHA-256 hashes to
detect corpus drift between ingest runs.

Import strategy (tried in order):
    1. ``squish.squash.rag``   — importable when squish is pip-installed.
    2. Path-injected fallback  — inserts ``~/squish`` onto sys.path scoped
       to this module so the package resolves without a prior install.
    3. Passthrough             — ``_SQUISH_AVAILABLE = False``, all public
       functions return degraded-mode dicts (K3 graceful degradation).

RagScanner API contract (squish.squash.rag):
    RagScanner.index(corpus_dir: str | Path) -> RagManifest
        .file_count: int
        .manifest_hash: str
        .indexed_at: str (ISO-8601)
        .write(path: str | Path)
    RagScanner.verify(corpus_dir: str | Path) -> RagVerifyResult
        .ok: bool
        .total_files: int
        .drift_count: int
        .drift: list[RagDriftItem]
            .path: str
            .status: "added" | "removed" | "modified"
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path registration — try pip-installed first; fall back to source tree.
# We insert ~/squish (not ~) to scope the path addition to Squish only.
# ---------------------------------------------------------------------------
_SQUISH_SRC = Path.home() / "squish"
_SQUISH_SRC_STR = str(_SQUISH_SRC)
if _SQUISH_SRC.is_dir() and _SQUISH_SRC_STR not in sys.path:
    sys.path.insert(0, _SQUISH_SRC_STR)

# ---------------------------------------------------------------------------
# Availability probe (cached after first call)
# ---------------------------------------------------------------------------

_SQUISH_AVAILABLE: bool | None = None
_rag_scanner_cls: Any = None


def _check_squish() -> bool:
    """Return True if the Squish RagScanner is importable.

    Caches the result on first call so subsequent calls are O(1) dictionary
    lookups — important for hot-path ingest.
    """
    global _SQUISH_AVAILABLE, _rag_scanner_cls  # noqa: PLW0603

    if _SQUISH_AVAILABLE is not None:
        return _SQUISH_AVAILABLE

    try:
        from squish.squash.rag import RagScanner  # type: ignore[import]
        _rag_scanner_cls = RagScanner
        _SQUISH_AVAILABLE = True
        logger.debug("RagScanner available (squish found)")
    except Exception:
        _SQUISH_AVAILABLE = False
        logger.debug("Squish not available; corpus integrity checks disabled (K3)")

    return _SQUISH_AVAILABLE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def index_corpus(corpus_dir: str) -> dict[str, Any]:
    """Hash every file in *corpus_dir* and write a .rag_manifest.json.

    Returns a dict with keys: ``file_count``, ``manifest_hash``,
    ``indexed_at``, ``corpus_dir``, ``available``.

    When Squish is unavailable, returns ``{"available": False, ...}``
    without raising (K3).
    """
    if not _check_squish():
        return {
            "available": False,
            "corpus_dir": corpus_dir,
            "file_count": 0,
            "manifest_hash": "",
            "indexed_at": "",
        }

    scanner = _rag_scanner_cls()
    manifest = scanner.index(corpus_dir)
    return {
        "available": True,
        "corpus_dir": corpus_dir,
        "file_count": getattr(manifest, "file_count", 0),
        "manifest_hash": getattr(manifest, "manifest_hash", ""),
        "indexed_at": getattr(manifest, "indexed_at", ""),
    }


def verify_corpus(corpus_dir: str) -> dict[str, Any]:
    """Compare the current corpus against the stored manifest.

    Returns a dict with keys: ``ok``, ``total_files``, ``drift_count``,
    ``drift`` (list of ``{"path": str, "status": str}``), ``available``.

    When Squish is unavailable, returns ``{"available": False, "ok": None}``
    without raising (K3).
    """
    if not _check_squish():
        return {
            "available": False,
            "ok": None,
            "total_files": 0,
            "drift_count": 0,
            "drift": [],
        }

    scanner = _rag_scanner_cls()
    result = scanner.verify(corpus_dir)
    drift = [
        {"path": getattr(item, "path", ""), "status": getattr(item, "status", "")}
        for item in getattr(result, "drift", [])
    ]
    return {
        "available": True,
        "ok": getattr(result, "ok", None),
        "total_files": getattr(result, "total_files", 0),
        "drift_count": getattr(result, "drift_count", 0),
        "drift": drift,
    }
