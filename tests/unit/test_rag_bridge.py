"""Unit tests for konjoai.ingest.rag_bridge.

Tests cover:
  - index_corpus / verify_corpus when Squish IS available (mocked RagScanner)
  - index_corpus / verify_corpus when Squish is NOT available (degraded mode)
  - Endpoint: POST /ingest/manifest
  - Endpoint: GET /ingest/verify
"""
from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scanner_cls(*, file_count=3, manifest_hash="abc123", indexed_at="2026-04-16T00:00:00Z",
                      ok=True, total_files=3, drift_count=0, drift=None):
    """Return a mock RagScanner class with the full Squish API surface."""
    drift = drift or []

    manifest = MagicMock()
    manifest.file_count = file_count
    manifest.manifest_hash = manifest_hash
    manifest.indexed_at = indexed_at

    verify_result = MagicMock()
    verify_result.ok = ok
    verify_result.total_files = total_files
    verify_result.drift_count = drift_count
    verify_result.drift = drift

    scanner = MagicMock()
    scanner.index.return_value = manifest
    scanner.verify.return_value = verify_result

    scanner_cls = MagicMock(return_value=scanner)
    return scanner_cls


def _reset_bridge():
    """Force rag_bridge availability probe to re-run on next call."""
    import konjoai.ingest.rag_bridge as bridge
    bridge._SQUISH_AVAILABLE = None
    bridge._rag_scanner_cls = None


# ---------------------------------------------------------------------------
# index_corpus — squish available
# ---------------------------------------------------------------------------

class TestIndexCorpus:
    def test_returns_file_count_and_hash(self):
        _reset_bridge()
        scanner_cls = _make_scanner_cls(file_count=5, manifest_hash="deadbeef")
        with patch("konjoai.ingest.rag_bridge._check_squish", return_value=True), \
             patch("konjoai.ingest.rag_bridge._rag_scanner_cls", scanner_cls):
            from konjoai.ingest.rag_bridge import index_corpus
            result = index_corpus("/tmp/corpus")
        assert result["available"] is True
        assert result["file_count"] == 5
        assert result["manifest_hash"] == "deadbeef"
        assert result["corpus_dir"] == "/tmp/corpus"

    def test_indexed_at_present(self):
        _reset_bridge()
        scanner_cls = _make_scanner_cls(indexed_at="2026-04-16T00:00:00Z")
        with patch("konjoai.ingest.rag_bridge._check_squish", return_value=True), \
             patch("konjoai.ingest.rag_bridge._rag_scanner_cls", scanner_cls):
            from konjoai.ingest.rag_bridge import index_corpus
            result = index_corpus("/tmp/corpus")
        assert result["indexed_at"] == "2026-04-16T00:00:00Z"


# ---------------------------------------------------------------------------
# index_corpus — squish NOT available
# ---------------------------------------------------------------------------

class TestIndexCorpusDegraded:
    def test_available_false_no_raise(self):
        _reset_bridge()
        with patch("konjoai.ingest.rag_bridge._check_squish", return_value=False):
            from konjoai.ingest.rag_bridge import index_corpus
            result = index_corpus("/tmp/corpus")
        assert result["available"] is False
        assert result["file_count"] == 0
        assert result["manifest_hash"] == ""

    def test_returns_corpus_dir(self):
        _reset_bridge()
        with patch("konjoai.ingest.rag_bridge._check_squish", return_value=False):
            from konjoai.ingest.rag_bridge import index_corpus
            result = index_corpus("/tmp/my_corpus")
        assert result["corpus_dir"] == "/tmp/my_corpus"


# ---------------------------------------------------------------------------
# verify_corpus — squish available
# ---------------------------------------------------------------------------

class TestVerifyCorpus:
    def test_ok_no_drift(self):
        _reset_bridge()
        scanner_cls = _make_scanner_cls(ok=True, total_files=10, drift_count=0)
        with patch("konjoai.ingest.rag_bridge._check_squish", return_value=True), \
             patch("konjoai.ingest.rag_bridge._rag_scanner_cls", scanner_cls):
            from konjoai.ingest.rag_bridge import verify_corpus
            result = verify_corpus("/tmp/corpus")
        assert result["available"] is True
        assert result["ok"] is True
        assert result["drift_count"] == 0
        assert result["drift"] == []

    def test_drift_detected(self):
        _reset_bridge()
        drift_item = MagicMock()
        drift_item.path = "/tmp/corpus/doc.txt"
        drift_item.status = "modified"
        scanner_cls = _make_scanner_cls(ok=False, drift_count=1, drift=[drift_item])
        with patch("konjoai.ingest.rag_bridge._check_squish", return_value=True), \
             patch("konjoai.ingest.rag_bridge._rag_scanner_cls", scanner_cls):
            from konjoai.ingest.rag_bridge import verify_corpus
            result = verify_corpus("/tmp/corpus")
        assert result["ok"] is False
        assert result["drift_count"] == 1
        assert result["drift"][0]["status"] == "modified"


# ---------------------------------------------------------------------------
# verify_corpus — squish NOT available
# ---------------------------------------------------------------------------

class TestVerifyCorpusDegraded:
    def test_available_false_no_raise(self):
        _reset_bridge()
        with patch("konjoai.ingest.rag_bridge._check_squish", return_value=False):
            from konjoai.ingest.rag_bridge import verify_corpus
            result = verify_corpus("/tmp/corpus")
        assert result["available"] is False
        assert result["ok"] is None
        assert result["drift_count"] == 0
        assert result["drift"] == []


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """TestClient with singleton reset between tests."""
    from konjoai.api.app import create_app
    return TestClient(create_app())


class TestManifestEndpoint:
    def test_returns_manifest_response_when_available(self, client):
        _reset_bridge()
        scanner_cls = _make_scanner_cls(file_count=7, manifest_hash="ff00")
        with patch("konjoai.ingest.rag_bridge._check_squish", return_value=True), \
             patch("konjoai.ingest.rag_bridge._rag_scanner_cls", scanner_cls):
            resp = client.post("/ingest/manifest", json={"corpus_dir": "/tmp/corpus"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["available"] is True
        assert body["file_count"] == 7

    def test_returns_degraded_when_squish_absent(self, client):
        _reset_bridge()
        with patch("konjoai.ingest.rag_bridge._check_squish", return_value=False):
            resp = client.post("/ingest/manifest", json={"corpus_dir": "/tmp/corpus"})
        assert resp.status_code == 200
        assert resp.json()["available"] is False


class TestVerifyEndpoint:
    def test_returns_verify_response_when_available(self, client):
        _reset_bridge()
        scanner_cls = _make_scanner_cls(ok=True, total_files=4, drift_count=0)
        with patch("konjoai.ingest.rag_bridge._check_squish", return_value=True), \
             patch("konjoai.ingest.rag_bridge._rag_scanner_cls", scanner_cls):
            resp = client.get("/ingest/verify", params={"corpus_dir": "/tmp/corpus"})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_returns_degraded_when_squish_absent(self, client):
        _reset_bridge()
        with patch("konjoai.ingest.rag_bridge._check_squish", return_value=False):
            resp = client.get("/ingest/verify", params={"corpus_dir": "/tmp/corpus"})
        assert resp.status_code == 200
        assert resp.json()["available"] is False
