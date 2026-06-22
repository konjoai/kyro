"""Unit tests for konjoai.services.vectro_pipeline_service.

Test taxonomy:
- Pure unit: no I/O, deterministic, no real subprocesses.
- Uses monkeypatch to intercept subprocess.run and shutil.which.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_BINARY = "/usr/local/bin/vectro"


def _make_fake_run(returncode: int = 0, stdout: str = "", stderr: str = "") -> Any:
    """Return a mock for subprocess.run that returns a fake CompletedProcess."""

    def _fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    return _fake_run


# ---------------------------------------------------------------------------
# _find_vectro_binary
# ---------------------------------------------------------------------------


class TestFindVectroBinary:
    def test_raises_when_not_on_path_and_no_known_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from konjoai.services import vectro_pipeline_service as svc

        monkeypatch.setattr("shutil.which", lambda _: None)
        # Override _BINARY_SEARCH_PATHS so none exist
        monkeypatch.setattr(svc, "_BINARY_SEARCH_PATHS", ("/nonexistent/vectro",))

        with pytest.raises(svc.VectroBinaryNotFoundError, match="cargo build"):
            svc._find_vectro_binary()

    def test_finds_binary_on_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from konjoai.services import vectro_pipeline_service as svc

        monkeypatch.setattr("shutil.which", lambda _: _FAKE_BINARY)

        result = svc._find_vectro_binary()
        assert result == _FAKE_BINARY

    def test_finds_binary_at_search_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from konjoai.services import vectro_pipeline_service as svc

        binary = tmp_path / "vectro"
        binary.touch(mode=0o755)

        monkeypatch.setattr("shutil.which", lambda _: None)
        monkeypatch.setattr(svc, "_BINARY_SEARCH_PATHS", (str(binary),))

        result = svc._find_vectro_binary()
        assert result == str(binary)


# ---------------------------------------------------------------------------
# Stub format guard
# ---------------------------------------------------------------------------


class TestStubFormatError:
    @pytest.mark.parametrize("fmt", ["rq", "auto"])
    def test_stub_format_raises(self, fmt: str, monkeypatch: pytest.MonkeyPatch) -> None:
        from konjoai.services import vectro_pipeline_service as svc

        monkeypatch.setattr("shutil.which", lambda _: _FAKE_BINARY)
        monkeypatch.setattr(svc, "_BINARY_SEARCH_PATHS", ())

        with pytest.raises(svc.VectroStubFormatError, match="v5.0"):
            svc.run_pipeline(
                input_jsonl="/tmp/fake.jsonl",
                out_dir="/tmp/fake_out",
                format=fmt,
            )

    def test_valid_formats_do_not_raise_stub_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from konjoai.services import vectro_pipeline_service as svc

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        # Create fake index so index_size_bytes resolves
        (out_dir / "index.bin").write_bytes(b"\x00" * 64)

        monkeypatch.setattr("shutil.which", lambda _: _FAKE_BINARY)
        monkeypatch.setattr(svc, "_BINARY_SEARCH_PATHS", ())
        monkeypatch.setattr(
            "subprocess.run",
            _make_fake_run(
                returncode=0,
                stderr="compressed 3 vectors",
                stdout="",
            ),
        )

        # Write a minimal JSONL input so dims can be inferred
        input_file = tmp_path / "input.jsonl"
        input_file.write_text(json.dumps({"id": "a", "vector": [0.1, 0.2, 0.3]}) + "\n")

        result = svc.run_pipeline(
            input_jsonl=str(input_file),
            out_dir=str(out_dir),
            format="nf4",
        )
        assert result.n_vectors == 3
        assert result.dims == 3


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------


class TestRunPipeline:
    def test_raises_binary_not_found_when_binary_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from konjoai.services import vectro_pipeline_service as svc

        monkeypatch.setattr("shutil.which", lambda _: None)
        monkeypatch.setattr(svc, "_BINARY_SEARCH_PATHS", ())

        with pytest.raises(svc.VectroBinaryNotFoundError):
            svc.run_pipeline(input_jsonl="/x.jsonl", out_dir="/out", format="nf4")

    def test_subprocess_invoked_with_correct_args(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from konjoai.services import vectro_pipeline_service as svc

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "index.bin").write_bytes(b"\x00" * 128)

        input_file = tmp_path / "in.jsonl"
        input_file.write_text(json.dumps({"id": "x", "vector": [1.0, 2.0]}) + "\n")

        captured: dict[str, Any] = {}

        def _capture_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="compressed 5 vectors")

        monkeypatch.setattr("shutil.which", lambda _: _FAKE_BINARY)
        monkeypatch.setattr(svc, "_BINARY_SEARCH_PATHS", ())
        monkeypatch.setattr("subprocess.run", _capture_run)

        svc.run_pipeline(
            input_jsonl=str(input_file),
            out_dir=str(out_dir),
            format="int8",
            m=32,
            ef_construction=100,
            ef_search=25,
        )

        cmd = captured["cmd"]
        assert cmd[0] == _FAKE_BINARY
        assert "pipeline" in cmd
        assert "--format" in cmd and cmd[cmd.index("--format") + 1] == "int8"
        assert "--m" in cmd and cmd[cmd.index("--m") + 1] == "32"
        assert "--ef-construction" in cmd
        assert "--ef-search" in cmd

    def test_subprocess_nonzero_exit_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from konjoai.services import vectro_pipeline_service as svc

        input_file = tmp_path / "in.jsonl"
        input_file.write_text(json.dumps({"id": "x", "vector": [0.1]}) + "\n")

        monkeypatch.setattr("shutil.which", lambda _: _FAKE_BINARY)
        monkeypatch.setattr(svc, "_BINARY_SEARCH_PATHS", ())
        monkeypatch.setattr(
            "subprocess.run",
            _make_fake_run(returncode=1, stderr="error: bad input"),
        )

        with pytest.raises(svc.VectroPipelineError, match="code 1"):
            svc.run_pipeline(
                input_jsonl=str(input_file),
                out_dir=str(tmp_path / "out"),
                format="nf4",
            )

    def test_query_results_parsed_from_stdout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from konjoai.services import vectro_pipeline_service as svc

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "index.bin").write_bytes(b"\x00" * 32)

        input_file = tmp_path / "in.jsonl"
        input_file.write_text(json.dumps({"id": "a", "vector": [1.0, 0.0]}) + "\n")

        stdout_lines = "\n".join(
            [
                json.dumps({"query_id": "q1", "results": [{"id": "a", "score": 0.99}]}),
                json.dumps({"query_id": "q2", "results": [{"id": "b", "score": 0.88}]}),
            ]
        )

        monkeypatch.setattr("shutil.which", lambda _: _FAKE_BINARY)
        monkeypatch.setattr(svc, "_BINARY_SEARCH_PATHS", ())
        monkeypatch.setattr(
            "subprocess.run",
            _make_fake_run(returncode=0, stdout=stdout_lines, stderr="compressed 1 vectors"),
        )

        result = svc.run_pipeline(str(input_file), str(out_dir), format="nf4")
        assert len(result.query_results) == 2
        assert result.query_results[0]["query_id"] == "q1"


# ---------------------------------------------------------------------------
# run_pipeline_from_embeddings
# ---------------------------------------------------------------------------


class TestRunPipelineFromEmbeddings:
    def test_dtype_assertion_raises_for_non_float32(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from konjoai.services import vectro_pipeline_service as svc

        monkeypatch.setattr("shutil.which", lambda _: _FAKE_BINARY)
        monkeypatch.setattr(svc, "_BINARY_SEARCH_PATHS", ())

        bad = np.zeros((4, 8), dtype=np.float64)
        with pytest.raises(AssertionError, match="float32"):
            svc.run_pipeline_from_embeddings(bad, out_dir=str(tmp_path / "out"))

    def test_tempfile_cleaned_up_on_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from konjoai.services import vectro_pipeline_service as svc

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "index.bin").write_bytes(b"\x00" * 32)

        created: list[str] = []

        orig_jsonl = svc.embeddings_to_jsonl

        def _tracking_jsonl(embeddings: np.ndarray, ids: Any = None) -> str:
            path = orig_jsonl(embeddings, ids)
            created.append(path)
            return path

        monkeypatch.setattr(svc, "embeddings_to_jsonl", _tracking_jsonl)
        monkeypatch.setattr("shutil.which", lambda _: _FAKE_BINARY)
        monkeypatch.setattr(svc, "_BINARY_SEARCH_PATHS", ())
        monkeypatch.setattr(
            "subprocess.run",
            _make_fake_run(returncode=0, stderr="compressed 5 vectors"),
        )

        vecs = np.random.randn(5, 16).astype(np.float32)
        svc.run_pipeline_from_embeddings(vecs, out_dir=str(out_dir))

        assert created, "embeddings_to_jsonl was never called"
        for path in created:
            assert not os.path.exists(path), f"Tempfile not cleaned up: {path}"

    def test_tempfile_cleaned_up_on_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from konjoai.services import vectro_pipeline_service as svc

        created: list[str] = []
        orig_jsonl = svc.embeddings_to_jsonl

        def _tracking_jsonl(embeddings: np.ndarray, ids: Any = None) -> str:
            path = orig_jsonl(embeddings, ids)
            created.append(path)
            return path

        monkeypatch.setattr(svc, "embeddings_to_jsonl", _tracking_jsonl)
        monkeypatch.setattr("shutil.which", lambda _: _FAKE_BINARY)
        monkeypatch.setattr(svc, "_BINARY_SEARCH_PATHS", ())
        monkeypatch.setattr(
            "subprocess.run",
            _make_fake_run(returncode=2, stderr="fatal error"),
        )

        vecs = np.random.randn(3, 8).astype(np.float32)
        with pytest.raises(svc.VectroPipelineError):
            svc.run_pipeline_from_embeddings(vecs, out_dir=str(tmp_path / "out"))

        for path in created:
            assert not os.path.exists(path), f"Tempfile not cleaned up on error: {path}"


# ---------------------------------------------------------------------------
# embeddings_to_jsonl
# ---------------------------------------------------------------------------


class TestEmbeddingsToJsonl:
    def test_creates_valid_jsonl(self, tmp_path: Path) -> None:
        from konjoai.services import vectro_pipeline_service as svc

        vecs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        path = svc.embeddings_to_jsonl(vecs, ids=["alpha", "beta"])
        try:
            lines = Path(path).read_text().strip().split("\n")
            assert len(lines) == 2
            rec = json.loads(lines[0])
            assert rec["id"] == "alpha"
            assert rec["vector"] == pytest.approx([1.0, 2.0])
        finally:
            os.unlink(path)

    def test_auto_ids_when_not_provided(self) -> None:
        from konjoai.services import vectro_pipeline_service as svc

        vecs = np.zeros((3, 4), dtype=np.float32)
        path = svc.embeddings_to_jsonl(vecs)
        try:
            lines = Path(path).read_text().strip().split("\n")
            ids = [json.loads(line)["id"] for line in lines]
            assert len(ids) == 3
            assert all(isinstance(i, str) for i in ids)
        finally:
            os.unlink(path)
