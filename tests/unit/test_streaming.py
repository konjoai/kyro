"""Unit tests for Phase C streaming additions.

Covers:
- generate_stream() on OpenAIGenerator, AnthropicGenerator, SquishGenerator
- /query/stream SSE endpoint (happy path + chat-intent short-circuit + fallback)
"""
from __future__ import annotations

import asyncio
import inspect
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Generator streaming tests
# ---------------------------------------------------------------------------

class TestOpenAIGeneratorStream:
    """generate_stream() yields tokens from the OpenAI streaming API."""

    def _make_generator(self) -> "OpenAIGenerator":  # noqa: F821
        from konjoai.generate.generator import OpenAIGenerator

        gen = OpenAIGenerator.__new__(OpenAIGenerator)
        gen._model = "gpt-4o-mini"
        gen._max_tokens = 512
        return gen

    def _make_chunk(self, content: str | None) -> MagicMock:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = content
        return chunk

    def test_yields_token_strings(self):
        from konjoai.generate.generator import OpenAIGenerator

        gen = self._make_generator()
        tokens = ["Hello", " world", "!"]
        mock_stream = iter([self._make_chunk(t) for t in tokens])
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_stream
        gen._client = mock_client

        result = list(gen.generate_stream(question="hi", context="ctx"))
        assert result == tokens

    def test_none_delta_yields_empty_string(self):
        from konjoai.generate.generator import OpenAIGenerator

        gen = self._make_generator()
        mock_stream = iter([self._make_chunk(None)])
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_stream
        gen._client = mock_client

        result = list(gen.generate_stream(question="q", context="c"))
        assert result == [""]

    def test_stream_flag_passed_to_client(self):
        from konjoai.generate.generator import OpenAIGenerator

        gen = self._make_generator()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([])
        gen._client = mock_client

        list(gen.generate_stream(question="q", context="c"))
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True

    def test_returns_iterator(self):
        from konjoai.generate.generator import OpenAIGenerator

        gen = self._make_generator()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([])
        gen._client = mock_client

        result = gen.generate_stream(question="q", context="c")
        # Must be an iterator (not a list) — lazy evaluation
        import inspect
        assert inspect.isgenerator(result)


class TestSquishGeneratorStream:
    """generate_stream() on SquishGenerator has identical code path to OpenAIGenerator."""

    def _make_chunk(self, content: str | None) -> MagicMock:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = content
        return chunk

    def test_yields_tokens(self):
        from konjoai.generate.generator import SquishGenerator

        gen = SquishGenerator.__new__(SquishGenerator)
        gen._model = "qwen3:8b"
        gen._max_tokens = 512
        mock_client = MagicMock()
        tokens = ["tok1", " tok2"]
        mock_client.chat.completions.create.return_value = iter(
            [self._make_chunk(t) for t in tokens]
        )
        gen._client = mock_client

        result = list(gen.generate_stream(question="q", context="c"))
        assert result == tokens

    def test_stream_flag_set(self):
        from konjoai.generate.generator import SquishGenerator

        gen = SquishGenerator.__new__(SquishGenerator)
        gen._model = "qwen3:8b"
        gen._max_tokens = 512
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([])
        gen._client = mock_client

        list(gen.generate_stream(question="q", context="c"))
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True


class TestAnthropicGeneratorStream:
    """generate_stream() uses the Anthropic text_stream context manager."""

    def test_yields_text_tokens(self):
        from konjoai.generate.generator import AnthropicGenerator

        gen = AnthropicGenerator.__new__(AnthropicGenerator)
        gen._model = "claude-3-5-haiku-latest"
        gen._max_tokens = 512

        tokens = ["Alpha", " Beta", " Gamma"]
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__ = MagicMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__exit__ = MagicMock(return_value=False)
        mock_stream_ctx.text_stream = iter(tokens)

        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream_ctx
        gen._client = mock_client

        result = list(gen.generate_stream(question="q", context="c"))
        assert result == tokens

    def test_stream_method_called(self):
        from konjoai.generate.generator import AnthropicGenerator

        gen = AnthropicGenerator.__new__(AnthropicGenerator)
        gen._model = "claude-3-5-haiku-latest"
        gen._max_tokens = 512

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__ = MagicMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__exit__ = MagicMock(return_value=False)
        mock_stream_ctx.text_stream = iter([])

        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream_ctx
        gen._client = mock_client

        list(gen.generate_stream(question="q", context="c"))
        mock_client.messages.stream.assert_called_once()


# ---------------------------------------------------------------------------
# /query/stream endpoint tests
# ---------------------------------------------------------------------------

def _make_app():
    """Build a minimal FastAPI app with the query router mounted.

    The query router already declares prefix="/query", so we include it
    without any extra prefix to get /query and /query/stream routes.
    """
    from fastapi import FastAPI
    from konjoai.api.routes.query import router
    app = FastAPI()
    app.include_router(router)  # router already has prefix="/query"
    return app


class _StubResult:
    answer = "stub answer"
    model = "stub-model"
    usage = {}


def _mock_retrieval_chain() -> dict:
    """Return a dict of all the lazy-import patch targets used by /query/stream."""
    return {
        "konjoai.api.routes.query.hybrid_search": MagicMock(return_value=[]),
        "konjoai.api.routes.query.rerank": MagicMock(return_value=[]),
        "konjoai.api.routes.query.classify_intent": MagicMock(),
        "konjoai.api.routes.query.get_settings": MagicMock(),
        "konjoai.api.routes.query.get_generator": MagicMock(),
    }


class TestQueryStreamEndpoint:
    """Tests for POST /query/stream SSE endpoint."""

    @pytest.fixture
    def client(self, monkeypatch):
        # httpx picks up ALL proxy env vars (HTTP_PROXY, HTTPS_PROXY, ALL_PROXY,
        # GRPC_PROXY …) set by the VS Code / CI host environment.  TestClient
        # uses an in-process ASGI transport and never makes a real network call,
        # so all proxy vars are irrelevant but cause ProxyError or ImportError
        # (SOCKS / socksio absent).  monkeypatch.delenv restores originals after
        # each test — no permanent env mutation.
        for _var in (
            "HTTP_PROXY", "http_proxy",
            "HTTPS_PROXY", "https_proxy",
            "ALL_PROXY", "all_proxy",
            "GRPC_PROXY", "grpc_proxy",
            "FTP_PROXY", "ftp_proxy",
            "RSYNC_PROXY",
        ):
            monkeypatch.delenv(_var, raising=False)
        return TestClient(_make_app(), raise_server_exceptions=True)

    def _sse_events(self, body: str) -> list[dict]:
        """Parse raw SSE body into a list of dicts."""
        events = []
        for line in body.strip().splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
        return events

    def test_streaming_response_content_type(self, client: TestClient):
        """Response Content-Type must be text/event-stream."""
        mock_gen = MagicMock()
        mock_gen.generate_stream = MagicMock(return_value=iter(["hello"]))
        mock_gen._model = "test-model"

        mock_settings = MagicMock()
        mock_settings.enable_query_router = False
        mock_settings.enable_hyde = False
        mock_settings.enable_telemetry = False
        mock_settings.use_vectro_retriever = False
        mock_settings.use_colbert = False

        with (
            patch("konjoai.generate.generator.get_generator", return_value=mock_gen),
            patch("konjoai.retrieve.hybrid.hybrid_search", return_value=[]),
            patch("konjoai.retrieve.reranker.rerank", return_value=[]),
            patch("konjoai.api.routes.query.get_settings", return_value=mock_settings),
        ):
            resp = client.post("/query/stream", json={"question": "what is X?"})

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_token_frames_have_done_false(self, client: TestClient):
        """Each mid-stream token frame should have done=false."""
        mock_gen = MagicMock()
        mock_gen.generate_stream = MagicMock(return_value=iter(["Hello", " world"]))
        mock_gen._model = "test-model"

        mock_settings = MagicMock()
        mock_settings.enable_query_router = False
        mock_settings.enable_hyde = False
        mock_settings.enable_telemetry = False
        mock_settings.use_vectro_retriever = False
        mock_settings.use_colbert = False

        with (
            patch("konjoai.generate.generator.get_generator", return_value=mock_gen),
            patch("konjoai.retrieve.hybrid.hybrid_search", return_value=[]),
            patch("konjoai.retrieve.reranker.rerank", return_value=[]),
            patch("konjoai.api.routes.query.get_settings", return_value=mock_settings),
        ):
            resp = client.post("/query/stream", json={"question": "q?"})

        events = self._sse_events(resp.text)
        mid = [e for e in events if not e["done"]]
        assert len(mid) >= 1
        assert all(e["done"] is False for e in mid)

    def test_final_frame_has_done_true(self, client: TestClient):
        """Final frame must have done=true with model and sources fields."""
        mock_gen = MagicMock()
        mock_gen.generate_stream = MagicMock(return_value=iter(["tok"]))
        mock_gen._model = "my-model"

        mock_settings = MagicMock()
        mock_settings.enable_query_router = False
        mock_settings.enable_hyde = False
        mock_settings.enable_telemetry = False
        mock_settings.use_vectro_retriever = False
        mock_settings.use_colbert = False

        with (
            patch("konjoai.generate.generator.get_generator", return_value=mock_gen),
            patch("konjoai.retrieve.hybrid.hybrid_search", return_value=[]),
            patch("konjoai.retrieve.reranker.rerank", return_value=[]),
            patch("konjoai.api.routes.query.get_settings", return_value=mock_settings),
        ):
            resp = client.post("/query/stream", json={"question": "q?"})

        events = self._sse_events(resp.text)
        final = events[-1]
        assert final["done"] is True
        assert "model" in final
        assert "sources" in final

    def test_fallback_when_no_generate_stream(self, client: TestClient):
        """Generators without generate_stream should work via synchronous fallback."""
        # Generator with no generate_stream attribute
        mock_gen = MagicMock(spec=["generate"])
        mock_gen.generate.return_value = _StubResult()

        mock_settings = MagicMock()
        mock_settings.enable_query_router = False
        mock_settings.enable_hyde = False
        mock_settings.enable_telemetry = False
        mock_settings.use_vectro_retriever = False
        mock_settings.use_colbert = False
        mock_settings.use_vectro_retriever = False
        mock_settings.use_colbert = False

        with (
            patch("konjoai.generate.generator.get_generator", return_value=mock_gen),
            patch("konjoai.retrieve.hybrid.hybrid_search", return_value=[]),
            patch("konjoai.retrieve.reranker.rerank", return_value=[]),
            patch("konjoai.api.routes.query.get_settings", return_value=mock_settings),
        ):
            resp = client.post("/query/stream", json={"question": "q?"})

        assert resp.status_code == 200
        events = self._sse_events(resp.text)
        final = events[-1]
        assert final["done"] is True

    def test_schema_stream_field_accepted(self, client: TestClient):
        """QueryRequest.stream field should not cause a validation error."""
        mock_gen = MagicMock()
        mock_gen.generate_stream = MagicMock(return_value=iter([]))
        mock_gen._model = "m"

        mock_settings = MagicMock()
        mock_settings.enable_query_router = False
        mock_settings.use_vectro_retriever = False
        mock_settings.use_colbert = False
        mock_settings.enable_hyde = False
        mock_settings.enable_telemetry = False
        mock_settings.use_vectro_retriever = False
        mock_settings.use_colbert = False

        with (
            patch("konjoai.generate.generator.get_generator", return_value=mock_gen),
            patch("konjoai.retrieve.hybrid.hybrid_search", return_value=[]),
            patch("konjoai.retrieve.reranker.rerank", return_value=[]),
            patch("konjoai.api.routes.query.get_settings", return_value=mock_settings),
        ):
            resp = client.post("/query/stream", json={"question": "q?", "stream": True})

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Sprint 9: async stream() sentinel + streaming_enabled config + CLI --stream
# ---------------------------------------------------------------------------


class TestAsyncGeneratorStream:
    """Verify async stream() wraps generate_stream() correctly for all backends."""

    def test_openai_stream_is_async_generator(self):
        from konjoai.generate.generator import OpenAIGenerator
        gen = OpenAIGenerator.__new__(OpenAIGenerator)
        gen._model = "gpt-4o-mini"
        gen._max_tokens = 1024
        with patch.object(gen, "generate_stream", return_value=iter(["tok"])):
            agen = gen.stream(question="q", context="c")
        assert inspect.isasyncgen(agen)

    def test_openai_stream_yields_all_tokens(self):
        from konjoai.generate.generator import OpenAIGenerator
        gen = OpenAIGenerator.__new__(OpenAIGenerator)
        gen._model = "gpt-4o-mini"
        gen._max_tokens = 1024
        tokens = ["Hello", " ", "world"]
        with patch.object(gen, "generate_stream", return_value=iter(tokens)):
            collected: list[str] = []

            async def _run() -> None:
                async for tok in gen.stream(question="q", context="c"):
                    collected.append(tok)

            asyncio.run(_run())
        assert collected == tokens

    def test_openai_stream_empty_yields_nothing(self):
        from konjoai.generate.generator import OpenAIGenerator
        gen = OpenAIGenerator.__new__(OpenAIGenerator)
        gen._model = "gpt-4o-mini"
        gen._max_tokens = 1024
        with patch.object(gen, "generate_stream", return_value=iter([])):
            collected: list[str] = []

            async def _run() -> None:
                async for tok in gen.stream(question="q", context="c"):
                    collected.append(tok)

            asyncio.run(_run())
        assert collected == []

    def test_anthropic_stream_is_async_generator(self):
        from konjoai.generate.generator import AnthropicGenerator
        gen = AnthropicGenerator.__new__(AnthropicGenerator)
        gen._model = "claude-3-haiku-20240307"
        gen._max_tokens = 1024
        with patch.object(gen, "generate_stream", return_value=iter(["x"])):
            agen = gen.stream(question="q", context="c")
        assert inspect.isasyncgen(agen)

    def test_anthropic_stream_yields_all_tokens(self):
        from konjoai.generate.generator import AnthropicGenerator
        gen = AnthropicGenerator.__new__(AnthropicGenerator)
        gen._model = "claude-3-haiku-20240307"
        gen._max_tokens = 1024
        tokens = ["Alpha", "Beta"]
        with patch.object(gen, "generate_stream", return_value=iter(tokens)):
            collected: list[str] = []

            async def _run() -> None:
                async for tok in gen.stream(question="q", context="c"):
                    collected.append(tok)

            asyncio.run(_run())
        assert collected == tokens

    def test_squish_stream_is_async_generator(self):
        from konjoai.generate.generator import SquishGenerator
        gen = SquishGenerator.__new__(SquishGenerator)
        gen._model = "qwen3:8b"
        gen._max_tokens = 1024
        with patch.object(gen, "generate_stream", return_value=iter(["y"])):
            agen = gen.stream(question="q", context="c")
        assert inspect.isasyncgen(agen)

    def test_squish_stream_yields_all_tokens(self):
        from konjoai.generate.generator import SquishGenerator
        gen = SquishGenerator.__new__(SquishGenerator)
        gen._model = "qwen3:8b"
        gen._max_tokens = 1024
        tokens = ["Gamma", "Delta"]
        with patch.object(gen, "generate_stream", return_value=iter(tokens)):
            collected: list[str] = []

            async def _run() -> None:
                async for tok in gen.stream(question="q", context="c"):
                    collected.append(tok)

            asyncio.run(_run())
        assert collected == tokens

    def test_squish_stream_empty_yields_nothing(self):
        from konjoai.generate.generator import SquishGenerator
        gen = SquishGenerator.__new__(SquishGenerator)
        gen._model = "qwen3:8b"
        gen._max_tokens = 1024
        with patch.object(gen, "generate_stream", return_value=iter([])):
            collected: list[str] = []

            async def _run() -> None:
                async for tok in gen.stream(question="q", context="c"):
                    collected.append(tok)

            asyncio.run(_run())
        assert collected == []


class TestStreamingEnabledConfig:
    """Verify streaming_enabled config field defaults and override."""

    def test_streaming_enabled_default_is_true(self):
        from konjoai.config import Settings
        s = Settings()
        assert s.streaming_enabled is True

    def test_streaming_enabled_can_be_disabled(self):
        from konjoai.config import Settings
        s = Settings(streaming_enabled=False)
        assert s.streaming_enabled is False


class TestCLIStreamFlag:
    """Verify --stream / -s flag on the query command."""

    def test_stream_flag_calls_generate_stream(self):
        from click.testing import CliRunner
        from konjoai.cli.main import cli

        mock_gen = MagicMock()
        mock_gen.generate_stream.return_value = iter(["Hello", " world"])

        with (
            patch("konjoai.retrieve.hybrid.hybrid_search", return_value=[
                MagicMock(content="ctx", score=0.9, source="src.md")
            ]),
            patch("konjoai.retrieve.reranker.rerank", return_value=[
                MagicMock(content="ctx", score=0.9, source="src.md")
            ]),
            patch("konjoai.generate.generator.get_generator", return_value=mock_gen),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["query", "--stream", "What is RAG?"])

        mock_gen.generate_stream.assert_called_once()
        mock_gen.generate.assert_not_called()
        assert result.exit_code == 0

    def test_stream_short_form_flag(self):
        from click.testing import CliRunner
        from konjoai.cli.main import cli

        mock_gen = MagicMock()
        mock_gen.generate_stream.return_value = iter(["tok"])

        with (
            patch("konjoai.retrieve.hybrid.hybrid_search", return_value=[
                MagicMock(content="ctx", score=0.9, source="src.md")
            ]),
            patch("konjoai.retrieve.reranker.rerank", return_value=[
                MagicMock(content="ctx", score=0.9, source="src.md")
            ]),
            patch("konjoai.generate.generator.get_generator", return_value=mock_gen),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["query", "-s", "What is RAG?"])

        assert result.exit_code == 0
        mock_gen.generate_stream.assert_called_once()

    def test_no_stream_flag_calls_generate(self):
        from click.testing import CliRunner
        from konjoai.cli.main import cli

        mock_gen = MagicMock()
        mock_gen.generate.return_value = MagicMock(answer="The answer.", model="gpt-4o-mini")

        with (
            patch("konjoai.retrieve.hybrid.hybrid_search", return_value=[
                MagicMock(content="ctx", score=0.9, source="src.md")
            ]),
            patch("konjoai.retrieve.reranker.rerank", return_value=[
                MagicMock(content="ctx", score=0.9, source="src.md")
            ]),
            patch("konjoai.generate.generator.get_generator", return_value=mock_gen),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["query", "What is RAG?"])

        mock_gen.generate.assert_called_once()
        mock_gen.generate_stream.assert_not_called()
        assert result.exit_code == 0

    def test_stream_flag_outputs_tokens(self):
        from click.testing import CliRunner
        from konjoai.cli.main import cli

        mock_gen = MagicMock()
        mock_gen.generate_stream.return_value = iter(["Hello", " world"])

        with (
            patch("konjoai.retrieve.hybrid.hybrid_search", return_value=[
                MagicMock(content="ctx", score=0.9, source="src.md")
            ]),
            patch("konjoai.retrieve.reranker.rerank", return_value=[
                MagicMock(content="ctx", score=0.9, source="src.md")
            ]),
            patch("konjoai.generate.generator.get_generator", return_value=mock_gen),
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["query", "--stream", "What is RAG?"])

        assert "Hello world" in result.output
        assert result.exit_code == 0
