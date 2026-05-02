"""Sprint 19 — Python SDK tests.

Coverage targets:
- KonjoClient construction: headers, base_url, timeout
- query(): success, 401, 403, 429, 500, timeout, Retry-After parsing
- query_stream(): token iteration, SSE [DONE] sentinel, timeout
- ingest(): success, error propagation
- health(): success, typed response
- agent_query(): success, steps deserialization
- context-manager lifecycle (close)
- SDKSourceDoc / SDKStreamChunk / SDKIngestResponse / SDKHealthResponse / SDKAgentQueryResponse models
- Exception hierarchy: KyroError, KyroAuthError, KyroRateLimitError, KyroTimeoutError, KyroNotFoundError
"""
from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from konjoai.sdk.client import KonjoClient
from konjoai.sdk.exceptions import (
    KyroAuthError,
    KyroError,
    KyroNotFoundError,
    KyroRateLimitError,
    KyroTimeoutError,
)
from konjoai.sdk.models import (
    SDKAgentQueryResponse,
    SDKAgentStep,
    SDKHealthResponse,
    SDKIngestResponse,
    SDKQueryResponse,
    SDKSourceDoc,
    SDKStreamChunk,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _mock_response(
    status_code: int = 200,
    json_data: dict | None = None,
    headers: dict | None = None,
    text: str = "",
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text or str(json_data or {})
    resp.headers = headers or {}
    return resp


_QUERY_PAYLOAD = {
    "answer": "42",
    "sources": [
        {"source": "doc.md", "content_preview": "hello", "score": 0.9}
    ],
    "model": "gpt-4o-mini",
    "usage": {"total_tokens": 100},
    "intent": "retrieval",
    "cache_hit": False,
}

_HEALTH_PAYLOAD = {
    "status": "ok",
    "vector_count": 1000,
    "bm25_built": True,
}

_INGEST_PAYLOAD = {
    "chunks_indexed": 42,
    "sources_processed": 3,
    "chunks_deduplicated": 2,
}

_AGENT_PAYLOAD = {
    "answer": "Agent answer",
    "sources": [{"source": "s.md", "content_preview": "...", "score": 0.8}],
    "model": "gpt-4o-mini",
    "usage": {"total_tokens": 200},
    "steps": [
        {
            "thought": "I need to retrieve info",
            "action": "retrieve",
            "action_input": "What is X?",
            "observation": "X is Y",
        }
    ],
}


# ── Exception hierarchy ───────────────────────────────────────────────────────


class TestExceptionHierarchy:
    def test_kyro_error_is_exception(self) -> None:
        e = KyroError("oops", status_code=500)
        assert isinstance(e, Exception)
        assert e.status_code == 500

    def test_kyro_auth_error_is_kyro_error(self) -> None:
        assert issubclass(KyroAuthError, KyroError)

    def test_kyro_rate_limit_error_is_kyro_error(self) -> None:
        assert issubclass(KyroRateLimitError, KyroError)

    def test_kyro_rate_limit_error_retry_after(self) -> None:
        e = KyroRateLimitError("limited", retry_after=30.0)
        assert e.status_code == 429
        assert e.retry_after == 30.0

    def test_kyro_rate_limit_error_no_retry_after(self) -> None:
        e = KyroRateLimitError("limited")
        assert e.retry_after is None

    def test_kyro_timeout_error_is_kyro_error(self) -> None:
        assert issubclass(KyroTimeoutError, KyroError)

    def test_kyro_timeout_error_no_status_code(self) -> None:
        e = KyroTimeoutError("timed out")
        assert e.status_code is None

    def test_kyro_not_found_error_is_kyro_error(self) -> None:
        assert issubclass(KyroNotFoundError, KyroError)


# ── Model contracts ───────────────────────────────────────────────────────────


class TestSDKModels:
    def test_sdk_source_doc_fields(self) -> None:
        doc = SDKSourceDoc(source="a.md", content_preview="hello", score=0.95)
        assert doc.source == "a.md"
        assert doc.content_preview == "hello"
        assert doc.score == 0.95

    def test_sdk_stream_chunk_text(self) -> None:
        chunk = SDKStreamChunk(text="hello")
        assert chunk.text == "hello"

    def test_sdk_health_response_fields(self) -> None:
        h = SDKHealthResponse(status="ok", vector_count=42, bm25_built=True)
        assert h.status == "ok"
        assert h.vector_count == 42
        assert h.bm25_built is True

    def test_sdk_ingest_response_defaults(self) -> None:
        r = SDKIngestResponse(chunks_indexed=10, sources_processed=2)
        assert r.chunks_deduplicated == 0

    def test_sdk_query_response_default_intent(self) -> None:
        r = SDKQueryResponse(answer="a", sources=[], model="m", usage={})
        assert r.intent == "retrieval"
        assert r.cache_hit is False

    def test_sdk_agent_step_fields(self) -> None:
        s = SDKAgentStep(thought="t", action="retrieve", action_input="q", observation="o")
        assert s.action == "retrieve"


# ── KonjoClient construction ──────────────────────────────────────────────────


class TestKonjoClientConstruction:
    def test_api_key_header_set(self) -> None:
        with patch("konjoai.sdk.client.httpx.Client") as MockClient:
            KonjoClient("http://localhost:8000", api_key="sk-test")
            _, kwargs = MockClient.call_args
            assert kwargs["headers"].get("X-API-Key") == "sk-test"

    def test_jwt_token_header_set(self) -> None:
        with patch("konjoai.sdk.client.httpx.Client") as MockClient:
            KonjoClient("http://localhost:8000", jwt_token="tok.123")
            _, kwargs = MockClient.call_args
            assert kwargs["headers"].get("Authorization") == "Bearer tok.123"

    def test_no_auth_headers_by_default(self) -> None:
        with patch("konjoai.sdk.client.httpx.Client") as MockClient:
            KonjoClient("http://localhost:8000")
            _, kwargs = MockClient.call_args
            assert "X-API-Key" not in kwargs["headers"]
            assert "Authorization" not in kwargs["headers"]

    def test_default_timeout_is_30(self) -> None:
        with patch("konjoai.sdk.client.httpx.Client") as MockClient:
            KonjoClient("http://localhost:8000")
            _, kwargs = MockClient.call_args
            assert kwargs["timeout"] == 30.0

    def test_custom_timeout(self) -> None:
        with patch("konjoai.sdk.client.httpx.Client") as MockClient:
            KonjoClient("http://localhost:8000", timeout=60.0)
            _, kwargs = MockClient.call_args
            assert kwargs["timeout"] == 60.0

    def test_trailing_slash_stripped_from_base_url(self) -> None:
        with patch("konjoai.sdk.client.httpx.Client") as MockClient:
            KonjoClient("http://localhost:8000/")
            _, kwargs = MockClient.call_args
            assert kwargs["base_url"] == "http://localhost:8000"


# ── query() ───────────────────────────────────────────────────────────────────


class TestKonjoClientQuery:
    def _client(self) -> KonjoClient:
        with patch("konjoai.sdk.client.httpx.Client"):
            c = KonjoClient("http://localhost:8000")
        return c

    def test_query_success_returns_typed_response(self) -> None:
        client = self._client()
        client._client.post.return_value = _mock_response(200, _QUERY_PAYLOAD)
        result = client.query("What is 6×7?")
        assert isinstance(result, SDKQueryResponse)
        assert result.answer == "42"

    def test_query_sources_parsed(self) -> None:
        client = self._client()
        client._client.post.return_value = _mock_response(200, _QUERY_PAYLOAD)
        result = client.query("q")
        assert len(result.sources) == 1
        assert isinstance(result.sources[0], SDKSourceDoc)
        assert result.sources[0].score == 0.9

    def test_query_default_top_k_5(self) -> None:
        client = self._client()
        client._client.post.return_value = _mock_response(200, _QUERY_PAYLOAD)
        client.query("q")
        call_json = client._client.post.call_args.kwargs["json"]
        assert call_json["top_k"] == 5

    def test_query_custom_top_k(self) -> None:
        client = self._client()
        client._client.post.return_value = _mock_response(200, _QUERY_PAYLOAD)
        client.query("q", top_k=10)
        assert client._client.post.call_args.kwargs["json"]["top_k"] == 10

    def test_query_raises_kyro_auth_error_on_401(self) -> None:
        client = self._client()
        client._client.post.return_value = _mock_response(401, text="Unauthorized")
        with pytest.raises(KyroAuthError):
            client.query("q")

    def test_query_raises_kyro_auth_error_on_403(self) -> None:
        client = self._client()
        client._client.post.return_value = _mock_response(403, text="Forbidden")
        with pytest.raises(KyroAuthError):
            client.query("q")

    def test_query_raises_kyro_rate_limit_on_429(self) -> None:
        client = self._client()
        client._client.post.return_value = _mock_response(
            429, text="Too Many Requests", headers={"Retry-After": "30"}
        )
        with pytest.raises(KyroRateLimitError) as exc_info:
            client.query("q")
        assert exc_info.value.retry_after == 30.0

    def test_query_raises_kyro_error_on_500(self) -> None:
        client = self._client()
        client._client.post.return_value = _mock_response(500, text="Internal Server Error")
        with pytest.raises(KyroError) as exc_info:
            client.query("q")
        assert exc_info.value.status_code == 500

    def test_query_raises_kyro_timeout_on_timeout(self) -> None:
        import httpx
        client = self._client()
        client._client.post.side_effect = httpx.TimeoutException("timed out")
        with pytest.raises(KyroTimeoutError):
            client.query("q")


# ── query_stream() ────────────────────────────────────────────────────────────


class TestKonjoClientQueryStream:
    def _client(self) -> KonjoClient:
        with patch("konjoai.sdk.client.httpx.Client"):
            c = KonjoClient("http://localhost:8000")
        return c

    def _mock_stream_ctx(self, lines: list[str], status_code: int = 200) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status_code
        resp.text = ""
        resp.headers = {}
        resp.iter_lines.return_value = iter(lines)
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=resp)
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    def test_stream_yields_tokens(self) -> None:
        client = self._client()
        client._client.stream.return_value = self._mock_stream_ctx([
            'data: {"token": "Hello"}',
            'data: {"token": " world"}',
            "data: [DONE]",
        ])
        chunks = list(client.query_stream("q"))
        assert [c.text for c in chunks] == ["Hello", " world"]

    def test_stream_stops_at_done_sentinel(self) -> None:
        client = self._client()
        client._client.stream.return_value = self._mock_stream_ctx([
            'data: {"token": "tok"}',
            "data: [DONE]",
            'data: {"token": "extra"}',
        ])
        chunks = list(client.query_stream("q"))
        assert len(chunks) == 1

    def test_stream_skips_non_data_lines(self) -> None:
        client = self._client()
        client._client.stream.return_value = self._mock_stream_ctx([
            "event: message",
            ": comment",
            'data: {"token": "only"}',
            "data: [DONE]",
        ])
        chunks = list(client.query_stream("q"))
        assert len(chunks) == 1

    def test_stream_raises_timeout(self) -> None:
        import httpx
        client = self._client()
        client._client.stream.side_effect = httpx.TimeoutException("timeout")
        with pytest.raises(KyroTimeoutError):
            list(client.query_stream("q"))


# ── ingest() ─────────────────────────────────────────────────────────────────


class TestKonjoClientIngest:
    def _client(self) -> KonjoClient:
        with patch("konjoai.sdk.client.httpx.Client"):
            c = KonjoClient("http://localhost:8000")
        return c

    def test_ingest_success(self) -> None:
        client = self._client()
        client._client.post.return_value = _mock_response(200, _INGEST_PAYLOAD)
        result = client.ingest("/docs")
        assert isinstance(result, SDKIngestResponse)
        assert result.chunks_indexed == 42
        assert result.sources_processed == 3
        assert result.chunks_deduplicated == 2

    def test_ingest_path_sent(self) -> None:
        client = self._client()
        client._client.post.return_value = _mock_response(200, _INGEST_PAYLOAD)
        client.ingest("/path/to/docs")
        call_json = client._client.post.call_args.kwargs["json"]
        assert call_json["path"] == "/path/to/docs"

    def test_ingest_raises_on_server_error(self) -> None:
        client = self._client()
        client._client.post.return_value = _mock_response(500, text="fail")
        with pytest.raises(KyroError):
            client.ingest("/docs")

    def test_ingest_timeout(self) -> None:
        import httpx
        client = self._client()
        client._client.post.side_effect = httpx.TimeoutException("timeout")
        with pytest.raises(KyroTimeoutError):
            client.ingest("/docs")


# ── health() ─────────────────────────────────────────────────────────────────


class TestKonjoClientHealth:
    def _client(self) -> KonjoClient:
        with patch("konjoai.sdk.client.httpx.Client"):
            c = KonjoClient("http://localhost:8000")
        return c

    def test_health_success(self) -> None:
        client = self._client()
        client._client.get.return_value = _mock_response(200, _HEALTH_PAYLOAD)
        result = client.health()
        assert isinstance(result, SDKHealthResponse)
        assert result.status == "ok"
        assert result.vector_count == 1000
        assert result.bm25_built is True

    def test_health_timeout(self) -> None:
        import httpx
        client = self._client()
        client._client.get.side_effect = httpx.TimeoutException("timeout")
        with pytest.raises(KyroTimeoutError):
            client.health()

    def test_health_not_found(self) -> None:
        client = self._client()
        client._client.get.return_value = _mock_response(404, text="not found")
        with pytest.raises(KyroNotFoundError):
            client.health()


# ── agent_query() ─────────────────────────────────────────────────────────────


class TestKonjoClientAgentQuery:
    def _client(self) -> KonjoClient:
        with patch("konjoai.sdk.client.httpx.Client"):
            c = KonjoClient("http://localhost:8000")
        return c

    def test_agent_query_success(self) -> None:
        client = self._client()
        client._client.post.return_value = _mock_response(200, _AGENT_PAYLOAD)
        result = client.agent_query("What is X?")
        assert isinstance(result, SDKAgentQueryResponse)
        assert result.answer == "Agent answer"

    def test_agent_query_steps_deserialized(self) -> None:
        client = self._client()
        client._client.post.return_value = _mock_response(200, _AGENT_PAYLOAD)
        result = client.agent_query("q")
        assert len(result.steps) == 1
        assert isinstance(result.steps[0], SDKAgentStep)
        assert result.steps[0].action == "retrieve"

    def test_agent_query_default_max_steps(self) -> None:
        client = self._client()
        client._client.post.return_value = _mock_response(200, _AGENT_PAYLOAD)
        client.agent_query("q")
        call_json = client._client.post.call_args.kwargs["json"]
        assert call_json["max_steps"] == 5

    def test_agent_query_timeout(self) -> None:
        import httpx
        client = self._client()
        client._client.post.side_effect = httpx.TimeoutException("timeout")
        with pytest.raises(KyroTimeoutError):
            client.agent_query("q")


# ── Context manager ───────────────────────────────────────────────────────────


class TestKonjoClientLifecycle:
    def test_context_manager_calls_close(self) -> None:
        with patch("konjoai.sdk.client.httpx.Client"):
            with KonjoClient("http://localhost:8000") as client:
                client._client.post.return_value = _mock_response(200, _HEALTH_PAYLOAD)
            client._client.close.assert_called_once()

    def test_close_delegates_to_httpx_client(self) -> None:
        with patch("konjoai.sdk.client.httpx.Client"):
            client = KonjoClient("http://localhost:8000")
        client.close()
        client._client.close.assert_called_once()


# ── agent_query_stream() ─────────────────────────────────────────────────────


class TestKonjoClientAgentQueryStream:
    def _client(self) -> KonjoClient:
        with patch("konjoai.sdk.client.httpx.Client"):
            c = KonjoClient("http://localhost:8000")
        return c

    def _mock_stream_ctx(self, lines: list[str], status_code: int = 200) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status_code
        resp.text = ""
        resp.headers = {}
        resp.iter_lines.return_value = iter(lines)
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=resp)
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    def test_agent_stream_yields_typed_events(self) -> None:
        client = self._client()
        client._client.stream.return_value = self._mock_stream_ctx([
            'data: {"type":"step","index":1,"action":"retrieve","thought":"t","action_input":"x","observation":"[]"}',
            'data: {"type":"step","index":2,"action":"finish","thought":"t","action_input":"","observation":"completed"}',
            'data: {"type":"result","answer":"A","model":"m","usage":{},"steps":[],"sources":[]}',
            'data: {"type":"telemetry","telemetry":null}',
            "data: [DONE]",
        ])
        events = list(client.agent_query_stream("q", top_k=3, max_steps=4))
        assert [e.type for e in events] == ["step", "step", "result", "telemetry"]
        assert events[0].data["action"] == "retrieve"
        assert events[2].data["answer"] == "A"

    def test_agent_stream_stops_at_done_sentinel(self) -> None:
        client = self._client()
        client._client.stream.return_value = self._mock_stream_ctx([
            'data: {"type":"step","action":"retrieve"}',
            "data: [DONE]",
            'data: {"type":"step","action":"never"}',
        ])
        events = list(client.agent_query_stream("q"))
        assert len(events) == 1

    def test_agent_stream_skips_malformed_and_typeless_frames(self) -> None:
        client = self._client()
        client._client.stream.return_value = self._mock_stream_ctx([
            "event: message",
            "data: not json",
            'data: {"no_type":"x"}',
            'data: ["a","b"]',
            'data: {"type":"step","action":"retrieve"}',
            "data: [DONE]",
        ])
        events = list(client.agent_query_stream("q"))
        assert len(events) == 1
        assert events[0].type == "step"

    def test_agent_stream_raises_timeout(self) -> None:
        import httpx
        client = self._client()
        client._client.stream.side_effect = httpx.TimeoutException("timeout")
        with pytest.raises(KyroTimeoutError):
            list(client.agent_query_stream("q"))
