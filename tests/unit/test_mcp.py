"""Sprint 19 — MCP server tests.

Coverage targets:
- _HAS_MCP flag is a bool
- KyroMCPServer.from_url() factory
- KyroMCPServer.list_tools() — schema contract (name, description, inputSchema)
- TOOLS constant: all four tools present
- dispatch(): kyro_query, kyro_ingest, kyro_health, kyro_agent_query
- dispatch(): unknown tool raises ValueError
- dispatch(): KyroError from client propagates
- run_stdio() raises RuntimeError when _HAS_MCP=False
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import konjoai.mcp as _mcp_module
from konjoai.mcp import _HAS_MCP
from konjoai.mcp.server import TOOLS, KyroMCPServer
from konjoai.sdk.exceptions import KyroError
from konjoai.sdk.models import (
    SDKAgentQueryResponse,
    SDKAgentStep,
    SDKHealthResponse,
    SDKIngestResponse,
    SDKQueryResponse,
    SDKSourceDoc,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _fake_query_response() -> SDKQueryResponse:
    return SDKQueryResponse(
        answer="The answer is 42",
        sources=[SDKSourceDoc(source="doc.md", content_preview="...", score=0.95)],
        model="gpt-4o-mini",
        usage={"total_tokens": 50},
        intent="retrieval",
        cache_hit=False,
    )


def _fake_ingest_response() -> SDKIngestResponse:
    return SDKIngestResponse(chunks_indexed=10, sources_processed=2, chunks_deduplicated=1)


def _fake_health_response() -> SDKHealthResponse:
    return SDKHealthResponse(status="ok", vector_count=500, bm25_built=True)


def _fake_agent_response() -> SDKAgentQueryResponse:
    return SDKAgentQueryResponse(
        answer="Agent found it",
        sources=[],
        model="gpt-4o-mini",
        usage={},
        steps=[
            SDKAgentStep(
                thought="I need to look this up",
                action="retrieve",
                action_input="some query",
                observation="found relevant doc",
            )
        ],
    )


def _make_server() -> tuple[KyroMCPServer, MagicMock]:
    mock_client = MagicMock()
    server = KyroMCPServer(mock_client)
    return server, mock_client


# ── _HAS_MCP flag ─────────────────────────────────────────────────────────────


class TestHasMcpFlag:
    def test_has_mcp_is_bool(self) -> None:
        assert isinstance(_HAS_MCP, bool)


# ── TOOLS constant ────────────────────────────────────────────────────────────


class TestToolsConstant:
    def test_four_tools_defined(self) -> None:
        assert len(TOOLS) == 4

    def test_tool_names(self) -> None:
        names = {t["name"] for t in TOOLS}
        assert names == {"kyro_query", "kyro_ingest", "kyro_health", "kyro_agent_query"}

    def test_each_tool_has_description(self) -> None:
        for t in TOOLS:
            assert "description" in t and t["description"]

    def test_each_tool_has_input_schema(self) -> None:
        for t in TOOLS:
            assert "inputSchema" in t
            assert t["inputSchema"]["type"] == "object"

    def test_kyro_query_requires_question(self) -> None:
        tool = next(t for t in TOOLS if t["name"] == "kyro_query")
        assert "question" in tool["inputSchema"]["required"]

    def test_kyro_ingest_requires_path(self) -> None:
        tool = next(t for t in TOOLS if t["name"] == "kyro_ingest")
        assert "path" in tool["inputSchema"]["required"]

    def test_kyro_health_has_no_required_fields(self) -> None:
        tool = next(t for t in TOOLS if t["name"] == "kyro_health")
        assert tool["inputSchema"]["required"] == []

    def test_kyro_agent_query_requires_question(self) -> None:
        tool = next(t for t in TOOLS if t["name"] == "kyro_agent_query")
        assert "question" in tool["inputSchema"]["required"]


# ── KyroMCPServer construction ─────────────────────────────────────────────────


class TestKyroMCPServerConstruction:
    def test_list_tools_returns_all_tools(self) -> None:
        server, _ = _make_server()
        tools = server.list_tools()
        assert len(tools) == 4

    def test_list_tools_returns_same_as_tools_constant(self) -> None:
        server, _ = _make_server()
        assert server.list_tools() is TOOLS

    def test_from_url_creates_server_with_client(self) -> None:
        with patch("konjoai.mcp.server.KonjoClient") as MockClient:
            server = KyroMCPServer.from_url("http://localhost:8000", api_key="sk-x")
        assert isinstance(server, KyroMCPServer)
        MockClient.assert_called_once()

    def test_from_url_passes_api_key(self) -> None:
        with patch("konjoai.mcp.server.KonjoClient") as MockClient:
            KyroMCPServer.from_url("http://localhost:8000", api_key="sk-test")
        _, kwargs = MockClient.call_args
        assert kwargs.get("api_key") == "sk-test"

    def test_from_url_passes_timeout(self) -> None:
        with patch("konjoai.mcp.server.KonjoClient") as MockClient:
            KyroMCPServer.from_url("http://localhost:8000", timeout=60.0)
        _, kwargs = MockClient.call_args
        assert kwargs.get("timeout") == 60.0


# ── dispatch(): kyro_query ────────────────────────────────────────────────────


class TestDispatchKyroQuery:
    @pytest.mark.asyncio
    async def test_dispatch_query_calls_client_query(self) -> None:
        server, mock_client = _make_server()
        mock_client.query.return_value = _fake_query_response()
        await server.dispatch("kyro_query", {"question": "What is 6×7?"})
        mock_client.query.assert_called_once_with("What is 6×7?", top_k=5, use_hyde=False)

    @pytest.mark.asyncio
    async def test_dispatch_query_returns_json_with_answer(self) -> None:
        server, mock_client = _make_server()
        mock_client.query.return_value = _fake_query_response()
        result = await server.dispatch("kyro_query", {"question": "Q"})
        data = json.loads(result)
        assert data["answer"] == "The answer is 42"

    @pytest.mark.asyncio
    async def test_dispatch_query_passes_top_k(self) -> None:
        server, mock_client = _make_server()
        mock_client.query.return_value = _fake_query_response()
        await server.dispatch("kyro_query", {"question": "Q", "top_k": 10})
        mock_client.query.assert_called_once_with("Q", top_k=10, use_hyde=False)

    @pytest.mark.asyncio
    async def test_dispatch_query_passes_use_hyde(self) -> None:
        server, mock_client = _make_server()
        mock_client.query.return_value = _fake_query_response()
        await server.dispatch("kyro_query", {"question": "Q", "use_hyde": True})
        mock_client.query.assert_called_once_with("Q", top_k=5, use_hyde=True)

    @pytest.mark.asyncio
    async def test_dispatch_query_sources_in_result(self) -> None:
        server, mock_client = _make_server()
        mock_client.query.return_value = _fake_query_response()
        result = await server.dispatch("kyro_query", {"question": "Q"})
        data = json.loads(result)
        assert len(data["sources"]) == 1
        assert data["sources"][0]["source"] == "doc.md"


# ── dispatch(): kyro_ingest ───────────────────────────────────────────────────


class TestDispatchKyroIngest:
    @pytest.mark.asyncio
    async def test_dispatch_ingest_calls_client_ingest(self) -> None:
        server, mock_client = _make_server()
        mock_client.ingest.return_value = _fake_ingest_response()
        await server.dispatch("kyro_ingest", {"path": "/docs"})
        mock_client.ingest.assert_called_once_with("/docs", strategy="recursive", chunk_size=512)

    @pytest.mark.asyncio
    async def test_dispatch_ingest_returns_chunks_indexed(self) -> None:
        server, mock_client = _make_server()
        mock_client.ingest.return_value = _fake_ingest_response()
        result = await server.dispatch("kyro_ingest", {"path": "/docs"})
        data = json.loads(result)
        assert data["chunks_indexed"] == 10
        assert data["sources_processed"] == 2
        assert data["chunks_deduplicated"] == 1


# ── dispatch(): kyro_health ───────────────────────────────────────────────────


class TestDispatchKyroHealth:
    @pytest.mark.asyncio
    async def test_dispatch_health_calls_client_health(self) -> None:
        server, mock_client = _make_server()
        mock_client.health.return_value = _fake_health_response()
        await server.dispatch("kyro_health", {})
        mock_client.health.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_dispatch_health_returns_status(self) -> None:
        server, mock_client = _make_server()
        mock_client.health.return_value = _fake_health_response()
        result = await server.dispatch("kyro_health", {})
        data = json.loads(result)
        assert data["status"] == "ok"
        assert data["vector_count"] == 500
        assert data["bm25_built"] is True


# ── dispatch(): kyro_agent_query ──────────────────────────────────────────────


class TestDispatchKyroAgentQuery:
    @pytest.mark.asyncio
    async def test_dispatch_agent_query_calls_client(self) -> None:
        server, mock_client = _make_server()
        mock_client.agent_query.return_value = _fake_agent_response()
        await server.dispatch("kyro_agent_query", {"question": "Find X"})
        mock_client.agent_query.assert_called_once_with("Find X", top_k=5, max_steps=5)

    @pytest.mark.asyncio
    async def test_dispatch_agent_query_returns_answer_and_steps(self) -> None:
        server, mock_client = _make_server()
        mock_client.agent_query.return_value = _fake_agent_response()
        result = await server.dispatch("kyro_agent_query", {"question": "Find X"})
        data = json.loads(result)
        assert data["answer"] == "Agent found it"
        assert len(data["steps"]) == 1
        assert data["steps"][0]["action"] == "retrieve"

    @pytest.mark.asyncio
    async def test_dispatch_agent_query_passes_max_steps(self) -> None:
        server, mock_client = _make_server()
        mock_client.agent_query.return_value = _fake_agent_response()
        await server.dispatch("kyro_agent_query", {"question": "Q", "max_steps": 3})
        mock_client.agent_query.assert_called_once_with("Q", top_k=5, max_steps=3)


# ── dispatch(): error cases ───────────────────────────────────────────────────


class TestDispatchErrors:
    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool_raises_value_error(self) -> None:
        server, _ = _make_server()
        with pytest.raises(ValueError, match="Unknown tool"):
            await server.dispatch("kyro_nonexistent", {})

    @pytest.mark.asyncio
    async def test_dispatch_propagates_kyro_error(self) -> None:
        server, mock_client = _make_server()
        mock_client.query.side_effect = KyroError("backend failed", status_code=500)
        with pytest.raises(KyroError):
            await server.dispatch("kyro_query", {"question": "Q"})


# ── run_stdio() guard ─────────────────────────────────────────────────────────


class TestRunStdioGuard:
    @pytest.mark.asyncio
    async def test_run_stdio_raises_when_mcp_absent(self) -> None:
        server, _ = _make_server()
        with patch.object(_mcp_module, "_HAS_MCP", False):
            with pytest.raises(RuntimeError, match="mcp"):
                await server.run_stdio()
