"""KyroMCPServer — Model Context Protocol server wrapping the Kyro SDK.

Architecture
------------
``KyroMCPServer`` owns a ``KonjoClient`` and exposes four MCP tools:

* ``kyro_query``        — full RAG pipeline query
* ``kyro_ingest``       — ingest a file or directory
* ``kyro_health``       — system health check
* ``kyro_agent_query``  — bounded ReAct agent loop

The ``dispatch`` method routes tool calls and is fully testable without the
``mcp`` package installed.  Transport-level code (``run_stdio``) is isolated
in a single method that imports ``mcp`` lazily so the rest of the module is
always importable (K3/K5 pattern).
"""

from __future__ import annotations

import json
from typing import Any

from konjoai.sdk.client import KonjoClient
from konjoai.sdk.exceptions import KyroError

# ── Tool definitions (JSON Schema for MCP list_tools) ────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "name": "kyro_query",
        "description": (
            "Query the Kyro RAG pipeline. Returns an answer synthesised from "
            "retrieved document chunks, source citations, and optional metadata."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The natural-language question to answer.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of source chunks to retrieve (1–50).",
                    "default": 5,
                },
                "use_hyde": {
                    "type": "boolean",
                    "description": "Enable HyDE hypothesis embedding.",
                    "default": False,
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "kyro_ingest",
        "description": (
            "Ingest a local file or directory into the Kyro vector store so it can be retrieved by future queries."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file or directory to ingest.",
                },
                "strategy": {
                    "type": "string",
                    "description": "Chunking strategy: 'recursive' or 'sentence_window'.",
                    "default": "recursive",
                },
                "chunk_size": {
                    "type": "integer",
                    "description": "Target chunk size in tokens.",
                    "default": 512,
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "kyro_health",
        "description": ("Check Kyro API health. Returns status, vector count, and BM25 index readiness."),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "kyro_agent_query",
        "description": (
            "Run the Kyro ReAct agent loop. The agent iterates retrieve → "
            "reason → act cycles up to max_steps and returns a final answer "
            "with a full step trace."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question for the agent to answer.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Sources per retrieval step.",
                    "default": 5,
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Maximum ReAct iterations (1–20).",
                    "default": 5,
                },
            },
            "required": ["question"],
        },
    },
]


class KyroMCPServer:
    """MCP server that exposes Kyro RAG capabilities as callable tools.

    The server is constructed with a ``KonjoClient`` (or a URL + credentials)
    and can be driven via ``run_stdio()`` for process-level MCP transport.

    :param client: Pre-constructed ``KonjoClient`` instance.
    """

    def __init__(self, client: KonjoClient) -> None:
        self._client = client

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_url(
        cls,
        base_url: str,
        *,
        api_key: str | None = None,
        jwt_token: str | None = None,
        timeout: float = 30.0,
    ) -> KyroMCPServer:
        """Construct a server from a base URL and optional credentials."""
        return cls(
            KonjoClient(
                base_url,
                api_key=api_key,
                jwt_token=jwt_token,
                timeout=timeout,
            )
        )

    # ── Tool registry ─────────────────────────────────────────────────────────

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the tool definitions understood by this server."""
        return TOOLS

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch a tool call and return the result as a JSON string.

        This method contains no mcp-specific imports and is fully testable
        without the ``mcp`` package installed.

        :raises ValueError: When ``name`` is not a recognised tool.
        :raises KyroError: When the underlying Kyro SDK call fails.
        """
        if name == "kyro_query":
            response = self._client.query(
                arguments["question"],
                top_k=int(arguments.get("top_k", 5)),
                use_hyde=bool(arguments.get("use_hyde", False)),
            )
            return json.dumps(
                {
                    "answer": response.answer,
                    "model": response.model,
                    "intent": response.intent,
                    "cache_hit": response.cache_hit,
                    "sources": [{"source": s.source, "score": s.score} for s in response.sources],
                }
            )

        if name == "kyro_ingest":
            response = self._client.ingest(
                arguments["path"],
                strategy=str(arguments.get("strategy", "recursive")),
                chunk_size=int(arguments.get("chunk_size", 512)),
            )
            return json.dumps(
                {
                    "chunks_indexed": response.chunks_indexed,
                    "sources_processed": response.sources_processed,
                    "chunks_deduplicated": response.chunks_deduplicated,
                }
            )

        if name == "kyro_health":
            response = self._client.health()
            return json.dumps(
                {
                    "status": response.status,
                    "vector_count": response.vector_count,
                    "bm25_built": response.bm25_built,
                }
            )

        if name == "kyro_agent_query":
            response = self._client.agent_query(
                arguments["question"],
                top_k=int(arguments.get("top_k", 5)),
                max_steps=int(arguments.get("max_steps", 5)),
            )
            return json.dumps(
                {
                    "answer": response.answer,
                    "model": response.model,
                    "steps": [
                        {
                            "thought": s.thought,
                            "action": s.action,
                            "action_input": s.action_input,
                            "observation": s.observation,
                        }
                        for s in response.steps
                    ],
                }
            )

        raise ValueError(f"Unknown tool: {name!r}")

    # ── MCP transport ─────────────────────────────────────────────────────────

    async def run_stdio(self) -> None:  # pragma: no cover
        """Run the MCP server over stdin/stdout (requires ``mcp`` package).

        :raises RuntimeError: When the ``mcp`` package is not installed.
        """
        from konjoai.mcp import _HAS_MCP

        if not _HAS_MCP:
            raise RuntimeError("The 'mcp' package is required to run the MCP server. Install it with: pip install mcp")

        import mcp.types as types
        from mcp.server import Server
        from mcp.server.stdio import stdio_server

        server = Server("kyro-mcp")

        @server.list_tools()
        async def _list_tools() -> list[types.Tool]:
            """Return the MCP tool definitions exposed by this server."""
            return [
                types.Tool(
                    name=t["name"],
                    description=t["description"],
                    inputSchema=t["inputSchema"],
                )
                for t in self.list_tools()
            ]

        @server.call_tool()
        async def _call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
            """Dispatch an MCP tool call, returning its result as text content."""
            try:
                text = await self.dispatch(name, arguments or {})
            except KyroError as exc:
                text = json.dumps({"error": str(exc), "status_code": exc.status_code})
            return [types.TextContent(type="text", text=text)]

        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
