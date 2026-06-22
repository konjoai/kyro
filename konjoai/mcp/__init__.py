"""Kyro MCP server — exposes Kyro RAG tools via the Model Context Protocol.

Requires the optional ``mcp`` package::

    pip install mcp

Run as a stdio server::

    python -m konjoai.mcp --base-url http://localhost:8000 --api-key sk-...
"""

from __future__ import annotations

try:
    import mcp  # noqa: F401

    _HAS_MCP: bool = True
except ImportError:
    _HAS_MCP = False

from konjoai.mcp.server import KyroMCPServer

__all__ = ["KyroMCPServer", "_HAS_MCP"]
