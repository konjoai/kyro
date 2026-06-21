"""Entry point for ``python -m konjoai.mcp``.

Run the Kyro MCP server over stdio::

    python -m konjoai.mcp --base-url http://localhost:8000
    python -m konjoai.mcp --base-url http://localhost:8000 --api-key sk-xxx
    python -m konjoai.mcp --base-url http://localhost:8000 --timeout 60
"""

from __future__ import annotations

import asyncio
import sys

import click

from konjoai.mcp import _HAS_MCP
from konjoai.mcp.server import KyroMCPServer


@click.command()
@click.option(
    "--base-url",
    default="http://localhost:8000",
    show_default=True,
    help="Base URL of the Kyro API.",
)
@click.option(
    "--api-key",
    default=None,
    envvar="KYRO_API_KEY",
    help="API key (also read from KYRO_API_KEY env var).",
)
@click.option(
    "--jwt-token",
    default=None,
    envvar="KYRO_JWT_TOKEN",
    help="JWT Bearer token (also read from KYRO_JWT_TOKEN env var).",
)
@click.option(
    "--timeout",
    default=30.0,
    show_default=True,
    type=float,
    help="HTTP request timeout in seconds.",
)
def main(base_url: str, api_key: str | None, jwt_token: str | None, timeout: float) -> None:
    """Run the Kyro MCP server over stdin/stdout."""
    if not _HAS_MCP:
        click.echo(
            "ERROR: The 'mcp' package is required. Install it with:\n  pip install mcp",
            err=True,
        )
        sys.exit(1)

    server = KyroMCPServer.from_url(
        base_url,
        api_key=api_key,
        jwt_token=jwt_token,
        timeout=timeout,
    )
    asyncio.run(server.run_stdio())


if __name__ == "__main__":
    main()
