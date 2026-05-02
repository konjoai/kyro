"""Kyro Python SDK — typed synchronous client for the Kyro RAG API.

Quick start::

    from konjoai.sdk import KonjoClient

    client = KonjoClient("http://localhost:8000", api_key="sk-...")
    response = client.query("What is the capital of Ethiopia?")
    print(response.answer)
"""
from __future__ import annotations

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
    SDKAgentStreamEvent,
    SDKHealthResponse,
    SDKIngestResponse,
    SDKQueryResponse,
    SDKSourceDoc,
    SDKStreamChunk,
)

__all__ = [
    "KonjoClient",
    "KyroError",
    "KyroAuthError",
    "KyroNotFoundError",
    "KyroRateLimitError",
    "KyroTimeoutError",
    "SDKQueryResponse",
    "SDKIngestResponse",
    "SDKHealthResponse",
    "SDKAgentQueryResponse",
    "SDKAgentStep",
    "SDKAgentStreamEvent",
    "SDKSourceDoc",
    "SDKStreamChunk",
]
