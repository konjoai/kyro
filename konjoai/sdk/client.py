from __future__ import annotations

import json
from typing import Iterator

import httpx

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


class KonjoClient:
    """Synchronous HTTP client for the Kyro RAG API.

    Usage::

        client = KonjoClient("http://localhost:8000", api_key="sk-...")
        response = client.query("What is the capital of Ethiopia?")
        print(response.answer)

        # Streaming tokens
        for chunk in client.query_stream("Explain transformers"):
            print(chunk.text, end="", flush=True)

    :param base_url: Base URL of the Kyro API (no trailing slash required).
    :param api_key: Optional API key sent as ``X-API-Key`` header.
    :param jwt_token: Optional JWT sent as ``Authorization: Bearer`` header.
    :param timeout: HTTP request timeout in seconds (default 30).
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        jwt_token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        headers: dict[str, str] = {}
        if api_key:
            headers["X-API-Key"] = api_key
        if jwt_token:
            headers["Authorization"] = f"Bearer {jwt_token}"
        self._client = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=timeout,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _raise_for_status(self, response: httpx.Response) -> None:
        code = response.status_code
        if code in (401, 403):
            raise KyroAuthError(
                f"Authentication failed ({code}): {response.text}",
                status_code=code,
            )
        if code == 404:
            raise KyroNotFoundError(
                f"Not found: {response.text}",
                status_code=404,
            )
        if code == 429:
            retry_after_raw = response.headers.get("Retry-After")
            raise KyroRateLimitError(
                f"Rate limit exceeded: {response.text}",
                retry_after=float(retry_after_raw) if retry_after_raw else None,
            )
        if code >= 500:
            raise KyroError(
                f"Server error {code}: {response.text}",
                status_code=code,
            )
        if code >= 400:
            raise KyroError(
                f"Client error {code}: {response.text}",
                status_code=code,
            )

    @staticmethod
    def _parse_sources(raw: list) -> list:
        return [
            SDKSourceDoc(
                source=s["source"],
                content_preview=s["content_preview"],
                score=float(s["score"]),
            )
            for s in raw
        ]

    # ── Public API ────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        *,
        top_k: int = 5,
        use_hyde: bool = False,
        use_crag: bool = False,
        use_self_rag: bool = False,
        use_decomposition: bool = False,
        use_graph_rag: bool = False,
    ) -> SDKQueryResponse:
        """Run the full RAG pipeline and return a typed response."""
        try:
            resp = self._client.post(
                "/query",
                json={
                    "question": question,
                    "top_k": top_k,
                    "use_hyde": use_hyde,
                    "use_crag": use_crag,
                    "use_self_rag": use_self_rag,
                    "use_decomposition": use_decomposition,
                    "use_graph_rag": use_graph_rag,
                },
            )
        except httpx.TimeoutException as exc:
            raise KyroTimeoutError(f"query timed out: {exc}") from exc
        self._raise_for_status(resp)
        data = resp.json()
        return SDKQueryResponse(
            answer=data["answer"],
            sources=self._parse_sources(data.get("sources", [])),
            model=data["model"],
            usage=data.get("usage", {}),
            telemetry=data.get("telemetry"),
            intent=data.get("intent", "retrieval"),
            cache_hit=data.get("cache_hit", False),
        )

    def query_stream(
        self,
        question: str,
        *,
        top_k: int = 5,
        use_hyde: bool = False,
    ) -> Iterator[SDKStreamChunk]:
        """Stream tokens from the RAG pipeline as Server-Sent Events.

        Each yielded ``SDKStreamChunk.text`` is one raw token string.
        """
        try:
            with self._client.stream(
                "POST",
                "/query/stream",
                json={"question": question, "top_k": top_k, "use_hyde": use_hyde},
            ) as resp:
                self._raise_for_status(resp)
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        text = chunk.get("token", payload) if isinstance(chunk, dict) else str(chunk)
                    except json.JSONDecodeError:
                        text = payload
                    yield SDKStreamChunk(text=text)
        except httpx.TimeoutException as exc:
            raise KyroTimeoutError(f"stream timed out: {exc}") from exc

    def ingest(
        self,
        path: str,
        *,
        strategy: str = "recursive",
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> SDKIngestResponse:
        """Ingest a file or directory path into the Kyro vector store."""
        try:
            resp = self._client.post(
                "/ingest",
                json={
                    "path": path,
                    "strategy": strategy,
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                },
            )
        except httpx.TimeoutException as exc:
            raise KyroTimeoutError(f"ingest timed out: {exc}") from exc
        self._raise_for_status(resp)
        data = resp.json()
        return SDKIngestResponse(
            chunks_indexed=data["chunks_indexed"],
            sources_processed=data["sources_processed"],
            chunks_deduplicated=data.get("chunks_deduplicated", 0),
        )

    def health(self) -> SDKHealthResponse:
        """Return a typed health-check response from ``GET /health``."""
        try:
            resp = self._client.get("/health")
        except httpx.TimeoutException as exc:
            raise KyroTimeoutError(f"health check timed out: {exc}") from exc
        self._raise_for_status(resp)
        data = resp.json()
        return SDKHealthResponse(
            status=data["status"],
            vector_count=data["vector_count"],
            bm25_built=data["bm25_built"],
        )

    def agent_query(
        self,
        question: str,
        *,
        top_k: int = 5,
        max_steps: int = 5,
    ) -> SDKAgentQueryResponse:
        """Run the bounded ReAct agent loop and return a typed response."""
        try:
            resp = self._client.post(
                "/agent/query",
                json={"question": question, "top_k": top_k, "max_steps": max_steps},
            )
        except httpx.TimeoutException as exc:
            raise KyroTimeoutError(f"agent query timed out: {exc}") from exc
        self._raise_for_status(resp)
        data = resp.json()
        return SDKAgentQueryResponse(
            answer=data["answer"],
            sources=self._parse_sources(data.get("sources", [])),
            model=data["model"],
            usage=data.get("usage", {}),
            steps=[
                SDKAgentStep(
                    thought=s["thought"],
                    action=s["action"],
                    action_input=s["action_input"],
                    observation=s["observation"],
                )
                for s in data.get("steps", [])
            ],
            telemetry=data.get("telemetry"),
        )

    def agent_query_stream(
        self,
        question: str,
        *,
        top_k: int = 5,
        max_steps: int = 5,
    ) -> Iterator[SDKAgentStreamEvent]:
        """Stream the bounded ReAct agent loop as Server-Sent Events.

        Each yielded :class:`SDKAgentStreamEvent` carries a ``type`` discriminator
        (``"step"``, ``"result"``, or ``"telemetry"``) and the decoded JSON ``data``
        payload. The terminal ``[DONE]`` sentinel is consumed silently and ends
        iteration.
        """
        try:
            with self._client.stream(
                "POST",
                "/agent/query/stream",
                json={"question": question, "top_k": top_k, "max_steps": max_steps},
            ) as resp:
                self._raise_for_status(resp)
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break
                    try:
                        frame = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(frame, dict):
                        continue
                    event_type = str(frame.get("type", ""))
                    if not event_type:
                        continue
                    yield SDKAgentStreamEvent(type=event_type, data=frame)
        except httpx.TimeoutException as exc:
            raise KyroTimeoutError(f"agent stream timed out: {exc}") from exc

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the underlying ``httpx.Client`` and release connections."""
        self._client.close()

    def __enter__(self) -> KonjoClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
