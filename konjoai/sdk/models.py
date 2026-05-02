from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SDKSourceDoc:
    source: str
    content_preview: str
    score: float


@dataclass(frozen=True)
class SDKQueryResponse:
    answer: str
    sources: list
    model: str
    usage: dict
    telemetry: object = None
    intent: str = "retrieval"
    cache_hit: bool = False


@dataclass(frozen=True)
class SDKStreamChunk:
    """A single streamed token chunk from ``KonjoClient.query_stream``."""
    text: str


@dataclass(frozen=True)
class SDKIngestResponse:
    chunks_indexed: int
    sources_processed: int
    chunks_deduplicated: int = 0


@dataclass(frozen=True)
class SDKHealthResponse:
    status: str
    vector_count: int
    bm25_built: bool


@dataclass(frozen=True)
class SDKAgentStep:
    thought: str
    action: str
    action_input: str
    observation: str


@dataclass(frozen=True)
class SDKAgentQueryResponse:
    answer: str
    sources: list
    model: str
    usage: dict
    steps: list
    telemetry: object = None


@dataclass(frozen=True)
class SDKAgentStreamEvent:
    """A single Server-Sent Event from ``KonjoClient.agent_query_stream``.

    ``type`` is one of ``"step"``, ``"result"``, or ``"telemetry"``. The
    ``data`` payload mirrors the JSON frame emitted by the server, decoded
    into a plain dict so callers can branch on ``type`` without depending on
    a specific schema version.
    """
    type: str
    data: dict = field(default_factory=dict)
