"""Shared scaffolding for ``/query`` route unit tests.

Centralises the settings/generator stubs, the FastAPI test-app builder, the
proxy-env scrubber, and the default hybrid/rerank fixtures that every route
test would otherwise duplicate verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.testclient import TestClient

from konjoai.api.routes.query import router
from konjoai.generate.generator import GenerationResult
from konjoai.retrieve.hybrid import HybridResult
from konjoai.retrieve.reranker import RerankResult

# Every proxy env var that can make TestClient attempt a real network call.
_PROXY_VARS = (
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
    "GRPC_PROXY",
    "grpc_proxy",
)


@dataclass
class SettingsStub:
    """Stand-in for ``KyroConfig`` covering the flags the query route reads."""

    enable_query_router: bool = True
    enable_hyde: bool = False
    enable_telemetry: bool = True
    use_vectro_retriever: bool = False
    use_colbert: bool = False
    enable_crag: bool = False
    enable_self_rag: bool = False
    enable_query_decomposition: bool = False
    decomposition_max_sub_queries: int = 4
    self_rag_max_iterations: int = 3
    top_k_dense: int = 8
    top_k_sparse: int = 8
    openai_model: str = "stub-model"
    request_timeout_seconds: float = 30.0
    enable_graph_rag: bool = False
    graph_rag_max_communities: int = 5
    graph_rag_similarity_threshold: float = 0.3
    otel_enabled: bool = False
    audit_enabled: bool = False


class GeneratorStub:
    """Generator that returns a fixed ``base answer`` regardless of input."""

    def generate(self, question: str, context: str) -> GenerationResult:
        """Return a deterministic stub completion."""
        _ = (question, context)
        return GenerationResult(
            answer="base answer",
            model="stub-model",
            usage={"prompt_tokens": 10, "completion_tokens": 2},
        )


def make_app() -> FastAPI:
    """Return a FastAPI app with only the query router mounted."""
    app = FastAPI()
    app.include_router(router)
    return app


def make_client() -> TestClient:
    """Return a TestClient wrapping a fresh query-router app."""
    return TestClient(make_app())


def clear_proxy_env(monkeypatch) -> None:
    """Remove every proxy env var so TestClient never hits the network."""
    for var in _PROXY_VARS:
        monkeypatch.delenv(var, raising=False)


def default_hybrid_results() -> list[HybridResult]:
    """Return the two-item hybrid-search result list used across route tests."""
    return [
        HybridResult(content="h1", source="a.txt", rrf_score=0.8, metadata={}),
        HybridResult(content="h2", source="b.txt", rrf_score=0.7, metadata={}),
    ]


def default_reranked_results() -> list[RerankResult]:
    """Return the single-item rerank result list used across route tests."""
    return [RerankResult(score=0.95, content="h1", source="a.txt", metadata={})]
