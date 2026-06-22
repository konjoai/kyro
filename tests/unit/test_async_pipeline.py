"""Unit tests for the async query pipeline (Sprint 8)."""

from __future__ import annotations

import inspect


class TestAsyncSignatures:
    """Verify query endpoints are declared as coroutines."""

    def test_query_is_coroutine(self) -> None:
        from konjoai.api.routes.query import query

        assert inspect.iscoroutinefunction(query)

    def test_query_stream_is_coroutine(self) -> None:
        from konjoai.api.routes.query import query_stream

        assert inspect.iscoroutinefunction(query_stream)


class TestAsyncSourceInspection:
    """Verify async patterns exist in query module source."""

    def test_to_thread_used_in_query_module(self) -> None:
        import konjoai.api.routes.query as q_mod

        src = inspect.getsource(q_mod)
        assert "asyncio.to_thread" in src

    def test_gather_used_for_aggregation(self) -> None:
        import konjoai.api.routes.query as q_mod
        import konjoai.retrieve.decomposition as d_mod

        query_src = inspect.getsource(q_mod)
        decomposition_src = inspect.getsource(d_mod)
        assert "asyncio.gather" in query_src or "asyncio.gather" in decomposition_src

    def test_asyncio_imported_in_query_module(self) -> None:
        import konjoai.api.routes.query as q_mod

        src = inspect.getsource(q_mod)
        assert "import asyncio" in src


class TestConfigAsyncFields:
    """Verify the three Sprint 8 async config fields exist with correct defaults."""

    def test_async_enabled_default(self) -> None:
        from konjoai.config import Settings

        s = Settings()
        assert hasattr(s, "async_enabled")
        assert s.async_enabled is True

    def test_request_timeout_seconds_default(self) -> None:
        from konjoai.config import Settings

        s = Settings()
        assert hasattr(s, "request_timeout_seconds")
        assert isinstance(s.request_timeout_seconds, (int, float))
        assert s.request_timeout_seconds > 0

    def test_qdrant_max_connections_default(self) -> None:
        from konjoai.config import Settings

        s = Settings()
        assert hasattr(s, "qdrant_max_connections")
        assert isinstance(s.qdrant_max_connections, int)
        assert s.qdrant_max_connections > 0


class TestAsyncQdrantStore:
    """Structural tests for AsyncQdrantStore."""

    def test_async_store_class_exists(self) -> None:
        from konjoai.store.qdrant import AsyncQdrantStore

        assert AsyncQdrantStore is not None

    def test_get_async_store_callable(self) -> None:
        from konjoai.store.qdrant import get_async_store

        assert callable(get_async_store)

    def test_async_store_module_source(self) -> None:
        import konjoai.store.qdrant as qdrant_mod

        src = inspect.getsource(qdrant_mod)
        assert "AsyncQdrantStore" in src
        assert "get_async_store" in src
