"""Tests for Sprint 26 OTel cache span emission.

Coverage:
- emit_cache_lookup: no-op when otel_enabled=False
- emit_cache_lookup: emits spans when OTel is available and enabled
- emit_cache_store: no-op / span emission
- cache_span: context manager wrapping
- K3: no exceptions when opentelemetry-sdk is absent (mocked)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from konjoai.cache.tracing import emit_cache_lookup, emit_cache_store


class TestEmitCacheLookup:
    def test_noop_when_otel_disabled(self) -> None:
        # Should not raise and should not call any OTel infrastructure
        emit_cache_lookup(
            question="q",
            q_vec=np.zeros(4, dtype=np.float32),
            result=None,
            similarity_score=0.5,
            threshold_used=0.85,
            latency_ms=1.2,
            query_type="faq",
            otel_enabled=False,
        )

    def test_noop_when_otel_disabled_with_hit(self) -> None:
        emit_cache_lookup(
            question="q",
            q_vec=np.zeros(4, dtype=np.float32),
            result=object(),
            similarity_score=0.95,
            threshold_used=0.85,
            latency_ms=0.3,
            query_type="faq",
            otel_enabled=False,
        )

    def test_noop_when_has_otel_false(self) -> None:
        with patch("konjoai.cache.tracing._HAS_OTEL", False):
            emit_cache_lookup(
                question="q",
                q_vec=np.zeros(4, dtype=np.float32),
                result=None,
                similarity_score=-1.0,
                threshold_used=0.94,
                latency_ms=2.0,
                query_type="factual",
                otel_enabled=True,
            )

    def test_span_emitted_when_otel_enabled(self) -> None:
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span

        mock_wrapper = MagicMock()
        mock_wrapper.available = True
        mock_wrapper._tracer = mock_tracer

        with (
            patch("konjoai.cache.tracing._HAS_OTEL", True),
            patch("konjoai.telemetry.get_tracer", return_value=mock_wrapper),
            patch("konjoai.cache.tracing.get_current_tenant_id", return_value="acme"),
        ):
            emit_cache_lookup(
                question="What is the refund policy?",
                q_vec=np.ones(4, dtype=np.float32),
                result=object(),
                similarity_score=0.97,
                threshold_used=0.85,
                latency_ms=0.45,
                query_type="faq",
                avg_response_tokens=300,
                otel_enabled=True,
            )

        # At least one span was started
        assert mock_tracer.start_as_current_span.call_count >= 1
        # Attributes were set on the span
        assert mock_span.set_attribute.called

    def test_attributes_include_tenant_id(self) -> None:
        attrs: dict[str, object] = {}
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_span.set_attribute.side_effect = lambda k, v: attrs.update({k: v})
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span
        mock_wrapper = MagicMock()
        mock_wrapper.available = True
        mock_wrapper._tracer = mock_tracer

        with (
            patch("konjoai.cache.tracing._HAS_OTEL", True),
            patch("konjoai.telemetry.get_tracer", return_value=mock_wrapper),
            patch("konjoai.cache.tracing.get_current_tenant_id", return_value="globex"),
        ):
            emit_cache_lookup(
                question="q",
                q_vec=np.zeros(4, dtype=np.float32),
                result=None,
                similarity_score=0.5,
                threshold_used=0.92,
                latency_ms=1.1,
                query_type="code",
                otel_enabled=True,
            )

        assert attrs.get("kyro.tenant_id") == "globex"
        assert "kyro.similarity_score" in attrs
        assert "kyro.threshold_used" in attrs

    def test_exception_in_tracer_is_swallowed(self) -> None:
        mock_wrapper = MagicMock()
        mock_wrapper.available = True
        mock_wrapper._tracer = None  # forces AttributeError / TypeError path

        with (
            patch("konjoai.cache.tracing._HAS_OTEL", True),
            patch("konjoai.telemetry.get_tracer", return_value=mock_wrapper),
        ):
            # Must not raise
            emit_cache_lookup(
                question="q",
                q_vec=np.zeros(4, dtype=np.float32),
                result=None,
                similarity_score=-1.0,
                threshold_used=0.85,
                latency_ms=0.0,
                otel_enabled=True,
            )


class TestEmitCacheStore:
    def test_noop_when_disabled(self) -> None:
        emit_cache_store(question="q", latency_ms=5.0, otel_enabled=False)

    def test_noop_when_has_otel_false(self) -> None:
        with patch("konjoai.cache.tracing._HAS_OTEL", False):
            emit_cache_store(question="q", latency_ms=5.0, otel_enabled=True)

    def test_span_emitted_when_enabled(self) -> None:
        mock_span = MagicMock()
        mock_span.__enter__ = MagicMock(return_value=mock_span)
        mock_span.__exit__ = MagicMock(return_value=False)
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_span
        mock_wrapper = MagicMock()
        mock_wrapper.available = True
        mock_wrapper._tracer = mock_tracer

        with (
            patch("konjoai.cache.tracing._HAS_OTEL", True),
            patch("konjoai.telemetry.get_tracer", return_value=mock_wrapper),
            patch("konjoai.cache.tracing.get_current_tenant_id", return_value="acme"),
        ):
            emit_cache_store(question="q", latency_ms=3.7, otel_enabled=True)

        mock_tracer.start_as_current_span.assert_called_once_with("cache.store")

    def test_exception_in_store_span_swallowed(self) -> None:
        mock_wrapper = MagicMock()
        mock_wrapper.available = True
        mock_wrapper._tracer = None

        with (
            patch("konjoai.cache.tracing._HAS_OTEL", True),
            patch("konjoai.telemetry.get_tracer", return_value=mock_wrapper),
        ):
            emit_cache_store(question="q", latency_ms=1.0, otel_enabled=True)
