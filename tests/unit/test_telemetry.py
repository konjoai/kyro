"""Unit tests for konjoai.telemetry — StepTiming, PipelineTelemetry, timed(),
KyroMetrics, KyroTracer, get_metrics(), get_tracer(), record_pipeline_metrics().
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

import konjoai.telemetry as _tel_module
from konjoai.telemetry import (
    _HAS_OTEL,
    _HAS_PROMETHEUS,
    KyroMetrics,
    KyroTracer,
    PipelineTelemetry,
    StepTiming,
    _noop_span,
    get_metrics,
    get_tracer,
    record_pipeline_metrics,
    timed,
)

# ---------------------------------------------------------------------------
# StepTiming
# ---------------------------------------------------------------------------


class TestStepTiming:
    def test_as_dict_contains_duration(self) -> None:
        st = StepTiming(step="foo", duration_ms=12.345)
        d = st.as_dict()
        assert "duration_ms" in d
        assert d["duration_ms"] == 12.345

    def test_as_dict_rounds_to_3dp(self) -> None:
        st = StepTiming(step="foo", duration_ms=1.23456789)
        # round(1.23456789, 3) == 1.235
        assert st.as_dict()["duration_ms"] == 1.235

    def test_as_dict_includes_metadata(self) -> None:
        st = StepTiming(step="search", duration_ms=5.0, metadata={"top_k": 10})
        d = st.as_dict()
        assert d["top_k"] == 10

    def test_as_dict_excludes_step_key(self) -> None:
        st = StepTiming(step="search", duration_ms=5.0)
        assert "step" not in st.as_dict()

    def test_empty_metadata_produces_minimal_dict(self) -> None:
        st = StepTiming(step="a", duration_ms=0.0)
        assert set(st.as_dict().keys()) == {"duration_ms"}


# ---------------------------------------------------------------------------
# PipelineTelemetry
# ---------------------------------------------------------------------------


class TestPipelineTelemetry:
    def test_empty_telemetry_as_dict(self) -> None:
        tel = PipelineTelemetry()
        d = tel.as_dict()
        assert d["steps"] == {}
        assert d["total_ms"] == 0.0

    def test_record_single_step(self) -> None:
        tel = PipelineTelemetry()
        tel.record("route", 1.5)
        d = tel.as_dict()
        assert "route" in d["steps"]
        assert d["steps"]["route"]["duration_ms"] == 1.5

    def test_total_ms_sums_steps(self) -> None:
        tel = PipelineTelemetry()
        tel.record("a", 10.0)
        tel.record("b", 20.0)
        assert tel.total_ms() == 30.0

    def test_as_dict_total_ms_rounded(self) -> None:
        tel = PipelineTelemetry()
        tel.record("a", 10.0)
        tel.record("b", 20.5)
        d = tel.as_dict()
        assert d["total_ms"] == pytest.approx(30.5, abs=1e-3)

    def test_record_with_metadata(self) -> None:
        tel = PipelineTelemetry()
        tel.record("search", 5.0, top_k=10, model="encoder")
        step = tel.as_dict()["steps"]["search"]
        assert step["top_k"] == 10
        assert step["model"] == "encoder"

    def test_duplicate_step_name_last_writer_wins(self) -> None:
        # as_dict uses {s.step: s.as_dict() for s in self.steps} → last entry wins
        tel = PipelineTelemetry()
        tel.record("search", 5.0)
        tel.record("search", 8.0)
        d = tel.as_dict()
        assert d["steps"]["search"]["duration_ms"] == 8.0

    def test_multiple_distinct_steps_all_present(self) -> None:
        tel = PipelineTelemetry()
        for name in ("route", "hyde", "hybrid_search", "rerank", "generate"):
            tel.record(name, 1.0)
        d = tel.as_dict()
        for name in ("route", "hyde", "hybrid_search", "rerank", "generate"):
            assert name in d["steps"]

    def test_total_ms_is_sum_of_all_step_durations(self) -> None:
        tel = PipelineTelemetry()
        tel.record("a", 7.5)
        tel.record("b", 2.5)
        assert tel.total_ms() == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# timed() context manager
# ---------------------------------------------------------------------------


class TestTimedContextManager:
    def test_timed_records_step(self) -> None:
        tel = PipelineTelemetry()
        with timed(tel, "test_step"):
            pass
        assert "test_step" in tel.as_dict()["steps"]

    def test_timed_duration_is_nonnegative(self) -> None:
        tel = PipelineTelemetry()
        with timed(tel, "work"):
            pass
        assert tel.as_dict()["steps"]["work"]["duration_ms"] >= 0.0

    def test_timed_captures_elapsed_time(self) -> None:
        tel = PipelineTelemetry()
        with timed(tel, "work"):
            time.sleep(0.01)  # 10 ms sleep
        # At least 8 ms (generous lower bound — CI can be slow)
        assert tel.as_dict()["steps"]["work"]["duration_ms"] >= 8.0

    def test_timed_records_metadata(self) -> None:
        tel = PipelineTelemetry()
        with timed(tel, "search", top_k=5, model="x"):
            pass
        step = tel.as_dict()["steps"]["search"]
        assert step["top_k"] == 5
        assert step["model"] == "x"

    def test_timed_records_on_exception(self) -> None:
        """Step must be recorded even when the wrapped block raises."""
        tel = PipelineTelemetry()
        with pytest.raises(ValueError):
            with timed(tel, "failing_step"):
                raise ValueError("intentional")
        assert "failing_step" in tel.as_dict()["steps"]

    def test_multiple_sequential_steps(self) -> None:
        tel = PipelineTelemetry()
        with timed(tel, "step_a"):
            pass
        with timed(tel, "step_b"):
            pass
        d = tel.as_dict()
        assert "step_a" in d["steps"]
        assert "step_b" in d["steps"]

    def test_total_ms_equals_sum_of_timed_steps(self) -> None:
        tel = PipelineTelemetry()
        with timed(tel, "step_a"):
            pass
        with timed(tel, "step_b"):
            pass
        d = tel.as_dict()
        expected = d["steps"]["step_a"]["duration_ms"] + d["steps"]["step_b"]["duration_ms"]
        assert d["total_ms"] == pytest.approx(expected, abs=1e-3)


# ---------------------------------------------------------------------------
# Sprint 16: _noop_span
# ---------------------------------------------------------------------------


class TestNoopSpan:
    def test_noop_span_is_context_manager(self) -> None:
        with _noop_span():
            pass  # must not raise

    def test_noop_span_yields_none(self) -> None:
        with _noop_span() as val:
            assert val is None


# ---------------------------------------------------------------------------
# Sprint 16: KyroMetrics — disabled path (prometheus-client absent or enabled=False)
# ---------------------------------------------------------------------------


class TestKyroMetricsDisabled:
    def test_available_false_when_disabled(self) -> None:
        m = KyroMetrics(enabled=False)
        assert m.available is False

    def test_record_step_noop_when_disabled(self) -> None:
        m = KyroMetrics(enabled=False)
        m.record_step("hybrid_search", 42.0)  # must not raise

    def test_inc_query_noop_when_disabled(self) -> None:
        m = KyroMetrics(enabled=False)
        m.inc_query("retrieval")  # must not raise

    def test_inc_error_noop_when_disabled(self) -> None:
        m = KyroMetrics(enabled=False)
        m.inc_error("generate")  # must not raise

    def test_inc_cache_hit_noop_when_disabled(self) -> None:
        m = KyroMetrics(enabled=False)
        m.inc_cache_hit()  # must not raise

    def test_exposition_returns_empty_when_disabled(self) -> None:
        m = KyroMetrics(enabled=False)
        assert m.exposition() == ""

    def test_available_false_when_prometheus_absent(self) -> None:
        with patch.object(_tel_module, "_HAS_PROMETHEUS", False):
            m = KyroMetrics(enabled=True)
            assert m.available is False

    def test_exposition_empty_when_prometheus_absent(self) -> None:
        with patch.object(_tel_module, "_HAS_PROMETHEUS", False):
            m = KyroMetrics(enabled=True)
            assert m.exposition() == ""


# ---------------------------------------------------------------------------
# Sprint 16: KyroMetrics — enabled path (prometheus-client present)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PROMETHEUS, reason="prometheus-client not installed")
class TestKyroMetricsEnabled:
    # Share a single KyroMetrics instance across all methods in this class so
    # we only register the prometheus counters once — registering the same
    # metric name twice raises ValueError in prometheus-client >= 0.20.
    _m: KyroMetrics | None = None

    @classmethod
    def setup_class(cls) -> None:
        cls._m = KyroMetrics(enabled=True)

    def test_available_true_when_enabled(self) -> None:
        assert self._m.available is True

    def test_exposition_non_empty(self) -> None:
        text = self._m.exposition()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_record_step_does_not_raise(self) -> None:
        self._m.record_step("rerank", 5.0)

    def test_inc_query_does_not_raise(self) -> None:
        self._m.inc_query("aggregation")

    def test_inc_error_does_not_raise(self) -> None:
        self._m.inc_error("generate")

    def test_inc_cache_hit_does_not_raise(self) -> None:
        self._m.inc_cache_hit()


# ---------------------------------------------------------------------------
# Sprint 16: KyroTracer
# ---------------------------------------------------------------------------


class TestKyroTracerDisabled:
    def test_available_false_when_otel_absent(self) -> None:
        with patch.object(_tel_module, "_HAS_OTEL", False):
            t = KyroTracer(endpoint="http://localhost:4317")
            assert t.available is False

    def test_available_false_when_no_endpoint(self) -> None:
        t = KyroTracer(endpoint="")
        assert t.available is False

    def test_start_span_returns_context_manager_when_unavailable(self) -> None:
        t = KyroTracer(endpoint="")
        ctx = t.start_span("test_span")
        with ctx:
            pass  # must not raise

    def test_available_false_without_otel(self) -> None:
        with patch.object(_tel_module, "_HAS_OTEL", False):
            t = KyroTracer(endpoint="http://localhost:4317", service_name="kyro")
            assert not t.available


# ---------------------------------------------------------------------------
# Sprint 16: get_metrics() singleton
# ---------------------------------------------------------------------------


class TestGetMetricsSingleton:
    def setup_method(self) -> None:
        _tel_module._metrics = None  # reset singleton before each test

    def teardown_method(self) -> None:
        _tel_module._metrics = None  # clean up

    def test_get_metrics_returns_kyro_metrics(self) -> None:
        m = get_metrics()
        assert isinstance(m, KyroMetrics)

    def test_get_metrics_singleton_same_object(self) -> None:
        m1 = get_metrics()
        m2 = get_metrics()
        assert m1 is m2

    def test_get_metrics_disabled_when_otel_disabled(self) -> None:
        stub_settings = MagicMock()
        stub_settings.otel_enabled = False
        with patch("konjoai.telemetry.get_metrics") as mock_gm:
            mock_gm.return_value = KyroMetrics(enabled=False)
            m = mock_gm()
        assert not m.available


# ---------------------------------------------------------------------------
# Sprint 16: get_tracer() singleton
# ---------------------------------------------------------------------------


class TestGetTracerSingleton:
    def setup_method(self) -> None:
        _tel_module._tracer = None

    def teardown_method(self) -> None:
        _tel_module._tracer = None

    def test_get_tracer_returns_kyro_tracer(self) -> None:
        t = get_tracer()
        assert isinstance(t, KyroTracer)

    def test_get_tracer_singleton_same_object(self) -> None:
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2


# ---------------------------------------------------------------------------
# Sprint 16: record_pipeline_metrics
# ---------------------------------------------------------------------------


class TestRecordPipelineMetrics:
    def test_noop_when_enabled_false(self) -> None:
        tel = PipelineTelemetry()
        tel.record("hybrid_search", 10.0)
        # Must not raise; metrics instance not created at all
        record_pipeline_metrics(tel, "retrieval", enabled=False)

    def test_calls_inc_query_with_intent(self) -> None:
        tel = PipelineTelemetry()
        mock_m = MagicMock(spec=KyroMetrics)
        with patch("konjoai.telemetry.get_metrics", return_value=mock_m):
            record_pipeline_metrics(tel, "aggregation", enabled=True)
        mock_m.inc_query.assert_called_once_with("aggregation")

    def test_calls_record_step_per_step(self) -> None:
        tel = PipelineTelemetry()
        tel.record("route", 1.0)
        tel.record("rerank", 5.0)
        tel.record("generate", 50.0)
        mock_m = MagicMock(spec=KyroMetrics)
        with patch("konjoai.telemetry.get_metrics", return_value=mock_m):
            record_pipeline_metrics(tel, "retrieval", enabled=True)
        assert mock_m.record_step.call_count == 3

    def test_default_intent_is_retrieval(self) -> None:
        tel = PipelineTelemetry()
        mock_m = MagicMock(spec=KyroMetrics)
        with patch("konjoai.telemetry.get_metrics", return_value=mock_m):
            record_pipeline_metrics(tel, enabled=True)
        mock_m.inc_query.assert_called_once_with("retrieval")

    def test_empty_telemetry_no_step_calls(self) -> None:
        tel = PipelineTelemetry()
        mock_m = MagicMock(spec=KyroMetrics)
        with patch("konjoai.telemetry.get_metrics", return_value=mock_m):
            record_pipeline_metrics(tel, enabled=True)
        mock_m.record_step.assert_not_called()


# ---------------------------------------------------------------------------
# Sprint 16: _HAS_PROMETHEUS / _HAS_OTEL module flags
# ---------------------------------------------------------------------------


class TestModuleFlags:
    def test_has_prometheus_is_bool(self) -> None:
        assert isinstance(_HAS_PROMETHEUS, bool)

    def test_has_otel_is_bool(self) -> None:
        assert isinstance(_HAS_OTEL, bool)
