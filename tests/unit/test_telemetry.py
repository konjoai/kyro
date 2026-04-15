"""Unit tests for ragos.telemetry — StepTiming, PipelineTelemetry, timed()."""
from __future__ import annotations

import time

import pytest

from ragos.telemetry import PipelineTelemetry, StepTiming, timed


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
