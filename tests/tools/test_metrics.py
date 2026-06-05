"""Tests for server.tools.metrics — ToolMetrics."""

from __future__ import annotations

import pytest

from server.tools.spec import ToolRuntime, ToolSpec, ToolStatus
from server.tools.metrics import ToolMetrics


@pytest.fixture()
def metrics(registry, telemetry):
    """ToolMetrics backed by the shared fixtures."""
    return ToolMetrics(registry=registry, telemetry=telemetry)


class TestEmptyTelemetry:
    """All metrics return 0/None with empty telemetry."""

    def test_tool_invocation_count_zero(self, metrics):
        assert metrics.get_tool_invocation_count() == 0

    def test_created_tool_count_zero(self, metrics):
        assert metrics.get_created_tool_count() == 0

    def test_registered_tool_count_zero(self, metrics):
        assert metrics.get_registered_tool_count() == 0

    def test_candidate_count_zero(self, metrics):
        assert metrics.get_candidate_count() == 0

    def test_active_generated_tool_count_zero(self, metrics):
        assert metrics.get_active_generated_tool_count() == 0

    def test_tool_request_count_zero(self, metrics):
        assert metrics.get_tool_request_count() == 0

    def test_tool_request_consumed_count_zero(self, metrics):
        assert metrics.get_tool_request_consumed_count() == 0


class TestGetEgl:
    """get_egl returns None when no invocations."""

    def test_egl_none_with_no_data(self, metrics):
        assert metrics.get_egl() is None

    def test_egl_zero_when_no_created_events(self, metrics, telemetry):
        telemetry.record_tool_invoked("some_tool")
        egl = metrics.get_egl()
        assert egl is not None
        assert egl == 0.0


class TestGetToolSuccessRate:
    """get_tool_success_rate returns None with no data."""

    def test_success_rate_none_with_no_data(self, metrics):
        assert metrics.get_tool_success_rate() is None

    def test_success_rate_none_for_unknown_tool(self, metrics, telemetry):
        telemetry.record_tool_succeeded("known_tool")
        assert metrics.get_tool_success_rate(tool_name="unknown_tool") is None

    def test_success_rate_computed(self, metrics, telemetry):
        telemetry.record_tool_succeeded("my_tool")
        telemetry.record_tool_succeeded("my_tool")
        telemetry.record_tool_failed("my_tool")
        rate = metrics.get_tool_success_rate(tool_name="my_tool")
        assert rate is not None
        assert rate == pytest.approx(2 / 3)


class TestAllMetrics:
    """all_metrics returns a dict with expected keys."""

    def test_all_metrics_keys(self, metrics):
        result = metrics.get_all_metrics()
        expected_keys = {
            "tool_invocation_count",
            "created_tool_count",
            "registered_tool_count",
            "candidate_count",
            "active_generated_tool_count",
            "tool_request_count",
            "tool_request_consumed_count",
            "tool_success_rate",
            "remote_escalation_rate",
            "egl",
            "window",
        }
        assert set(result.keys()) == expected_keys

    def test_all_metrics_default_values(self, metrics):
        result = metrics.get_all_metrics()
        assert result["tool_invocation_count"] == 0
        assert result["created_tool_count"] == 0
        assert result["registered_tool_count"] == 0
        assert result["candidate_count"] == 0
        assert result["active_generated_tool_count"] == 0
        assert result["tool_request_count"] == 0
        assert result["tool_request_consumed_count"] == 0
        assert result["tool_success_rate"] is None
        assert result["remote_escalation_rate"] is None
        assert result["egl"] is None
        assert result["window"] == "all_time"

    def test_all_metrics_reflects_registered_tools(self, metrics, registry):
        spec = ToolSpec(
            name="active_tool",
            description="An active tool",
            input_schema={"type": "object", "properties": {}},
            runtime=ToolRuntime.MEMORY_METHOD,
            status=ToolStatus.ACTIVE,
        )
        registry.register(spec)

        result = metrics.get_all_metrics()
        assert result["registered_tool_count"] == 1
