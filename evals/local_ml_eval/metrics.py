"""Metrics computation for local_ml_eval."""

from __future__ import annotations

from typing import Any


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics from eval results.

    Each result dict should have:
        - task_success: bool
        - expected_tool_hit: bool (expected tool was called)
        - forbidden_tool_called: bool (a forbidden tool was called)
        - tool_failed: bool (tool execution failed)
        - tool_requested: bool (a ToolRequest was recorded)
        - generated_tool_success: bool (generated tool executed successfully)
        - latency_ms: float
    """
    total = len(results)
    if total == 0:
        return _empty_metrics()

    task_successes = sum(1 for r in results if r.get("task_success"))
    expected_hits = sum(1 for r in results if r.get("expected_tool_hit"))
    forbidden_calls = sum(1 for r in results if r.get("forbidden_tool_called"))
    tool_failures = sum(1 for r in results if r.get("tool_failed"))
    tool_requests = sum(1 for r in results if r.get("tool_requested"))
    generated_successes = sum(1 for r in results if r.get("generated_tool_success"))
    generated_total = sum(1 for r in results if r.get("category") == "generated_tool")
    latencies = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]

    return {
        "total_tasks": total,
        "task_success_rate": round(task_successes / total, 4),
        "expected_tool_hit_rate": round(expected_hits / total, 4),
        "forbidden_tool_call_rate": round(forbidden_calls / total, 4),
        "tool_failure_rate": round(tool_failures / total, 4),
        "tool_request_rate": round(tool_requests / total, 4),
        "generated_tool_success_rate": round(
            generated_successes / generated_total, 4
        ) if generated_total > 0 else None,
        "egl": _compute_egl(results),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
    }


def _compute_egl(results: list[dict[str, Any]]) -> float | None:
    """EGL = generated tools successfully used / total tool invocations."""
    tool_invocations = sum(
        1 for r in results
        if r.get("tool_called") and not r.get("tool_requested")
    )
    generated_used = sum(
        1 for r in results
        if r.get("category") == "generated_tool" and r.get("generated_tool_success")
    )
    if tool_invocations == 0:
        return None
    return round(generated_used / tool_invocations, 4)


def _empty_metrics() -> dict[str, Any]:
    return {
        "total_tasks": 0,
        "task_success_rate": 0.0,
        "expected_tool_hit_rate": 0.0,
        "forbidden_tool_call_rate": 0.0,
        "tool_failure_rate": 0.0,
        "tool_request_rate": 0.0,
        "generated_tool_success_rate": None,
        "egl": None,
        "avg_latency_ms": None,
    }
