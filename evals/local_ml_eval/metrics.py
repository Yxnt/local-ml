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
        - generated_tools_created: int (generated tools created in the run)
        - tool_invocation_count: int (tool invocations in the run)
        - in_situ_generation_attempted: bool
        - in_situ_generation_success: bool
        - warm_start_reused_generated_tool: bool
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
    evolution_growth_level = _compute_evolution_growth_level(results)

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
        "evolution_growth_level": evolution_growth_level,
        "egl": evolution_growth_level,
        "generated_tool_use_rate": _compute_generated_tool_use_rate(results),
        "in_situ_generation_success_rate": _compute_in_situ_generation_success_rate(results),
        "warm_start_reuse_rate": _compute_warm_start_reuse_rate(results),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
    }


def placeholder_metrics(total_tasks: int) -> dict[str, Any]:
    """Return placeholder metrics for dry-run output.

    Dry-run validates task structure only, so execution metrics should remain
    explicitly unset instead of looking like genuine zero scores.
    """
    return {
        "total_tasks": total_tasks,
        "task_success_rate": None,
        "expected_tool_hit_rate": None,
        "forbidden_tool_call_rate": None,
        "tool_failure_rate": None,
        "tool_request_rate": None,
        "generated_tool_success_rate": None,
        "evolution_growth_level": None,
        "egl": None,
        "generated_tool_use_rate": None,
        "in_situ_generation_success_rate": None,
        "warm_start_reuse_rate": None,
        "avg_latency_ms": None,
    }


def _tool_invocations(results: list[dict[str, Any]]) -> int:
    return sum(
        int(r.get("tool_invocation_count") or (1 if r.get("tool_called") else 0))
        for r in results
    )


def _compute_evolution_growth_level(results: list[dict[str, Any]]) -> float | None:
    """EGL = generated tools created per total tool invocation."""
    tool_invocations = _tool_invocations(results)
    if tool_invocations == 0:
        return None
    created = sum(int(r.get("generated_tools_created") or 0) for r in results)
    return round(created / tool_invocations, 4)


def _compute_generated_tool_use_rate(results: list[dict[str, Any]]) -> float | None:
    """Generated tools successfully used per total tool invocation."""
    tool_invocations = _tool_invocations(results)
    if tool_invocations == 0:
        return None
    generated_used = sum(
        1 for r in results
        if r.get("generated_tool_success")
    )
    return round(generated_used / tool_invocations, 4)


def _compute_in_situ_generation_success_rate(results: list[dict[str, Any]]) -> float | None:
    attempts = sum(1 for r in results if r.get("in_situ_generation_attempted"))
    if attempts == 0:
        return None
    successes = sum(1 for r in results if r.get("in_situ_generation_success"))
    return round(successes / attempts, 4)


def _compute_warm_start_reuse_rate(results: list[dict[str, Any]]) -> float | None:
    warm_tasks = [r for r in results if r.get("mode") == "warm-start"]
    if not warm_tasks:
        return None
    reused = sum(1 for r in warm_tasks if r.get("warm_start_reused_generated_tool"))
    return round(reused / len(warm_tasks), 4)


def _empty_metrics() -> dict[str, Any]:
    return {
        "total_tasks": 0,
        "task_success_rate": 0.0,
        "expected_tool_hit_rate": 0.0,
        "forbidden_tool_call_rate": 0.0,
        "tool_failure_rate": 0.0,
        "tool_request_rate": 0.0,
        "generated_tool_success_rate": None,
        "evolution_growth_level": None,
        "egl": None,
        "generated_tool_use_rate": None,
        "in_situ_generation_success_rate": None,
        "warm_start_reuse_rate": None,
        "avg_latency_ms": None,
    }
