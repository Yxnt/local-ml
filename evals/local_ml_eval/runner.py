"""Runner for local_ml_eval.

Usage:
    python -m evals.local_ml_eval.runner --tasks evals/local_ml_eval/tasks.jsonl --output /tmp/report.json --dry-run
"""

from __future__ import annotations

import asyncio
import argparse
import inspect
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from evals.local_ml_eval.fixtures import (
    OfflineToolDeveloper,
    install_fixture_integrations,
    install_generated_tool_fixtures,
)
from evals.local_ml_eval.metrics import compute_metrics, placeholder_metrics
from evals.local_ml_eval.report import write_json_report, write_markdown_summary


def load_tasks(path: str) -> list[dict[str, Any]]:
    """Load tasks from a JSONL file."""
    tasks = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                task = json.loads(line)
                task["_line"] = i
                tasks.append(task)
            except json.JSONDecodeError as e:
                print(f"WARNING: skipping line {i}: {e}", file=sys.stderr)
    return tasks


def validate_task(task: dict[str, Any]) -> list[str]:
    """Validate a single task. Returns list of errors (empty = valid)."""
    errors = []
    for field in ("id", "category", "input", "expected_tools", "success_check"):
        if field not in task:
            errors.append(f"missing required field: {field}")
    if "expected_tools" in task and not isinstance(task["expected_tools"], list):
        errors.append("expected_tools must be a list")
    if "forbidden_tools" in task and not isinstance(task["forbidden_tools"], list):
        errors.append("forbidden_tools must be a list")
    if task.get("mode") is not None and task["mode"] not in {"zero-start", "warm-start"}:
        errors.append("mode must be one of: warm-start, zero-start")
    return errors


def run_dry(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Dry-run: validate tasks and produce a skeleton report."""
    results = []
    all_valid = True

    for task in tasks:
        errors = validate_task(task)
        if errors:
            all_valid = False
            print(f"  INVALID {task.get('id', '?')}: {'; '.join(errors)}", file=sys.stderr)
            continue

        results.append({
            "id": task["id"],
            "category": task.get("category", "unknown"),
            "mode": task.get("mode"),
            "task_success": None,
            "expected_tool_hit": None,
            "forbidden_tool_called": None,
            "tool_failed": None,
            "tool_requested": None,
            "generated_tool_success": None,
            "tool_called": None,
            "tool_invocation_count": None,
            "generated_tools_created": None,
            "in_situ_generation_attempted": None,
            "in_situ_generation_success": None,
            "warm_start_reused_generated_tool": None,
            "latency_ms": None,
            "note": "dry-run — not executed",
        })

    metrics = placeholder_metrics(len(results))
    return {"metrics": metrics, "results": results, "valid": all_valid, "total_tasks": len(tasks)}


def run_live(
    tasks: list[dict[str, Any]],
    *,
    agent_factory: Callable[[], Any] | None = None,
    model: str = "gemma-4-e2b-it-4bit",
    data_dir: str | None = None,
    integration_fixtures: str = "offline",
    generated_tool_fixtures: bool = True,
    evolution_mode: str = "fixture",
) -> dict[str, Any]:
    """Execute eval tasks through the real Agent.run path."""
    return asyncio.run(
        _run_live_async(
            tasks,
            agent_factory=agent_factory,
            model=model,
            data_dir=data_dir,
            integration_fixtures=integration_fixtures,
            generated_tool_fixtures=generated_tool_fixtures,
            evolution_mode=evolution_mode,
        )
    )


async def _run_live_async(
    tasks: list[dict[str, Any]],
    *,
    agent_factory: Callable[[], Any] | None,
    model: str,
    data_dir: str | None,
    integration_fixtures: str,
    generated_tool_fixtures: bool,
    evolution_mode: str,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    all_valid = True
    selected_tasks = _filter_tasks_by_evolution_mode(tasks, evolution_mode)

    agent = agent_factory() if agent_factory is not None else _build_live_agent(
        data_dir=data_dir,
        model=model,
        evolution_mode=evolution_mode,
    )
    try:
        _prepare_generated_executor(agent, data_dir)
        await install_fixture_integrations(agent, mode=integration_fixtures)
        if evolution_mode == "warm-start":
            install_generated_tool_fixtures(agent)
        elif evolution_mode == "fixture" and generated_tool_fixtures:
            install_generated_tool_fixtures(agent)

        for task in selected_tasks:
            errors = validate_task(task)
            if errors:
                all_valid = False
                results.append(_build_invalid_result(task, errors))
                continue

            results.append(await _execute_task(agent, task, model=model))
    finally:
        await agent.close()

    metrics = compute_metrics(results)
    return {
        "metrics": metrics,
        "results": results,
        "valid": all_valid,
        "total_tasks": len(selected_tasks),
    }


def _filter_tasks_by_evolution_mode(
    tasks: list[dict[str, Any]],
    evolution_mode: str,
) -> list[dict[str, Any]]:
    if evolution_mode in {"zero-start", "warm-start"}:
        return [task for task in tasks if task.get("mode") == evolution_mode]
    return tasks


def _build_live_agent(*, data_dir: str | None, model: str, evolution_mode: str = "fixture"):
    from server.agent import Agent

    resolved_data_dir = data_dir or tempfile.mkdtemp(prefix="local_ml_eval_")
    kwargs: dict[str, Any] = {
        "data_dir": resolved_data_dir,
        "default_model": model,
        "use_tool_registry": True,
        "tool_retrieval_mode": "all",
    }

    if evolution_mode == "zero-start":
        supported = set(inspect.signature(Agent).parameters)
        sandbox_dir = str(Path(resolved_data_dir) / "generated_tools")
        in_situ_kwargs: dict[str, Any] = {
            "enable_in_situ_tool_generation": True,
            "tool_sandbox_dir": sandbox_dir,
            "tool_developer": OfflineToolDeveloper(),
        }
        if {"tool_verifier", "tool_risk_classifier"} & supported:
            try:
                from server.tools.risk_classifier import CapabilityRiskClassifier
                from server.tools.verifier import ToolVerifier

                in_situ_kwargs.update(
                    {
                        "tool_verifier": ToolVerifier(sandbox_dir=sandbox_dir),
                        "tool_risk_classifier": CapabilityRiskClassifier(),
                    }
                )
            except ImportError:
                pass
        kwargs.update({key: value for key, value in in_situ_kwargs.items() if key in supported})

    return Agent(**kwargs)


def _prepare_generated_executor(agent: Any, data_dir: str | None) -> None:
    router = getattr(agent, "_tool_router", None)
    if router is None:
        return

    from server.tools.spec import ToolRuntime

    generated_executor = router._executors.get(ToolRuntime.PYTHON_GENERATED)
    if generated_executor is None:
        return

    base_dir = Path(data_dir or getattr(agent, "_data_dir", tempfile.mkdtemp(prefix="local_ml_eval_")))
    sandbox_dir = base_dir / "generated_tools"
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    generated_executor._sandbox_dir = str(sandbox_dir)


async def _execute_task(agent: Any, task: dict[str, Any], *, model: str) -> dict[str, Any]:
    telemetry = getattr(agent, "_tool_telemetry", None)
    if telemetry is None or telemetry._conn is None:
        raise RuntimeError("Agent telemetry is not available")

    agent.clear_context()
    before_event_id = _last_row_id(telemetry._conn, "tool_events")
    before_request_id = _last_row_id(telemetry._conn, "tool_requests")
    start = time.monotonic()
    response = await agent.run(task["input"], model=model)
    latency_ms = round((time.monotonic() - start) * 1000, 2)

    events = _rows_since(telemetry._conn, "tool_events", before_event_id)
    requests = _rows_since(telemetry._conn, "tool_requests", before_request_id)

    invoked_tools = [
        row["tool_name"]
        for row in events
        if row["event_type"] == "tool_invoked" and row.get("tool_name")
    ]
    failed_tools = [
        row["tool_name"]
        for row in events
        if row["event_type"] == "tool_failed" and row.get("tool_name")
    ]
    requested_tools = [row["candidate_name"] for row in requests if row.get("candidate_name")]

    expected_tools = task.get("expected_tools", [])
    forbidden_tools = task.get("forbidden_tools", [])
    expected_tool_hit = bool(expected_tools) and all(name in invoked_tools for name in expected_tools)
    forbidden_tool_called = any(name in invoked_tools for name in forbidden_tools)
    tool_failed = bool(failed_tools)
    tool_requested = bool(requested_tools)
    tool_called = bool(invoked_tools)
    generated_tool_success = (
        task.get("category") == "generated_tool"
        and expected_tool_hit
        and not tool_failed
    )
    tool_invocation_count = len(invoked_tools)
    created_events = [
        row for row in events
        if row.get("event_type") == "tool_created"
    ]
    generated_tools_created = sum(
        1 for row in created_events
        if _metadata(row).get("source") in {"in_situ_local", "absorber"}
    )
    in_situ_generation_attempted = any(
        row.get("event_type") == "tool_generation_attempted"
        and _metadata(row).get("source") == "in_situ_local"
        for row in events
    )
    in_situ_generation_success = any(
        _metadata(row).get("source") == "in_situ_local"
        for row in created_events
    )
    warm_start_reused_generated_tool = (
        task.get("mode") == "warm-start"
        and generated_tool_success
        and generated_tools_created == 0
    )

    return {
        "id": task["id"],
        "category": task.get("category", "unknown"),
        "mode": task.get("mode"),
        "task_success": _is_task_success(
            task,
            expected_tool_hit=expected_tool_hit,
            forbidden_tool_called=forbidden_tool_called,
            tool_failed=tool_failed,
            tool_requested=tool_requested,
        ),
        "expected_tool_hit": expected_tool_hit,
        "forbidden_tool_called": forbidden_tool_called,
        "tool_failed": tool_failed,
        "tool_requested": tool_requested,
        "generated_tool_success": generated_tool_success,
        "tool_called": tool_called,
        "tool_invocation_count": tool_invocation_count,
        "generated_tools_created": generated_tools_created,
        "in_situ_generation_attempted": in_situ_generation_attempted,
        "in_situ_generation_success": in_situ_generation_success,
        "warm_start_reused_generated_tool": warm_start_reused_generated_tool,
        "latency_ms": latency_ms,
        "response": response,
        "invoked_tools": invoked_tools,
        "requested_tools": requested_tools,
        "note": "executed",
    }


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata") or "{}"
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _is_task_success(
    task: dict[str, Any],
    *,
    expected_tool_hit: bool,
    forbidden_tool_called: bool,
    tool_failed: bool,
    tool_requested: bool,
) -> bool:
    if forbidden_tool_called:
        return False
    if task.get("success_check") == "tool_request_recorded":
        return tool_requested
    return expected_tool_hit and not tool_failed


def _build_invalid_result(task: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    return {
        "id": task.get("id", "unknown"),
        "category": task.get("category", "unknown"),
        "mode": task.get("mode"),
        "task_success": False,
        "expected_tool_hit": False,
        "forbidden_tool_called": False,
        "tool_failed": False,
        "tool_requested": False,
        "generated_tool_success": False,
        "tool_called": False,
        "tool_invocation_count": 0,
        "generated_tools_created": 0,
        "in_situ_generation_attempted": False,
        "in_situ_generation_success": False,
        "warm_start_reused_generated_tool": False,
        "latency_ms": None,
        "note": f"invalid task: {'; '.join(errors)}",
    }


def _last_row_id(conn: Any, table: str) -> int:
    row = conn.execute(f"SELECT COALESCE(MAX(id), 0) FROM {table}").fetchone()
    return int(row[0]) if row else 0


def _rows_since(conn: Any, table: str, row_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(f"SELECT * FROM {table} WHERE id > ? ORDER BY id", (row_id,)).fetchall()
    return [dict(row) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="local_ml_eval runner")
    parser.add_argument("--tasks", required=True, help="Path to tasks.jsonl")
    parser.add_argument("--output", default="/tmp/local_ml_eval_report.json", help="Output JSON path")
    parser.add_argument("--markdown", default=None, help="Output Markdown summary path")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, don't execute")
    parser.add_argument("--model", default="gemma-4-e2b-it-4bit", help="Model name for live execution.")
    parser.add_argument("--data-dir", default=None, help="Data directory for live execution.")
    parser.add_argument(
        "--integration-fixtures",
        choices=("offline", "none"),
        default="offline",
        help="Fixture integrations to connect during live execution.",
    )
    parser.add_argument(
        "--no-generated-tool-fixtures",
        action="store_true",
        help="Do not pre-register generated tool fixtures before live execution.",
    )
    parser.add_argument(
        "--evolution-mode",
        choices=("fixture", "zero-start", "warm-start"),
        default="fixture",
        help="Select fixture smoke mode or self-evolution zero/warm-start mode.",
    )
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    tasks = _filter_tasks_by_evolution_mode(tasks, args.evolution_mode)
    print(f"Loaded {len(tasks)} tasks from {args.tasks}")

    if args.dry_run:
        print("Running in dry-run mode (validation only)...")
        result = run_dry(tasks)
        metrics = result["metrics"]
        results = result["results"]

        if result["valid"]:
            print(f"All {result['total_tasks']} tasks are valid.")
        else:
            print(f"Some tasks have validation errors (see stderr).")

        # Write skeleton report.
        write_json_report(metrics, results, args.output)
        print(f"Report written to {args.output}")

        if args.markdown:
            write_markdown_summary(metrics, args.markdown)
            print(f"Markdown summary written to {args.markdown}")

        # Print metrics summary.
        print("\nMetrics (skeleton):")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        print("Running live execution mode...")
        result = run_live(
            tasks,
            model=args.model,
            data_dir=args.data_dir,
            integration_fixtures=args.integration_fixtures,
            generated_tool_fixtures=not args.no_generated_tool_fixtures,
            evolution_mode=args.evolution_mode,
        )
        metrics = result["metrics"]
        results = result["results"]

        write_json_report(metrics, results, args.output)
        print(f"Report written to {args.output}")

        if args.markdown:
            write_markdown_summary(metrics, args.markdown)
            print(f"Markdown summary written to {args.markdown}")

        print("\nMetrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        if not result["valid"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
