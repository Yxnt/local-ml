"""Runner for local_ml_eval.

Usage:
    python -m evals.local_ml_eval.runner --tasks evals/local_ml_eval/tasks.jsonl --output /tmp/report.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evals.local_ml_eval.metrics import compute_metrics
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
            "task_success": None,
            "expected_tool_hit": None,
            "forbidden_tool_called": None,
            "tool_failed": None,
            "tool_requested": None,
            "generated_tool_success": None,
            "tool_called": None,
            "latency_ms": None,
            "note": "dry-run — not executed",
        })

    metrics = compute_metrics(results)
    return {"metrics": metrics, "results": results, "valid": all_valid, "total_tasks": len(tasks)}


def main() -> None:
    parser = argparse.ArgumentParser(description="local_ml_eval runner")
    parser.add_argument("--tasks", required=True, help="Path to tasks.jsonl")
    parser.add_argument("--output", default="/tmp/local_ml_eval_report.json", help="Output JSON path")
    parser.add_argument("--markdown", default=None, help="Output Markdown summary path")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, don't execute")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
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
        print("Full execution not yet implemented. Use --dry-run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
