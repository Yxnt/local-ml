"""Report generation for local_ml_eval."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_json_report(metrics: dict[str, Any], results: list[dict[str, Any]], output_path: str) -> None:
    """Write a JSON report."""
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "task_count": len(results),
        "results": results,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown_summary(metrics: dict[str, Any], output_path: str) -> None:
    """Write a Markdown summary."""
    lines = [
        "# local_ml_eval Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Task success rate | {metrics.get('task_success_rate', 'N/A')} |",
        f"| Expected tool hit rate | {metrics.get('expected_tool_hit_rate', 'N/A')} |",
        f"| Forbidden tool call rate | {metrics.get('forbidden_tool_call_rate', 'N/A')} |",
        f"| Tool failure rate | {metrics.get('tool_failure_rate', 'N/A')} |",
        f"| Tool request rate | {metrics.get('tool_request_rate', 'N/A')} |",
        f"| Generated tool success rate | {metrics.get('generated_tool_success_rate', 'N/A')} |",
        f"| Evolution growth level (EGL) | {metrics.get('evolution_growth_level', 'N/A')} |",
        f"| Generated tool use rate | {metrics.get('generated_tool_use_rate', 'N/A')} |",
        f"| In-situ generation success rate | {metrics.get('in_situ_generation_success_rate', 'N/A')} |",
        f"| Warm-start reuse rate | {metrics.get('warm_start_reuse_rate', 'N/A')} |",
        f"| Avg latency (ms) | {metrics.get('avg_latency_ms', 'N/A')} |",
        "",
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
