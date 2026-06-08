"""Tests for local_ml_eval framework."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


TASKS_PATH = "evals/local_ml_eval/tasks.jsonl"


class TestTasksJsonl:
    def test_tasks_file_is_valid_jsonl(self):
        """Every line in tasks.jsonl should be valid JSON."""
        with open(TASKS_PATH, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                task = json.loads(line)  # will raise on invalid JSON
                assert "id" in task, f"Line {i}: missing 'id'"
                assert "category" in task, f"Line {i}: missing 'category'"
                assert "input" in task, f"Line {i}: missing 'input'"

    def test_tasks_have_required_fields(self):
        """All tasks should have the required fields."""
        from evals.local_ml_eval.runner import load_tasks, validate_task

        tasks = load_tasks(TASKS_PATH)
        assert len(tasks) >= 20, f"Expected >= 20 tasks, got {len(tasks)}"

        for task in tasks:
            errors = validate_task(task)
            assert errors == [], f"Task {task.get('id')}: {errors}"

    def test_tasks_cover_all_categories(self):
        """Should have tasks for all four categories."""
        from evals.local_ml_eval.runner import load_tasks

        tasks = load_tasks(TASKS_PATH)
        categories = {t["category"] for t in tasks}
        assert "memory" in categories
        assert "tool_retrieval" in categories
        assert "missing_tool" in categories
        assert "generated_tool" in categories


class TestDryRun:
    def test_dry_run_produces_report(self, tmp_path):
        """Dry-run should produce a valid JSON report."""
        from evals.local_ml_eval.runner import load_tasks, run_dry
        from evals.local_ml_eval.report import write_json_report

        tasks = load_tasks(TASKS_PATH)
        result = run_dry(tasks)

        assert result["valid"] is True
        assert result["total_tasks"] >= 20
        assert len(result["results"]) >= 20

        output = str(tmp_path / "report.json")
        write_json_report(result["metrics"], result["results"], output)

        report = json.loads(Path(output).read_text())
        assert "generated_at" in report
        assert "metrics" in report
        assert "results" in report

    def test_dry_run_metrics_have_all_fields(self, tmp_path):
        """Report metrics should contain all required fields."""
        from evals.local_ml_eval.runner import load_tasks, run_dry

        tasks = load_tasks(TASKS_PATH)
        result = run_dry(tasks)
        metrics = result["metrics"]

        required_fields = [
            "total_tasks",
            "task_success_rate",
            "expected_tool_hit_rate",
            "forbidden_tool_call_rate",
            "tool_failure_rate",
            "tool_request_rate",
            "generated_tool_success_rate",
            "egl",
            "avg_latency_ms",
        ]
        for field in required_fields:
            assert field in metrics, f"Missing metric: {field}"

    def test_dry_run_results_have_all_fields(self):
        """Each result should have all required fields."""
        from evals.local_ml_eval.runner import load_tasks, run_dry

        tasks = load_tasks(TASKS_PATH)
        result = run_dry(tasks)

        for r in result["results"]:
            assert "id" in r
            assert "category" in r
            assert "note" in r


class TestMetrics:
    def test_compute_metrics_empty(self):
        """Empty results should produce zero/None metrics."""
        from evals.local_ml_eval.metrics import compute_metrics

        m = compute_metrics([])
        assert m["total_tasks"] == 0
        assert m["task_success_rate"] == 0.0
        assert m["egl"] is None

    def test_compute_metrics_with_data(self):
        """Metrics should compute correctly from result data."""
        from evals.local_ml_eval.metrics import compute_metrics

        results = [
            {"task_success": True, "expected_tool_hit": True, "forbidden_tool_called": False,
             "tool_failed": False, "tool_requested": False, "generated_tool_success": False,
             "tool_called": True, "category": "memory", "latency_ms": 100},
            {"task_success": False, "expected_tool_hit": False, "forbidden_tool_called": True,
             "tool_failed": True, "tool_requested": True, "generated_tool_success": False,
             "tool_called": False, "category": "missing_tool", "latency_ms": 200},
            {"task_success": True, "expected_tool_hit": True, "forbidden_tool_called": False,
             "tool_failed": False, "tool_requested": False, "generated_tool_success": True,
             "tool_called": True, "category": "generated_tool", "latency_ms": 150},
        ]

        m = compute_metrics(results)
        assert m["total_tasks"] == 3
        assert m["task_success_rate"] == round(2 / 3, 4)
        assert m["expected_tool_hit_rate"] == round(2 / 3, 4)
        assert m["forbidden_tool_call_rate"] == round(1 / 3, 4)
        assert m["tool_failure_rate"] == round(1 / 3, 4)
        assert m["tool_request_rate"] == round(1 / 3, 4)
        assert m["generated_tool_success_rate"] == 1.0  # 1/1
        assert m["avg_latency_ms"] == 150.0


class TestReport:
    def test_json_report_contains_all_fields(self, tmp_path):
        """JSON report should contain all required top-level fields."""
        from evals.local_ml_eval.report import write_json_report

        metrics = {
            "total_tasks": 20,
            "task_success_rate": 0.85,
            "expected_tool_hit_rate": 0.9,
            "forbidden_tool_call_rate": 0.0,
            "tool_failure_rate": 0.05,
            "tool_request_rate": 0.15,
            "generated_tool_success_rate": 0.8,
            "egl": 0.1,
            "avg_latency_ms": 150.0,
        }
        output = str(tmp_path / "report.json")
        write_json_report(metrics, [], output)

        report = json.loads(Path(output).read_text())
        assert "generated_at" in report
        assert report["metrics"]["egl"] == 0.1
        assert report["task_count"] == 0

    def test_markdown_report_is_valid(self, tmp_path):
        """Markdown report should be readable."""
        from evals.local_ml_eval.report import write_markdown_summary

        metrics = {"task_success_rate": 0.85, "egl": 0.1}
        output = str(tmp_path / "report.md")
        write_markdown_summary(metrics, output)

        content = Path(output).read_text()
        assert "# local_ml_eval Report" in content
        assert "0.85" in content
