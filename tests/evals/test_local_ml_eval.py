"""Tests for local_ml_eval framework."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from backends.base import ModelBackend


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

    def test_dry_run_metrics_are_placeholders(self):
        """Dry-run metrics should not look like real execution metrics."""
        from evals.local_ml_eval.runner import load_tasks, run_dry

        tasks = load_tasks(TASKS_PATH)
        result = run_dry(tasks)
        metrics = result["metrics"]

        assert metrics["total_tasks"] == len(result["results"])
        assert metrics["task_success_rate"] is None
        assert metrics["expected_tool_hit_rate"] is None
        assert metrics["tool_request_rate"] is None
        assert metrics["avg_latency_ms"] is None

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


class ScriptedEvalBackend(ModelBackend):
    """Deterministic backend that emits tool calls based on the latest user input."""

    def __init__(self) -> None:
        self._messages: list[dict] = []

    def load(self, model_id: str) -> None:
        pass

    def unload(self) -> None:
        pass

    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, top_p: float = 0.9) -> str:
        if self._messages and self._messages[-1].get("role") == "tool":
            return "done"

        user_input = next(
            (m.get("content", "") for m in reversed(self._messages) if m.get("role") == "user"),
            "",
        )
        mapping = {
            "记住我喜欢简洁回答": ("memory_remember", {"content": "我喜欢简洁回答", "type": "fact", "importance": 0.8}),
            "搜索Obsidian笔记中关于机器学习的内容": ("obsidian_search", {"query": "机器学习", "limit": 3}),
            "解析这个ICS日历文件": ("parse_ics_calendar", {"text": "BEGIN:VCALENDAR"}),
            "统计这段文本的单词数": ("text_count_words", {"text": "one two three four"}),
        }
        for needle, (name, arguments) in mapping.items():
            if needle in user_input:
                return json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
        return "No tool needed"

    def apply_chat_template(self, messages, tools=None, enable_thinking=False) -> str:
        self._messages = list(messages)
        return "prompt"

    def parse_tool_calls(self, text: str) -> list[dict]:
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return []

        if "name" not in obj:
            return []

        return [{
            "id": "call_0",
            "type": "function",
            "function": {
                "name": obj["name"],
                "arguments": json.dumps(obj.get("arguments", {}), ensure_ascii=False),
            },
        }]

    def warmup(self) -> None:
        pass


def _make_eval_agent(tmp_path: Path):
    """Create an Agent wired to the deterministic eval backend."""
    with patch("server.agent.ModelRegistry") as MockRegistry, patch("server.agent.EmbeddingClient") as MockEmbedding:
        mock_registry = MockRegistry.return_value
        backend = ScriptedEvalBackend()
        mock_registry.register_defaults.return_value = None
        mock_registry.get_backend.return_value = backend
        mock_registry.get_or_load = AsyncMock(return_value=backend)
        mock_registry.list_models.return_value = []

        mock_embedding = MockEmbedding.return_value
        mock_embedding.close = AsyncMock()

        from server.agent import Agent

        agent = Agent(
            data_dir=str(tmp_path / "eval_data"),
            default_model="test",
            use_tool_registry=True,
            tool_retrieval_mode="all",
        )
        agent._registry = mock_registry
        agent._embedding = mock_embedding
        return agent


class TestLiveRun:
    def test_run_live_executes_agent_and_collects_results(self, tmp_path):
        """Live runner should execute tasks through Agent.run and inspect telemetry."""
        from evals.local_ml_eval.runner import load_tasks, run_live

        tasks = [t for t in load_tasks(TASKS_PATH) if t["id"] in {
            "memory_001",
            "retrieval_001",
            "missing_001",
            "generated_001",
        }]

        result = run_live(
            tasks,
            agent_factory=lambda: _make_eval_agent(tmp_path),
            model="test",
        )

        by_id = {row["id"]: row for row in result["results"]}

        assert result["valid"] is True
        assert len(by_id) == 4
        assert by_id["memory_001"]["task_success"] is True
        assert by_id["retrieval_001"]["expected_tool_hit"] is True
        assert by_id["missing_001"]["tool_requested"] is True
        assert by_id["generated_001"]["generated_tool_success"] is True
        assert result["metrics"]["generated_tool_success_rate"] == 1.0
        assert result["metrics"]["tool_request_rate"] == 0.25
        assert result["metrics"]["avg_latency_ms"] is not None
