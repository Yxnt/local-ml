"""Tests for local_ml_eval framework."""

from __future__ import annotations

import asyncio
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

    def test_tasks_include_self_evolution_modes(self):
        """Self-evolution tasks should distinguish zero-start from warm-start."""
        from evals.local_ml_eval.runner import load_tasks

        tasks = load_tasks(TASKS_PATH)
        modes = {task.get("mode") for task in tasks if task.get("mode")}

        assert "zero-start" in modes
        assert "warm-start" in modes

    def test_task_modes_are_known_when_present(self):
        """Task mode should only use supported self-evolution values."""
        from evals.local_ml_eval.runner import load_tasks, validate_task

        tasks = load_tasks(TASKS_PATH)

        for task in tasks:
            errors = validate_task(task)
            assert "mode must be one of: warm-start, zero-start" not in errors


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
            "evolution_growth_level",
            "egl",
            "generated_tool_use_rate",
            "in_situ_generation_success_rate",
            "warm_start_reuse_rate",
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
        assert m["evolution_growth_level"] is None
        assert m["generated_tool_use_rate"] is None

    def test_compute_metrics_with_data(self):
        """Metrics should compute correctly from result data."""
        from evals.local_ml_eval.metrics import compute_metrics

        results = [
            {"task_success": True, "expected_tool_hit": True, "forbidden_tool_called": False,
             "tool_failed": False, "tool_requested": False, "generated_tool_success": False,
             "tool_called": True, "tool_invocation_count": 1, "category": "memory", "latency_ms": 100},
            {"task_success": False, "expected_tool_hit": False, "forbidden_tool_called": True,
             "tool_failed": True, "tool_requested": True, "generated_tool_success": False,
             "tool_called": False, "tool_invocation_count": 0, "category": "missing_tool", "latency_ms": 200},
            {"task_success": True, "expected_tool_hit": True, "forbidden_tool_called": False,
             "tool_failed": False, "tool_requested": False, "generated_tool_success": True,
             "tool_called": True, "tool_invocation_count": 1, "generated_tools_created": 0,
             "category": "generated_tool", "latency_ms": 150},
        ]

        m = compute_metrics(results)
        assert m["total_tasks"] == 3
        assert m["task_success_rate"] == round(2 / 3, 4)
        assert m["expected_tool_hit_rate"] == round(2 / 3, 4)
        assert m["forbidden_tool_call_rate"] == round(1 / 3, 4)
        assert m["tool_failure_rate"] == round(1 / 3, 4)
        assert m["tool_request_rate"] == round(1 / 3, 4)
        assert m["generated_tool_success_rate"] == 1.0  # 1/1
        assert m["evolution_growth_level"] == 0.0
        assert m["egl"] == 0.0
        assert m["generated_tool_use_rate"] == 0.5
        assert m["avg_latency_ms"] == 150.0

    def test_compute_metrics_with_self_evolution_fields(self):
        """Creation pressure and generated-tool use should be distinct signals."""
        from evals.local_ml_eval.metrics import compute_metrics

        results = [
            {
                "task_success": True,
                "expected_tool_hit": True,
                "forbidden_tool_called": False,
                "tool_failed": False,
                "tool_requested": True,
                "tool_called": True,
                "tool_invocation_count": 1,
                "generated_tools_created": 1,
                "generated_tool_success": True,
                "in_situ_generation_attempted": True,
                "in_situ_generation_success": True,
                "mode": "zero-start",
                "category": "generated_tool",
                "latency_ms": 100,
            },
            {
                "task_success": True,
                "expected_tool_hit": True,
                "forbidden_tool_called": False,
                "tool_failed": False,
                "tool_requested": False,
                "tool_called": True,
                "tool_invocation_count": 1,
                "generated_tools_created": 0,
                "generated_tool_success": True,
                "in_situ_generation_attempted": False,
                "in_situ_generation_success": False,
                "mode": "warm-start",
                "warm_start_reused_generated_tool": True,
                "category": "generated_tool",
                "latency_ms": 50,
            },
        ]

        m = compute_metrics(results)

        assert m["evolution_growth_level"] == 0.5
        assert m["egl"] == 0.5
        assert m["generated_tool_use_rate"] == 1.0
        assert m["in_situ_generation_success_rate"] == 1.0
        assert m["warm_start_reuse_rate"] == 1.0
        assert m["tool_request_rate"] == 0.5


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
            "evolution_growth_level": 0.1,
            "egl": 0.1,
            "generated_tool_use_rate": 0.7,
            "in_situ_generation_success_rate": 0.5,
            "warm_start_reuse_rate": 0.9,
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
        assert "Evolution growth level (EGL)" in content
        assert "Generated tool use rate" in content
        assert "In-situ generation success rate" in content
        assert "Warm-start reuse rate" in content


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
    def test_zero_start_mode_does_not_preinstall_generated_fixtures(self, tmp_path):
        """Zero-start should rely on in-situ generation rather than fixture install."""
        from evals.local_ml_eval.runner import _build_live_agent

        agent = _build_live_agent(data_dir=str(tmp_path), model="test", evolution_mode="zero-start")
        try:
            assert agent._tool_registry.get_tool("text_reverse") is None
            assert agent._tool_retrieval_mode == "all"
        finally:
            asyncio.run(agent.close())

    def test_warm_start_mode_uses_existing_generated_fixtures(self, tmp_path):
        """Warm-start should make generated tools available for reuse."""
        from evals.local_ml_eval.fixtures import install_generated_tool_fixtures
        from evals.local_ml_eval.runner import _build_live_agent, _prepare_generated_executor

        agent = _build_live_agent(data_dir=str(tmp_path), model="test", evolution_mode="warm-start")
        try:
            _prepare_generated_executor(agent, str(tmp_path))
            install_generated_tool_fixtures(agent)
            assert agent._tool_registry.get_tool("text_reverse") is not None
        finally:
            asyncio.run(agent.close())

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
