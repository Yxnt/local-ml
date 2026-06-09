"""E2E smoke test: ToolRequest → Generate → Verify → Candidate → Promote → Execute → EGL.

Uses a FakeToolDeveloper (no real LLM) and tmp_path isolation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from server.tools.spec import (
    RiskLevel,
    ToolContext,
    ToolRequest,
    ToolRuntime,
    ToolSpec,
    ToolStatus,
)
from server.tools.telemetry import TelemetryService
from server.tools.registry import ToolRegistry
from server.tools.router import (
    GeneratedPythonExecutor,
    IntegrationToolExecutor,
    MemoryToolExecutor,
    ComputerUseToolExecutor,
    ToolRuntimeRouter,
)
from server.tools.verifier import ToolVerifier
from server.tools.orchestrator import ToolEvolutionOrchestrator
from server.tools.metrics import ToolMetrics


# ---------------------------------------------------------------------------
# Fake tool developer
# ---------------------------------------------------------------------------

_VALID_TOOL_SOURCE = '''\
from pydantic import BaseModel

class InputModel(BaseModel):
    text: str

class OutputModel(BaseModel):
    count: int

def run(input: InputModel) -> OutputModel:
    words = input.text.split()
    return OutputModel(count=len(words))
'''


class FakeToolDeveloper:
    """Returns a valid Python tool without calling any LLM."""

    async def generate(self, request: ToolRequest) -> dict[str, Any]:
        sandbox_dir = Path(tempfile.mkdtemp(prefix="fake_dev_"))
        file_path = sandbox_dir / f"{request.candidate_name}.py"
        file_path.write_text(_VALID_TOOL_SOURCE, encoding="utf-8")
        return {
            "success": True,
            "tool_name": request.candidate_name,
            "source_code": _VALID_TOOL_SOURCE,
            "file_path": str(file_path),
            "error": None,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry(tmp_path: Path) -> tuple[ToolRegistry, TelemetryService, ToolRuntimeRouter]:
    """Create isolated registry, telemetry, and router."""
    db_path = str(tmp_path / "test.db")

    telemetry = TelemetryService(db_path=db_path)
    telemetry.connect()

    registry = ToolRegistry(db_path=db_path, telemetry=telemetry)
    registry.connect()

    class _FakeMemoryManager:
        def remember(self, content, mtype, importance):
            return 1
        def recall(self, query, mtype, limit):
            return []
        class store:
            @staticmethod
            def get_stats():
                return {"total": 0}
        def get_tools(self):
            return []

    class _FakeModelRegistry:
        def get_backend(self, name):
            return None

    memory_executor = MemoryToolExecutor(_FakeMemoryManager())
    integration_executor = IntegrationToolExecutor(lambda name: None)
    computer_executor = ComputerUseToolExecutor(_FakeModelRegistry())

    router = ToolRuntimeRouter(
        memory_executor=memory_executor,
        integration_executor=integration_executor,
        computer_executor=computer_executor,
        generated_executor=GeneratedPythonExecutor(
            sandbox_dir=str(tmp_path / "sandbox"),
            execution_mode="subprocess",
        ),
        telemetry=telemetry,
    )
    registry._router = router

    return registry, telemetry, router


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolEvolutionSmoke:
    """Full E2E: request → generate → verify → candidate → promote → execute → EGL."""

    def test_full_pipeline(self, tmp_path):
        registry, telemetry, router = _make_registry(tmp_path)
        sandbox_dir = str(tmp_path / "sandbox")
        verifier = ToolVerifier(sandbox_dir=sandbox_dir)
        developer = FakeToolDeveloper()

        orchestrator = ToolEvolutionOrchestrator(
            registry=registry,
            telemetry=telemetry,
            developer=developer,
            verifier=verifier,
            absorber=None,
        )

        # 1. Write a ToolRequest into telemetry.
        request = ToolRequest(
            task_id="task_001",
            session_id="sess_001",
            reason="No word counting tool available",
            missing_capability="Count words in text",
            candidate_name="text_count_words",
            candidate_description="Count words in input text",
            candidate_input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            candidate_output_schema={
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            },
            risk_level=RiskLevel.L0,
        )
        telemetry.record_tool_request(request)

        # 2. Process pending requests (not dry-run).
        result = asyncio.run(orchestrator.process_pending_requests(limit=5, dry_run=False))
        assert result["processed"] == 1, f"Expected 1 processed, got {result}"
        assert result["errors"] == 0

        # 3. Verify tool is in registry as CANDIDATE.
        spec = registry.get_tool("text_count_words")
        assert spec is not None
        assert spec.status == ToolStatus.CANDIDATE
        assert spec.runtime == ToolRuntime.PYTHON_GENERATED

        # 4. Candidate tools are not normal dispatch targets.
        ctx = ToolContext(session_id="sess_001", task_id="task_001")
        candidate_result = asyncio.run(
            registry.dispatch("text_count_words", {"text": "hello world"}, ctx)
        )
        assert not candidate_result.success
        assert candidate_result.error_type == "tool_candidate"

        # 5. Internal validation can still execute the candidate and record telemetry.
        validation_result = asyncio.run(
            router.dispatch(spec, {"text": "hello world"}, ctx)
        )
        assert validation_result.success, f"Validation failed: {validation_result.content}"
        data = json.loads(validation_result.content)
        assert data["count"] == 2, f"Expected count=2, got {data}"

        # 6. Verify telemetry events.
        events = telemetry.get_recent_events(limit=50)
        event_types = {e["event_type"] for e in events}
        assert "tool_request" in event_types
        assert "tool_invoked" in event_types
        assert "tool_succeeded" in event_types

        # 7. Promote candidates (min_success_count=1, min_success_rate=1.0).
        promote_result = orchestrator.promote_candidates(
            min_success_count=1, min_success_rate=1.0
        )
        assert promote_result["promoted"] == 1, f"Expected 1 promoted, got {promote_result}"

        # 8. Verify tool is now ACTIVE and normal dispatch works.
        spec_after = registry.get_tool("text_count_words")
        assert spec_after is not None
        assert spec_after.status == ToolStatus.ACTIVE

        exec_result = asyncio.run(
            registry.dispatch("text_count_words", {"text": "hello world"}, ctx)
        )
        assert exec_result.success, f"Execution failed: {exec_result.content}"
        data = json.loads(exec_result.content)
        assert data["count"] == 2, f"Expected count=2, got {data}"

        # 9. Verify EGL > 0.
        metrics = ToolMetrics(registry=registry, telemetry=telemetry)
        egl = metrics.get_egl()
        assert egl is not None, "EGL should not be None"
        assert egl > 0, f"EGL should be > 0, got {egl}"

        # Cleanup.
        registry.disconnect()
        telemetry.disconnect()

    def test_dry_run_does_not_register(self, tmp_path):
        registry, telemetry, router = _make_registry(tmp_path)
        sandbox_dir = str(tmp_path / "sandbox")
        verifier = ToolVerifier(sandbox_dir=sandbox_dir)
        developer = FakeToolDeveloper()

        orchestrator = ToolEvolutionOrchestrator(
            registry=registry, telemetry=telemetry,
            developer=developer, verifier=verifier, absorber=None,
        )

        request = ToolRequest(
            candidate_name="dry_run_tool",
            candidate_description="test",
            candidate_input_schema={"type": "object", "properties": {}},
            risk_level=RiskLevel.L0,
        )
        telemetry.record_tool_request(request)

        result = asyncio.run(orchestrator.process_pending_requests(limit=5, dry_run=True))
        assert result["processed"] == 1

        # Tool should NOT be in registry.
        spec = registry.get_tool("dry_run_tool")
        assert spec is None

        registry.disconnect()
        telemetry.disconnect()

    def test_l2_request_skipped(self, tmp_path):
        registry, telemetry, router = _make_registry(tmp_path)
        sandbox_dir = str(tmp_path / "sandbox")
        verifier = ToolVerifier(sandbox_dir=sandbox_dir)
        developer = FakeToolDeveloper()

        orchestrator = ToolEvolutionOrchestrator(
            registry=registry, telemetry=telemetry,
            developer=developer, verifier=verifier, absorber=None,
        )

        request = ToolRequest(
            candidate_name="risky_tool",
            candidate_description="test",
            candidate_input_schema={"type": "object", "properties": {}},
            risk_level=RiskLevel.L3,
        )
        telemetry.record_tool_request(request)

        result = asyncio.run(orchestrator.process_pending_requests(limit=5, dry_run=False))
        assert result["skipped"] >= 1
        assert result["processed"] == 0

        registry.disconnect()
        telemetry.disconnect()


import asyncio
