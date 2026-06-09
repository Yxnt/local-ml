"""End-to-end test: ToolRequest -> Fake Developer -> Verifier -> Registry -> Executor.

Covers the full lifecycle:
  1. Construct a ToolRequest for 'text_count_words' (L0 risk)
  2. Fake ToolDeveloper that returns valid Python source code
  3. Orchestrator.process_pending_requests(dry_run=False)
  4. Verify the tool is registered as CANDIDATE
  5. Verify CANDIDATE tools are not callable through normal registry dispatch
  6. Validate the candidate through the runtime router and record telemetry
  7. promote_candidates -> verify status changes to ACTIVE
  8. Execute promoted ACTIVE tool via registry.dispatch()
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from typing import Any

import pytest

from server.tools.spec import RiskLevel, ToolStatus
from server.tools.registry import ToolRegistry
from server.tools.router import (
    GeneratedPythonExecutor,
    ToolRuntimeRouter,
    MemoryToolExecutor,
    IntegrationToolExecutor,
    ComputerUseToolExecutor,
)
from server.tools.telemetry import TelemetryService
from server.tools.orchestrator import ToolEvolutionOrchestrator


# ---------------------------------------------------------------------------
# Fake Developer
# ---------------------------------------------------------------------------

WORD_COUNT_SOURCE = textwrap.dedent('''\
    """Count the number of words in a text string."""

    from pydantic import BaseModel


    class InputModel(BaseModel):
        text: str


    class OutputModel(BaseModel):
        word_count: int


    def run(input: InputModel) -> OutputModel:
        words = input.text.split()
        return OutputModel(word_count=len(words))
''')


class FakeDeveloper:
    """Returns valid Python source code for tool generation."""

    def __init__(self, source_code: str, sandbox_dir: str) -> None:
        self._source_code = source_code
        self._sandbox_dir = sandbox_dir

    async def generate(self, tool_request) -> dict[str, Any]:
        file_path = f"{self._sandbox_dir}/{tool_request.candidate_name}.py"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self._source_code)
        return {
            "success": True,
            "source_code": self._source_code,
            "file_path": file_path,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def telemetry(tmp_path):
    svc = TelemetryService(db_path=str(tmp_path / "telemetry.db"))
    svc.connect()
    yield svc
    svc.disconnect()


@pytest.fixture()
def registry_with_router(tmp_path, telemetry):
    db_path = str(tmp_path / "registry.db")
    sandbox_dir = str(tmp_path / "sandbox")

    generated_executor = GeneratedPythonExecutor(sandbox_dir=sandbox_dir)
    router = ToolRuntimeRouter(
        memory_executor=MemoryToolExecutor(memory_manager=None),
        integration_executor=IntegrationToolExecutor(get_integration=lambda _: None),
        computer_executor=ComputerUseToolExecutor(model_registry=None),
        generated_executor=generated_executor,
        telemetry=telemetry,
    )
    reg = ToolRegistry(db_path=db_path, router=router, telemetry=telemetry)
    reg.connect()
    yield reg
    reg.disconnect()


@pytest.fixture()
def sandbox_dir(tmp_path):
    d = tmp_path / "sandbox"
    d.mkdir()
    return str(d)


@pytest.fixture()
def orchestrator(registry_with_router, telemetry, sandbox_dir):
    from server.tools.verifier import ToolVerifier

    verifier = ToolVerifier(sandbox_dir=sandbox_dir)
    developer = FakeDeveloper(source_code=WORD_COUNT_SOURCE, sandbox_dir=sandbox_dir)
    return ToolEvolutionOrchestrator(
        registry=registry_with_router,
        telemetry=telemetry,
        developer=developer,
        verifier=verifier,
        absorber=None,
        retriever=None,
    )


def _seed_tool_request(telemetry: TelemetryService) -> None:
    """Insert a tool_request row into the telemetry DB for the orchestrator to consume."""
    conn = telemetry._conn
    assert conn is not None
    now = "2025-01-01T00:00:00+00:00"
    conn.execute(
        """
        INSERT INTO tool_requests
            (task_id, session_id, reason, missing_capability,
             candidate_name, candidate_desc, candidate_input, candidate_output,
             risk_level, privacy_notes, metadata, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "task-001",
            "session-001",
            "No tool to count words",
            "count_words_in_text",
            "text_count_words",
            "Count the number of words in a text string",
            json.dumps({
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            }),
            json.dumps({
                "type": "object",
                "properties": {"word_count": {"type": "integer"}},
            }),
            "L0",
            "",
            "{}",
            now,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_e2e_lifecycle(orchestrator, registry_with_router, telemetry, sandbox_dir):
    """Full lifecycle: request -> generate -> verify -> candidate -> promote -> dispatch."""

    # 1. Seed telemetry with a tool request
    _seed_tool_request(telemetry)

    # 2. Process pending requests (not dry_run)
    result = await orchestrator.process_pending_requests(dry_run=False)
    assert result["processed"] == 1
    assert result["errors"] == 0
    assert result["skipped"] == 0
    detail = result["details"][0]
    assert detail["action"] == "registered"
    assert detail["candidate_name"] == "text_count_words"

    # 3. Verify registered as CANDIDATE
    spec = registry_with_router.get_tool("text_count_words")
    assert spec is not None
    assert spec.status == ToolStatus.CANDIDATE
    assert spec.risk_level == RiskLevel.L0
    assert spec.runtime.value == "python_generated"

    # 4. Candidate tools are absorber/promotion inputs, not normal dispatch targets.
    from server.tools.spec import ToolContext
    ctx = ToolContext(session_id="test-session", task_id="test-task")
    candidate_result = await registry_with_router.dispatch(
        "text_count_words", {"text": "hello world test"}, ctx
    )
    assert candidate_result.success is False
    assert candidate_result.error_type == "tool_candidate"

    # 5. Internal validation can still execute the candidate via the router,
    # which records invocation/success telemetry used by promotion.
    assert registry_with_router._router is not None
    validation_result = await registry_with_router._router.dispatch(
        spec, {"text": "hello world test"}, ctx
    )
    assert validation_result.success is True
    payload = json.loads(validation_result.content)
    assert payload["word_count"] == 3

    # 6. Verify telemetry events
    stats = telemetry.get_tool_stats("text_count_words")
    assert stats.get("tool_created", 0) >= 1
    assert stats.get("tool_invoked", 0) >= 1
    assert stats.get("tool_succeeded", 0) >= 1

    # 7. Promote candidates -> ACTIVE
    promo_result = orchestrator.promote_candidates(
        min_success_count=1, min_success_rate=0.5
    )
    assert promo_result["promoted"] >= 1

    promoted_spec = registry_with_router.get_tool("text_count_words")
    assert promoted_spec is not None
    assert promoted_spec.status == ToolStatus.ACTIVE

    # 8. Promoted tools are callable through normal registry dispatch.
    exec_result = await registry_with_router.dispatch(
        "text_count_words", {"text": "hello world test"}, ctx
    )
    assert exec_result.success is True
    payload = json.loads(exec_result.content)
    assert payload["word_count"] == 3


@pytest.mark.asyncio
async def test_dry_run_does_not_register(orchestrator, registry_with_router, telemetry):
    """dry_run=True should NOT register the tool."""
    _seed_tool_request(telemetry)

    result = await orchestrator.process_pending_requests(dry_run=True)
    assert result["processed"] == 1
    detail = result["details"][0]
    assert detail["action"] == "dry_run"

    # Tool should NOT be in registry
    spec = registry_with_router.get_tool("text_count_words")
    assert spec is None


@pytest.mark.asyncio
async def test_skip_high_risk_request(orchestrator, registry_with_router, telemetry):
    """L2+ risk level requests should be skipped."""
    conn = telemetry._conn
    assert conn is not None
    conn.execute(
        """
        INSERT INTO tool_requests
            (task_id, session_id, reason, missing_capability,
             candidate_name, candidate_desc, candidate_input, candidate_output,
             risk_level, privacy_notes, metadata, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "task-high", "sess-high", "needs network", "fetch_url",
            "url_fetcher", "Fetch a URL",
            json.dumps({"type": "object", "properties": {"url": {"type": "string"}}}),
            json.dumps({"type": "object"}),
            "L3", "", "{}", "2025-01-01T00:00:00+00:00",
        ),
    )
    conn.commit()

    result = await orchestrator.process_pending_requests(dry_run=False)
    assert result["processed"] == 0
    assert result["skipped"] == 1
    assert result["details"][0]["action"] == "skipped"
    assert "L3" in result["details"][0]["reason"]


@pytest.mark.asyncio
async def test_skip_duplicate_tool_name(orchestrator, registry_with_router, telemetry):
    """Requests whose candidate_name already exists should be skipped."""
    _seed_tool_request(telemetry)

    # First pass: register
    await orchestrator.process_pending_requests(dry_run=False)

    # Insert another request with the same candidate_name
    _seed_tool_request(telemetry)

    # Second pass: both old and new rows are skipped (orchestrator doesn't
    # delete processed rows from tool_requests, so the original row is still
    # there and is also a duplicate now).
    result = await orchestrator.process_pending_requests(dry_run=False)
    assert result["skipped"] >= 1
    assert all("already exists" in d.get("reason", "") for d in result["details"])


@pytest.mark.asyncio
async def test_generation_failure(orchestrator, registry_with_router, telemetry, sandbox_dir):
    """Generation failure should be reported as an error."""
    _seed_tool_request(telemetry)

    # Replace developer with one that fails
    class FailingDeveloper:
        async def generate(self, tool_request):
            return {"success": False, "error": "LLM unavailable"}

    orchestrator._developer = FailingDeveloper()

    result = await orchestrator.process_pending_requests(dry_run=False)
    assert result["errors"] == 1
    assert result["details"][0]["action"] == "generation_failed"


@pytest.mark.asyncio
async def test_dispatch_single_word(registry_with_router, telemetry, sandbox_dir):
    """Dispatch with single word returns count=1."""
    from server.tools.spec import ToolSpec, ToolRuntime, ToolContext

    # Write tool file directly
    with open(f"{sandbox_dir}/text_count_words.py", "w") as f:
        f.write(WORD_COUNT_SOURCE)

    spec = ToolSpec(
        name="text_count_words",
        description="Count words",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        runtime=ToolRuntime.PYTHON_GENERATED,
        provider="generated",
        status=ToolStatus.ACTIVE,
    )
    registry_with_router.register(spec)

    ctx = ToolContext(session_id="s1", task_id="t1")
    result = await registry_with_router.dispatch(
        "text_count_words", {"text": "hello"}, ctx
    )
    assert result.success is True
    assert json.loads(result.content)["word_count"] == 1


@pytest.mark.asyncio
async def test_dispatch_empty_string(registry_with_router, telemetry, sandbox_dir):
    """Dispatch with empty string returns count=0."""
    from server.tools.spec import ToolSpec, ToolRuntime, ToolContext

    with open(f"{sandbox_dir}/text_count_words.py", "w") as f:
        f.write(WORD_COUNT_SOURCE)

    spec = ToolSpec(
        name="text_count_words",
        description="Count words",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        runtime=ToolRuntime.PYTHON_GENERATED,
        provider="generated",
        status=ToolStatus.ACTIVE,
    )
    registry_with_router.register(spec)

    ctx = ToolContext(session_id="s1", task_id="t1")
    result = await registry_with_router.dispatch(
        "text_count_words", {"text": ""}, ctx
    )
    assert result.success is True
    assert json.loads(result.content)["word_count"] == 0
