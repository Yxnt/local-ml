from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backends.base import ModelBackend
from server.tools.spec import RiskLevel, ToolResult


GENERATED_REVERSE_SOURCE = """
from pydantic import BaseModel

class InputModel(BaseModel):
    text: str

class OutputModel(BaseModel):
    result: str

def run(input: InputModel) -> OutputModel:
    return OutputModel(result=input.text[::-1])
""".strip()


class FakeDecision:
    def __init__(self, auto_generatable: bool) -> None:
        self.auto_generatable = auto_generatable
        self.risk_level = RiskLevel.L0 if auto_generatable else None
        self.reason = "pure_local_computation" if auto_generatable else "blocked_integration_or_private_data"


class FakeRiskClassifier:
    def annotate_request(self, request):
        auto_generatable = "calendar" not in request.candidate_name
        decision = FakeDecision(auto_generatable)
        request.metadata = {
            **request.metadata,
            "risk_classifier": {
                "auto_generatable": decision.auto_generatable,
                "risk_level": decision.risk_level.value if decision.risk_level else None,
                "reason": decision.reason,
                "blocked_terms": [] if decision.auto_generatable else ["calendar"],
            },
        }
        if decision.risk_level is not None:
            request.risk_level = decision.risk_level
        return decision


class FakeDeveloper:
    def __init__(self) -> None:
        self.calls = []

    async def generate(self, request):
        self.calls.append(request)
        return {
            "success": True,
            "tool_name": request.candidate_name,
            "source_code": GENERATED_REVERSE_SOURCE,
            "file_path": None,
            "error": None,
        }


class FakeVerifyResult:
    passed = True
    errors: list[str] = []


class FakeVerifier:
    def verify(self, tool_name: str, source_code: str):
        return FakeVerifyResult()


class FakeExporter:
    def __init__(self) -> None:
        self.calls = []

    def export(self, pool, *, task_id: str, batch_id: str = ""):
        specs = pool.list_specs()
        self.calls.append((specs, task_id, batch_id))
        return specs


class ToolCallingBackend(ModelBackend):
    def __init__(self, tool_name: str = "text_reverse", final_response: str = "Done!") -> None:
        self._tool_name = tool_name
        self._final_response = final_response
        self._generate_count = 0
        self.tool_names_by_round: list[list[str]] = []

    def load(self, model_id: str) -> None:
        pass

    def unload(self) -> None:
        pass

    def warmup(self) -> None:
        pass

    def apply_chat_template(self, messages, tools=None, enable_thinking=False) -> str:
        self.tool_names_by_round.append([
            tool["function"]["name"] for tool in tools or []
        ])
        return "\n".join(f"{m['role']}: {m.get('content', '')}" for m in messages)

    def generate(self, prompt: str, **kwargs) -> str:
        self._generate_count += 1
        if self._generate_count == 1:
            return json.dumps({"name": self._tool_name, "arguments": {"text": "abc"}})
        return self._final_response

    def parse_tool_calls(self, text: str) -> list[dict]:
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return []
        if "name" not in payload:
            return []
        return [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": payload["name"],
                    "arguments": json.dumps(payload.get("arguments", {})),
                },
            }
        ]


@pytest.fixture
def agent(tmp_path: Path):
    developer = FakeDeveloper()
    exporter = FakeExporter()
    backend = ToolCallingBackend()
    with patch("server.agent.ModelRegistry") as MockRegistry, \
         patch("server.agent.EmbeddingClient") as MockEmbedding:
        model_registry = MockRegistry.return_value
        model_registry.get_backend.return_value = backend
        model_registry.get_or_load = AsyncMock(return_value=backend)
        model_registry.list_models.return_value = []

        embedding = MockEmbedding.return_value
        embedding.close = AsyncMock()

        from server.agent import Agent

        agent = Agent(
            data_dir=str(tmp_path / "data"),
            default_model="test",
            use_tool_registry=True,
            enable_in_situ_tool_generation=True,
            tool_sandbox_dir=str(tmp_path / "sandbox"),
            tool_candidate_dir=str(tmp_path / "candidates"),
            tool_developer=developer,
            tool_verifier=FakeVerifier(),
            tool_risk_classifier=FakeRiskClassifier(),
            tool_local_candidate_exporter=exporter,
        )
        agent._registry = model_registry
        agent._embedding = embedding
        agent._test_backend = backend
        agent._test_developer = developer
        agent._test_exporter = exporter
        yield agent


@pytest.mark.asyncio
async def test_dispatch_generates_safe_missing_tool_and_retries_locally(agent):
    dispatch_ctx = SimpleNamespace(
        task_id="task1",
        user_input="Reverse abc",
        local_tool_pool=agent._create_local_tool_pool(),
        attempted_generation=set(),
    )
    created_events = []
    generation_events = []

    async def missing_dispatch(name, args, tool_ctx):
        return ToolResult(content=f"未知工具: {name}", success=False, error_type="tool_not_found")

    agent._tool_registry.dispatch = missing_dispatch
    agent._tool_telemetry.record = (
        lambda event_type, **kw: generation_events.append((event_type, kw)) or 1
    )
    agent._tool_telemetry.record_tool_created = (
        lambda name, version="", **kw: created_events.append((name, version, kw)) or 1
    )

    content = await agent._dispatch_tool("text_reverse", {"text": "abc"}, dispatch_ctx)

    assert json.loads(content) == {"result": "cba"}
    assert (Path(agent._tool_sandbox_dir) / "text_reverse.py").exists()
    assert "text_reverse" in dispatch_ctx.attempted_generation
    assert generation_events[0][0] == "tool_generation_attempted"
    assert generation_events[0][1]["metadata"]["source"] == "in_situ_local"
    assert created_events[0][0] == "text_reverse"
    assert created_events[0][2]["metadata"]["source"] == "in_situ_local"


@pytest.mark.asyncio
async def test_dispatch_records_request_only_when_classifier_blocks(agent):
    dispatch_ctx = SimpleNamespace(
        task_id="task1",
        user_input="Read calendar",
        local_tool_pool=agent._create_local_tool_pool(),
        attempted_generation=set(),
    )
    recorded = []

    async def missing_dispatch(name, args, tool_ctx):
        return ToolResult(content=f"未知工具: {name}", success=False, error_type="tool_not_found")

    agent._tool_registry.dispatch = missing_dispatch
    agent._tool_telemetry.record_tool_request = recorded.append

    content = await agent._dispatch_tool("calendar_read", {"query": "today"}, dispatch_ctx)

    assert content == "未知工具: calendar_read"
    assert recorded[0].metadata["risk_classifier"]["auto_generatable"] is False
    assert agent._test_developer.calls == []


@pytest.mark.asyncio
async def test_in_situ_generation_attempt_is_bounded_per_tool(agent):
    dispatch_ctx = SimpleNamespace(
        task_id="task1",
        user_input="Reverse abc",
        local_tool_pool=agent._create_local_tool_pool(),
        attempted_generation={"text_reverse"},
    )

    async def missing_dispatch(name, args, tool_ctx):
        return ToolResult(content=f"未知工具: {name}", success=False, error_type="tool_not_found")

    agent._tool_registry.dispatch = missing_dispatch

    content = await agent._dispatch_tool("text_reverse", {"text": "abc"}, dispatch_ctx)

    assert content == "未知工具: text_reverse"
    assert agent._test_developer.calls == []


@pytest.mark.asyncio
async def test_run_exports_query_local_tools_as_candidates(agent):
    response = await agent.run("Reverse abc")

    assert response == "Done!"
    assert agent._test_backend.tool_names_by_round[1][0] == "text_reverse"
    assert len(agent._test_exporter.calls) == 1
    specs, task_id, batch_id = agent._test_exporter.calls[0]
    assert specs[0].name == "text_reverse"
    assert task_id.startswith("task_")
    assert batch_id.startswith("session_")
