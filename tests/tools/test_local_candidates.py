from pathlib import Path

from server.tools.local_candidates import LocalCandidateExporter
from server.tools.local_pool import LocalToolPool
from server.tools.spec import RiskLevel, ToolRuntime, ToolSpec, ToolStatus


class FakeRouter:
    async def dispatch(self, spec, arguments, ctx):
        raise AssertionError("dispatch is not used by candidate export")


class FakeRegistry:
    def __init__(self) -> None:
        self.registered = []

    def get_tool(self, name: str):
        return None

    def register(self, spec: ToolSpec):
        self.registered.append(spec)


class FakeTelemetry:
    def __init__(self) -> None:
        self.events = []

    def record_tool_registered(self, name, version="", **kw):
        self.events.append((name, version, kw))
        return len(self.events)


def _spec(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Reverse provided text",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        output_schema={
            "type": "object",
            "properties": {"result": {"type": "string"}},
        },
        runtime=ToolRuntime.PYTHON_GENERATED,
        provider="query_local",
        risk_level=RiskLevel.L0,
        metadata={
            "replay_cases": [
                {
                    "arguments": {"text": "abc"},
                    "expect": {"result": "cba"},
                    "match": "subset",
                }
            ]
        },
    )


def test_export_registers_local_tools_as_candidates(tmp_path: Path):
    pool = LocalToolPool(router=FakeRouter(), sandbox_dir=str(tmp_path / "local"))
    pool.register(_spec("text_reverse"), "source code")
    registry = FakeRegistry()
    telemetry = FakeTelemetry()

    exported = LocalCandidateExporter(
        registry=registry,
        telemetry=telemetry,
        candidate_dir=str(tmp_path / "candidates"),
    ).export(pool, task_id="task1", batch_id="batch1")

    assert [spec.name for spec in exported] == ["text_reverse"]
    assert registry.registered[0].status == ToolStatus.CANDIDATE
    assert registry.registered[0].provider == "query_local_candidate"
    assert registry.registered[0].metadata["absorber_candidate"] is True
    assert registry.registered[0].metadata["source"] == "local_candidate_export"
    assert registry.registered[0].metadata["task_id"] == "task1"
    assert registry.registered[0].metadata["batch_id"] == "batch1"
    assert registry.registered[0].metadata["replay_cases"] == _spec("text_reverse").metadata["replay_cases"]
    assert Path(registry.registered[0].metadata["source_file"]).read_text(encoding="utf-8") == "source code"
    assert telemetry.events[0][2]["metadata"]["source"] == "local_candidate_export"


def test_export_skips_names_that_already_exist_in_registry(tmp_path: Path):
    pool = LocalToolPool(router=FakeRouter(), sandbox_dir=str(tmp_path / "local"))
    pool.register(_spec("text_reverse"), "source code")

    class RegistryWithExisting(FakeRegistry):
        def get_tool(self, name: str):
            return _spec(name)

    exported = LocalCandidateExporter(
        registry=RegistryWithExisting(),
        telemetry=None,
        candidate_dir=str(tmp_path / "candidates"),
    ).export(pool, task_id="task1")

    assert exported == []
