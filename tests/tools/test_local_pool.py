from pathlib import Path

import pytest

from server.tools.local_pool import LocalToolPool
from server.tools.spec import RiskLevel, ToolContext, ToolResult, ToolRuntime, ToolSpec


class FakeGlobalRegistry:
    def __init__(self, specs: list[ToolSpec] | None = None) -> None:
        self._specs = {spec.name: spec for spec in specs or []}

    def get_tool(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def list_openai_tools(self) -> list[dict]:
        return [spec.to_openai_schema() for spec in self._specs.values()]


class FakeRouter:
    def __init__(self) -> None:
        self.calls = []

    async def dispatch(self, spec: ToolSpec, arguments: dict, ctx: ToolContext) -> ToolResult:
        self.calls.append((spec, arguments, ctx))
        return ToolResult(content=f"local:{spec.name}:{arguments['text']}")


def _spec(name: str, description: str = "Reverse provided text") -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
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
        provider="generated",
        risk_level=RiskLevel.L0,
    )


def test_register_materializes_source_and_returns_local_tool(tmp_path: Path):
    pool = LocalToolPool(router=FakeRouter(), sandbox_dir=str(tmp_path))
    source = "from pydantic import BaseModel\n"

    registered = pool.register(_spec("text_reverse"), source)

    assert registered.name == "text_reverse"
    assert registered.provider == "query_local"
    assert registered.metadata["local_pool"] is True
    assert registered.metadata["source_file"] == str(tmp_path / "text_reverse.py")
    assert (tmp_path / "text_reverse.py").read_text(encoding="utf-8") == source
    assert pool.get_tool("text_reverse") is registered


def test_get_tool_is_local_first_over_global_registry(tmp_path: Path):
    global_spec = _spec("text_reverse", description="Global reverse")
    local_spec = _spec("text_reverse", description="Local reverse")
    pool = LocalToolPool(
        router=FakeRouter(),
        sandbox_dir=str(tmp_path),
        global_registry=FakeGlobalRegistry([global_spec]),
    )

    registered = pool.register(local_spec, "source")

    assert pool.get_tool("text_reverse") is registered
    assert pool.get_tool("missing") is None


def test_list_openai_tools_is_local_first_and_dedupes_global_names(tmp_path: Path):
    pool = LocalToolPool(router=FakeRouter(), sandbox_dir=str(tmp_path))
    pool.register(_spec("text_reverse"), "source")

    global_tools = [
        {"type": "function", "function": {"name": "text_reverse", "description": "global", "parameters": {}}},
        {"type": "function", "function": {"name": "memory_recall", "description": "memory", "parameters": {}}},
    ]

    tools = pool.list_openai_tools(global_tools)

    assert [tool["function"]["name"] for tool in tools] == ["text_reverse", "memory_recall"]
    assert tools[0]["function"]["description"] == "Reverse provided text"


def test_list_openai_tools_can_pull_from_global_registry(tmp_path: Path):
    pool = LocalToolPool(
        router=FakeRouter(),
        sandbox_dir=str(tmp_path),
        global_registry=FakeGlobalRegistry([_spec("memory_recall", description="Memory recall")]),
    )
    pool.register(_spec("text_reverse"), "source")

    tools = pool.list_openai_tools()

    assert [tool["function"]["name"] for tool in tools] == ["text_reverse", "memory_recall"]


@pytest.mark.asyncio
async def test_dispatch_uses_router_with_local_spec(tmp_path: Path):
    router = FakeRouter()
    pool = LocalToolPool(router=router, sandbox_dir=str(tmp_path))
    pool.register(_spec("text_reverse"), "source")

    result = await pool.dispatch("text_reverse", {"text": "abc"}, ToolContext(session_id="s1"))

    assert result.success is True
    assert result.content == "local:text_reverse:abc"
    assert router.calls[0][0].name == "text_reverse"


@pytest.mark.asyncio
async def test_dispatch_returns_tool_not_found_for_missing_local_tool(tmp_path: Path):
    pool = LocalToolPool(router=FakeRouter(), sandbox_dir=str(tmp_path))

    result = await pool.dispatch("missing_tool", {}, ToolContext(session_id="s1"))

    assert result.success is False
    assert result.error_type == "tool_not_found"
