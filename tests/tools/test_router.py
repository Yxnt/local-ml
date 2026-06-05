"""Tests for server.tools.router — MemoryToolExecutor, GeneratedPythonExecutor, ToolRuntimeRouter."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from server.tools.spec import (
    ToolContext,
    ToolRuntime,
    ToolSpec,
)
from server.tools.router import (
    GeneratedPythonExecutor,
    MemoryToolExecutor,
    ToolRuntimeRouter,
    IntegrationToolExecutor,
    ComputerUseToolExecutor,
)


class TestMemoryToolExecutor:
    """MemoryToolExecutor handles memory_remember."""

    @pytest.mark.asyncio
    async def test_memory_remember(self, fake_memory_manager, sample_memory_tool):
        executor = MemoryToolExecutor(fake_memory_manager)
        ctx = ToolContext(session_id="s1", task_id="t1")
        result = await executor.execute(
            sample_memory_tool,
            {"content": "The sky is blue", "type": "fact", "importance": 0.8},
            ctx,
        )
        assert result.success is True
        data = json.loads(result.content)
        assert data["status"] == "ok"
        assert data["id"] >= 1

    @pytest.mark.asyncio
    async def test_unknown_memory_tool(self, fake_memory_manager):
        spec = ToolSpec(
            name="memory_unknown",
            description="Unknown memory tool",
            input_schema={"type": "object", "properties": {}},
            runtime=ToolRuntime.MEMORY_METHOD,
        )
        executor = MemoryToolExecutor(fake_memory_manager)
        ctx = ToolContext(session_id="s1", task_id="t1")
        result = await executor.execute(spec, {}, ctx)
        assert result.success is False
        assert result.error_type == "unknown_tool"


class TestGeneratedPythonExecutor:
    """GeneratedPythonExecutor runs tools in subprocess."""

    @pytest.mark.asyncio
    async def test_subprocess_happy_path(self, sandbox_dir, sample_generated_tool):
        """Write a valid tool .py, execute it via subprocess."""
        tool_source = textwrap.dedent("""\
            from pydantic import BaseModel

            class InputModel(BaseModel):
                name: str = "world"

            class OutputModel(BaseModel):
                greeting: str

            def run(input: InputModel) -> OutputModel:
                return OutputModel(greeting=f"Hello, {input.name}!")
        """)
        tool_file = Path(sandbox_dir) / "hello_tool.py"
        tool_file.write_text(tool_source)

        executor = GeneratedPythonExecutor(sandbox_dir=sandbox_dir, execution_mode="subprocess")
        ctx = ToolContext(session_id="s1", task_id="t1")
        result = await executor.execute(sample_generated_tool, {"name": "Alice"}, ctx)
        assert result.success is True
        data = json.loads(result.content)
        assert data["greeting"] == "Hello, Alice!"

    @pytest.mark.asyncio
    async def test_subprocess_file_not_found(self, sandbox_dir, sample_generated_tool):
        """Tool file doesn't exist → file_not_found error."""
        executor = GeneratedPythonExecutor(sandbox_dir=sandbox_dir, execution_mode="subprocess")
        ctx = ToolContext(session_id="s1", task_id="t1")
        result = await executor.execute(sample_generated_tool, {"name": "Alice"}, ctx)
        assert result.success is False
        assert result.error_type == "file_not_found"

    @pytest.mark.asyncio
    async def test_subprocess_bad_code(self, sandbox_dir):
        """Tool with syntax error → execution_error."""
        bad_source = "def broken(\n"
        tool_file = Path(sandbox_dir) / "broken_tool.py"
        tool_file.write_text(bad_source)

        spec = ToolSpec(
            name="broken_tool",
            description="Broken tool",
            input_schema={"type": "object", "properties": {}},
            runtime=ToolRuntime.PYTHON_GENERATED,
        )
        executor = GeneratedPythonExecutor(sandbox_dir=sandbox_dir, execution_mode="subprocess")
        ctx = ToolContext(session_id="s1", task_id="t1")
        result = await executor.execute(spec, {}, ctx)
        assert result.success is False
        # Could be execution_error or subprocess_error depending on how it fails
        assert result.error_type in ("execution_error", "subprocess_error")

    def test_invalid_execution_mode_raises(self, sandbox_dir):
        with pytest.raises(ValueError, match="execution_mode"):
            GeneratedPythonExecutor(sandbox_dir=sandbox_dir, execution_mode="invalid")


class TestToolRuntimeRouter:
    """ToolRuntimeRouter dispatches to the correct executor."""

    @pytest.mark.asyncio
    async def test_dispatches_to_memory_executor(self, fake_memory_manager, sample_memory_tool):
        mm_executor = MemoryToolExecutor(fake_memory_manager)
        # Use dummy executors for the others
        dummy_integ = MagicMock(spec=IntegrationToolExecutor)
        dummy_cu = MagicMock(spec=ComputerUseToolExecutor)
        router = ToolRuntimeRouter(
            memory_executor=mm_executor,
            integration_executor=dummy_integ,
            computer_executor=dummy_cu,
        )

        ctx = ToolContext(session_id="s1", task_id="t1")
        result = await router.dispatch(
            sample_memory_tool,
            {"content": "test memory", "type": "fact", "importance": 0.5},
            ctx,
        )
        assert result.success is True
        data = json.loads(result.content)
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_no_executor_returns_error(self):
        """A runtime with no registered executor returns no_executor error."""
        mm_executor = MemoryToolExecutor(MagicMock())
        dummy_integ = MagicMock(spec=IntegrationToolExecutor)
        dummy_cu = MagicMock(spec=ComputerUseToolExecutor)
        router = ToolRuntimeRouter(
            memory_executor=mm_executor,
            integration_executor=dummy_integ,
            computer_executor=dummy_cu,
            generated_executor=None,
        )
        # JS_TOOL has no executor registered
        spec = ToolSpec(
            name="js_tool",
            description="A JS tool",
            input_schema={"type": "object", "properties": {}},
            runtime=ToolRuntime.JS_TOOL,
        )
        ctx = ToolContext(session_id="s1", task_id="t1")
        result = await router.dispatch(spec, {}, ctx)
        assert result.success is False
        assert result.error_type == "no_executor"
