"""Tests for Agent tool system integration.

Verifies that Agent.run() uses ToolRegistry / ToolRetriever / ToolRuntimeRouter
/ TelemetryService when use_tool_registry=True, and falls back to legacy when
disabled or when the tool system fails.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backends.base import ModelBackend


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubBackend(ModelBackend):
    """Minimal backend that returns canned responses."""

    def __init__(self, response: str = "Hello!"):
        self._response = response
        self._loaded = True

    def load(self, model_id: str) -> None:
        pass

    def unload(self) -> None:
        pass

    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, top_p: float = 0.9) -> str:
        return self._response

    def apply_chat_template(self, messages, tools=None, enable_thinking=False) -> str:
        parts = []
        for m in messages:
            parts.append(f"{m['role']}: {m.get('content', '')}")
        return "\n".join(parts)

    def parse_tool_calls(self, text: str) -> list[dict]:
        return []

    def warmup(self) -> None:
        pass


class ToolCallingBackend(ModelBackend):
    """Backend that returns a tool call on first generate, then text."""

    def __init__(self, tool_call: dict, final_response: str = "Done!"):
        self._tool_call = tool_call
        self._final_response = final_response
        self._call_count = 0

    def load(self, model_id: str) -> None:
        pass

    def unload(self) -> None:
        pass

    def generate(self, prompt: str, **kw) -> str:
        self._call_count += 1
        if self._call_count <= 1:
            return json.dumps(self._tool_call)
        return self._final_response

    def apply_chat_template(self, messages, tools=None, **kw) -> str:
        parts = []
        for m in messages:
            parts.append(f"{m['role']}: {m.get('content', '')}")
        return "\n".join(parts)

    def parse_tool_calls(self, text: str) -> list[dict]:
        try:
            obj = json.loads(text)
            if "name" in obj and "arguments" in obj:
                return [{
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": obj["name"],
                        "arguments": json.dumps(obj["arguments"]) if isinstance(obj["arguments"], dict) else obj["arguments"],
                    },
                }]
        except (json.JSONDecodeError, TypeError):
            pass
        return []

    def warmup(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_data_dir(tmp_path):
    return str(tmp_path / "data")


@pytest.fixture
def agent_with_registry(tmp_data_dir):
    """Agent with tool registry enabled."""
    with patch("server.agent.ModelRegistry") as MockRegistry, \
         patch("server.agent.EmbeddingClient") as MockEmbedding:
        mock_registry = MockRegistry.return_value
        stub = StubBackend()
        mock_registry.get_backend.return_value = stub
        mock_registry.get_or_load = AsyncMock(return_value=stub)
        mock_registry.list_models.return_value = []

        mock_embedding = MockEmbedding.return_value
        mock_embedding.close = AsyncMock()

        from server.agent import Agent
        a = Agent(data_dir=tmp_data_dir, default_model="test", use_tool_registry=True)
        a._registry = mock_registry
        a._embedding = mock_embedding
        yield a


@pytest.fixture
def agent_legacy(tmp_data_dir):
    """Agent with tool registry disabled (legacy mode)."""
    with patch("server.agent.ModelRegistry") as MockRegistry, \
         patch("server.agent.EmbeddingClient") as MockEmbedding:
        mock_registry = MockRegistry.return_value
        stub = StubBackend()
        mock_registry.get_backend.return_value = stub
        mock_registry.get_or_load = AsyncMock(return_value=stub)
        mock_registry.list_models.return_value = []

        mock_embedding = MockEmbedding.return_value
        mock_embedding.close = AsyncMock()

        from server.agent import Agent
        a = Agent(data_dir=tmp_data_dir, default_model="test", use_tool_registry=False)
        a._registry = mock_registry
        a._embedding = mock_embedding
        yield a


# ---------------------------------------------------------------------------
# Phase 1: Agent bootstraps memory tools to registry
# ---------------------------------------------------------------------------


class TestBootstrapMemoryTools:
    def test_agent_bootstraps_memory_tools_to_registry(self, agent_with_registry):
        """Memory tools should be in the registry after init."""
        reg = agent_with_registry._tool_registry
        assert reg is not None
        tools = reg.list_tools()
        names = {t.name for t in tools}
        assert "memory_remember" in names
        assert "memory_recall" in names
        assert "memory_stats" in names

    def test_registry_tool_count_at_least_three(self, agent_with_registry):
        """At least memory + computer_use tools registered."""
        reg = agent_with_registry._tool_registry
        tools = reg.list_tools()
        assert len(tools) >= 3


# ---------------------------------------------------------------------------
# Phase 3: get_tools_async
# ---------------------------------------------------------------------------


class TestGetToolsAsync:
    @pytest.mark.asyncio
    async def test_get_tools_async_all_uses_registry(self, agent_with_registry):
        """Mode 'all' should return all tools from registry."""
        agent_with_registry._tool_retrieval_mode = "all"
        tools = await agent_with_registry.get_tools_async()
        assert len(tools) >= 3
        names = {t["function"]["name"] for t in tools}
        assert "memory_remember" in names

    @pytest.mark.asyncio
    async def test_get_tools_async_top_k_calls_retriever(self, agent_with_registry):
        """Mode 'top_k' should call retriever.retrieve()."""
        agent_with_registry._tool_retrieval_mode = "top_k"

        # Mock retriever to return a subset.
        mock_specs = [MagicMock()]
        mock_specs[0].to_openai_schema.return_value = {
            "type": "function",
            "function": {"name": "memory_recall", "description": "recall", "parameters": {}},
        }
        mock_specs[0].name = "memory_recall"

        agent_with_registry._tool_retriever = AsyncMock()
        agent_with_registry._tool_retriever.retrieve = AsyncMock(return_value=mock_specs)

        tools = await agent_with_registry.get_tools_async("test query")
        agent_with_registry._tool_retriever.retrieve.assert_awaited_once_with("test query", limit=12)
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "memory_recall"

    @pytest.mark.asyncio
    async def test_get_tools_async_retriever_failure_falls_back_to_all(self, agent_with_registry):
        """When retriever raises, should fall back to all tools."""
        agent_with_registry._tool_retrieval_mode = "top_k"
        agent_with_registry._tool_retriever = AsyncMock()
        agent_with_registry._tool_retriever.retrieve = AsyncMock(side_effect=RuntimeError("embed down"))
        agent_with_registry._tool_retriever._keyword_search = MagicMock(side_effect=RuntimeError("also down"))

        tools = await agent_with_registry.get_tools_async("test")
        # Should still return tools (fallback to all).
        assert len(tools) >= 3

    @pytest.mark.asyncio
    async def test_get_tools_async_auto_mode_small_registry(self, agent_with_registry):
        """Auto mode with <= top_k tools should return all."""
        agent_with_registry._tool_retrieval_mode = "auto"
        agent_with_registry._tool_retrieval_top_k = 100  # larger than tool count
        tools = await agent_with_registry.get_tools_async()
        assert len(tools) >= 3


# ---------------------------------------------------------------------------
# Phase 4: Dispatch via registry
# ---------------------------------------------------------------------------


class TestDispatchViaRegistry:
    @pytest.mark.asyncio
    async def test_dispatch_memory_tool_via_registry(self, agent_with_registry):
        """memory_stats should be dispatched via registry."""
        result = await agent_with_registry._dispatch_tool("memory_stats", {})
        parsed = json.loads(result)
        assert "total_memories" in parsed

    @pytest.mark.asyncio
    async def test_unknown_tool_records_tool_request(self, agent_with_registry):
        """Unknown tool should record a ToolRequest."""
        result = await agent_with_registry._dispatch_tool("totally_fake_tool", {})
        assert "未知工具" in result

        # Check that a tool_request was recorded.
        if agent_with_registry._tool_telemetry is not None:
            requests = agent_with_registry._tool_telemetry.get_tool_requests(limit=5)
            names = [r.get("candidate_name") for r in requests]
            assert "totally_fake_tool" in names

    @pytest.mark.asyncio
    async def test_integration_tool_falls_back_to_legacy(self, agent_with_registry):
        """Integration tool not in registry should fall back to legacy dispatch."""
        mock_obsidian = AsyncMock()
        mock_obsidian.query.return_value = [{"path": "note.md", "title": "Test"}]
        mock_obsidian.get_tools.return_value = []

        await agent_with_registry.connect_integration("obsidian", mock_obsidian)

        result = await agent_with_registry._dispatch_tool("obsidian_search", {"query": "test"})
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["path"] == "note.md"


# ---------------------------------------------------------------------------
# Phase 5: Telemetry covers online tool calls
# ---------------------------------------------------------------------------


class TestTelemetryIntegration:
    @pytest.mark.asyncio
    async def test_run_tool_call_records_telemetry(self, agent_with_registry):
        """Agent.run() with a tool call should record tool_invoked/tool_succeeded."""
        tool_call = {"name": "memory_stats", "arguments": {}}
        backend = ToolCallingBackend(tool_call=tool_call, final_response="Stats done")
        agent_with_registry._registry.get_or_load = AsyncMock(return_value=backend)

        response = await agent_with_registry.run("show stats")
        assert response == "Stats done"

        # Check telemetry events.
        if agent_with_registry._tool_telemetry is not None:
            events = agent_with_registry._tool_telemetry.get_recent_events(limit=20)
            event_types = [e["event_type"] for e in events]
            assert "tool_invoked" in event_types
            assert "tool_succeeded" in event_types

    @pytest.mark.asyncio
    async def test_run_records_task_started_finished(self, agent_with_registry):
        """Agent.run() should record task_started and task_finished."""
        response = await agent_with_registry.run("Hello!")
        assert response == "Hello!"

        if agent_with_registry._tool_telemetry is not None:
            events = agent_with_registry._tool_telemetry.get_recent_events(limit=20)
            event_types = [e["event_type"] for e in events]
            assert "task_started" in event_types
            assert "task_finished" in event_types


# ---------------------------------------------------------------------------
# Phase 6: connect_integration registers tools
# ---------------------------------------------------------------------------


class TestConnectIntegration:
    @pytest.mark.asyncio
    async def test_connect_integration_registers_tools(self, agent_with_registry):
        """After connecting an integration, its tools should be in the registry."""
        mock_cal = MagicMock()
        mock_cal.connect = AsyncMock()
        mock_cal.get_tools.return_value = [
            {"type": "function", "function": {"name": "calendar_search", "description": "search", "parameters": {}}},
            {"type": "function", "function": {"name": "calendar_upcoming", "description": "upcoming", "parameters": {}}},
        ]

        await agent_with_registry.connect_integration("calendar", mock_cal)

        reg = agent_with_registry._tool_registry
        tool_names = {t.name for t in reg.list_tools()}
        assert "calendar_search" in tool_names
        assert "calendar_upcoming" in tool_names


# ---------------------------------------------------------------------------
# Legacy mode
# ---------------------------------------------------------------------------


class TestLegacyMode:
    @pytest.mark.asyncio
    async def test_legacy_mode_still_works(self, agent_legacy):
        """Legacy mode should use old dispatch logic."""
        assert agent_legacy._use_tool_registry is False
        assert agent_legacy._tool_registry is None

        result = await agent_legacy._dispatch_tool("memory_stats", {})
        parsed = json.loads(result)
        assert "total_memories" in parsed

    @pytest.mark.asyncio
    async def test_legacy_get_tools_returns_all(self, agent_legacy):
        """Legacy get_tools should return memory tools."""
        tools = agent_legacy._legacy_get_tools()
        names = {t["function"]["name"] for t in tools}
        assert "memory_remember" in names

    @pytest.mark.asyncio
    async def test_legacy_run_works(self, agent_legacy):
        """Legacy agent.run() should work normally."""
        response = await agent_legacy.run("Hello!")
        assert response == "Hello!"


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------


class TestImportSafety:
    def test_import_has_no_optional_optimization_dependency(self):
        """importing server.agent should not require DSPy or GEPA."""
        import importlib
        mod = importlib.import_module("server.agent")
        assert hasattr(mod, "Agent")


# ---------------------------------------------------------------------------
# Registry failure fallback
# ---------------------------------------------------------------------------


class TestRegistryFailureFallback:
    @pytest.mark.asyncio
    async def test_registry_dispatch_failure_falls_back_to_legacy(self, agent_with_registry):
        """If registry.dispatch raises, should fall back to legacy."""
        # Break the registry dispatch.
        agent_with_registry._tool_registry.dispatch = AsyncMock(side_effect=RuntimeError("registry broken"))

        result = await agent_with_registry._dispatch_tool("memory_stats", {})
        parsed = json.loads(result)
        assert "total_memories" in parsed

    def test_init_tool_system_failure_disables_registry(self, tmp_data_dir):
        """If tool system init fails, should fall back to legacy."""
        with patch("server.agent.ModelRegistry") as MockRegistry, \
             patch("server.agent.EmbeddingClient") as MockEmbedding:
            mock_registry = MockRegistry.return_value
            stub = StubBackend()
            mock_registry.get_backend.return_value = stub
            mock_registry.get_or_load = AsyncMock(return_value=stub)
            mock_registry.list_models.return_value = []

            mock_embedding = MockEmbedding.return_value
            mock_embedding.close = AsyncMock()

            # Make TelemetryService.connect() fail.
            with patch("server.tools.telemetry.TelemetryService.connect", side_effect=RuntimeError("db locked")):
                from server.agent import Agent
                a = Agent(data_dir=tmp_data_dir, default_model="test", use_tool_registry=True)
                a._registry = mock_registry
                a._embedding = mock_embedding

                # Should have fallen back to legacy.
                assert a._use_tool_registry is False
                assert a._tool_registry is None
