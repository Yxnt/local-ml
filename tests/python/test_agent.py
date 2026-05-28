"""Tests for server.agent.Agent.

All external dependencies (model backends, integrations, embedding service)
are mocked so the tests run without any external services.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backends.base import ModelBackend


# ---------------------------------------------------------------------------
# Stub backend for testing
# ---------------------------------------------------------------------------


class StubBackend(ModelBackend):
    """Minimal backend that returns canned responses."""

    def __init__(self, response: str = "Hello!"):
        self._response = response
        self._loaded = True

    def load(self, model_id: str) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, top_p: float = 0.9) -> str:
        return self._response

    def apply_chat_template(self, messages, tools=None, enable_thinking=False) -> str:
        # Return a simple concatenation for testing.
        parts = []
        for m in messages:
            parts.append(f"{m['role']}: {m['content']}")
        return "\n".join(parts)

    def parse_tool_calls(self, text: str) -> list[dict]:
        return []

    def warmup(self) -> None:
        pass


class ToolCallingBackend(ModelBackend):
    """Backend that returns tool calls on first generate, then text."""

    def __init__(self, tool_call: dict, final_response: str = "Done!"):
        self._tool_call = tool_call
        self._final_response = final_response
        self._call_count = 0

    def load(self, model_id: str) -> None:
        pass

    def unload(self) -> None:
        pass

    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7, top_p: float = 0.9) -> str:
        self._call_count += 1
        if self._call_count <= 1:
            return json.dumps(self._tool_call)
        return self._final_response

    def apply_chat_template(self, messages, tools=None, enable_thinking=False) -> str:
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
    """Provide a temporary data directory for MemoryManager."""
    return str(tmp_path / "memory_data")


@pytest.fixture
def agent(tmp_data_dir):
    """Create an Agent with a stub backend, using a temp data dir."""
    with patch("server.agent.ModelRegistry") as MockRegistry, \
         patch("server.agent.EmbeddingClient") as MockEmbedding:

        # Set up mock registry.
        mock_registry_instance = MockRegistry.return_value
        stub_backend = StubBackend()
        mock_registry_instance.get_backend.return_value = stub_backend
        mock_registry_instance.get_or_load = AsyncMock(return_value=stub_backend)
        mock_registry_instance.list_models.return_value = [
            {"id": "gemma-4-e2b-it-4bit", "backend": "mlx_vlm"},
        ]

        # Mock embedding client.
        mock_embedding = MockEmbedding.return_value
        mock_embedding.close = AsyncMock()

        from server.agent import Agent
        a = Agent(data_dir=tmp_data_dir, default_model="gemma-4-e2b-it-4bit")

        # Inject the stub backend into the registry mock.
        a._registry = mock_registry_instance
        a._embedding = mock_embedding

        yield a


@pytest.fixture
def tool_agent(tmp_data_dir):
    """Create an Agent with a tool-calling backend."""
    tool_call = {
        "name": "memory_stats",
        "arguments": {},
    }
    backend = ToolCallingBackend(tool_call=tool_call, final_response="Memory has 5 entries.")

    with patch("server.agent.ModelRegistry") as MockRegistry, \
         patch("server.agent.EmbeddingClient") as MockEmbedding:

        mock_registry_instance = MockRegistry.return_value
        mock_registry_instance.get_backend.return_value = backend
        mock_registry_instance.get_or_load = AsyncMock(return_value=backend)

        mock_embedding = MockEmbedding.return_value
        mock_embedding.close = AsyncMock()

        from server.agent import Agent
        a = Agent(data_dir=tmp_data_dir, default_model="gemma-4-e2b-it-4bit")
        a._registry = mock_registry_instance
        a._embedding = mock_embedding

        yield a


# ---------------------------------------------------------------------------
# Tests: Initialization
# ---------------------------------------------------------------------------


class TestAgentInit:
    def test_agent_creates_successfully(self, agent):
        """Agent should initialize without errors."""
        assert agent is not None
        assert agent.get_current_model() == "gemma-4-e2b-it-4bit"

    def test_agent_has_session_id(self, agent):
        """Agent should have a session ID after init."""
        session_id = agent.get_session_id()
        assert session_id
        assert isinstance(session_id, str)

    def test_agent_memory_connected(self, agent):
        """Memory manager should be connected after init."""
        assert agent._memory._soul is not None
        assert agent._memory._user is not None
        assert agent._memory._store is not None

    def test_agent_collector_connected(self, agent):
        """Usage collector should be connected after init."""
        assert agent._collector._conn is not None

    def test_agent_default_model(self, tmp_data_dir):
        """Default model should be configurable."""
        with patch("server.agent.ModelRegistry") as MockRegistry, \
             patch("server.agent.EmbeddingClient"):
            mock_reg = MockRegistry.return_value
            mock_reg.get_backend.return_value = StubBackend()
            mock_reg.get_or_load = AsyncMock(return_value=StubBackend())

            from server.agent import Agent
            a = Agent(data_dir=tmp_data_dir, default_model="minicpm5-1b-mlx")
            assert a.get_current_model() == "minicpm5-1b-mlx"


# ---------------------------------------------------------------------------
# Tests: System prompt generation
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_system_prompt_includes_soul(self, agent):
        """System prompt should contain soul personality info."""
        prompt = agent._build_system_prompt()
        # Soul defaults include name "Local ML Assistant"
        assert "Local ML Assistant" in prompt or "性格特点" in prompt

    def test_system_prompt_includes_user_context(self, agent):
        """System prompt should include user profile context."""
        prompt = agent._build_system_prompt()
        # Default user profile has devices info
        assert "Mac Mini" in prompt or "用户" in prompt

    def test_system_prompt_set_on_first_run(self, agent):
        """First call to run() should insert a system message."""
        import asyncio
        response = asyncio.get_event_loop().run_until_complete(agent.run("Hi"))
        assert agent._messages[0]["role"] == "system"
        assert len(agent._messages[0]["content"]) > 0


# ---------------------------------------------------------------------------
# Tests: Tool dispatch
# ---------------------------------------------------------------------------


class TestToolDispatch:
    @pytest.mark.asyncio
    async def test_memory_remember(self, agent):
        """memory_remember tool should store a memory."""
        result = await agent._dispatch_tool("memory_remember", {
            "content": "用户喜欢咖啡",
            "type": "preference",
            "importance": 0.8,
        })
        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert "id" in parsed

    @pytest.mark.asyncio
    async def test_memory_recall(self, agent):
        """memory_recall tool should return search results."""
        # First store something.
        await agent._dispatch_tool("memory_remember", {
            "content": "用户的生日是 1990-01-01",
        })
        # Then recall.
        result = await agent._dispatch_tool("memory_recall", {"query": "生日"})
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    @pytest.mark.asyncio
    async def test_memory_stats(self, agent):
        """memory_stats tool should return statistics."""
        result = await agent._dispatch_tool("memory_stats", {})
        parsed = json.loads(result)
        assert "total_memories" in parsed

    @pytest.mark.asyncio
    async def test_obsidian_not_connected(self, agent):
        """Obsidian tool should return helpful error when not connected."""
        result = await agent._dispatch_tool("obsidian_search", {"query": "test"})
        assert "请先配置 Obsidian" in result

    @pytest.mark.asyncio
    async def test_calendar_not_connected(self, agent):
        """Calendar tool should return helpful error when not connected."""
        result = await agent._dispatch_tool("calendar_search", {"query": "meeting"})
        assert "请先配置日历" in result

    @pytest.mark.asyncio
    async def test_email_not_connected(self, agent):
        """Email tool should return helpful error when not connected."""
        result = await agent._dispatch_tool("email_search", {"query": "hello"})
        assert "请先配置邮箱" in result

    @pytest.mark.asyncio
    async def test_unknown_tool(self, agent):
        """Unknown tool name should return an error message."""
        result = await agent._dispatch_tool("totally_fake_tool", {})
        assert "未知工具" in result

    @pytest.mark.asyncio
    async def test_obsidian_connected(self, agent):
        """Obsidian tool should work when integration is connected."""
        mock_obsidian = AsyncMock()
        mock_obsidian.query.return_value = [{"path": "note.md", "title": "Test"}]
        mock_obsidian.get_tools.return_value = []

        await agent.connect_integration("obsidian", mock_obsidian)

        result = await agent._dispatch_tool("obsidian_search", {"query": "test"})
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["path"] == "note.md"
        mock_obsidian.query.assert_called_once_with("test", 10)

    @pytest.mark.asyncio
    async def test_obsidian_read(self, agent):
        """obsidian_read should return note content."""
        mock_obsidian = AsyncMock()
        mock_obsidian.read_note.return_value = {"path": "test.md", "content": "# Hello"}
        mock_obsidian.get_tools.return_value = []

        await agent.connect_integration("obsidian", mock_obsidian)

        result = await agent._dispatch_tool("obsidian_read", {"path": "test.md"})
        parsed = json.loads(result)
        assert parsed["content"] == "# Hello"

    @pytest.mark.asyncio
    async def test_obsidian_read_not_found(self, agent):
        """obsidian_read should handle missing notes gracefully."""
        mock_obsidian = AsyncMock()
        mock_obsidian.read_note.return_value = None
        mock_obsidian.get_tools.return_value = []

        await agent.connect_integration("obsidian", mock_obsidian)

        result = await agent._dispatch_tool("obsidian_read", {"path": "missing.md"})
        assert "笔记不存在" in result

    @pytest.mark.asyncio
    async def test_calendar_upcoming(self, agent):
        """calendar_upcoming should return events when connected."""
        mock_calendar = AsyncMock()
        mock_calendar.get_upcoming.return_value = [
            {"summary": "Team meeting", "start": "2026-05-29T10:00:00"},
        ]
        mock_calendar.get_tools.return_value = []

        await agent.connect_integration("calendar", mock_calendar)

        result = await agent._dispatch_tool("calendar_upcoming", {"days": 7})
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["summary"] == "Team meeting"

    @pytest.mark.asyncio
    async def test_email_recent(self, agent):
        """email_recent should return emails when connected."""
        mock_email = AsyncMock()
        mock_email.get_recent.return_value = [
            {"from": "test@example.com", "subject": "Hello"},
        ]
        mock_email.get_tools.return_value = []

        await agent.connect_integration("email", mock_email)

        result = await agent._dispatch_tool("email_recent", {"limit": 5})
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["subject"] == "Hello"


# ---------------------------------------------------------------------------
# Tests: Integration connection
# ---------------------------------------------------------------------------


class TestIntegrationConnection:
    @pytest.mark.asyncio
    async def test_connect_integration_success(self, agent):
        """connect_integration should return True on success."""
        mock_integ = AsyncMock()
        result = await agent.connect_integration("obsidian", mock_integ)
        assert result is True
        assert "obsidian" in agent._integrations
        mock_integ.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_integration_failure(self, agent):
        """connect_integration should return False and store error on failure."""
        mock_integ = AsyncMock()
        mock_integ.connect.side_effect = ValueError("vault not found")

        result = await agent.connect_integration("obsidian", mock_integ)
        assert result is False
        assert "obsidian" not in agent._integrations
        assert "obsidian" in agent._integration_errors
        assert "vault not found" in agent._integration_errors["obsidian"]

    @pytest.mark.asyncio
    async def test_integration_status(self, agent):
        """get_integration_status should report all known integrations."""
        status = agent.get_integration_status()
        assert "obsidian" in status
        assert "calendar" in status
        assert "email" in status
        assert "computer" in status
        # All should be disconnected initially.
        for name, info in status.items():
            assert info["connected"] is False


# ---------------------------------------------------------------------------
# Tests: Model switching
# ---------------------------------------------------------------------------


class TestModelSwitching:
    @pytest.mark.asyncio
    async def test_switch_model(self, agent):
        """switch_model should update the current model."""
        new_backend = StubBackend("Switched!")
        agent._registry.get_backend.return_value = new_backend
        agent._registry.get_or_load = AsyncMock(return_value=new_backend)

        await agent.switch_model("minicpm5-1b-mlx")
        assert agent.get_current_model() == "minicpm5-1b-mlx"

    @pytest.mark.asyncio
    async def test_switch_model_unknown_raises(self, agent):
        """switch_model should raise ValueError for unknown models."""
        agent._registry.get_backend.side_effect = ValueError("Unknown model: fake")

        with pytest.raises(ValueError, match="Unknown model"):
            await agent.switch_model("fake-model")

    @pytest.mark.asyncio
    async def test_run_with_model_override(self, agent):
        """run() should accept a model parameter to override the default."""
        other_backend = StubBackend("From other model!")
        agent._registry.get_or_load = AsyncMock(return_value=other_backend)

        response = await agent.run("Hello", model="minicpm5-1b-mlx")
        assert response == "From other model!"
        # Should have called get_or_load with the override model.
        agent._registry.get_or_load.assert_called_with("minicpm5-1b-mlx")


# ---------------------------------------------------------------------------
# Tests: Tool call loop
# ---------------------------------------------------------------------------


class TestToolCallLoop:
    @pytest.mark.asyncio
    async def test_run_executes_tool_calls(self, tool_agent):
        """run() should detect tool calls, execute them, and continue."""
        response = await tool_agent.run("Check memory stats")
        # The tool agent calls memory_stats, then returns the final response.
        assert response == "Memory has 5 entries."
        # Should have recorded the interaction.
        assert tool_agent._collector._conn is not None

    @pytest.mark.asyncio
    async def test_execute_tools_returns_results(self, agent):
        """execute_tools should return structured results."""
        tool_calls = [{
            "id": "call_0",
            "type": "function",
            "function": {
                "name": "memory_stats",
                "arguments": "{}",
            },
        }]
        results = await agent.execute_tools(tool_calls)
        assert len(results) == 1
        assert results[0]["tool_call_id"] == "call_0"
        assert results[0]["name"] == "memory_stats"
        parsed = json.loads(results[0]["content"])
        assert "total_memories" in parsed

    @pytest.mark.asyncio
    async def test_execute_tools_handles_error(self, agent):
        """execute_tools should catch errors and return them as content."""
        tool_calls = [{
            "id": "call_bad",
            "type": "function",
            "function": {
                "name": "obsidian_search",
                "arguments": "INVALID_JSON{{{",
            },
        }]
        results = await agent.execute_tools(tool_calls)
        assert len(results) == 1
        # Should not crash -- error should be in the content.
        assert "请先配置 Obsidian" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_execute_tools_mixed(self, agent):
        """execute_tools should handle a mix of memory and unconfigured tools."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "memory_stats", "arguments": "{}"},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "email_search", "arguments": '{"query": "test"}'},
            },
        ]
        results = await agent.execute_tools(tool_calls)
        assert len(results) == 2
        # Memory tool succeeds.
        assert "total_memories" in results[0]["content"]
        # Email tool returns config error.
        assert "请先配置邮箱" in results[1]["content"]


# ---------------------------------------------------------------------------
# Tests: get_tools
# ---------------------------------------------------------------------------


class TestGetTools:
    def test_tools_include_memory(self, agent):
        """get_tools should always include memory tools."""
        tools = agent.get_tools()
        names = {t["function"]["name"] for t in tools}
        assert "memory_remember" in names
        assert "memory_recall" in names
        assert "memory_stats" in names

    @pytest.mark.asyncio
    async def test_tools_include_connected_integration(self, agent):
        """get_tools should include tools from connected integrations."""
        mock_obsidian = AsyncMock()
        # get_tools is a sync method on integrations -- use MagicMock for it.
        mock_obsidian.get_tools = MagicMock(return_value=[
            {"type": "function", "function": {"name": "obsidian_search", "parameters": {}}},
            {"type": "function", "function": {"name": "obsidian_read", "parameters": {}}},
        ])
        await agent.connect_integration("obsidian", mock_obsidian)

        tools = agent.get_tools()
        names = {t["function"]["name"] for t in tools}
        assert "obsidian_search" in names
        assert "obsidian_read" in names


# ---------------------------------------------------------------------------
# Tests: Context management
# ---------------------------------------------------------------------------


class TestContextManagement:
    def test_clear_context(self, agent):
        """clear_context should keep system prompt but remove messages."""
        agent._messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        agent.clear_context()
        assert len(agent._messages) == 1
        assert agent._messages[0]["role"] == "system"

    def test_clear_context_no_system(self, agent):
        """clear_context should handle empty message list."""
        agent._messages = []
        agent.clear_context()
        assert agent._messages == []


# ---------------------------------------------------------------------------
# Tests: Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_close_cleans_up(self, agent):
        """close() should disconnect all components."""
        await agent.close()
        # Memory should be disconnected.
        assert agent._memory._store._conn is None
        # Collector should be disconnected.
        assert agent._collector._conn is None

    @pytest.mark.asyncio
    async def test_context_manager(self, tmp_data_dir):
        """Agent should work as an async context manager."""
        with patch("server.agent.ModelRegistry") as MockRegistry, \
             patch("server.agent.EmbeddingClient") as MockEmbedding:

            mock_reg = MockRegistry.return_value
            backend = StubBackend()
            mock_reg.get_backend.return_value = backend
            mock_reg.get_or_load = AsyncMock(return_value=backend)

            mock_emb = MockEmbedding.return_value
            mock_emb.close = AsyncMock()

            from server.agent import Agent
            async with Agent(data_dir=tmp_data_dir) as a:
                a._registry = mock_reg
                a._embedding = mock_emb
                response = await a.run("test")
                assert response == "Hello!"

            # After exit, memory should be disconnected.
            assert a._memory._store._conn is None


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_run_records_on_success(self, agent):
        """Successful run should record an interaction."""
        await agent.run("Hello")
        examples = agent._collector.get_training_examples()
        assert len(examples) >= 1
        assert examples[0]["user_input"] == "Hello"

    @pytest.mark.asyncio
    async def test_tool_dispatch_json_error(self, agent):
        """Tool with invalid JSON arguments should not crash."""
        results = await agent.execute_tools([{
            "id": "call_x",
            "type": "function",
            "function": {"name": "memory_recall", "arguments": "not-json"},
        }])
        # Should still work -- args defaults to {}.
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_integration_tool_error_returns_message(self, agent):
        """Integration tool errors should be caught and returned as text."""
        mock_obsidian = AsyncMock()
        mock_obsidian.query.side_effect = RuntimeError("disk full")
        mock_obsidian.get_tools.return_value = []
        await agent.connect_integration("obsidian", mock_obsidian)

        result = await agent._dispatch_tool("obsidian_search", {"query": "test"})
        assert "工具调用失败" in result or "disk full" in result
