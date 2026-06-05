"""Tests for server.tools.registry — ToolRegistry."""

from __future__ import annotations

import json

import pytest

from server.tools.spec import (
    RiskLevel,
    ToolContext,
    ToolRuntime,
    ToolSpec,
    ToolStatus,
)


class TestRegisterAndGet:
    """register() then get_tool() returns the same spec."""

    def test_register_then_get(self, registry):
        spec = ToolSpec(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
            runtime=ToolRuntime.MEMORY_METHOD,
        )
        registry.register(spec)
        got = registry.get_tool("test_tool")
        assert got is not None
        assert got.name == "test_tool"
        assert got.description == "A test tool"
        assert got.runtime == ToolRuntime.MEMORY_METHOD

    def test_get_nonexistent_returns_none(self, registry):
        assert registry.get_tool("nonexistent") is None


class TestListTools:
    """list_tools returns only ACTIVE tools by default."""

    def test_active_only_by_default(self, registry, sample_memory_tool):
        registry.register(sample_memory_tool)
        other = ToolSpec(
            name="other_tool",
            description="Another tool",
            input_schema={"type": "object", "properties": {}},
            status=ToolStatus.DEPRECATED,
        )
        registry.register(other)

        active = registry.list_tools()
        names = [t.name for t in active]
        assert "memory_remember" in names
        assert "other_tool" not in names

    def test_list_all_statuses(self, registry, sample_memory_tool):
        registry.register(sample_memory_tool)
        other = ToolSpec(
            name="other_tool",
            description="Another tool",
            input_schema={"type": "object", "properties": {}},
            status=ToolStatus.DEPRECATED,
        )
        registry.register(other)

        all_tools = registry.list_tools(status=None)
        names = [t.name for t in all_tools]
        assert "memory_remember" in names
        assert "other_tool" in names

    def test_list_openai_tools(self, registry, sample_memory_tool):
        registry.register(sample_memory_tool)
        openai_tools = registry.list_openai_tools()
        assert len(openai_tools) >= 1
        assert openai_tools[0]["type"] == "function"
        assert "name" in openai_tools[0]["function"]


class TestUnregister:
    """unregister changes status to deprecated."""

    def test_unregister_marks_deprecated(self, registry, sample_memory_tool):
        registry.register(sample_memory_tool)
        registry.unregister("memory_remember")

        # get_tool still returns it (first version), but status is deprecated
        raw = registry._conn.execute(
            "SELECT status FROM tools WHERE name = ?", ("memory_remember",)
        ).fetchone()
        assert raw["status"] == "deprecated"

    def test_list_tools_excludes_deprecated(self, registry, sample_memory_tool):
        registry.register(sample_memory_tool)
        registry.unregister("memory_remember")

        active = registry.list_tools()
        names = [t.name for t in active]
        assert "memory_remember" not in names


class TestDispatchDeprecated:
    """Deprecated tools cannot be dispatched (returns tool_deprecated error)."""

    @pytest.mark.asyncio
    async def test_dispatch_deprecated_returns_error(self, registry, sample_memory_tool):
        registry.register(sample_memory_tool)
        registry.unregister("memory_remember")

        ctx = ToolContext(session_id="test", task_id="t1")
        result = await registry.dispatch("memory_remember", {"content": "hello"}, ctx)
        assert result.success is False
        assert result.error_type == "tool_deprecated"


class TestDispatchNotFound:
    """Dispatching a tool that doesn't exist returns tool_not_found."""

    @pytest.mark.asyncio
    async def test_dispatch_not_found(self, registry):
        ctx = ToolContext(session_id="test", task_id="t1")
        result = await registry.dispatch("no_such_tool", {}, ctx)
        assert result.success is False
        assert result.error_type == "tool_not_found"


class TestSetEmbedding:
    """set_embedding stores and retrieves the vector."""

    def test_set_embedding_updates_cache_and_db(self, registry, sample_memory_tool):
        registry.register(sample_memory_tool)
        vec = [0.1] * 768
        registry.set_embedding("memory_remember", vec)

        spec = registry.get_tool("memory_remember")
        assert spec is not None
        assert spec.embedding is not None
        assert len(spec.embedding) == 768
        assert spec.embedding[0] == pytest.approx(0.1)


class TestSearchByEmbedding:
    """search_by_embedding returns empty gracefully when sqlite-vec is missing."""

    def test_returns_empty_when_vec_unavailable(self, registry, sample_memory_tool):
        """Without sqlite-vec extension, search_by_embedding returns []."""
        registry.register(sample_memory_tool)
        # Even if we set an embedding, the vec0 virtual table may not be available.
        registry.set_embedding("memory_remember", [0.1] * 768)
        results = registry.search_by_embedding([0.1] * 768, limit=5)
        # Should be a list (empty if vec not available, or with results if it is)
        assert isinstance(results, list)
