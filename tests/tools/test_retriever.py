"""Tests for server.tools.retriever — ToolRetriever."""

from __future__ import annotations

import pytest

from server.tools.spec import ToolRuntime, ToolSpec, ToolStatus
from server.tools.retriever import ToolRetriever


@pytest.fixture()
def retriever(registry):
    """ToolRetriever with no embedding client (keyword-only)."""
    return ToolRetriever(registry=registry, embedding_client=None)


@pytest.fixture()
def populated_registry(registry):
    """Register several tools for retrieval tests."""
    tools = [
        ToolSpec(
            name="memory_remember",
            description="Store a long-term memory about the user or facts",
            input_schema={"type": "object", "properties": {}},
            runtime=ToolRuntime.MEMORY_METHOD,
            provider="memory",
            tags=["memory", "storage"],
        ),
        ToolSpec(
            name="obsidian_search",
            description="Search Obsidian vault notes by keyword",
            input_schema={"type": "object", "properties": {}},
            runtime=ToolRuntime.INTEGRATION_METHOD,
            provider="obsidian",
            tags=["search", "notes"],
        ),
        ToolSpec(
            name="calendar_search",
            description="Search calendar events",
            input_schema={"type": "object", "properties": {}},
            runtime=ToolRuntime.INTEGRATION_METHOD,
            provider="calendar",
            tags=["calendar", "events"],
        ),
        ToolSpec(
            name="weather_check",
            description="Check current weather for a location",
            input_schema={"type": "object", "properties": {}},
            runtime=ToolRuntime.INTEGRATION_METHOD,
            provider="weather",
            tags=["weather"],
        ),
    ]
    for t in tools:
        registry.register(t)
    return registry


class TestKeywordFallback:
    """keyword fallback returns matching tools."""

    @pytest.mark.asyncio
    async def test_keyword_match_by_name(self, populated_registry, retriever):
        results = await retriever.retrieve("memory", limit=5)
        names = [t.name for t in results]
        assert "memory_remember" in names

    @pytest.mark.asyncio
    async def test_keyword_match_by_description(self, populated_registry, retriever):
        results = await retriever.retrieve("calendar events", limit=5)
        names = [t.name for t in results]
        assert "calendar_search" in names

    @pytest.mark.asyncio
    async def test_keyword_match_by_tag(self, populated_registry, retriever):
        results = await retriever.retrieve("notes", limit=5)
        names = [t.name for t in results]
        assert "obsidian_search" in names

    @pytest.mark.asyncio
    async def test_keyword_match_by_provider(self, populated_registry, retriever):
        results = await retriever.retrieve("obsidian", limit=5)
        names = [t.name for t in results]
        assert "obsidian_search" in names


class TestNoMatch:
    """retrieve returns empty gracefully when no tools match."""

    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self, populated_registry, retriever):
        results = await retriever.retrieve("zzz_nonexistent_xyz", limit=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_empty_registry_returns_empty(self, registry, retriever):
        results = await retriever.retrieve("anything", limit=5)
        assert results == []
