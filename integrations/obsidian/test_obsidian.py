"""Tests for Obsidian integration."""

import pytest
import tempfile
from pathlib import Path

from integrations.base import IntegrationConfig
from integrations.obsidian import ObsidianIntegration


@pytest.fixture
def vault_dir(tmp_path):
    """Create a temporary vault with test notes."""
    vault = tmp_path / "vault"
    vault.mkdir()

    # Create a note with frontmatter
    note1 = vault / "test-note.md"
    note1.write_text("""---
tags: [test, demo]
created: 2024-01-01
---

# Test Note

This is a test note for unit testing.
""")

    # Create a note in subdirectory
    subdir = vault / "daily"
    subdir.mkdir()
    note2 = subdir / "2024-01-01.md"
    note2.write_text("""# Daily Note

- Did some coding
- Read a book
""")

    return vault


@pytest.mark.asyncio
async def test_connect(vault_dir):
    config = IntegrationConfig(name="obsidian", config={"vault_path": str(vault_dir)})
    integration = ObsidianIntegration(config)
    await integration.connect()
    assert integration._connected


@pytest.mark.asyncio
async def test_sync(vault_dir):
    config = IntegrationConfig(name="obsidian", config={"vault_path": str(vault_dir)})
    integration = ObsidianIntegration(config)
    await integration.connect()

    stats = await integration.sync()
    assert stats["total"] == 2
    assert stats["new"] == 2


@pytest.mark.asyncio
async def test_query(vault_dir):
    config = IntegrationConfig(name="obsidian", config={"vault_path": str(vault_dir)})
    integration = ObsidianIntegration(config)
    await integration.connect()
    await integration.sync()

    results = await integration.query("coding")
    assert len(results) > 0
    assert any("coding" in r.get("snippet", "") for r in results)


@pytest.mark.asyncio
async def test_read_note(vault_dir):
    config = IntegrationConfig(name="obsidian", config={"vault_path": str(vault_dir)})
    integration = ObsidianIntegration(config)
    await integration.connect()

    note = await integration.read_note("test-note.md")
    assert note is not None
    assert "Test Note" in note["content"]
    assert "test" in note["metadata"].get("tags", [])


def test_get_tools():
    config = IntegrationConfig(name="obsidian", config={})
    integration = ObsidianIntegration(config)
    tools = integration.get_tools()
    assert len(tools) == 2
    assert tools[0]["function"]["name"] == "obsidian_search"
    assert tools[1]["function"]["name"] == "obsidian_read"
