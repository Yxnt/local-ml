"""Shared fixtures for server.tools tests.

All fixtures use tmp_path so no real databases are touched.
Imports are deferred to fixture bodies to avoid module cache conflicts
with tests/python/test_agent.py.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest

# Ensure project root is on sys.path so server.agent can import optimization.collector
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Telemetry / Registry
# ---------------------------------------------------------------------------


@pytest.fixture()
def telemetry(tmp_path):
    """TelemetryService backed by a temporary SQLite DB."""
    from server.tools.telemetry import TelemetryService

    db_path = str(tmp_path / "telemetry.db")
    svc = TelemetryService(db_path=db_path)
    svc.connect()
    yield svc
    svc.disconnect()


@pytest.fixture()
def registry(tmp_path, telemetry):
    """ToolRegistry backed by a temporary SQLite DB with a TelemetryService."""
    from server.tools.registry import ToolRegistry

    db_path = str(tmp_path / "registry.db")
    reg = ToolRegistry(db_path=db_path, telemetry=telemetry)
    reg.connect()
    yield reg
    reg.disconnect()


# ---------------------------------------------------------------------------
# Memory mock
# ---------------------------------------------------------------------------


class FakeMemoryManager:
    """Minimal stand-in for MemoryManager used by MemoryToolExecutor tests."""

    def __init__(self):
        self._store: list[dict[str, Any]] = []

    def remember(self, content, mtype, importance):
        mid = len(self._store) + 1
        self._store.append({"id": mid, "content": content, "type": str(mtype), "importance": importance})
        return mid

    class _InnerStore:
        @staticmethod
        def get_stats():
            return {"total": 0}

    store = _InnerStore()


@pytest.fixture()
def fake_memory_manager():
    return FakeMemoryManager()


# ---------------------------------------------------------------------------
# Sample ToolSpecs
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_memory_tool():
    """A tool spec representing memory_remember."""
    from server.tools.spec import ToolRuntime, ToolSpec

    return ToolSpec(
        name="memory_remember",
        description="Store a memory",
        input_schema={
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "type": {"type": "string", "enum": ["fact", "preference"]},
                "importance": {"type": "number"},
            },
            "required": ["content"],
        },
        runtime=ToolRuntime.MEMORY_METHOD,
        provider="memory",
    )


@pytest.fixture()
def sample_generated_tool():
    """A tool spec representing a python_generated tool."""
    from server.tools.spec import ToolRuntime, ToolSpec

    return ToolSpec(
        name="hello_tool",
        description="A simple generated tool",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        runtime=ToolRuntime.PYTHON_GENERATED,
        provider="generated",
    )


# ---------------------------------------------------------------------------
# Sandbox helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def sandbox_dir(tmp_path):
    """A temporary sandbox directory for generated tool files."""
    d = tmp_path / "sandbox"
    d.mkdir()
    return str(d)


@pytest.fixture()
def verifier(sandbox_dir):
    """ToolVerifier pointing at the temporary sandbox."""
    from server.tools.verifier import ToolVerifier

    return ToolVerifier(sandbox_dir=sandbox_dir)
