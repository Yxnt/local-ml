"""Tests for DarwinianEvolverAdapter.

Covers:
  1. EvolverConfig(enabled=False) -> evolve_candidates returns empty
  2. CLI missing (fake command) -> graceful skip
  3. Mock CLI that returns evolved code -> verify code goes through verifier
  4. Evolved tool registered as new version
  5. Original candidate NOT destroyed
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from server.tools.evolver_adapter import DarwinianEvolverAdapter, EvolverConfig
from server.tools.spec import (
    RiskLevel,
    ToolRuntime,
    ToolSpec,
    ToolStatus,
)
from server.tools.registry import ToolRegistry
from server.tools.telemetry import TelemetryService
from server.tools.verifier import ToolVerifier


# ---------------------------------------------------------------------------
# Sample tool source
# ---------------------------------------------------------------------------

SAMPLE_TOOL_SOURCE = textwrap.dedent('''\
    """A simple greeting tool."""

    from pydantic import BaseModel


    class InputModel(BaseModel):
        name: str

    class OutputModel(BaseModel):
        greeting: str

    def run(input: InputModel) -> OutputModel:
        return OutputModel(greeting=f"Hello, {input.name}!")
''')

EVOLVED_TOOL_SOURCE = textwrap.dedent('''\
    """An evolved greeting tool with time awareness."""

    from pydantic import BaseModel


    class InputModel(BaseModel):
        name: str
        formal: bool = False

    class OutputModel(BaseModel):
        greeting: str

    def run(input: InputModel) -> OutputModel:
        prefix = "Dear" if input.formal else "Hello"
        return OutputModel(greeting=f"{prefix}, {input.name}!")
''')


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def telemetry(tmp_path):
    svc = TelemetryService(db_path=str(tmp_path / "telemetry.db"))
    svc.connect()
    yield svc
    svc.disconnect()


@pytest.fixture()
def registry(tmp_path, telemetry):
    db_path = str(tmp_path / "registry.db")
    reg = ToolRegistry(db_path=db_path, telemetry=telemetry)
    reg.connect()
    yield reg
    reg.disconnect()


@pytest.fixture()
def sandbox_dir(tmp_path):
    d = tmp_path / "sandbox"
    d.mkdir()
    return str(d)


@pytest.fixture()
def verifier(sandbox_dir):
    return ToolVerifier(sandbox_dir=sandbox_dir)


def _register_candidate(registry: ToolRegistry, sandbox_dir: str) -> ToolSpec:
    """Register a L0 CANDIDATE tool with source in sandbox."""
    with open(f"{sandbox_dir}/greet_tool.py", "w") as f:
        f.write(SAMPLE_TOOL_SOURCE)

    spec = ToolSpec(
        name="greet_tool",
        description="A greeting tool",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        output_schema={
            "type": "object",
            "properties": {"greeting": {"type": "string"}},
        },
        runtime=ToolRuntime.PYTHON_GENERATED,
        provider="generated",
        risk_level=RiskLevel.L0,
        status=ToolStatus.CANDIDATE,
        version="1.0.0",
    )
    registry.register(spec)
    return spec


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_returns_empty(registry, verifier, telemetry):
    """EvolverConfig(enabled=False) -> evolve_candidates returns empty list."""
    config = EvolverConfig(enabled=False)
    adapter = DarwinianEvolverAdapter(
        config=config, registry=registry, verifier=verifier, telemetry=telemetry
    )

    results = await adapter.evolve_candidates(limit=5)
    assert results == []


@pytest.mark.asyncio
async def test_cli_missing_graceful_skip(registry, verifier, telemetry, sandbox_dir, tmp_path):
    """CLI not found on PATH -> graceful skip, returns empty list."""
    _register_candidate(registry, sandbox_dir)

    config = EvolverConfig(
        enabled=True,
        cli_command="nonexistent_evolver_command_xyz",
        work_dir=str(tmp_path / "evolver_work"),
    )
    adapter = DarwinianEvolverAdapter(
        config=config, registry=registry, verifier=verifier, telemetry=telemetry
    )

    results = await adapter.evolve_candidates(limit=5)
    assert results == []


@pytest.mark.asyncio
async def test_evolved_code_through_verifier(
    registry, verifier, telemetry, sandbox_dir, tmp_path
):
    """Mock CLI that returns evolved code -> verify code goes through verifier."""
    _register_candidate(registry, sandbox_dir)

    work_dir = str(tmp_path / "evolver_work")

    config = EvolverConfig(
        enabled=True,
        cli_command="echo",  # 'echo' exists on PATH
        work_dir=work_dir,
    )
    adapter = DarwinianEvolverAdapter(
        config=config, registry=registry, verifier=verifier, telemetry=telemetry
    )

    # Pre-create the output directory and write evolved code
    # The _read_evolved method checks: output/<tool>.py, evolved_<tool>.py, <tool>.py
    def _mock_run_evolver(wd: str) -> dict[str, Any]:
        # Write evolved code to the expected location
        output_dir = Path(wd) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "greet_tool.py").write_text(EVOLVED_TOOL_SOURCE, encoding="utf-8")
        return {
            "success": True,
            "best_code_path": str(output_dir / "greet_tool.py"),
            "score": 0.95,
            "error": None,
        }

    with patch.object(adapter, "_run_evolver", side_effect=_mock_run_evolver):
        results = await adapter.evolve_candidates(limit=5)

    assert len(results) == 1
    r = results[0]
    assert r["tool_name"] == "greet_tool"
    assert r["status"] == "evolved"


@pytest.mark.asyncio
async def test_evolved_tool_registered_as_new_version(
    registry, verifier, telemetry, sandbox_dir, tmp_path
):
    """Evolved tool gets registered as a new version (bumped patch)."""
    _register_candidate(registry, sandbox_dir)

    work_dir = str(tmp_path / "evolver_work")
    config = EvolverConfig(enabled=True, cli_command="echo", work_dir=work_dir)
    adapter = DarwinianEvolverAdapter(
        config=config, registry=registry, verifier=verifier, telemetry=telemetry
    )

    def _mock_run_evolver(wd: str) -> dict[str, Any]:
        output_dir = Path(wd) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "greet_tool.py").write_text(EVOLVED_TOOL_SOURCE, encoding="utf-8")
        return {"success": True, "best_code_path": str(output_dir / "greet_tool.py"), "score": 0.9, "error": None}

    with patch.object(adapter, "_run_evolver", side_effect=_mock_run_evolver):
        results = await adapter.evolve_candidates(limit=5)

    # New version registered
    evolved = registry.get_tool("greet_tool", version="1.0.1")
    assert evolved is not None
    assert evolved.status == ToolStatus.CANDIDATE
    assert "darwinian_evolved" in evolved.tags
    assert evolved.metadata.get("evolved_by") == "darwinian_evolver"
    assert evolved.metadata.get("evolved_from_version") == "1.0.0"


@pytest.mark.asyncio
async def test_original_candidate_not_destroyed(
    registry, verifier, telemetry, sandbox_dir, tmp_path
):
    """Original candidate tool survives evolution (not overwritten or removed)."""
    _register_candidate(registry, sandbox_dir)

    work_dir = str(tmp_path / "evolver_work")
    config = EvolverConfig(enabled=True, cli_command="echo", work_dir=work_dir)
    adapter = DarwinianEvolverAdapter(
        config=config, registry=registry, verifier=verifier, telemetry=telemetry
    )

    def _mock_run_evolver(wd: str) -> dict[str, Any]:
        output_dir = Path(wd) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "greet_tool.py").write_text(EVOLVED_TOOL_SOURCE, encoding="utf-8")
        return {"success": True, "best_code_path": str(output_dir / "greet_tool.py"), "score": 0.9, "error": None}

    with patch.object(adapter, "_run_evolver", side_effect=_mock_run_evolver):
        await adapter.evolve_candidates(limit=5)

    # Original still exists
    original = registry.get_tool("greet_tool", version="1.0.0")
    assert original is not None
    assert original.status == ToolStatus.CANDIDATE

    # Evolved also exists
    evolved = registry.get_tool("greet_tool", version="1.0.1")
    assert evolved is not None


@pytest.mark.asyncio
async def test_evolver_cli_failure(registry, verifier, telemetry, sandbox_dir, tmp_path):
    """CLI returning failure -> tool_name returned with status 'failed'."""
    _register_candidate(registry, sandbox_dir)

    work_dir = str(tmp_path / "evolver_work")
    config = EvolverConfig(enabled=True, cli_command="echo", work_dir=work_dir)
    adapter = DarwinianEvolverAdapter(
        config=config, registry=registry, verifier=verifier, telemetry=telemetry
    )

    def _mock_run_evolver(wd: str) -> dict[str, Any]:
        return {
            "success": False,
            "best_code_path": None,
            "score": None,
            "error": "evolver crashed",
        }

    with patch.object(adapter, "_run_evolver", side_effect=_mock_run_evolver):
        results = await adapter.evolve_candidates(limit=5)

    assert len(results) == 1
    assert results[0]["status"] == "failed"
    assert "evolver crashed" in results[0]["details"]


@pytest.mark.asyncio
async def test_no_evolved_code_found(registry, verifier, telemetry, sandbox_dir, tmp_path):
    """CLI succeeds but no evolved code found -> status 'failed'."""
    _register_candidate(registry, sandbox_dir)

    work_dir = str(tmp_path / "evolver_work")
    config = EvolverConfig(enabled=True, cli_command="echo", work_dir=work_dir)
    adapter = DarwinianEvolverAdapter(
        config=config, registry=registry, verifier=verifier, telemetry=telemetry
    )

    def _mock_run_evolver(wd: str) -> dict[str, Any]:
        # CLI says success but no files written
        return {"success": True, "best_code_path": None, "score": 0.5, "error": None}

    # Also mock _read_evolved to return None, since _export_candidate writes
    # the original source to <work_dir>/<tool_name>.py and _read_evolved
    # would find it there.
    with patch.object(adapter, "_run_evolver", side_effect=_mock_run_evolver), \
         patch.object(adapter, "_read_evolved", return_value=None):
        results = await adapter.evolve_candidates(limit=5)

    assert len(results) == 1
    assert results[0]["status"] == "failed"
    assert "no evolved code" in results[0]["details"]


@pytest.mark.asyncio
async def test_evolved_code_fails_verifier(
    registry, telemetry, sandbox_dir, tmp_path
):
    """Evolved code that fails verification -> status 'failed'."""
    _register_candidate(registry, sandbox_dir)

    # Verifier that rejects everything
    class RejectingVerifier:
        def verify(self, tool_name, source_code):
            class R:
                passed = False
                errors = ["Rejected by test verifier"]
            return R()

    work_dir = str(tmp_path / "evolver_work")
    config = EvolverConfig(enabled=True, cli_command="echo", work_dir=work_dir)
    adapter = DarwinianEvolverAdapter(
        config=config,
        registry=registry,
        verifier=RejectingVerifier(),
        telemetry=telemetry,
    )

    def _mock_run_evolver(wd: str) -> dict[str, Any]:
        output_dir = Path(wd) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "greet_tool.py").write_text(EVOLVED_TOOL_SOURCE, encoding="utf-8")
        return {"success": True, "best_code_path": str(output_dir / "greet_tool.py"), "score": 0.8, "error": None}

    with patch.object(adapter, "_run_evolver", side_effect=_mock_run_evolver):
        results = await adapter.evolve_candidates(limit=5)

    assert len(results) == 1
    assert results[0]["status"] == "failed"
    assert "verification failed" in results[0]["details"]


@pytest.mark.asyncio
async def test_only_l0_l1_candidates_evolved(
    registry, verifier, telemetry, sandbox_dir, tmp_path
):
    """Only L0/L1 risk level candidates are eligible for evolution."""
    # Register L0 candidate (eligible)
    with open(f"{sandbox_dir}/l0_tool.py", "w") as f:
        f.write(SAMPLE_TOOL_SOURCE)
    l0_spec = ToolSpec(
        name="l0_tool",
        description="L0 tool",
        input_schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        output_schema={"type": "object", "properties": {"greeting": {"type": "string"}}},
        runtime=ToolRuntime.PYTHON_GENERATED,
        provider="generated",
        risk_level=RiskLevel.L0,
        status=ToolStatus.CANDIDATE,
    )
    registry.register(l0_spec)

    # Register L3 candidate (NOT eligible)
    l3_spec = ToolSpec(
        name="l3_tool",
        description="L3 tool",
        input_schema={"type": "object", "properties": {}},
        runtime=ToolRuntime.PYTHON_GENERATED,
        provider="generated",
        risk_level=RiskLevel.L3,
        status=ToolStatus.CANDIDATE,
    )
    registry.register(l3_spec)

    config = EvolverConfig(enabled=True, cli_command="echo", work_dir=str(tmp_path / "work"))
    adapter = DarwinianEvolverAdapter(
        config=config, registry=registry, verifier=verifier, telemetry=telemetry
    )

    # Mock to capture which tools were processed
    processed_tools: list[str] = []

    async def _mock_evolve_one(spec):
        processed_tools.append(spec.name)
        return {"tool_name": spec.name, "status": "skipped", "details": "mock"}

    with patch.object(adapter, "_evolve_one", side_effect=_mock_evolve_one):
        await adapter.evolve_candidates(limit=10)

    # Only L0 tool processed
    assert "l0_tool" in processed_tools
    assert "l3_tool" not in processed_tools


def test_bump_version():
    """Version bumping works correctly."""
    assert DarwinianEvolverAdapter._bump_version("1.0.0") == "1.0.1"
    assert DarwinianEvolverAdapter._bump_version("2.3.9") == "2.3.10"
    assert DarwinianEvolverAdapter._bump_version("abc") == "abc.1"
