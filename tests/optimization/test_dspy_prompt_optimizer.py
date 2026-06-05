"""Tests for DSPyPromptOptimizer (dspy prompt optimizer).

Covers:
  1. DSPy not installed -> graceful skip (returns empty)
  2. DSPy installed but GEPA unavailable -> falls back to BootstrapFewShot
  3. dry_run=True does NOT persist to PromptStore
  4. Mock LM -> verify optimization produces a result
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from server.optimization.prompt_store import PromptStore
from server.optimization.trace_dataset import TraceDatasetBuilder
from server.optimization.gepa_tool_evolution import DSPyPromptOptimizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def prompt_store(tmp_path):
    db_path = str(tmp_path / "prompts.db")
    s = PromptStore(db_path=db_path)
    s.connect()
    yield s
    s.disconnect()


@pytest.fixture()
def telemetry(tmp_path):
    from server.tools.telemetry import TelemetryService

    svc = TelemetryService(db_path=str(tmp_path / "telemetry.db"))
    svc.connect()
    yield svc
    svc.disconnect()


@pytest.fixture()
def trace_builder(telemetry):
    return TraceDatasetBuilder(telemetry=telemetry)


def _seed_training_data(telemetry):
    """Seed telemetry with enough data to build training examples."""
    # Tool selection events
    for i in range(10):
        task_id = f"task-{i}"
        tool_name = "bash" if i % 2 == 0 else "search"
        telemetry.record(
            "tool_invoked",
            tool_name=tool_name,
            task_id=task_id,
            session_id=f"sess-{i}",
            metadata={"user_input": f"do something {i}"},
        )
        telemetry.record(
            "tool_succeeded" if i % 3 != 0 else "tool_failed",
            tool_name=tool_name,
            task_id=task_id,
            session_id=f"sess-{i}",
        )

    # Tool requests
    conn = telemetry._conn
    assert conn is not None
    for i in range(5):
        conn.execute(
            """
            INSERT INTO tool_requests
                (task_id, session_id, reason, missing_capability,
                 candidate_name, candidate_desc, candidate_input, candidate_output,
                 risk_level, privacy_notes, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"task-req-{i}", f"sess-{i}", f"reason {i}",
                f"capability_{i}", f"tool_{i}", f"description_{i}",
                '{}', '{}', "L0", "", "{}",
                f"2025-01-01T0{i}:00:00+00:00",
            ),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Test: DSPy not installed
# ---------------------------------------------------------------------------


class TestDSPyNotInstalled:
    """When DSPy is not installed, optimizer returns empty result."""

    def test_graceful_skip_when_no_dspy(self, prompt_store, trace_builder):
        """Returns empty dict when HAS_DSPY is False."""
        optimizer = DSPyPromptOptimizer(
            prompt_store=prompt_store,
            trace_builder=trace_builder,
            lm_model="openai/gpt-4o-mini",
        )

        with patch("server.optimization.gepa_tool_evolution.HAS_DSPY", False):
            result = optimizer.optimize(target="manager", dry_run=True)

        assert result == {}

    def test_no_persistence_when_no_dspy(self, prompt_store, trace_builder):
        """Nothing persisted to store when DSPy unavailable."""
        optimizer = DSPyPromptOptimizer(
            prompt_store=prompt_store,
            trace_builder=trace_builder,
        )

        with patch("server.optimization.gepa_tool_evolution.HAS_DSPY", False):
            optimizer.optimize(target="manager", dry_run=False)

        active = prompt_store.get_active("tool_manager_prompt")
        assert active is None


# ---------------------------------------------------------------------------
# Test: DSPy installed, no API key
# ---------------------------------------------------------------------------


class TestNoAPIKey:
    """When DSPy is installed but no API key -> returns empty."""

    def test_no_api_key_returns_empty(self, prompt_store, trace_builder, telemetry):
        """Returns empty dict when no API key is found."""
        _seed_training_data(telemetry)

        optimizer = DSPyPromptOptimizer(
            prompt_store=prompt_store,
            trace_builder=trace_builder,
        )

        # Clear all API keys
        env_patch = {
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "DATABRICKS_TOKEN": "",
        }
        with patch("server.optimization.gepa_tool_evolution.HAS_DSPY", True):
            with patch.dict(os.environ, env_patch, clear=False):
                result = optimizer.optimize(target="manager", dry_run=True)

        assert result == {}


# ---------------------------------------------------------------------------
# Test: dry_run does NOT persist
# ---------------------------------------------------------------------------


class TestDryRunNoPersist:
    """dry_run=True does NOT write to PromptStore."""

    def test_dry_run_no_persist(self, prompt_store, trace_builder, telemetry):
        """Optimization with dry_run=True should not create any store entries."""
        _seed_training_data(telemetry)

        optimizer = DSPyPromptOptimizer(
            prompt_store=prompt_store,
            trace_builder=trace_builder,
        )

        # Mock HAS_DSPY and API key so optimization proceeds
        with patch("server.optimization.gepa_tool_evolution.HAS_DSPY", True):
            with patch.object(optimizer, "_has_api_key", return_value=True):
                with patch.object(optimizer, "_build_examples") as mock_build:
                    # Provide enough examples
                    mock_build.return_value = [
                        {"task": f"task {i}", "available_tools": "bash, search",
                         "selected_tools": "bash", "outcome": "tool_succeeded",
                         "score": 1.0, "risk_level": "L0"}
                        for i in range(10)
                    ]
                    with patch.object(optimizer, "_run_dspy_optimization") as mock_opt:
                        mock_opt.return_value = {
                            "best_prompt": "optimized prompt text",
                            "score": 0.85,
                            "eval_lines": ["Method: test"],
                        }
                        result = optimizer.optimize(target="manager", dry_run=True)

        assert result["best_prompt"] == "optimized prompt text"
        assert result["score"] == 0.85

        # Verify nothing persisted
        active = prompt_store.get_active("tool_manager_prompt")
        assert active is None
        versions = prompt_store.list_versions("tool_manager_prompt")
        assert len(versions) == 0


# ---------------------------------------------------------------------------
# Test: Mock LM -> optimization produces result
# ---------------------------------------------------------------------------


class TestMockLM:
    """With mocked LM, optimization produces a valid result."""

    def test_optimization_produces_result(self, prompt_store, trace_builder, telemetry):
        """Mocked DSPy optimization returns a result with best_prompt and score."""
        _seed_training_data(telemetry)

        optimizer = DSPyPromptOptimizer(
            prompt_store=prompt_store,
            trace_builder=trace_builder,
        )

        with patch("server.optimization.gepa_tool_evolution.HAS_DSPY", True):
            with patch.object(optimizer, "_has_api_key", return_value=True):
                with patch.object(optimizer, "_build_examples") as mock_build:
                    mock_build.return_value = [
                        {"task": f"task {i}", "available_tools": "bash, search",
                         "selected_tools": "bash", "outcome": "tool_succeeded",
                         "score": 1.0, "risk_level": "L0"}
                        for i in range(10)
                    ]
                    with patch.object(optimizer, "_run_dspy_optimization") as mock_opt:
                        mock_opt.return_value = {
                            "best_prompt": "You are an optimized manager. Select tools wisely.",
                            "score": 0.92,
                            "eval_lines": ["Method: BootstrapFewShot", "Val avg score: 0.920"],
                        }
                        result = optimizer.optimize(target="manager", dry_run=False)

        assert result["best_prompt"] != ""
        assert result["score"] > 0
        assert "eval_summary" in result

    def test_persists_as_candidate(self, prompt_store, trace_builder, telemetry):
        """Optimization with dry_run=False persists as candidate."""
        _seed_training_data(telemetry)

        optimizer = DSPyPromptOptimizer(
            prompt_store=prompt_store,
            trace_builder=trace_builder,
        )

        with patch("server.optimization.gepa_tool_evolution.HAS_DSPY", True):
            with patch.object(optimizer, "_has_api_key", return_value=True):
                with patch.object(optimizer, "_build_examples") as mock_build:
                    mock_build.return_value = [
                        {"task": f"task {i}", "available_tools": "bash",
                         "selected_tools": "bash", "outcome": "tool_succeeded",
                         "score": 1.0, "risk_level": "L0"}
                        for i in range(10)
                    ]
                    with patch.object(optimizer, "_run_dspy_optimization") as mock_opt:
                        mock_opt.return_value = {
                            "best_prompt": "optimized developer instructions",
                            "score": 0.88,
                            "eval_lines": ["test"],
                        }
                        optimizer.optimize(target="developer", dry_run=False)

        # Should be saved as candidate, NOT active
        versions = prompt_store.list_versions("tool_developer_prompt")
        assert len(versions) == 1
        assert versions[0].status == "candidate"
        assert versions[0].optimizer == "dspy"
        assert versions[0].score == 0.88

        # Not automatically active
        active = prompt_store.get_active("tool_developer_prompt")
        assert active is None


# ---------------------------------------------------------------------------
# Test: Fallback to manual optimization
# ---------------------------------------------------------------------------


class TestManualFallback:
    """When DSPy optimization fails, falls back to manual loop."""

    def test_fallback_to_manual(self, prompt_store, trace_builder, telemetry):
        """DSPy failure triggers manual fallback optimization."""
        _seed_training_data(telemetry)

        optimizer = DSPyPromptOptimizer(
            prompt_store=prompt_store,
            trace_builder=trace_builder,
        )

        with patch("server.optimization.gepa_tool_evolution.HAS_DSPY", True):
            with patch.object(optimizer, "_has_api_key", return_value=True):
                with patch.object(optimizer, "_build_examples") as mock_build:
                    mock_build.return_value = [
                        {"task": f"task {i}", "available_tools": "bash",
                         "selected_tools": "bash", "outcome": "tool_succeeded",
                         "score": 1.0, "risk_level": "L0",
                         "rationale": "bash is good for this"}
                        for i in range(10)
                    ]
                    # DSPy optimization fails
                    with patch.object(
                        optimizer, "_run_dspy_optimization", side_effect=Exception("DSPy failed")
                    ):
                        result = optimizer.optimize(target="manager", dry_run=True)

        # Manual fallback should still produce a result
        assert result["best_prompt"] != ""
        assert result["score"] >= 0


# ---------------------------------------------------------------------------
# Test: Invalid target
# ---------------------------------------------------------------------------


class TestInvalidTarget:
    """Unknown optimization target returns empty."""

    def test_unknown_target(self, prompt_store, trace_builder):
        optimizer = DSPyPromptOptimizer(
            prompt_store=prompt_store,
            trace_builder=trace_builder,
        )

        result = optimizer.optimize(target="nonexistent_target", dry_run=True)
        assert result == {}
