"""Tests for server.hybrid_agent."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure project root is on sys.path.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from server.hybrid_agent import AgentResponse, HybridAgent
from server.local_processor import LocalResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_memory_manager():
    """Provide a mock MemoryManager."""
    mm = MagicMock()
    mm.connect = MagicMock()
    mm.disconnect = MagicMock()
    mm.store = MagicMock()
    mm.store.add_conversation = MagicMock()
    mm.recall = MagicMock(return_value=[])
    mm.remember = MagicMock(return_value=1)
    return mm


@pytest.fixture()
def mock_registry():
    """Provide a mock ModelRegistry."""
    registry = MagicMock()
    registry.register_defaults = MagicMock()

    mock_backend = MagicMock()
    mock_backend.generate = MagicMock(return_value="mock response")

    async def _get_or_load(name):
        return mock_backend

    registry.get_or_load = AsyncMock(side_effect=_get_or_load)
    return registry


@pytest.fixture()
def sample_result():
    """Provide a sample LocalResult with high confidence."""
    return LocalResult(
        answer="Test answer",
        confidence=0.95,
        reasoning="High confidence result.",
        rules_applied=["test_rule"],
    )


@pytest.fixture()
def low_confidence_result():
    """Provide a sample LocalResult with low confidence."""
    return LocalResult(
        answer="Uncertain answer",
        confidence=0.5,
        reasoning="Low confidence result.",
        rules_applied=["test_rule"],
    )


def _make_agent(
    mock_memory_manager: MagicMock,
    mock_registry: MagicMock,
    remote_enabled: bool = False,
) -> HybridAgent:
    """Build a HybridAgent with mocked dependencies via patching."""
    config = {"remote": {"enabled": remote_enabled}, "models": {"default": "test-model"}}

    with (
        patch("server.hybrid_agent.ModelRegistry", return_value=mock_registry),
        patch("server.hybrid_agent.MemoryManager", return_value=mock_memory_manager),
        patch("server.hybrid_agent.RuleManager") as MockRM,
        patch("server.hybrid_agent.SessionSanitizer") as MockSS,
        patch("server.hybrid_agent.LocalProcessor") as MockLP,
    ):
        MockRM.return_value = MagicMock()
        MockSS.return_value = MagicMock()

        agent = HybridAgent(config=config, data_dir="/tmp/test_hybrid")
        # Replace the local processor with a controllable mock.
        agent._local_processor = MagicMock()
        agent._local_processor.process = AsyncMock()
        # Replace rule_manager mock.
        agent._rule_manager = MagicMock()
        agent._rule_manager.get_relevant_rules = MagicMock(return_value=[])
        agent._rule_manager.update_rule_stats = MagicMock()
        return agent


# ---------------------------------------------------------------------------
# 1. test_run_returns_agent_response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_agent_response(
    mock_memory_manager: MagicMock,
    mock_registry: MagicMock,
    sample_result: LocalResult,
):
    """run() should return an AgentResponse with the local result's answer and confidence."""
    agent = _make_agent(mock_memory_manager, mock_registry, remote_enabled=False)
    agent._local_processor.process = AsyncMock(return_value=sample_result)

    response = await agent.run("Hello")

    assert isinstance(response, AgentResponse)
    assert response.answer == "Test answer"
    assert response.confidence == 0.95
    assert response.is_learning is False
    assert response.learning_task is None


# ---------------------------------------------------------------------------
# 2. test_followup_question_detected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_followup_question_detected(
    mock_memory_manager: MagicMock,
    mock_registry: MagicMock,
    low_confidence_result: LocalResult,
):
    """run() should detect follow-up questions and wait for learning before reprocessing."""
    agent = _make_agent(mock_memory_manager, mock_registry, remote_enabled=True)

    # Set up a mock learning controller.
    mock_lc = MagicMock()
    mock_lc.learn = AsyncMock(return_value=True)
    agent._learning_controller = mock_lc

    # First call returns low confidence; after learning, second call returns high confidence.
    high_confidence_result = LocalResult(
        answer="Better answer after learning",
        confidence=0.95,
        reasoning="Improved result.",
        rules_applied=[],
    )
    agent._local_processor.process = AsyncMock(
        side_effect=[low_confidence_result, high_confidence_result]
    )

    # "她的名字是什么" contains "她" -- a follow-up pronoun.
    response = await agent.run("她的名字是什么")

    assert isinstance(response, AgentResponse)
    assert response.answer == "Better answer after learning"
    assert response.is_learning is False  # Learning completed synchronously for follow-ups.

    # Learning should have been called.
    mock_lc.learn.assert_called_once()
    # Processor should have been called twice (initial + reprocess).
    assert agent._local_processor.process.call_count == 2


# ---------------------------------------------------------------------------
# 3. test_run_with_progress_callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_with_progress_callback(
    mock_memory_manager: MagicMock,
    mock_registry: MagicMock,
    sample_result: LocalResult,
):
    """run_with_progress() should invoke the callback at key stages and return the answer."""
    agent = _make_agent(mock_memory_manager, mock_registry, remote_enabled=False)
    agent._local_processor.process = AsyncMock(return_value=sample_result)

    stages: list[str] = []

    def on_progress(stage: str) -> None:
        stages.append(stage)

    answer = await agent.run_with_progress("Hello", on_progress)

    assert answer == "Test answer"
    assert "processing" in stages
    assert "done" in stages
