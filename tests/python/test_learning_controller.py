"""Tests for server.learning_controller."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Ensure project root is on sys.path.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from server.learning_controller import LearningController
from server.local_processor import LocalResult
from server.remote_analyzer import RemoteAnalyzer, RemoteFeedback, SanitizedContext
from server.rule_manager import RuleManager
from server.sanitizer import SanitizedText, SensitivityLevel
from server.session_sanitizer import SessionSanitizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_remote_analyzer():
    """Provide a mock RemoteAnalyzer that returns high-confidence feedback."""
    analyzer = AsyncMock(spec=RemoteAnalyzer)
    analyzer.analyze = AsyncMock(return_value=RemoteFeedback(
        is_correct=False,
        confidence=0.85,
        reasoning="The local answer could be improved.",
        better_answer="A better answer.",
        logic_rule="When answering geography questions, always use official country names.",
        rule_type="geography",
    ))
    return analyzer


@pytest.fixture()
def mock_rule_manager():
    """Provide a mock RuleManager."""
    rm = MagicMock(spec=RuleManager)
    rm.add_rule = MagicMock(return_value=42)
    return rm


@pytest.fixture()
def mock_session_sanitizer():
    """Provide a mock SessionSanitizer that passes text through."""
    sanitizer = MagicMock(spec=SessionSanitizer)
    sanitizer.sanitize = MagicMock(side_effect=lambda text: SanitizedText(
        original=text,
        sanitized=text,
        mapping={},
        level=SensitivityLevel.PUBLIC,
    ))
    return sanitizer


@pytest.fixture()
def controller(mock_remote_analyzer, mock_rule_manager, mock_session_sanitizer):
    """Provide a LearningController with mocked dependencies."""
    return LearningController(
        remote_analyzer=mock_remote_analyzer,
        rule_manager=mock_rule_manager,
        session_sanitizer=mock_session_sanitizer,
        hourly_limit=10,
        daily_limit=100,
    )


@pytest.fixture()
def sample_result():
    """Provide a sample LocalResult."""
    return LocalResult(
        answer="Paris is the capital of France.",
        confidence=0.85,
        reasoning="Well-known geographical fact.",
        rules_applied=["geography"],
    )


# ---------------------------------------------------------------------------
# 1. test_learn_stores_rule
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_learn_stores_rule(
    controller: LearningController,
    mock_rule_manager: MagicMock,
    sample_result: LocalResult,
):
    """learn() should store a rule when remote feedback confidence >= 0.7."""
    result = await controller.learn("What is the capital of France?", sample_result)

    assert result is True
    mock_rule_manager.add_rule.assert_called_once_with(
        rule_type="geography",
        logic="When answering geography questions, always use official country names.",
        confidence=0.85,
        source="remote_feedback",
    )


# ---------------------------------------------------------------------------
# 2. test_budget_blocks_learning
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_budget_blocks_learning(
    controller: LearningController,
    mock_remote_analyzer: AsyncMock,
    mock_rule_manager: MagicMock,
    sample_result: LocalResult,
):
    """learn() should skip learning when the hourly budget is exhausted."""
    # Exhaust the hourly budget.
    controller._hourly.count = controller._hourly_limit

    result = await controller.learn("What is the capital of France?", sample_result)

    assert result is False
    mock_rule_manager.add_rule.assert_not_called()
    mock_remote_analyzer.analyze.assert_not_called()


# ---------------------------------------------------------------------------
# 3. test_duplicate_learning_skipped
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_duplicate_learning_skipped(
    controller: LearningController,
    mock_remote_analyzer: AsyncMock,
    mock_rule_manager: MagicMock,
    sample_result: LocalResult,
):
    """learn() should skip if the same input is already being learned."""
    # Manually mark the key as pending.
    key = controller._make_key("What is the capital of France?")
    controller._pending[key] = True

    result = await controller.learn("What is the capital of France?", sample_result)

    assert result is False
    mock_rule_manager.add_rule.assert_not_called()
    mock_remote_analyzer.analyze.assert_not_called()
