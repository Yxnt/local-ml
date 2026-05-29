"""Tests for server.remote_analyzer."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# Ensure project root is on sys.path.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from server.remote_analyzer import Correction, RemoteAnalyzer, RemoteFeedback, SanitizedContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_model():
    """Provide a mock model with generate_async."""
    model = AsyncMock()
    model.generate_async = AsyncMock(return_value=json.dumps({
        "is_correct": True,
        "confidence": 0.95,
        "reasoning": "The local answer is factually accurate.",
        "better_answer": "",
        "logic_rule": "When asked about country capitals, the answer should be the official capital city.",
        "rule_type": "geography",
        "corrections": [],
    }))
    return model


@pytest.fixture()
def analyzer(mock_model):
    """Provide a RemoteAnalyzer with a mock model."""
    return RemoteAnalyzer(model=mock_model)


@pytest.fixture()
def sample_context():
    """Provide a sample SanitizedContext."""
    return SanitizedContext(
        user_input="What is the capital of France?",
        history=[{"role": "user", "content": "Hello"}],
        entities=["France", "Paris"],
        local_answer="Paris is the capital of France.",
        confidence=0.85,
        reasoning="Well-known geographical fact.",
        rules_applied=["geography"],
        timestamp="2026-05-28T10:00:00Z",
    )


# ---------------------------------------------------------------------------
# 1. test_analyze_returns_feedback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_analyze_returns_feedback(analyzer: RemoteAnalyzer, sample_context: SanitizedContext):
    """analyze() should return a RemoteFeedback with all expected fields."""
    feedback = await analyzer.analyze(sample_context)

    assert isinstance(feedback, RemoteFeedback)
    assert feedback.is_correct is True
    assert feedback.confidence == 0.95
    assert feedback.reasoning == "The local answer is factually accurate."
    assert feedback.better_answer == ""
    assert "capitals" in feedback.logic_rule
    assert feedback.rule_type == "geography"
    assert isinstance(feedback.corrections, list)
    assert len(feedback.corrections) == 0


# ---------------------------------------------------------------------------
# 2. test_analyze_handles_invalid_json
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_analyze_handles_invalid_json():
    """analyze() should gracefully handle invalid JSON from the remote model."""
    model = AsyncMock()
    model.generate_async = AsyncMock(return_value="This is not valid JSON at all.")

    analyzer = RemoteAnalyzer(model=model)
    context = SanitizedContext(user_input="test", local_answer="answer")

    feedback = await analyzer.analyze(context)

    assert isinstance(feedback, RemoteFeedback)
    assert feedback.is_correct is False
    assert feedback.confidence == 0.0
    assert "Failed to parse feedback" in feedback.reasoning
    assert feedback.logic_rule == ""
    assert feedback.corrections == []


# ---------------------------------------------------------------------------
# 3. test_analyze_extracts_rule
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_analyze_extracts_rule():
    """analyze() should correctly extract the logic rule and rule type from feedback."""
    model = AsyncMock()
    model.generate_async = AsyncMock(return_value=json.dumps({
        "is_correct": False,
        "confidence": 0.9,
        "reasoning": "The local answer used a wrong date format.",
        "better_answer": "The event is on 2026-01-15.",
        "logic_rule": "When dates appear in answers, prefer ISO 8601 format (YYYY-MM-DD).",
        "rule_type": "date_format",
        "corrections": [
            {
                "field": "answer",
                "original": "January 15th",
                "corrected": "2026-01-15",
                "reason": "ISO 8601 is unambiguous",
            }
        ],
    }))

    analyzer = RemoteAnalyzer(model=model)
    context = SanitizedContext(
        user_input="When is the meeting?",
        local_answer="The meeting is on January 15th.",
        confidence=0.6,
    )

    feedback = await analyzer.analyze(context)

    assert feedback.is_correct is False
    assert "ISO 8601" in feedback.logic_rule
    assert feedback.rule_type == "date_format"
    assert len(feedback.corrections) == 1
    assert feedback.corrections[0].field == "answer"
    assert feedback.corrections[0].original == "January 15th"
    assert feedback.corrections[0].corrected == "2026-01-15"
    assert feedback.corrections[0].reason == "ISO 8601 is unambiguous"
    assert feedback.better_answer == "The event is on 2026-01-15."
