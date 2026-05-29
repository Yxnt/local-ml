"""Tests for server.local_processor."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Ensure project root is on sys.path.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from memory.manager import MemoryManager
from server.local_processor import Entity, LocalProcessor, LocalResult, TimeRange


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def memory_manager(tmp_path):
    """Provide a MemoryManager backed by an ephemeral database."""
    data_dir = str(tmp_path / "memory_data")
    with MemoryManager(data_dir=data_dir) as mm:
        yield mm


@pytest.fixture()
def mock_model():
    """Provide a mock model with generate_async."""
    model = AsyncMock()
    model.generate_async = AsyncMock(return_value=json.dumps({
        "answer": "Paris is the capital of France.",
        "reasoning": "Well-known geographical fact.",
        "entities_found": [
            {"name": "Paris", "type": "place", "confidence": 0.95, "source": "knowledge"},
        ],
        "pronouns_resolved": [],
        "time_parsed": None,
        "ambiguity_detected": False,
    }))
    return model


@pytest.fixture()
def processor(memory_manager, mock_model):
    """Provide a LocalProcessor with real memory and mock model."""
    return LocalProcessor(
        memory_manager=memory_manager,
        model=mock_model,
        confidence_threshold=0.9,
        session_id="test-session",
    )


# ---------------------------------------------------------------------------
# 1. test_process_returns_local_result
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_process_returns_local_result(processor: LocalProcessor):
    """process() should return a LocalResult with all expected fields."""
    result = await processor.process("What is the capital of France?")

    assert isinstance(result, LocalResult)
    assert result.answer == "Paris is the capital of France."
    assert result.reasoning == "Well-known geographical fact."
    assert isinstance(result.entities_found, list)
    assert isinstance(result.pronouns_resolved, list)
    assert isinstance(result.rules_applied, list)
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# 2. test_process_with_rules
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_process_with_rules(memory_manager, mock_model):
    """When rules exist, they should be applied and reflected in the result."""
    from server.rule_manager import RuleManager

    rm = RuleManager(memory_manager)
    rm.add_rule(
        rule_type="geography",
        logic="France is in Europe; its capital is Paris",
        confidence=0.95,
        source="user_taught",
        pattern="capital of France",
    )

    processor = LocalProcessor(
        memory_manager=memory_manager,
        model=mock_model,
        confidence_threshold=0.9,
        session_id="test-session",
    )

    result = await processor.process("What is the capital of France?")

    assert "geography" in result.rules_applied
    assert result.confidence > 0.0


# ---------------------------------------------------------------------------
# 3. test_extract_pronoun_pattern
# ---------------------------------------------------------------------------

def test_extract_pronoun_pattern(processor: LocalProcessor):
    """_extract_success_pattern should return 'pronoun_resolution' when pronouns were resolved."""
    result = LocalResult(
        answer="He is a developer.",
        pronouns_resolved=["he"],
        entities_found=[],
        time_parsed=None,
    )
    pattern = processor._extract_success_pattern("Who is he?", result)
    assert pattern == "pronoun_resolution"


# ---------------------------------------------------------------------------
# 4. test_parse_response_json
# ---------------------------------------------------------------------------

def test_parse_response_json(processor: LocalProcessor):
    """_parse_response should correctly parse valid JSON into a LocalResult."""
    response = json.dumps({
        "answer": "The answer is 42.",
        "reasoning": "Mathematical calculation.",
        "entities_found": [
            {"name": "42", "type": "number", "confidence": 0.99, "source": "computation"},
        ],
        "pronouns_resolved": ["it"],
        "time_parsed": {"start": "2024-01-01", "end": "2024-12-31", "expression": "this year"},
        "ambiguity_detected": False,
    })

    result = processor._parse_response(response)

    assert result.answer == "The answer is 42."
    assert result.reasoning == "Mathematical calculation."
    assert len(result.entities_found) == 1
    assert result.entities_found[0].name == "42"
    assert result.entities_found[0].type == "number"
    assert result.pronouns_resolved == ["it"]
    assert result.time_parsed is not None
    assert result.time_parsed.start == "2024-01-01"
    assert result.time_parsed.expression == "this year"
    assert result.ambiguity_detected is False


# ---------------------------------------------------------------------------
# 5. test_parse_response_invalid_json
# ---------------------------------------------------------------------------

def test_parse_response_invalid_json(processor: LocalProcessor):
    """_parse_response should gracefully handle invalid JSON."""
    response = "This is not valid JSON at all."

    result = processor._parse_response(response)

    assert result.answer == response
    assert result.reasoning == "Failed to parse structured response"
    assert result.entities_found == []
    assert result.pronouns_resolved == []
    assert result.time_parsed is None
    assert result.ambiguity_detected is False
