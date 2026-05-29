"""Tests for server.rule_manager."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path so `memory` and `server` packages resolve.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from memory.manager import MemoryManager
from memory.memory import MemoryType
from server.rule_manager import RuleExample, RuleManager


@pytest.fixture()
def rule_manager(tmp_path):
    """Provide a RuleManager backed by an ephemeral in-memory database."""
    data_dir = str(tmp_path / "memory_data")
    with MemoryManager(data_dir=data_dir) as mm:
        rm = RuleManager(mm)
        yield rm


# ------------------------------------------------------------------
# 1. test_add_and_retrieve_rule
# ------------------------------------------------------------------

def test_add_and_retrieve_rule(rule_manager: RuleManager):
    """Adding a rule and searching for it should return the same rule."""
    rule_id = rule_manager.add_rule(
        rule_type="formatting",
        logic="Always use markdown headers for section titles",
        confidence=0.9,
        source="user_taught",
        pattern="section title",
    )
    assert rule_id > 0

    rules = rule_manager.get_relevant_rules("markdown headers section")
    assert len(rules) >= 1

    found = rules[0]
    assert found.id == rule_id
    assert found.rule_type == "formatting"
    assert found.logic == "Always use markdown headers for section titles"
    assert found.confidence == 0.9
    assert found.source == "user_taught"
    assert found.status == "active"


# ------------------------------------------------------------------
# 2. test_dynamic_rule_types
# ------------------------------------------------------------------

def test_dynamic_rule_types(rule_manager: RuleManager):
    """Any string can be used as rule_type; types are not restricted to an enum."""
    rule_manager.add_rule(rule_type="tone", logic="Be concise", source="user_taught")
    rule_manager.add_rule(rule_type="api_style", logic="Use snake_case", source="local_pattern")
    rule_manager.add_rule(rule_type="custom_xyz", logic="Do something special", source="remote_feedback")

    rules = rule_manager.get_relevant_rules("tone api style custom")
    types = {r.rule_type for r in rules}

    # All three custom types should appear.
    assert "tone" in types
    assert "api_style" in types
    assert "custom_xyz" in types


# ------------------------------------------------------------------
# 3. test_rule_priority_ordering
# ------------------------------------------------------------------

def test_rule_priority_ordering(rule_manager: RuleManager):
    """Rules with higher source priority should rank first.

    user_taught (100) > remote_feedback (80) > local_pattern (60).
    When multiple rules of different types exist, they should be sorted by
    composite score.  We add one rule per source with identical usage stats
    so the source base priority is the differentiator.
    """
    rule_manager.add_rule(rule_type="a", logic="local pattern rule", source="local_pattern", confidence=0.5)
    rule_manager.add_rule(rule_type="b", logic="remote feedback rule", source="remote_feedback", confidence=0.5)
    rule_manager.add_rule(rule_type="c", logic="user taught rule", source="user_taught", confidence=0.5)

    rules = rule_manager.get_relevant_rules("rule")
    assert len(rules) == 3

    # user_taught should be first, remote_feedback second, local_pattern third.
    assert rules[0].source == "user_taught"
    assert rules[1].source == "remote_feedback"
    assert rules[2].source == "local_pattern"


# ------------------------------------------------------------------
# 4. test_rule_deactivation
# ------------------------------------------------------------------

def test_rule_deactivation(rule_manager: RuleManager):
    """Archived rules should no longer appear in get_relevant_rules."""
    rule_id = rule_manager.add_rule(
        rule_type="temp",
        logic="This rule will be archived",
        source="local_pattern",
    )

    # Confirm it's retrievable.
    rules = rule_manager.get_relevant_rules("archived")
    assert any(r.id == rule_id for r in rules)

    # Archive it.
    rule_manager.archive_rule(rule_id)

    # Should no longer appear.
    rules = rule_manager.get_relevant_rules("archived")
    assert all(r.id != rule_id for r in rules)


# ------------------------------------------------------------------
# 5. test_only_active_rules_returned
# ------------------------------------------------------------------

def test_only_active_rules_returned(rule_manager: RuleManager):
    """get_relevant_rules must never return archived rules."""
    # Add two rules.
    id_active = rule_manager.add_rule(rule_type="x", logic="active rule", source="user_taught")
    id_archived = rule_manager.add_rule(rule_type="y", logic="archived rule", source="user_taught")

    # Archive one.
    rule_manager.archive_rule(id_archived)

    rules = rule_manager.get_relevant_rules("rule")
    returned_ids = {r.id for r in rules}

    assert id_active in returned_ids
    assert id_archived not in returned_ids

    # Verify status field on every returned rule.
    for r in rules:
        assert r.status == "active"
