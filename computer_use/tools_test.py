"""Tests for computer_use module."""

import json
import pytest

from computer_use.tools import COMPUTER_USE_TOOL, COMPUTER_USE_SYSTEM_PROMPT


class TestComputerUseTool:
    def test_tool_schema_valid(self):
        """Test that the tool schema is valid JSON Schema."""
        assert COMPUTER_USE_TOOL["type"] == "function"
        fn = COMPUTER_USE_TOOL["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"
        assert "action" in fn["parameters"]["properties"]

    def test_tool_has_all_actions(self):
        """Test that all expected actions are defined."""
        action_enum = COMPUTER_USE_TOOL["function"]["parameters"]["properties"]["action"]["enum"]
        expected = {"screenshot", "click", "double_click", "move", "drag", "scroll", "type", "keypress", "wait"}
        assert set(action_enum) == expected

    def test_tool_required_fields(self):
        """Test that action is required."""
        required = COMPUTER_USE_TOOL["function"]["parameters"]["required"]
        assert "action" in required

    def test_system_prompt_not_empty(self):
        """Test that system prompt is not empty."""
        assert len(COMPUTER_USE_SYSTEM_PROMPT) > 100
        assert "screenshot" in COMPUTER_USE_SYSTEM_PROMPT.lower()


class TestActions:
    def test_get_keycode_known_keys(self):
        """Test keycode lookup for known keys."""
        from computer_use.actions import _get_keycode
        assert _get_keycode("a") == 0
        assert _get_keycode("return") == 36
        assert _get_keycode("cmd") == 55

    def test_get_keycode_unknown_key(self):
        """Test keycode lookup raises for unknown keys."""
        from computer_use.actions import _get_keycode
        with pytest.raises(ValueError, match="Unknown key"):
            _get_keycode("nonexistent_key")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
