"""Tests for session sanitizer - consistent entity mapping across a session."""

import pytest
from server.session_sanitizer import (
    EntityMapping,
    SessionSanitizer,
)


class TestSessionSanitizer:
    """SessionSanitizer tests for consistent entity mapping."""

    def test_consistent_entity_mapping(self):
        """Same entity gets the same placeholder across multiple sanitize calls."""
        session = SessionSanitizer()

        result1 = session.sanitize("张伟的邮箱是zhangwei@test.com")
        result2 = session.sanitize("张伟今天请假了")

        # Extract the placeholder used for 张伟 in each call
        # Both calls should use the same placeholder for 张伟
        # Find the PERSON placeholder in result1
        person_placeholder_1 = None
        for ph, orig in result1.mapping.items():
            if orig == "张伟":
                person_placeholder_1 = ph
                break

        person_placeholder_2 = None
        for ph, orig in result2.mapping.items():
            if orig == "张伟":
                person_placeholder_2 = ph
                break

        assert person_placeholder_1 is not None, "张伟 should be detected in first call"
        assert person_placeholder_2 is not None, "张伟 should be detected in second call"
        assert person_placeholder_1 == person_placeholder_2, (
            f"Same entity should get same placeholder, got {person_placeholder_1} vs {person_placeholder_2}"
        )

    def test_multiple_entities_different_placeholders(self):
        """Different entities get different placeholders."""
        session = SessionSanitizer()

        result = session.sanitize("张伟和李娜都是同事，邮箱分别是zw@test.com和ln@test.com")

        # All placeholders should be unique
        placeholders = list(result.mapping.keys())
        assert len(placeholders) == len(set(placeholders)), "All placeholders should be unique"

        # Should have detected at least 2 persons and 2 emails
        person_placeholders = [ph for ph in placeholders if "PERSON" in ph]
        email_placeholders = [ph for ph in placeholders if "EMAIL" in ph]
        assert len(person_placeholders) >= 2, f"Expected 2+ person placeholders, got {person_placeholders}"
        assert len(email_placeholders) >= 2, f"Expected 2+ email placeholders, got {email_placeholders}"

    def test_desanitize_restores_original(self):
        """desanitize() restores all placeholders back to original values."""
        session = SessionSanitizer()

        original = "张伟的邮箱是zhangwei@test.com，电话13800138000"
        result = session.sanitize(original)

        # Simulate a model response that includes the sanitized text
        model_response = f"好的，我已经记录了{result.sanitized}的信息"
        restored = session.desanitize(model_response)

        assert "张伟" in restored, "Person name should be restored"
        assert "zhangwei@test.com" in restored, "Email should be restored"
        assert "13800138000" in restored, "Phone should be restored"
        # No placeholders should remain
        for ph in result.mapping:
            assert ph not in restored, f"Placeholder {ph} should have been replaced"

    def test_get_sanitized_history_uses_same_mapping(self):
        """Sanitizing a history list uses consistent mappings across all entries."""
        session = SessionSanitizer()

        history = [
            "张伟的邮箱是zhangwei@test.com",
            "张伟今天请假了",
            "帮张伟回复一下邮件",
        ]

        sanitized_history = session.get_sanitized_history(history)

        assert len(sanitized_history) == 3

        # Find the placeholder used for 张伟 (should be consistent)
        person_placeholder = None
        for text in sanitized_history:
            for ph in session._entity_map:
                mapping = session._entity_map[ph]
                if mapping.original == "张伟":
                    person_placeholder = ph
                    break
            if person_placeholder:
                break

        assert person_placeholder is not None, "张伟 should have a placeholder"
        # All history entries that contained 张伟 should now use the same placeholder
        assert person_placeholder in sanitized_history[0], "First entry should contain 张伟 placeholder"
        assert person_placeholder in sanitized_history[1], "Second entry should contain 张伟 placeholder"
        assert person_placeholder in sanitized_history[2], "Third entry should contain 张伟 placeholder"

    def test_get_entity_types_only(self):
        """get_entity_types_only returns entity types without leaking original values."""
        session = SessionSanitizer()

        session.sanitize("张伟的邮箱是zhangwei@test.com，电话13800138000")

        entity_types = session.get_entity_types_only()

        assert isinstance(entity_types, list)
        assert len(entity_types) >= 3, "Should have at least person, email, phone types"

        # Each entry should have entity_type and placeholder, but NOT original
        for entry in entity_types:
            assert "entity_type" in entry, "Each entry should have entity_type"
            assert "placeholder" in entry, "Each entry should have placeholder"
            assert "original" not in entry, "Should NOT leak original value"

        # Check that we have the expected types
        types = {e["entity_type"] for e in entity_types}
        assert "PERSON" in types, f"Should detect PERSON, got {types}"
        assert "EMAIL" in types, f"Should detect EMAIL, got {types}"
        assert "PHONE" in types, f"Should detect PHONE, got {types}"
