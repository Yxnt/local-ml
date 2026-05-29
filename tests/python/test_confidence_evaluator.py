"""Tests for ConfidenceEvaluator."""

import pytest

from server.confidence_evaluator import ConfidenceEvaluator


@pytest.fixture
def evaluator():
    return ConfidenceEvaluator()


class TestConfidenceEvaluator:
    def test_simple_query_high_confidence(self, evaluator):
        """A straightforward query with no pronouns, no time, no ambiguity
        should score high (but memory drag keeps it below 1.0)."""
        score = evaluator.evaluate(
            user_input="What is the capital of France?",
            response="The capital of France is Paris.",
        )
        # pronoun=0.9*0.20, time=0.9*0.15, rule=0.5*0.30, memory=0.5*0.20, ambiguity=0.9*0.15
        # = 0.18 + 0.135 + 0.15 + 0.10 + 0.135 = 0.70
        assert score == pytest.approx(0.70, abs=0.01)
        assert score >= 0.6

    def test_pronoun_unresolved_low_confidence(self, evaluator):
        """Unresolved pronouns should drag confidence down."""
        score = evaluator.evaluate(
            user_input="Can you fix it?",
            response="I can help. What would you like me to do?",
        )
        # pronoun=0.4*0.20=0.08, time=0.9*0.15=0.135, rule=0.5*0.30=0.15,
        # memory=0.5*0.20=0.10, ambiguity=0.9*0.15=0.135
        # = 0.08 + 0.135 + 0.15 + 0.10 + 0.135 = 0.60
        assert score == pytest.approx(0.60, abs=0.01)
        # Should be lower than the simple query case
        simple_score = evaluator.evaluate(
            user_input="What is the capital of France?",
            response="The capital of France is Paris.",
        )
        assert score < simple_score

    def test_time_expression_reduces_confidence(self, evaluator):
        """Time expressions in input should slightly reduce confidence."""
        score = evaluator.evaluate(
            user_input="Remind me tomorrow at 10:00",
            response="I will remind you tomorrow at 10:00.",
        )
        # time=0.8*0.15=0.12 (lower than default 0.9*0.15=0.135)
        # pronoun=0.9*0.20 (no pronouns), rule=0.5*0.30, memory=0.5*0.20, ambiguity=0.9*0.15
        # = 0.18 + 0.12 + 0.15 + 0.10 + 0.135 = 0.685
        assert score == pytest.approx(0.685, abs=0.01)

        # Compare with no time expression
        no_time_score = evaluator.evaluate(
            user_input="Set a reminder for me",
            response="Sure, when should I remind you?",
        )
        assert score < no_time_score

    def test_ambiguity_gives_low_confidence(self, evaluator):
        """Ambiguous input should significantly reduce confidence."""
        score = evaluator.evaluate(
            user_input="Maybe we should do something, I'm not sure",
            response="Let me help you clarify what you need.",
        )
        # ambiguity=0.3*0.15=0.045 (vs 0.9*0.15=0.135)
        # pronoun=0.9*0.20, time=0.9*0.15, rule=0.5*0.30, memory=0.5*0.20
        # = 0.18 + 0.135 + 0.15 + 0.10 + 0.045 = 0.61
        assert score == pytest.approx(0.61, abs=0.01)
        assert score < 0.65

    def test_with_rules_boosts_confidence(self, evaluator):
        """Matched rules with high confidence should boost overall score."""
        rules = [{"confidence": 0.95, "name": "greeting_rule"}]
        score = evaluator.evaluate(
            user_input="Hello there",
            response="Hello! How can I help you today?",
            rules=rules,
        )
        # rule=0.95*0.30=0.285 (vs 0.5*0.30=0.15)
        # pronoun=0.9*0.20, time=0.9*0.15, memory=0.5*0.20, ambiguity=0.9*0.15
        # = 0.18 + 0.135 + 0.285 + 0.10 + 0.135 = 0.835
        assert score == pytest.approx(0.835, abs=0.01)

        # Should be higher than no rules
        no_rules_score = evaluator.evaluate(
            user_input="Hello there",
            response="Hello! How can I help you today?",
        )
        assert score > no_rules_score
