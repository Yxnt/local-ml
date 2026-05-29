"""Multi-factor weighted confidence evaluator for local model processing.

Evaluates confidence of local model responses using five factors:
- Rule matching (weight 0.30)
- Pronoun resolution (weight 0.20)
- Time parsing (weight 0.15)
- Memory matching (weight 0.20)
- Ambiguity detection (weight 0.15)
"""

import re
from typing import List, Optional


class ConfidenceEvaluator:
    """Evaluates confidence of local model processing using multi-factor weighted scoring."""

    WEIGHTS = {
        "rule": 0.30,
        "pronoun": 0.20,
        "time": 0.15,
        "memory": 0.20,
        "ambiguity": 0.15,
    }

    PRONOUN_PATTERN = re.compile(
        r"\b(it|its|they|them|their|he|him|his|she|her|hers|this|that|these|those)\b",
        re.IGNORECASE,
    )

    TIME_PATTERN = re.compile(
        r"\b(today|tomorrow|yesterday|now|later|soon|tonight|morning|evening|afternoon|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"january|february|march|april|may|june|july|august|september|october|november|december|"
        r"\d{1,2}:\d{2}|\d{4}-\d{2}-\d{2}|next\s+\w+|last\s+\w+|in\s+\d+)\b",
        re.IGNORECASE,
    )

    AMBIGUITY_PATTERN = re.compile(
        r"\b(maybe|perhaps|might|could|possibly|not sure|uncertain|ambiguous|"
        r"i think|probably|kind of|sort of|somewhat)\b",
        re.IGNORECASE,
    )

    def evaluate(
        self,
        user_input: str,
        response: str,
        rules: Optional[List[dict]] = None,
        memories: Optional[List[dict]] = None,
    ) -> float:
        """Evaluate confidence using multi-factor weighted scoring.

        Args:
            user_input: The user's input text.
            response: The model's response text.
            rules: Optional list of matched rules (each with a 'confidence' key).
            memories: Optional list of relevant memories.

        Returns:
            Confidence score clamped to [0.0, 1.0].
        """
        scores = {
            "rule": self._score_rules(rules),
            "pronoun": self._score_pronouns(user_input, response),
            "time": self._score_time(user_input),
            "memory": self._score_memory(memories),
            "ambiguity": self._score_ambiguity(user_input),
        }

        weighted_sum = sum(
            scores[factor] * weight for factor, weight in self.WEIGHTS.items()
        )
        return max(0.0, min(1.0, weighted_sum))

    def _score_rules(self, rules: Optional[List[dict]]) -> float:
        """Rule matching: max confidence of rules, or 0.5 if no rules."""
        if not rules:
            return 0.5
        return max(r.get("confidence", 0.5) for r in rules)

    def _score_pronouns(self, user_input: str, response: str) -> float:
        """Pronoun resolution scoring.

        0.8 if all resolved, 0.4 if unresolved, 0.9 if no pronouns found.
        """
        pronouns_in_input = set(self.PRONOUN_PATTERN.findall(user_input.lower()))
        if not pronouns_in_input:
            return 0.9

        response_lower = response.lower()
        resolved = sum(
            1 for p in pronouns_in_input if p.lower() in response_lower
        )
        if resolved == len(pronouns_in_input):
            return 0.8
        return 0.4

    def _score_time(self, user_input: str) -> float:
        """Time parsing: 0.8 if time detected, 0.9 if not."""
        if self.TIME_PATTERN.search(user_input):
            return 0.8
        return 0.9

    def _score_memory(self, memories: Optional[List[dict]]) -> float:
        """Memory matching: 0.7 if memories found, 0.5 if not."""
        if memories:
            return 0.7
        return 0.5

    def _score_ambiguity(self, user_input: str) -> float:
        """Ambiguity: 0.3 if ambiguity detected, 0.9 if not."""
        if self.AMBIGUITY_PATTERN.search(user_input):
            return 0.3
        return 0.9
