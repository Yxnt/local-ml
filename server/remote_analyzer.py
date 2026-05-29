"""Remote Analyzer - analyze local model results via remote LLM for learning.

Sends sanitized context to a remote LLM and receives structured feedback
including correctness judgment, better answers, and generalizable rules.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Correction:
    """A specific correction to a local model's answer."""
    field: str
    original: str
    corrected: str
    reason: str


@dataclass
class RemoteFeedback:
    """Feedback from the remote LLM analyzing a local result."""
    is_correct: bool = False
    confidence: float = 0.0
    reasoning: str = ""
    better_answer: str = ""
    logic_rule: str = ""
    rule_type: str = ""
    corrections: list[Correction] = field(default_factory=list)


@dataclass
class SanitizedContext:
    """Context sent to the remote analyzer (no raw sensitive data)."""
    user_input: str = ""
    history: list[dict[str, str]] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    local_answer: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    rules_applied: list[str] = field(default_factory=list)
    timestamp: str = ""


# ---------------------------------------------------------------------------
# RemoteAnalyzer
# ---------------------------------------------------------------------------


class RemoteAnalyzer:
    """Analyzes local model results using a remote LLM for learning.

    Args:
        model: Remote model with a ``generate_async(prompt) -> str`` method.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    async def analyze(self, context: SanitizedContext) -> RemoteFeedback:
        """Analyze local result via the remote LLM.

        Args:
            context: Sanitized context containing the local model's answer
                     and surrounding conversation context.

        Returns:
            RemoteFeedback with correctness judgment, reasoning, and rules.
        """
        prompt = self._build_prompt(context)
        raw_response = await self._model.generate_async(prompt)
        return self._parse_feedback(raw_response)

    def _build_prompt(self, context: SanitizedContext) -> str:
        """Build a prompt asking the remote LLM to analyze the local result.

        The prompt requests JSON feedback with:
        - Whether the local answer is correct
        - Confidence in that judgment
        - Reasoning behind the judgment
        - A better answer if the local one is wrong
        - A generalizable rule extracted from the analysis
        """
        parts: list[str] = []

        parts.append(
            "You are an expert evaluator. A local AI assistant answered a user's "
            "question, and you need to analyze whether the answer is correct.\n"
            "Respond with a JSON object containing these keys:\n"
            "- is_correct: boolean, whether the local answer is correct\n"
            "- confidence: float 0.0-1.0, your confidence in this judgment\n"
            "- reasoning: string, explanation of your judgment\n"
            "- better_answer: string, a better answer if the local one is wrong "
            "(empty string if correct)\n"
            "- logic_rule: string, a generalizable rule learned from this analysis "
            "(e.g. 'When X, the answer is typically Y because Z')\n"
            "- rule_type: string, a descriptive category for the rule "
            "(any type you determine, e.g. 'geography', 'math', 'date_format', etc.)\n"
            "- corrections: list of {field, original, corrected, reason} for each "
            "specific correction needed (empty list if correct)"
        )

        # User input.
        parts.append(f"User asked: {context.user_input}")

        # Conversation history.
        if context.history:
            history_lines = [f"  {h.get('role', '?')}: {h.get('content', '')}" for h in context.history]
            parts.append("Conversation history:\n" + "\n".join(history_lines))

        # Entities.
        if context.entities:
            parts.append("Entities mentioned: " + ", ".join(context.entities))

        # Local answer and its metadata.
        parts.append(f"Local assistant answered: {context.local_answer}")
        parts.append(f"Local confidence: {context.confidence}")
        parts.append(f"Local reasoning: {context.reasoning}")

        if context.rules_applied:
            parts.append("Rules applied locally: " + ", ".join(context.rules_applied))

        parts.append("Respond with JSON only:")
        return "\n\n".join(parts)

    def _parse_feedback(self, response: str) -> RemoteFeedback:
        """Parse a JSON response from the remote LLM into RemoteFeedback.

        Handles markdown code fences and invalid JSON gracefully.
        """
        json_str = response.strip()

        # Strip markdown code fences if present.
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            json_lines: list[str] = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.strip() == "```" and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            json_str = "\n".join(json_lines)

        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse remote feedback JSON, using raw response")
            return RemoteFeedback(
                is_correct=False,
                confidence=0.0,
                reasoning=f"Failed to parse feedback: {response[:200]}",
            )

        # Parse corrections list.
        corrections: list[Correction] = []
        for c in data.get("corrections", []):
            if isinstance(c, dict):
                corrections.append(Correction(
                    field=c.get("field", ""),
                    original=c.get("original", ""),
                    corrected=c.get("corrected", ""),
                    reason=c.get("reason", ""),
                ))

        return RemoteFeedback(
            is_correct=bool(data.get("is_correct", False)),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=str(data.get("reasoning", "")),
            better_answer=str(data.get("better_answer", "")),
            logic_rule=str(data.get("logic_rule", "")),
            rule_type=str(data.get("rule_type", "")),
            corrections=corrections,
        )
