"""Local Processor - rules-and-memory driven local model processing.

Combines RuleManager, MemoryManager, and ConfidenceEvaluator to process
user input locally with full context awareness.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from memory.manager import MemoryManager
from memory.memory import MemoryType
from server.confidence_evaluator import ConfidenceEvaluator
from server.rule_manager import RuleManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """An entity extracted from user input."""
    name: str
    type: str  # person, place, thing, topic
    confidence: float = 0.5
    source: str = ""  # input, memory, rule


@dataclass
class TimeRange:
    """A parsed time range from user input."""
    start: str = ""
    end: str = ""
    expression: str = ""  # original time expression found


@dataclass
class LocalResult:
    """Result from local model processing."""
    answer: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    entities_found: list[Entity] = field(default_factory=list)
    pronouns_resolved: list[str] = field(default_factory=list)
    time_parsed: TimeRange | None = None
    rules_applied: list[str] = field(default_factory=list)
    ambiguity_detected: bool = False
    needs_remote_help: bool = False


# ---------------------------------------------------------------------------
# LocalProcessor
# ---------------------------------------------------------------------------

class LocalProcessor:
    """Processes user input using local model with rules and memory context.

    Args:
        memory_manager: Shared MemoryManager instance.
        model: Model with a ``generate_async(prompt) -> str`` method.
        confidence_threshold: Minimum confidence to accept local result (default 0.9).
        session_id: Conversation session ID for history retrieval.
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        model: Any,
        confidence_threshold: float = 0.9,
        session_id: str = "default",
    ) -> None:
        self._mm = memory_manager
        self._model = model
        self._confidence_threshold = confidence_threshold
        self._session_id = session_id

        self._rule_manager = RuleManager(memory_manager)
        self._confidence_evaluator = ConfidenceEvaluator()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process(self, user_input: str) -> LocalResult:
        """Process user input and return a LocalResult.

        Steps:
        1. Get relevant rules from RuleManager.
        2. Get relevant memories from MemoryManager.
        3. Get recent conversation history.
        4. Build prompt with all context.
        5. Generate response from model.
        6. Parse JSON response.
        7. Evaluate confidence.
        8. If successful (confidence > threshold), extract and store pattern.
        """
        # 1. Get relevant rules.
        rules = self._rule_manager.get_relevant_rules(user_input)

        # 2. Get relevant memories.
        memories = self._mm.recall(user_input, limit=5)

        # 3. Get recent conversation history.
        history = self._get_recent_history(limit=6)

        # 4. Build prompt.
        prompt = self._build_prompt(user_input, rules, memories, history)

        # 5. Generate response.
        raw_response = await self._model.generate_async(prompt)

        # 6. Parse response.
        result = self._parse_response(raw_response)

        # 7. Evaluate confidence.
        rule_dicts = [{"confidence": r.confidence, "name": r.rule_type} for r in rules]
        confidence = self._confidence_evaluator.evaluate(
            user_input=user_input,
            response=result.answer,
            rules=rule_dicts,
            memories=memories,
        )
        result.confidence = confidence

        # Track which rules were applied.
        result.rules_applied = [r.rule_type for r in rules]

        # Determine if remote help is needed.
        result.needs_remote_help = confidence < self._confidence_threshold

        # 8. If successful, extract and store pattern.
        if confidence > self._confidence_threshold:
            self._extract_and_store_pattern(user_input, result)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        user_input: str,
        rules: list[Any],
        memories: list[dict[str, Any]],
        history: list[dict[str, str]],
    ) -> str:
        """Build a prompt enriched with rules, memories, and history context."""
        parts: list[str] = []

        # System instructions.
        parts.append(
            "You are a local assistant. Answer the user's question using the provided context. "
            "Respond in JSON format with keys: answer, reasoning, entities_found, "
            "pronouns_resolved, time_parsed, ambiguity_detected.\n"
            "- entities_found: list of {name, type, confidence, source}\n"
            "- pronouns_resolved: list of pronouns you resolved\n"
            "- time_parsed: {start, end, expression} or null\n"
            "- ambiguity_detected: boolean"
        )

        # Rules context.
        if rules:
            rule_lines = []
            for r in rules:
                rule_lines.append(f"- [{r.rule_type}] {r.logic} (confidence: {r.confidence})")
            parts.append("Relevant rules:\n" + "\n".join(rule_lines))

        # Memory context.
        if memories:
            memory_lines = [f"- {m.get('content', '')}" for m in memories]
            parts.append("Relevant memories:\n" + "\n".join(memory_lines))

        # Conversation history.
        if history:
            history_lines = [f"{h['role']}: {h['content']}" for h in history]
            parts.append("Recent conversation:\n" + "\n".join(history_lines))

        # User input.
        parts.append(f"User: {user_input}")
        parts.append("Respond with JSON only:")

        return "\n\n".join(parts)

    def _parse_response(self, response: str) -> LocalResult:
        """Parse a JSON response from the model into a LocalResult."""
        result = LocalResult()

        # Try to extract JSON from the response (handle markdown fences).
        json_str = response.strip()
        if json_str.startswith("```"):
            # Strip markdown code fences.
            lines = json_str.split("\n")
            json_lines = []
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
            # If parsing fails, use the raw response as the answer.
            result.answer = response
            result.reasoning = "Failed to parse structured response"
            return result

        # Extract fields.
        result.answer = data.get("answer", "")
        result.reasoning = data.get("reasoning", "")

        # Parse entities.
        for ent_data in data.get("entities_found", []):
            if isinstance(ent_data, dict):
                result.entities_found.append(Entity(
                    name=ent_data.get("name", ""),
                    type=ent_data.get("type", ""),
                    confidence=ent_data.get("confidence", 0.5),
                    source=ent_data.get("source", ""),
                ))

        # Pronouns resolved.
        result.pronouns_resolved = data.get("pronouns_resolved", [])

        # Time parsed.
        time_data = data.get("time_parsed")
        if isinstance(time_data, dict):
            result.time_parsed = TimeRange(
                start=time_data.get("start", ""),
                end=time_data.get("end", ""),
                expression=time_data.get("expression", ""),
            )

        # Ambiguity.
        result.ambiguity_detected = data.get("ambiguity_detected", False)

        return result

    def _extract_and_store_pattern(self, user_input: str, result: LocalResult) -> None:
        """Store a successful processing pattern as a new rule."""
        pattern_type = self._extract_success_pattern(user_input, result)
        if not pattern_type:
            return

        try:
            self._rule_manager.add_rule(
                rule_type=pattern_type,
                logic=f"Input: {user_input[:200]} -> Answer: {result.answer[:200]}",
                confidence=result.confidence,
                source="local_pattern",
                pattern=user_input[:100],
            )
        except Exception:
            logger.debug("Failed to store pattern for input: %s", user_input[:50])

    def _extract_success_pattern(self, user_input: str, result: LocalResult) -> str:
        """Extract the type of successful pattern for storage."""
        # Pronoun resolution pattern.
        if result.pronouns_resolved:
            return "pronoun_resolution"

        # Entity extraction pattern.
        if result.entities_found:
            return "entity_extraction"

        # Time parsing pattern.
        if result.time_parsed and result.time_parsed.expression:
            return "time_parsing"

        # Question-answer pattern (fallback).
        if "?" in user_input or any(
            user_input.lower().startswith(w)
            for w in ("what", "who", "where", "when", "how", "why", "is", "are", "do", "does")
        ):
            return "question_answer"

        return "general"

    def _get_recent_history(self, limit: int = 6) -> list[dict[str, str]]:
        """Get recent conversation history from the memory store."""
        try:
            rows = self._mm.store.get_conversation_history(self._session_id, limit=limit)
            return [{"role": r.get("role", ""), "content": r.get("content", "")} for r in rows]
        except Exception:
            return []
