"""Learning Controller - orchestrates when and how to learn from the remote LLM.

Coordinates SessionSanitizer, RemoteAnalyzer, and RuleManager to learn
new rules from remote feedback while respecting budget and concurrency limits.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field

from server.local_processor import LocalResult
from server.remote_analyzer import RemoteAnalyzer, SanitizedContext
from server.rule_manager import RuleManager
from server.session_sanitizer import SessionSanitizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Budget tracking
# ---------------------------------------------------------------------------

@dataclass
class BudgetWindow:
    """Tracks usage within a time window."""
    count: int = 0
    window_start: float = field(default_factory=time.time)

    def reset_if_expired(self, window_seconds: float) -> None:
        """Reset the counter if the window has expired."""
        now = time.time()
        if now - self.window_start >= window_seconds:
            self.count = 0
            self.window_start = now


# ---------------------------------------------------------------------------
# LearningController
# ---------------------------------------------------------------------------

class LearningController:
    """Controls when and how to learn from the remote LLM.

    Manages budget (hourly/daily limits), concurrency (asyncio.Lock),
    and deduplication (pending dict) for the learning flow.

    Args:
        remote_analyzer: Analyzer that queries the remote LLM.
        rule_manager: Manager for storing learned rules.
        session_sanitizer: Sanitizer for removing sensitive data from context.
        hourly_limit: Maximum learning calls per hour (default 10).
        daily_limit: Maximum learning calls per day (default 100).
    """

    def __init__(
        self,
        remote_analyzer: RemoteAnalyzer,
        rule_manager: RuleManager,
        session_sanitizer: SessionSanitizer,
        hourly_limit: int = 10,
        daily_limit: int = 100,
    ) -> None:
        self._remote_analyzer = remote_analyzer
        self._rule_manager = rule_manager
        self._session_sanitizer = session_sanitizer
        self._hourly_limit = hourly_limit
        self._daily_limit = daily_limit

        # Concurrency control.
        self._lock = asyncio.Lock()

        # Deduplication: key -> True while a learning task is pending.
        self._pending: dict[str, bool] = {}

        # Budget tracking windows.
        self._hourly = BudgetWindow()
        self._daily = BudgetWindow()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def learn(self, user_input: str, result: LocalResult) -> bool:
        """Execute the learning flow for a user input and local result.

        Returns True if a rule was learned, False if skipped or failed.
        """
        # 1. Generate learning key from input.
        key = self._make_key(user_input)

        # 2. Skip if same learning already pending.
        if key in self._pending:
            logger.debug("Learning already pending for key: %s", key)
            return False

        # 3. Check budget.
        if not self._check_budget():
            logger.debug("Budget exceeded, skipping learning")
            return False

        # 4. Mark learning as pending.
        self._pending[key] = True

        try:
            # 5. Acquire lock.
            async with self._lock:
                # 6. Prepare sanitized context using SessionSanitizer.
                context = self._prepare_context(user_input, result)

                # 7. Call remote analyzer.
                feedback = await self._remote_analyzer.analyze(context)

                # 8. If rule found with confidence >= 0.7, store it.
                rule_stored = False
                if feedback.logic_rule and feedback.confidence >= 0.7:
                    self._rule_manager.add_rule(
                        rule_type=feedback.rule_type or "remote_feedback",
                        logic=feedback.logic_rule,
                        confidence=feedback.confidence,
                        source="remote_feedback",
                    )
                    rule_stored = True
                    logger.info(
                        "Learned rule [%s]: %s (confidence=%.2f)",
                        feedback.rule_type,
                        feedback.logic_rule[:80],
                        feedback.confidence,
                    )

                # 9. Update budget counters.
                self._hourly.count += 1
                self._daily.count += 1

                return rule_stored

        finally:
            # 10. Release pending mark.
            self._pending.pop(key, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_context(self, user_input: str, result: LocalResult) -> SanitizedContext:
        """Prepare a sanitized context from user input and local result."""
        sanitized = self._session_sanitizer.sanitize(user_input)

        return SanitizedContext(
            user_input=sanitized.sanitized,
            local_answer=result.answer,
            confidence=result.confidence,
            reasoning=result.reasoning,
            rules_applied=result.rules_applied,
        )

    def _check_budget(self) -> bool:
        """Check if learning is within hourly and daily limits."""
        self._hourly.reset_if_expired(window_seconds=3600)
        self._daily.reset_if_expired(window_seconds=86400)

        if self._hourly.count >= self._hourly_limit:
            return False
        if self._daily.count >= self._daily_limit:
            return False
        return True

    @staticmethod
    def _make_key(user_input: str) -> str:
        """Generate a unique key for deduplication from user input."""
        return hashlib.sha256(user_input.strip().encode()).hexdigest()[:16]
