"""Hybrid Agent - main entry point orchestrating local processing and remote learning.

Combines LocalProcessor, LearningController, RuleManager, and MemoryManager
into a single agent that processes user input locally, triggers async learning
when confidence is low, and handles follow-up questions.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from backends.registry import ModelRegistry
from memory.manager import MemoryManager
from memory.memory import MemoryType
from server.learning_controller import LearningController
from server.local_processor import LocalProcessor, LocalResult
from server.remote_analyzer import RemoteAnalyzer
from server.rule_manager import RuleManager
from server.session_sanitizer import SessionSanitizer

logger = logging.getLogger(__name__)

# Pronouns that indicate a follow-up question referencing prior context.
_FOLLOWUP_PRONOUNS: set[str] = {
    "他",  # 他
    "她",  # 她
    "它",  # 它
    "这个",  # 这个
    "那个",  # 那个
    "这",  # 这
    "那",  # 那
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AgentResponse:
    """Response returned by HybridAgent.run()."""

    answer: str = ""
    confidence: float = 0.0
    is_learning: bool = False
    learning_task: asyncio.Task | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# HybridAgent
# ---------------------------------------------------------------------------


class HybridAgent:
    """Main agent entry point that orchestrates all components.

    Flow:
    1. Process input locally via LocalProcessor.
    2. If confidence < 0.9 and remote is enabled, trigger async learning.
    3. If it is a follow-up question, wait for learning to finish then reprocess.
    4. Update rule stats.
    5. Save conversation to memory.

    Args:
        config: Application configuration (dict or AppConfig).
        data_dir: Directory for memory data files.
    """

    def __init__(self, config: Any, data_dir: str = "memory/data") -> None:
        self._config = config

        # -- Core components --------------------------------------------------
        self._registry = ModelRegistry()
        self._registry.register_defaults()

        self._memory = MemoryManager(data_dir=data_dir)
        self._memory.connect()

        self._rule_manager = RuleManager(self._memory)
        self._session_sanitizer = SessionSanitizer()

        # Determine whether remote learning is enabled.
        self._remote_enabled = self._resolve_remote_enabled(config)

        self._remote_analyzer: RemoteAnalyzer | None = None
        self._learning_controller: LearningController | None = None

        if self._remote_enabled:
            self._setup_remote_learning()

        # Local processor uses the default model from the registry.
        default_model_name = self._resolve_default_model(config)
        self._local_processor = LocalProcessor(
            memory_manager=self._memory,
            model=self._build_model_proxy(default_model_name),
            session_id="hybrid",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, user_input: str) -> AgentResponse:
        """Process user input and return an AgentResponse.

        Steps:
        1. Local processing via LocalProcessor.
        2. If confidence < 0.9 and remote enabled, trigger async learning.
        3. If follow-up question, wait for learning and reprocess.
        4. Update rule stats.
        5. Save conversation.
        """
        # 1. Local processing.
        result = await self._local_processor.process(user_input)

        is_learning = False
        learning_task: asyncio.Task | None = None

        # 2. Trigger async learning if confidence is low and remote is enabled.
        if result.confidence < 0.9 and self._remote_enabled and self._learning_controller:
            is_followup = self._is_followup_question(user_input)

            if is_followup:
                # 3a. Follow-up: wait for learning to complete, then reprocess.
                learned = await self._learning_controller.learn(user_input, result)
                if learned:
                    result = await self._local_processor.process(user_input)
                    is_learning = False
            else:
                # 3b. Non-follow-up: fire-and-forget learning task.
                learning_task = asyncio.create_task(
                    self._learning_controller.learn(user_input, result)
                )
                is_learning = True

        # 4. Update rule stats for applied rules.
        self._update_rule_stats(result)

        # 5. Save conversation.
        self._save_conversation(user_input, result.answer)

        return AgentResponse(
            answer=result.answer,
            confidence=result.confidence,
            is_learning=is_learning,
            learning_task=learning_task,
        )

    async def run_with_progress(
        self,
        user_input: str,
        callback: Callable[[str], None],
    ) -> str:
        """Process user input with progress callbacks, returning the answer string.

        Callbacks are emitted at key stages of the processing pipeline.

        Args:
            user_input: The user's message.
            callback: A callable that receives progress stage descriptions.

        Returns:
            The final answer string.
        """
        callback("processing")

        result = await self._local_processor.process(user_input)

        if result.confidence < 0.9 and self._remote_enabled and self._learning_controller:
            callback("learning")
            is_followup = self._is_followup_question(user_input)

            if is_followup:
                learned = await self._learning_controller.learn(user_input, result)
                if learned:
                    callback("reprocessing")
                    result = await self._local_processor.process(user_input)
            else:
                # Fire-and-forget; don't block the response.
                asyncio.create_task(
                    self._learning_controller.learn(user_input, result)
                )

        callback("done")

        self._update_rule_stats(result)
        self._save_conversation(user_input, result.answer)

        return result.answer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_followup_question(user_input: str) -> bool:
        """Detect whether the input is a follow-up question using pronouns."""
        return any(pronoun in user_input for pronoun in _FOLLOWUP_PRONOUNS)

    def _save_conversation(self, user_input: str, answer: str) -> None:
        """Save the user input and answer to the conversation memory."""
        try:
            self._memory.store.add_conversation("hybrid", "user", user_input)
            self._memory.store.add_conversation("hybrid", "assistant", answer)
        except Exception:
            logger.debug("Failed to save conversation", exc_info=True)

    def _update_rule_stats(self, result: LocalResult) -> None:
        """Update usage/success stats for rules that were applied."""
        for rule_type in result.rules_applied:
            try:
                rules = self._rule_manager.get_relevant_rules(rule_type, limit=1)
                if rules:
                    rule = rules[0]
                    if rule.id is not None:
                        self._rule_manager.update_rule_stats(
                            rule.id,
                            confidence=result.confidence if result.confidence >= 0.9 else None,
                        )
            except Exception:
                logger.debug("Failed to update rule stats for %s", rule_type, exc_info=True)

    def _setup_remote_learning(self) -> None:
        """Initialize RemoteAnalyzer and LearningController."""
        try:
            remote_model = self._build_model_proxy("remote")
            self._remote_analyzer = RemoteAnalyzer(model=remote_model)
            self._learning_controller = LearningController(
                remote_analyzer=self._remote_analyzer,
                rule_manager=self._rule_manager,
                session_sanitizer=self._session_sanitizer,
            )
        except Exception:
            logger.warning("Failed to setup remote learning, disabling", exc_info=True)
            self._remote_enabled = False

    def _build_model_proxy(self, model_name: str) -> Any:
        """Build a proxy object with generate_async that delegates to the registry."""

        class _ModelProxy:
            def __init__(self, registry: ModelRegistry, name: str) -> None:
                self._registry = registry
                self._name = name

            async def generate_async(self, prompt: str) -> str:
                backend = await self._registry.get_or_load(self._name)
                return backend.generate(prompt=prompt)

        return _ModelProxy(self._registry, model_name)

    @staticmethod
    def _resolve_default_model(config: Any) -> str:
        """Extract the default model name from config."""
        if isinstance(config, dict):
            return config.get("models", {}).get("default", "gemma-4-e2b-it-4bit")
        # AppConfig object
        if hasattr(config, "models"):
            return getattr(config.models, "default", "gemma-4-e2b-it-4bit")
        return "gemma-4-e2b-it-4bit"

    @staticmethod
    def _resolve_remote_enabled(config: Any) -> bool:
        """Determine whether remote learning is enabled from config."""
        if isinstance(config, dict):
            return config.get("remote", {}).get("enabled", False)
        if hasattr(config, "remote"):
            return getattr(config.remote, "enabled", False)
        return False
