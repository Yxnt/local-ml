"""Abstract base class for model backends."""

from __future__ import annotations

import abc
from typing import Any


class ModelBackend(abc.ABC):
    """Abstract base for a model backend.

    Lifecycle: construct -> load() -> generate() / apply_chat_template() / ...
               -> unload()  (optional, called before loading another model)
    """

    @abc.abstractmethod
    def load(self, model_id: str) -> None:
        """Load the model into memory."""

    @abc.abstractmethod
    def unload(self) -> None:
        """Release model resources."""

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Run inference and return the generated text."""

    @abc.abstractmethod
    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        enable_thinking: bool = False,
    ) -> str:
        """Convert messages (+ optional tool defs) into a prompt string."""

    @abc.abstractmethod
    def parse_tool_calls(self, text: str) -> list[dict]:
        """Extract tool calls from model output into OpenAI-compatible dicts."""

    @abc.abstractmethod
    def warmup(self) -> None:
        """Run a tiny generation to warm up the model (compilation, cache)."""
