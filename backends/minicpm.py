"""MiniCPM5-1B backend using mlx_lm."""

from __future__ import annotations

import logging
from typing import Any

from backends.base import ModelBackend
from backends.tool_parser import parse_mcp_tool_calls
from server.message_adapter import normalize_messages

logger = logging.getLogger(__name__)


class MiniCPMBackend(ModelBackend):
    """MiniCPM5-1B backend using mlx_lm.

    NOTE: mlx_lm is imported lazily inside load().  The standard API is:
        load(model_id) -> (model, tokenizer)
        generate(model, tokenizer, prompt=..., max_tokens=..., ...)
    If the actual installed version differs, adjust load() and generate().
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_id: str | None = None

    # -- lifecycle --

    def load(self, model_id: str) -> None:
        """Load MiniCPM5 via mlx_lm.  Raises ImportError if mlx_lm is absent."""
        from mlx_lm import load

        self._model_id = model_id
        self._model, self._tokenizer = load(model_id)
        logger.info("MiniCPM model loaded: %s", model_id)

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_id = None

    # -- inference --

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        from mlx_lm import generate as lm_generate

        return lm_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        enable_thinking: bool = False,
    ) -> str:
        messages = normalize_messages(messages)
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        if tools:
            kwargs["tools"] = tools
        return self._tokenizer.apply_chat_template(messages, **kwargs)

    def parse_tool_calls(self, text: str) -> list[dict]:
        return parse_mcp_tool_calls(text)

    def warmup(self) -> None:
        dummy = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        self.generate(dummy, max_tokens=1)
        logger.info("MiniCPM warmup done")
