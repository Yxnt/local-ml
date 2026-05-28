"""MiniCPM-V 4.6 VLM backend using mlx_vlm."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from backends.base import ModelBackend
from server.message_adapter import build_tool_prompt_prefix, normalize_messages

logger = logging.getLogger(__name__)


class MiniCPMVBackend(ModelBackend):
    """MiniCPM-V 4.6 VLM backend using mlx_vlm.

    MiniCPM-V 4.6 is a vision-language model (1.3B params) that uses mlx_vlm
    for loading and inference. Supports image input for computer use scenarios.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._model_id: str | None = None

    # -- lifecycle --

    def load(self, model_id: str) -> None:
        """Load MiniCPM-V via mlx_vlm."""
        import mlx.core as mx
        from mlx_vlm import load

        self._model_id = model_id
        self._model, self._processor = load(model_id)
        mx.eval(self._model.parameters())
        logger.info("MiniCPM-V model loaded: %s", model_id)

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._model_id = None

    # -- inference --

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        from mlx_vlm import generate as vlm_generate

        output = vlm_generate(
            model=self._model,
            processor=self._processor,
            prompt=prompt,
            image=None,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=False,
        )
        return getattr(output, "text", str(output))

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        enable_thinking: bool = False,
    ) -> str:
        tokenizer = self._processor.tokenizer
        messages = normalize_messages(messages)

        if tools:
            # Try native tool template
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return prompt
            except (TypeError, Exception):
                # Fallback to manual tool injection
                messages = build_tool_prompt_prefix(messages, tools)

        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def parse_tool_calls(self, text: str) -> list[dict]:
        # Try JSON format (MiniCPM-V outputs JSON for tool calls)
        calls: list[dict] = []
        json_pattern = r'(?:```(?:json)?\s*)?(\{(?:[^{}]|\{[^{}]*\})*\})(?:\s*```)?'
        for match in re.finditer(json_pattern, text):
            try:
                obj = json.loads(match.group(1))
                if "name" in obj and "arguments" in obj:
                    calls.append({
                        "id": f"call_{len(calls)}",
                        "type": "function",
                        "function": {
                            "name": obj["name"],
                            "arguments": json.dumps(obj["arguments"]),
                        },
                    })
            except json.JSONDecodeError:
                continue
        return calls

    def warmup(self) -> None:
        dummy = self._processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        self.generate(dummy, max_tokens=1)
        logger.info("MiniCPM-V warmup done")
