"""Gemma 4 VLM backend using mlx_vlm."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from backends.base import ModelBackend
from server.message_adapter import build_tool_prompt_prefix, normalize_messages

logger = logging.getLogger(__name__)


def _convert_openai_tools_to_gemma(tools: list[Any]) -> list[dict]:
    """Convert OpenAI-format tools to Gemma 4's apply_chat_template schema."""
    schemas = []
    for t in tools:
        # Support both dict and Pydantic model
        if hasattr(t, "function"):
            fn = t.function
            fn_name = fn.name if hasattr(fn, "name") else fn.get("name", "")
            fn_desc = fn.description if hasattr(fn, "description") else fn.get("description", "")
            fn_params = fn.parameters if hasattr(fn, "parameters") else fn.get("parameters")
        else:
            fn = t.get("function", {})
            fn_name = fn.get("name", "")
            fn_desc = fn.get("description", "")
            fn_params = fn.get("parameters")

        schemas.append({
            "type": "function",
            "function": {
                "name": fn_name,
                "description": fn_desc or "",
                "parameters": fn_params or {"type": "object", "properties": {}},
            },
        })
    return schemas


def _parse_gemma_tool_calls(text: str) -> list[dict]:
    """Parse Gemma 4 native <|tool_call|> format, with JSON fallback."""
    calls: list[dict] = []

    # 1. Native format: <|tool_call>call:name{args}<tool_call|>
    gemma_pattern = r'<\|tool_call\>call:(.+?)\{(.+?)\}<tool_call\|>'

    def _cast(v: str) -> Any:
        v = v.strip()
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
        return {"true": True, "false": False}.get(v.lower(), v.strip("'\""))

    for name, args_text in re.findall(gemma_pattern, text):
        args: dict[str, Any] = {}
        for k, v1, v2 in re.findall(
            r'(\w+):(?:<\|"\|>(.*?)<\|"\|>|([^,}]*))', args_text
        ):
            args[k] = _cast(v1 or v2)

        calls.append({
            "id": f"call_{len(calls)}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args),
            },
        })

    if calls:
        return calls

    # 2. Fallback: JSON blocks (handles one level of brace nesting)
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


class GemmaBackend(ModelBackend):
    """Gemma 4 VLM backend using mlx_vlm."""

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._model_id: str | None = None

    # -- lifecycle --

    def load(self, model_id: str) -> None:
        """Load Gemma 4 via mlx_vlm.  Raises ImportError if mlx_vlm is absent."""
        import mlx.core as mx
        from mlx_vlm import load

        self._model_id = model_id
        self._model, self._processor = load(model_id)
        mx.eval(self._model.parameters())
        mx.set_cache_limit(8 * 1024 * 1024 * 1024)
        logger.info("Gemma model loaded: %s", model_id)

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
        enable_thinking: bool = False,  # ignored by Gemma
    ) -> str:
        tokenizer = self._processor.tokenizer
        messages = normalize_messages(messages)

        if tools:
            gemma_tools = _convert_openai_tools_to_gemma(tools)
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tools=gemma_tools,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                logger.info("Gemma native tool template OK")
                return prompt
            except TypeError:
                logger.info("Gemma native tool template failed, using fallback")
                messages = build_tool_prompt_prefix(messages, tools)

        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def parse_tool_calls(self, text: str) -> list[dict]:
        return _parse_gemma_tool_calls(text)

    def warmup(self) -> None:
        dummy = self._processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        self.generate(dummy, max_tokens=1)
        logger.info("Gemma warmup done")
