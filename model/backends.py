"""Strategy-pattern model backends for multi-model inference.

Provides ModelBackend ABC, GemmaBackend (mlx_vlm), MiniCPMBackend (mlx_lm),
and ModelRegistry for lazy-loading with hot-switching (one model in memory).

mlx_vlm and mlx_lm are imported lazily inside load() so that the rest of
the module (including parse_tool_calls and registry logic) works without
the ML libraries installed.
"""

from __future__ import annotations

import abc
import asyncio
import json
import logging
import os
import re
from typing import Any

from server_message_adapter import build_tool_prompt_prefix
from model.minicpm_tool_parser import parse_mcp_tool_calls

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model catalogue
# ---------------------------------------------------------------------------

DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "gemma-4-e2b-it-4bit": {
        "backend": "gemma",
        "model_id": "mlx-community/gemma-4-e2b-it-4bit",
    },
    "minicpm5-1b-mlx": {
        "backend": "minicpm",
        "model_id": "openbmb/MiniCPM5-1B-MLX",
    },
    "minicpm-v-4_6": {
        "backend": "minicpmv",
        "model_id": "openbmb/MiniCPM-V-4_6",
    },
}

# Map internal backend names to public API names (matching design spec)
_BACKEND_DISPLAY_NAMES: dict[str, str] = {
    "gemma": "mlx_vlm",
    "minicpm": "mlx_lm",
    "minicpmv": "mlx_vlm",
}

# Backend name -> class mapping (populated at end of file)
_BACKEND_CLASSES: dict[str, type[ModelBackend]] = {}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


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
        messages: list[dict[str, str]],
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


# ---------------------------------------------------------------------------
# Gemma 4 backend  (mlx_vlm)
# ---------------------------------------------------------------------------


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
        messages: list[dict[str, str]],
        tools: list[Any] | None = None,
        enable_thinking: bool = False,  # ignored by Gemma
    ) -> str:
        tokenizer = self._processor.tokenizer

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


# ---------------------------------------------------------------------------
# MiniCPM backend  (mlx_lm)
# ---------------------------------------------------------------------------


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
        messages: list[dict[str, str]],
        tools: list[Any] | None = None,
        enable_thinking: bool = False,
    ) -> str:
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


# ---------------------------------------------------------------------------
# MiniCPM-V 4.6 backend  (mlx_vlm, VLM with vision)
# ---------------------------------------------------------------------------


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
        messages: list[dict[str, str]],
        tools: list[Any] | None = None,
        enable_thinking: bool = False,
    ) -> str:
        tokenizer = self._processor.tokenizer

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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """Manages backend instances with lazy loading and hot-switching.

    Only one model is in memory at a time: loading a new model unloads the
    previously active one.
    """

    def __init__(self) -> None:
        self._backends: dict[str, ModelBackend] = {}
        self._active: str | None = None
        self._lock = asyncio.Lock()

    # -- registration --

    def register(self, name: str, backend: ModelBackend) -> None:
        self._backends[name] = backend

    def register_defaults(self) -> None:
        """Register all entries from DEFAULT_MODELS, with optional MODELS_CONFIG override."""
        models = dict(DEFAULT_MODELS)

        # Override with MODELS_CONFIG env var if set
        config_env = os.environ.get("MODELS_CONFIG")
        if config_env:
            try:
                override = json.loads(config_env)
                models.update(override)
                logger.info("Loaded MODELS_CONFIG override: %s", list(override.keys()))
            except json.JSONDecodeError as e:
                logger.warning("Invalid MODELS_CONFIG JSON: %s", e)

        for name, cfg in models.items():
            backend_cls = _BACKEND_CLASSES.get(cfg["backend"])
            if backend_cls is None:
                raise ValueError(f"Unknown backend type: {cfg['backend']}")
            backend = backend_cls()
            # Attach model_id so get_or_load can use it
            backend._default_model_id = cfg["model_id"]  # type: ignore[attr-defined]
            self.register(name, backend)

    def list_models(self) -> list[dict[str, str]]:
        result = []
        for name in self._backends:
            cfg = DEFAULT_MODELS.get(name, {})
            backend_key = cfg.get("backend", "unknown")
            display_name = _BACKEND_DISPLAY_NAMES.get(backend_key, backend_key)
            result.append({"id": name, "backend": display_name})
        return result

    def get_backend(self, name: str) -> ModelBackend:
        if name not in self._backends:
            raise ValueError(f"Unknown model: {name}")
        return self._backends[name]

    # -- lazy loading with hot-switch --

    async def get_or_load(self, name: str) -> ModelBackend:
        """Return a ready-to-use backend, loading it if necessary.

        If a different model is currently loaded, it is unloaded first.
        """
        async with self._lock:
            backend = self.get_backend(name)

            # Already loaded?
            if self._active == name:
                return backend

            # Unload previous model
            if self._active is not None:
                prev = self._backends.get(self._active)
                if prev is not None:
                    logger.info("Unloading model: %s", self._active)
                    prev.unload()

            # Load requested model
            model_id: str = getattr(backend, "_default_model_id", name)
            logger.info("Loading model: %s (%s)", name, model_id)
            backend.load(model_id)
            backend.warmup()
            self._active = name
            return backend


# Populate the backend class map (must be after class definitions)
_BACKEND_CLASSES = {
    "gemma": GemmaBackend,
    "minicpm": MiniCPMBackend,
    "minicpmv": MiniCPMVBackend,
}
