"""Model registry with lazy loading and hot-switching."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from backends.base import ModelBackend
from backends.gemma import GemmaBackend
from backends.minicpm import MiniCPMBackend
from backends.minicpmv import MiniCPMVBackend

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
    "minicpm-v-4.6": {
        "backend": "minicpmv",
        "model_id": "openbmb/MiniCPM-V-4.6",
    },
}

# Map internal backend names to public API names (matching design spec)
_BACKEND_DISPLAY_NAMES: dict[str, str] = {
    "gemma": "mlx_vlm",
    "minicpm": "mlx_lm",
    "minicpmv": "mlx_vlm",
}

# Backend name -> class mapping
_BACKEND_CLASSES: dict[str, type[ModelBackend]] = {
    "gemma": GemmaBackend,
    "minicpm": MiniCPMBackend,
    "minicpmv": MiniCPMVBackend,
}


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
