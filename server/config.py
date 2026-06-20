"""Configuration system for local-ml.

Loads config from YAML, supports environment variable substitution
(e.g., ``${GMAIL_PASSWORD}``), validates structure, and provides typed
access to config sections.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Environment variable substitution
# ---------------------------------------------------------------------------

_ENV_VAR_RE = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")


def _substitute_env_vars(value: str) -> str:
    """Replace ``${VAR}`` or ``${VAR:default}`` with environment values."""

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default = match.group(2)
        env_val = os.environ.get(var_name)
        if env_val is not None:
            return env_val
        if default is not None:
            return default
        return match.group(0)  # leave as-is if no env and no default

    return _ENV_VAR_RE.sub(_replace, value)


def _walk_and_substitute(obj: Any) -> Any:
    """Recursively substitute env vars in all string values."""
    if isinstance(obj, str):
        return _substitute_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _walk_and_substitute(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_substitute(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Pydantic models for each config section
# ---------------------------------------------------------------------------


class ModelEntry(BaseModel):
    """A single model definition."""

    backend: str
    model_id: str


class ModelsConfig(BaseModel):
    """Models configuration section."""

    default: str = "gemma-4-e2b-it-4bit"
    available: dict[str, ModelEntry] = Field(default_factory=dict)


class EmbeddingConfig(BaseModel):
    """Embedding service configuration."""

    url: str = "http://localhost:8001"
    dimensions: int = 768


class ObsidianConfig(BaseModel):
    """Obsidian integration config."""

    vaults: dict[str, str] = Field(default_factory=dict)


class CalendarAccount(BaseModel):
    """A single calendar account."""

    url: str
    username: str
    password: str = ""


class CalendarConfig(BaseModel):
    """Calendar integration config."""

    accounts: dict[str, CalendarAccount] = Field(default_factory=dict)


class EmailAccount(BaseModel):
    """A single email account."""

    host: str
    username: str
    password: str = ""


class EmailConfig(BaseModel):
    """Email integration config."""

    accounts: dict[str, EmailAccount] = Field(default_factory=dict)


class PhotosConfig(BaseModel):
    """Apple Photos integration config."""

    enabled: bool = False
    photos_library: str = "~/Pictures/Photos Library.photoslibrary"
    db_path: str = "memory/photos.db"
    vlm_model: str = "minicpm-v-4.6"
    cache_analyses: bool = True
    max_results: int = 20


class IntegrationsConfig(BaseModel):
    """All third-party integrations."""

    obsidian: ObsidianConfig = Field(default_factory=ObsidianConfig)
    calendar: CalendarConfig = Field(default_factory=CalendarConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)
    photos: PhotosConfig = Field(default_factory=PhotosConfig)


class OptimizationConfig(BaseModel):
    """Optimization / training config."""

    auto_train_threshold: int = 50
    min_interactions: int = 20


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    """Root configuration model."""

    models: ModelsConfig = Field(default_factory=ModelsConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    integrations: IntegrationsConfig = Field(default_factory=IntegrationsConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)


# ---------------------------------------------------------------------------
# Default config (used when no config.yaml exists)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict[str, Any] = {
    "models": {
        "default": "gemma-4-e2b-it-4bit",
        "available": {
            "gemma-4-e2b-it-4bit": {
                "backend": "gemma",
                "model_id": "mlx-community/gemma-4-e2b-it-4bit",
            },
            "minicpm-v-4.6": {
                "backend": "minicpmv",
                "model_id": "openbmb/MiniCPM-V-4.6",
            },
        },
    },
    "embedding": {
        "url": "http://localhost:8001",
        "dimensions": 768,
    },
    "integrations": {
        "obsidian": {"vaults": {}},
        "calendar": {"accounts": {}},
        "email": {"accounts": {}},
        "photos": {},
    },
    "optimization": {
        "auto_train_threshold": 50,
        "min_interactions": 20,
    },
}


# ---------------------------------------------------------------------------
# Loading logic
# ---------------------------------------------------------------------------

_CONFIG: AppConfig | None = None


def load_config(path: str | Path | None = None, *, reload: bool = False) -> AppConfig:
    """Load and validate the application configuration.

    Parameters
    ----------
    path:
        Path to a YAML config file.  When *None* the function looks for
        ``config.yaml`` in the project root (one level above ``server/``).
    reload:
        Force re-reading even if a cached config exists (useful in tests).

    Returns
    -------
    AppConfig
        A fully validated, immutable configuration object.
    """
    global _CONFIG  # noqa: PLW0603
    if _CONFIG is not None and not reload:
        return _CONFIG

    raw: dict[str, Any]

    if path is not None:
        config_path = Path(path)
    else:
        # Default: project_root/config.yaml
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"

    if config_path.is_file():
        with open(config_path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    else:
        raw = {}

    # Merge with defaults (user values override defaults)
    merged = _deep_merge(_DEFAULT_CONFIG, raw)

    # Substitute environment variables in all string values
    merged = _walk_and_substitute(merged)

    config = AppConfig.model_validate(merged)
    _CONFIG = config
    return config


def get_config() -> AppConfig:
    """Return the cached config, loading defaults if :func:`load_config` was
    not called yet."""
    global _CONFIG  # noqa: PLW0603
    if _CONFIG is None:
        return load_config()
    return _CONFIG


def reset_config() -> None:
    """Clear cached config (for testing)."""
    global _CONFIG  # noqa: PLW0603
    _CONFIG = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
