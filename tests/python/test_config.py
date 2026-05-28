"""Tests for server.config -- YAML loading, env-var substitution, defaults, validation."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
import yaml

from server.config import (
    AppConfig,
    EmailAccount,
    EmbeddingConfig,
    IntegrationsConfig,
    ModelEntry,
    ModelsConfig,
    OptimizationConfig,
    _deep_merge,
    _substitute_env_vars,
    _walk_and_substitute,
    load_config,
    get_config,
    reset_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_global_config():
    """Ensure each test starts with a clean config cache."""
    reset_config()
    yield
    reset_config()


@pytest.fixture()
def config_dir(tmp_path: Path):
    """Return a tmp directory and a helper that writes a config.yaml into it."""
    def _write(content: str) -> Path:
        p = tmp_path / "config.yaml"
        p.write_text(textwrap.dedent(content), encoding="utf-8")
        return p

    return _write


# ---------------------------------------------------------------------------
# _substitute_env_vars
# ---------------------------------------------------------------------------


class TestSubstituteEnvVars:
    def test_simple_substitution(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MY_VAR", "hello")
        assert _substitute_env_vars("${MY_VAR}") == "hello"

    def test_substitution_with_default(self):
        # Variable that is NOT set -- default should be used
        result = _substitute_env_vars("${UNDEFINED_VAR_123:fallback}")
        assert result == "fallback"

    def test_env_overrides_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MY_VAR", "real")
        assert _substitute_env_vars("${MY_VAR:default}") == "real"

    def test_no_substitution_needed(self):
        assert _substitute_env_vars("plain string") == "plain string"

    def test_multiple_vars_in_one_string(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("A", "alpha")
        monkeypatch.setenv("B", "beta")
        assert _substitute_env_vars("${A}-${B}") == "alpha-beta"

    def test_unresolved_var_left_as_is(self):
        # No env var set, no default -- original expression stays
        assert _substitute_env_vars("${DOES_NOT_EXIST_XYZ}") == "${DOES_NOT_EXIST_XYZ}"


# ---------------------------------------------------------------------------
# _walk_and_substitute (recursive)
# ---------------------------------------------------------------------------


class TestWalkAndSubstitute:
    def test_nested_dict(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PASS", "s3cret")
        data = {"a": {"b": "${PASS}"}, "c": ["${PASS}", 42]}
        result = _walk_and_substitute(data)
        assert result == {"a": {"b": "s3cret"}, "c": ["s3cret", 42]}

    def test_non_string_values_unchanged(self):
        data = {"n": 42, "f": 3.14, "b": True, "none": None}
        assert _walk_and_substitute(data) == data


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_flat_merge(self):
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_override_wins(self):
        assert _deep_merge({"a": 1}, {"a": 99}) == {"a": 99}

    def test_recursive_merge(self):
        base = {"x": {"y": 1, "z": 2}}
        over = {"x": {"y": 10}}
        assert _deep_merge(base, over) == {"x": {"y": 10, "z": 2}}


# ---------------------------------------------------------------------------
# load_config -- YAML loading
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_load_full_config(self, config_dir, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GMAIL_PASS", "secret123")
        path = config_dir("""\
            models:
              default: minicpm-v-4.6
              available:
                minicpm-v-4.6:
                  backend: minicpmv
                  model_id: openbmb/MiniCPM-V-4.6
            embedding:
              url: http://localhost:9000
              dimensions: 1024
            email:
              host: imap.gmail.com
        """)
        # Should not raise
        cfg = load_config(path, reload=True)
        assert isinstance(cfg, AppConfig)

    def test_missing_file_uses_defaults(self):
        cfg = load_config("/nonexistent/path/config.yaml", reload=True)
        assert cfg.models.default == "gemma-4-e2b-it-4bit"
        assert cfg.embedding.url == "http://localhost:8001"
        assert cfg.embedding.dimensions == 768
        assert cfg.optimization.auto_train_threshold == 50
        assert cfg.optimization.min_interactions == 20

    def test_empty_file_uses_defaults(self, config_dir):
        path = config_dir("")  # empty YAML
        cfg = load_config(path, reload=True)
        assert cfg.models.default == "gemma-4-e2b-it-4bit"

    def test_partial_override(self, config_dir):
        path = config_dir("""\
            embedding:
              dimensions: 1024
        """)
        cfg = load_config(path, reload=True)
        # overridden
        assert cfg.embedding.dimensions == 1024
        # default preserved
        assert cfg.embedding.url == "http://localhost:8001"

    def test_env_var_in_yaml(self, config_dir, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MY_EMAIL_PASS", "hunter2")
        path = config_dir("""\
            integrations:
              email:
                accounts:
                  personal:
                    host: imap.gmail.com
                    username: me@gmail.com
                    password: ${MY_EMAIL_PASS}
        """)
        cfg = load_config(path, reload=True)
        acct = cfg.integrations.email.accounts["personal"]
        assert acct.password == "hunter2"

    def test_env_var_default_in_yaml(self, config_dir):
        path = config_dir("""\
            integrations:
              calendar:
                accounts:
                  personal:
                    url: https://caldav.icloud.com
                    username: user@icloud.com
                    password: ${MISSING_VAR:defaultpass}
        """)
        cfg = load_config(path, reload=True)
        acct = cfg.integrations.calendar.accounts["personal"]
        assert acct.password == "defaultpass"


# ---------------------------------------------------------------------------
# load_config -- validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_models_available_is_dict_of_model_entry(self, config_dir):
        path = config_dir("""\
            models:
              available:
                my-model:
                  backend: gemma
                  model_id: some/model
        """)
        cfg = load_config(path, reload=True)
        entry = cfg.models.available["my-model"]
        assert isinstance(entry, ModelEntry)
        assert entry.backend == "gemma"
        assert entry.model_id == "some/model"

    def test_embedding_dimensions_must_be_int(self, config_dir):
        path = config_dir("""\
            embedding:
              dimensions: "not_an_int"
        """)
        with pytest.raises(Exception):
            load_config(path, reload=True)

    def test_optimization_thresholds_are_ints(self, config_dir):
        path = config_dir("""\
            optimization:
              auto_train_threshold: 100
              min_interactions: 5
        """)
        cfg = load_config(path, reload=True)
        assert cfg.optimization.auto_train_threshold == 100
        assert cfg.optimization.min_interactions == 5


# ---------------------------------------------------------------------------
# get_config / caching
# ---------------------------------------------------------------------------


class TestGetConfig:
    def test_get_config_returns_same_instance(self):
        a = get_config()
        b = get_config()
        assert a is b

    def test_reset_config_clears_cache(self):
        a = get_config()
        reset_config()
        b = get_config()
        # New instance after reset (though values are same)
        assert a is not b


# ---------------------------------------------------------------------------
# Typed access to sections
# ---------------------------------------------------------------------------


class TestTypedAccess:
    def test_email_accounts(self, config_dir, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GMAIL_PW", "gpass")
        monkeypatch.setenv("OUTLOOK_PW", "opass")
        path = config_dir("""\
            integrations:
              email:
                accounts:
                  personal:
                    host: imap.gmail.com
                    username: user@gmail.com
                    password: ${GMAIL_PW}
                  work:
                    host: imap.outlook.com
                    username: user@work.com
                    password: ${OUTLOOK_PW}
        """)
        cfg = load_config(path, reload=True)

        assert len(cfg.integrations.email.accounts) == 2
        personal = cfg.integrations.email.accounts["personal"]
        assert isinstance(personal, EmailAccount)
        assert personal.host == "imap.gmail.com"
        assert personal.password == "gpass"

        work = cfg.integrations.email.accounts["work"]
        assert work.host == "imap.outlook.com"
        assert work.password == "opass"

    def test_obsidian_vaults(self, config_dir):
        path = config_dir("""\
            integrations:
              obsidian:
                vaults:
                  main: ~/Documents/Obsidian/Main
                  work: ~/Documents/Obsidian/Work
        """)
        cfg = load_config(path, reload=True)
        assert cfg.integrations.obsidian.vaults["main"] == "~/Documents/Obsidian/Main"
        assert cfg.integrations.obsidian.vaults["work"] == "~/Documents/Obsidian/Work"

    def test_model_entries(self, config_dir):
        path = config_dir("""\
            models:
              default: minicpm-v-4.6
              available:
                minicpm-v-4.6:
                  backend: minicpmv
                  model_id: openbmb/MiniCPM-V-4.6
                gemma-4-e2b-it-4bit:
                  backend: gemma
                  model_id: mlx-community/gemma-4-e2b-it-4bit
        """)
        cfg = load_config(path, reload=True)
        assert cfg.models.default == "minicpm-v-4.6"
        assert "minicpm-v-4.6" in cfg.models.available
        assert "gemma-4-e2b-it-4bit" in cfg.models.available
        assert cfg.models.available["minicpm-v-4.6"].backend == "minicpmv"
