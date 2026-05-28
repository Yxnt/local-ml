"""Tests for remote LLM backend."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from backends.remote import RemoteBackend, PREDEFINED_MODELS, create_remote_backend


class TestRemoteBackend:
    """远程后端测试"""

    def test_from_predefined(self):
        backend = RemoteBackend.from_predefined("mimo-v2.5-pro", "test-key")
        assert backend._model_id == "mimo-v2.5-pro"
        assert backend._provider == "xiaomi"
        assert "xiaomimimo.com" in backend._base_url

    def test_from_predefined_unknown(self):
        with pytest.raises(ValueError, match="Unknown model"):
            RemoteBackend.from_predefined("unknown-model", "test-key")

    def test_custom_config(self):
        backend = RemoteBackend(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model_id="custom-model",
        )
        assert backend._base_url == "https://api.example.com/v1"
        assert backend._model_id == "custom-model"

    def test_load_unload(self):
        backend = RemoteBackend(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model_id="test-model",
        )
        backend.load("test-model")
        assert backend._loaded is True

        backend.unload()
        assert backend._loaded is True  # 远程模型保持可用

    def test_warmup(self):
        backend = RemoteBackend(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model_id="test-model",
        )
        # warmup 不应该抛出异常
        backend.warmup()

    def test_parse_tool_calls_json(self):
        backend = RemoteBackend(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model_id="test-model",
        )
        text = '{"name": "bash", "arguments": {"command": "ls"}}'
        calls = backend.parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "bash"

    def test_parse_tool_calls_markdown(self):
        backend = RemoteBackend(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model_id="test-model",
        )
        text = '```json\n{"name": "bash", "arguments": {"command": "ls"}}\n```'
        calls = backend.parse_tool_calls(text)
        assert len(calls) == 1

    def test_parse_tool_calls_empty(self):
        backend = RemoteBackend(
            base_url="https://api.example.com/v1",
            api_key="test-key",
            model_id="test-model",
        )
        calls = backend.parse_tool_calls("No tool calls here")
        assert len(calls) == 0


class TestCreateRemoteBackend:
    """工厂函数测试"""

    def test_create_predefined(self):
        backend = create_remote_backend("mimo-v2.5-pro", "test-key")
        assert backend._model_id == "mimo-v2.5-pro"

    def test_create_custom(self):
        backend = create_remote_backend(
            "custom-model",
            "test-key",
            base_url="https://api.example.com/v1"
        )
        assert backend._model_id == "custom-model"
        assert backend._base_url == "https://api.example.com/v1"


class TestPredefinedModels:
    """预定义模型测试"""

    def test_xiaomi_models_exist(self):
        assert "mimo-v2.5-pro" in PREDEFINED_MODELS
        assert "mimo-v2.5-flash" in PREDEFINED_MODELS

    def test_openai_models_exist(self):
        assert "gpt-4o" in PREDEFINED_MODELS
        assert "gpt-4o-mini" in PREDEFINED_MODELS

    def test_model_config_structure(self):
        for name, config in PREDEFINED_MODELS.items():
            assert "provider" in config
            assert "base_url" in config
            assert "model_id" in config
            assert "description" in config
