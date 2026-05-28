"""Tests for model/backends.py -- strategy-pattern backends and ModelRegistry."""

import json
import pytest

from model.backends import (
    DEFAULT_MODELS,
    GemmaBackend,
    MiniCPMBackend,
    MiniCPMVBackend,
    ModelRegistry,
)

# Build XML closing tags at runtime to avoid literal '</' in source
_EF = chr(60) + "/" + "function" + chr(62)
_EP = chr(60) + "/" + "parameter" + chr(62)


# ---------------------------------------------------------------------------
# ModelRegistry basics
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_register_and_list_models(self):
        registry = ModelRegistry()
        gemma = GemmaBackend()
        minicpm = MiniCPMBackend()
        registry.register("gemma-4-e2b-it-4bit", gemma)
        registry.register("minicpm5-1b-mlx", minicpm)
        models = registry.list_models()
        ids = [m["id"] for m in models]
        assert "gemma-4-e2b-it-4bit" in ids
        assert "minicpm5-1b-mlx" in ids
        assert len(models) == 2

    def test_list_models_backend_display_names(self):
        registry = ModelRegistry()
        registry.register_defaults()
        models = registry.list_models()
        by_id = {m["id"]: m["backend"] for m in models}
        assert by_id["gemma-4-e2b-it-4bit"] == "mlx_vlm"
        assert by_id["minicpm5-1b-mlx"] == "mlx_lm"

    def test_unknown_model_raises(self):
        registry = ModelRegistry()
        with pytest.raises(ValueError, match="(?i)unknown"):
            registry.get_backend("nonexistent")

    def test_default_models_populated(self):
        assert "gemma-4-e2b-it-4bit" in DEFAULT_MODELS
        assert "minicpm5-1b-mlx" in DEFAULT_MODELS
        assert DEFAULT_MODELS["gemma-4-e2b-it-4bit"]["backend"] == "gemma"
        assert DEFAULT_MODELS["minicpm5-1b-mlx"]["backend"] == "minicpm"

    def test_register_default_models(self):
        registry = ModelRegistry()
        registry.register_defaults()
        models = registry.list_models()
        ids = [m["id"] for m in models]
        assert "gemma-4-e2b-it-4bit" in ids
        assert "minicpm5-1b-mlx" in ids


# ---------------------------------------------------------------------------
# GemmaBackend.parse_tool_calls
# ---------------------------------------------------------------------------


class TestGemmaBackendParseToolCalls:
    def setup_method(self):
        self.backend = GemmaBackend()

    def test_native_gemma_format_single(self):
        text = '<|tool_call>call:bash{command:"ls -la"}<tool_call|>'
        calls = self.backend.parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "bash"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["command"] == "ls -la"
        assert calls[0]["id"] == "call_0"
        assert calls[0]["type"] == "function"

    def test_native_gemma_format_multiple(self):
        text = (
            '<|tool_call>call:bash{command:"echo hello"}<tool_call|>'
            '<|tool_call>call:read_file{path:"/tmp/a.txt"}<tool_call|>'
        )
        calls = self.backend.parse_tool_calls(text)
        assert len(calls) == 2
        assert calls[0]["function"]["name"] == "bash"
        assert calls[1]["function"]["name"] == "read_file"
        assert calls[1]["id"] == "call_1"

    def test_native_gemma_format_int_args(self):
        text = '<|tool_call>call:search{query:"test",limit:10}<tool_call|>'
        calls = self.backend.parse_tool_calls(text)
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["limit"] == 10

    def test_native_gemma_format_bool_args(self):
        text = '<|tool_call>call:config{verbose:true}<tool_call|>'
        calls = self.backend.parse_tool_calls(text)
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["verbose"] is True

    def test_json_fallback_format(self):
        text = '```json\n{"name": "bash", "arguments": {"command": "pwd"}}\n```'
        calls = self.backend.parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "bash"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["command"] == "pwd"

    def test_json_fallback_no_markdown(self):
        text = '{"name": "read_file", "arguments": {"path": "/etc/hosts"}}'
        calls = self.backend.parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "read_file"

    def test_empty_text_returns_empty(self):
        assert self.backend.parse_tool_calls("") == []
        assert self.backend.parse_tool_calls("Just a regular response.") == []

    def test_native_takes_precedence_over_json(self):
        """When native format is found, JSON fallback is skipped."""
        text = (
            '<|tool_call>call:bash{command:"ls"}<tool_call|>'
            '\n```json\n{"name": "read_file", "arguments": {"path": "/tmp"}}\n```'
        )
        calls = self.backend.parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "bash"


# ---------------------------------------------------------------------------
# MiniCPMBackend.parse_tool_calls
# ---------------------------------------------------------------------------


def _xml(func_name, params):
    """Build an XML tool call string using chr() for angle brackets."""
    lt = chr(60)
    gt = chr(62)
    param_str = ""
    for key, val in params.items():
        param_str += lt + "parameter=" + key + gt + val + _EP + "\n"
    return lt + "function=" + func_name + gt + "\n" + param_str + _EF


class TestMiniCPMBackendParseToolCalls:
    def setup_method(self):
        self.backend = MiniCPMBackend()

    def test_xml_single_call(self):
        text = _xml("bash", {"command": "ls -la"})
        calls = self.backend.parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "bash"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["command"] == "ls -la"
        assert calls[0]["id"] == "call_0"
        assert calls[0]["type"] == "function"

    def test_xml_multiple_params(self):
        text = _xml("search", {"query": "hello world", "limit": "5"})
        calls = self.backend.parse_tool_calls(text)
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["query"] == "hello world"
        assert args["limit"] == "5"

    def test_xml_multiple_calls(self):
        text = (
            _xml("bash", {"command": "echo hi"}) + "\n"
            + _xml("read_file", {"path": "/tmp/a.txt"})
        )
        calls = self.backend.parse_tool_calls(text)
        assert len(calls) == 2
        assert calls[0]["function"]["name"] == "bash"
        assert calls[1]["function"]["name"] == "read_file"
        assert calls[1]["id"] == "call_1"

    def test_empty_text_returns_empty(self):
        assert self.backend.parse_tool_calls("") == []
        assert self.backend.parse_tool_calls("Just a regular response.") == []

    def test_no_function_tags_returns_empty(self):
        text = "Here is some text without any tool calls."
        assert self.backend.parse_tool_calls(text) == []


# ---------------------------------------------------------------------------
# MiniCPMVBackend.parse_tool_calls
# ---------------------------------------------------------------------------


class TestMiniCPMVBackendParseToolCalls:
    def setup_method(self):
        self.backend = MiniCPMVBackend()

    def test_json_format_single(self):
        text = '```json\n{"name": "bash", "arguments": {"command": "ls"}}\n```'
        calls = self.backend.parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "bash"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["command"] == "ls"

    def test_json_format_no_markdown(self):
        text = '{"name": "click", "arguments": {"x": 100, "y": 200}}'
        calls = self.backend.parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "click"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["x"] == 100
        assert args["y"] == 200

    def test_empty_text_returns_empty(self):
        assert self.backend.parse_tool_calls("") == []
        assert self.backend.parse_tool_calls("Just a regular response.") == []

    def test_default_models_includes_minicpmv(self):
        assert "minicpm-v-4.6" in DEFAULT_MODELS
        assert DEFAULT_MODELS["minicpm-v-4.6"]["backend"] == "minicpmv"
