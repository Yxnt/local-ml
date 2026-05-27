# Multi-Model Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MiniCPM5-1B-MLX support alongside existing Gemma 4, with request-parameter routing and single-model hot-switching.

**Architecture:** Strategy pattern with ModelBackend base class. ModelRegistry manages lazy loading and hot-switching. Each backend encapsulates its own loader, chat template, tool call parser, and generate logic. Streaming is handled at the routing layer (generate-then-chunk).

**Tech Stack:** Python (FastAPI, mlx_vlm, mlx_lm), JavaScript (pi-coding-agent SDK)

**Known Risk:** `mlx_lm.generate()` API signature may differ from `mlx_vlm.generate()`. The `mlx_vlm` version accepts `model`, `processor`, `prompt`, `image`, `max_tokens`, `temperature`, `top_p`, `verbose`. The `mlx_lm` version may use different parameter names or a different calling convention. Task 4 includes a verification step to check the actual API before writing the implementation. If the API differs, adapt the MiniCPMBackend.generate() method accordingly.

---

## File Structure

| File | Role |
|------|------|
| `model/backends.py` | ModelBackend ABC, GemmaBackend, MiniCPMBackend, ModelRegistry |
| `model/minicpm_tool_parser.py` | MiniCPM5 XML tool call parser (regex-based) |
| `model/ml.py` | Pydantic request models (ChatCompletionRequest, Tool, etc.) |
| `model/__init__.py` | Re-export backend classes |
| `main.py` | FastAPI routing layer, streaming logic |
| `server_message_adapter.py` | OpenAI message normalization (shared by all backends) |
| `server_message_adapter_test.py` | Tests for message normalization |
| `minimal_pi_session.js` | JS client session factory |
| `multiturn_replay.js` | JS multi-turn conversation driver |
| `multiturn_replay.test.js` | Tests for multi-turn replay |

---

### Task 1: Rename normalize_messages_for_gemma -> normalize_messages

**Files:**
- Modify: `server_message_adapter.py:72`
- Modify: `server_message_adapter_test.py:5,16,46,76`

- [ ] **Step 1: Rename function in server_message_adapter.py**

In `server_message_adapter.py`, rename the function on line 72:

```python
# Before:
def normalize_messages_for_gemma(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
# After:
def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
```

- [ ] **Step 2: Update test imports and references**

In `server_message_adapter_test.py`, update all references:

```python
# Line 5: change import
from server_message_adapter import (
    build_tool_prompt_prefix,
    normalize_messages,  # was normalize_messages_for_gemma
)

# Lines 16, 46, 76: change all calls
normalized = normalize_messages(messages)  # was normalize_messages_for_gemma
```

- [ ] **Step 3: Run existing tests to verify**

Run: `python -m pytest server_message_adapter_test.py -v`
Expected: All 3 tests PASS

- [ ] **Step 4: Commit**

```
git add server_message_adapter.py server_message_adapter_test.py
git commit -m "refactor: rename normalize_messages_for_gemma -> normalize_messages"
```

---

### Task 2: Update model/ml.py

**Files:**
- Modify: `model/ml.py`

- [ ] **Step 1: Add enable_thinking field and remove GenerateRequest**

Replace the entire content of `model/ml.py` with:

```python
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra='ignore')
    role: str
    content: Optional[str] = None


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = ""
    parameters: Optional[dict] = None


class Tool(BaseModel):
    type: Literal['function'] = 'function'
    function: ToolFunction


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra='ignore')
    model: str
    messages: List[ChatMessage]
    tools: Optional[List[Tool]] = None
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    enable_thinking: Optional[bool] = None
```

- [ ] **Step 2: Commit**

```
git add model/ml.py
git commit -m "feat: add enable_thinking to ChatCompletionRequest, remove dead GenerateRequest"
```

---

### Task 3: Create MiniCPM5 XML tool call parser

**Files:**
- Create: `model/minicpm_tool_parser.py`
- Create: `model/minicpm_tool_parser_test.py`

- [ ] **Step 1: Write failing tests**

```python
# model/minicpm_tool_parser_test.py
import unittest
from model.minicpm_tool_parser import parse_mcp_tool_calls


class ParseMcpToolCallsTests(unittest.TestCase):
    def test_single_tool_call_with_one_param(self):
        text = "<function=get_weather>"
        text += "\n  <parameter=city>Beijing</parameter>"
        text += "\n</function>"
        calls = parse_mcp_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        import json
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(args["city"], "Beijing")

    def test_multiple_tool_calls(self):
        text = "<function=get_weather>"
        text += "\n  <parameter=city>Beijing</parameter>"
        text += "\n</function>"
        text += "\n<function=get_time>"
        text += "\n  <parameter=timezone>Asia/Shanghai</parameter>"
        text += "\n</function>"
        calls = parse_mcp_tool_calls(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(calls[1]["function"]["name"], "get_time")

    def test_no_tool_calls(self):
        text = "Hello, how can I help you?"
        calls = parse_mcp_tool_calls(text)
        self.assertEqual(len(calls), 0)

    def test_mixed_text_and_tool_calls(self):
        text = "Let me check the weather.\n<function=get_weather>"
        text += "\n  <parameter=city>Shanghai</parameter>"
        text += "\n</function>\nDone!"
        calls = parse_mcp_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")

    def test_multiple_params(self):
        text = "<function=search>"
        text += "\n  <parameter=query>Python</parameter>"
        text += "\n  <parameter=limit>10</parameter>"
        text += "\n</function>"
        calls = parse_mcp_tool_calls(text)
        self.assertEqual(len(calls), 1)
        import json
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(args["query"], "Python")
        self.assertEqual(args["limit"], "10")


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest model/minicpm_tool_parser_test.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement the parser**

```python
# model/minicpm_tool_parser.py
import json
import re
from typing import List


def parse_mcp_tool_calls(text: str) -> List[dict]:
    """Parse MiniCPM5 XML-style tool call output into OpenAI-compatible format."""
    calls = []
    pattern = r"<function=(\w+)>(.*?)</function>"
    for match in re.finditer(pattern, text, re.DOTALL):
        name = match.group(1)
        body = match.group(2)
        args = {}
        param_pattern = r"<parameter=(\w+)>(.*?)</parameter>"
        for pm in re.finditer(param_pattern, body, re.DOTALL):
            args[pm.group(1)] = pm.group(2).strip()
        calls.append({
            "id": f"call_{len(calls)}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })
    return calls
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest model/minicpm_tool_parser_test.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```
git add model/minicpm_tool_parser.py model/minicpm_tool_parser_test.py
git commit -m "feat: add MiniCPM5 XML tool call parser with tests"
```

---

### Task 4: Create model/backends.py

**Files:**
- Create: `model/backends.py`
- Create: `model/backends_test.py`

**Risk:** `mlx_lm.generate()` API may differ from `mlx_vlm.generate()`. Step 3 includes verification.

- [ ] **Step 1: Write failing tests for ModelRegistry**

```python
# model/backends_test.py
import unittest
from unittest.mock import MagicMock
from model.backends import ModelRegistry, ModelBackend, GemmaBackend, MiniCPMBackend


class ModelRegistryTests(unittest.TestCase):
    def test_register_and_list_models(self):
        registry = ModelRegistry()
        registry.register("gemma", {"backend": "gemma", "model_id": "test/gemma"})
        registry.register("minicpm", {"backend": "minicpm", "model_id": "test/minicpm"})
        models = registry.list_models()
        self.assertEqual(len(models), 2)
        ids = [m["id"] for m in models]
        self.assertIn("gemma", ids)
        self.assertIn("minicpm", ids)

    def test_get_unknown_model_raises(self):
        registry = ModelRegistry()
        import asyncio
        with self.assertRaises(ValueError):
            asyncio.run(registry.get_or_load('nonexistent'))


class GemmaBackendTests(unittest.TestCase):
    def test_parse_tool_calls_gemma_format(self):
        backend = GemmaBackend.__new__(GemmaBackend)
        text = '</tool_call>call:bash{command:ls}<tool_call/'
        calls = backend.parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "bash")

    def test_parse_tool_calls_json_fallback(self):
        backend = GemmaBackend.__new__(GemmaBackend)
        text = '```json\n{"name":"bash","arguments":{"command":"ls"}}\n```'
        calls = backend.parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "bash")


class MiniCPMBackendTests(unittest.TestCase):
    def test_parse_tool_calls_xml_format(self):
        backend = MiniCPMBackend.__new__(MiniCPMBackend)
        text = '<function=get_weather>\n  <parameter=city>Beijing</parameter>\n</function>'
        calls = backend.parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")

    def test_parse_tool_calls_empty(self):
        backend = MiniCPMBackend.__new__(MiniCPMBackend)
        calls = backend.parse_tool_calls("Hello!")
        self.assertEqual(len(calls), 0)


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest model/backends_test.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Verify mlx_lm API (CRITICAL RISK MITIGATION)**

Before writing the implementation, check the actual `mlx_lm` API. Run:

```
python3 -c 'import inspect; from mlx_lm import load, generate; print(inspect.signature(load)); print(inspect.signature(generate))'
```

Expected output shows the function signatures. Compare with `mlx_vlm`:
- `mlx_vlm.load(model_id)` returns `(model, processor)`
- `mlx_vlm.generate(model=, processor=, prompt=, image=, max_tokens=, temperature=, top_p=, verbose=)`

If `mlx_lm.generate()` has a different signature (e.g. `generate(model, tokenizer, prompt, ...)`), adapt MiniCPMBackend accordingly.

If `mlx_lm` is not installed, install it: `pip install mlx-lm`

- [ ] **Step 4: Implement model/backends.py**

Write `model/backends.py`. Key design:
- `ModelBackend` ABC with load/unload/generate/apply_chat_template/parse_tool_calls/warmup
- `GemmaBackend` migrates all Gemma-specific logic from main.py (convert_openai_tools_to_gemma, parse_tool_calls with <|tool_call|> regex + JSON fallback)
- `MiniCPMBackend` uses mlx_lm for loading/generating, tokenizer.apply_chat_template(enable_thinking=...), and delegates parse_tool_calls to minicpm_tool_parser
- `ModelRegistry` with async get_or_load, asyncio.Lock, lazy loading, warmup after load

```python
# model/backends.py
import abc
import asyncio
import json
import os
import re
from typing import Optional

from server_message_adapter import normalize_messages, build_tool_prompt_prefix
from model.minicpm_tool_parser import parse_mcp_tool_calls


DEFAULT_MODELS = {
    "gemma-4-e2b-it-4bit": {"backend": "gemma", "model_id": "mlx-community/gemma-4-e2b-it-4bit"},
    "minicpm5-1b-mlx": {"backend": "minicpm", "model_id": "openbmb/MiniCPM5-1B-MLX"},
}


class ModelBackend(abc.ABC):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._loaded = False

    @abc.abstractmethod
    def load(self) -> None: ...
    @abc.abstractmethod
    def unload(self) -> None: ...
    @abc.abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str: ...
    @abc.abstractmethod
    def apply_chat_template(self, messages: list, tools: Optional[list] = None, **kwargs) -> str: ...
    @abc.abstractmethod
    def parse_tool_calls(self, text: str) -> list[dict]: ...

    def warmup(self) -> None:
        try:
            self.generate("hi", max_tokens=1, temperature=0.7, top_p=0.9)
        except Exception as e:
            print(f"[WARN] Warmup failed for {self.model_id}: {e}")


class GemmaBackend(ModelBackend):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self.model = None
        self.processor = None

    def load(self) -> None:
        import mlx.core as mx
        from mlx_vlm import load as vlm_load
        self.model, self.processor = vlm_load(self.model_id)
        mx.eval(self.model.parameters())
        mx.set_cache_limit(8 * 1024 * 1024 * 1024)
        self._loaded = True

    def unload(self) -> None:
        self.model = None
        self.processor = None
        self._loaded = False
        import gc; gc.collect()

    def generate(self, prompt, max_tokens, temperature, top_p) -> str:
        from mlx_vlm import generate as vlm_generate
        output = vlm_generate(model=self.model, processor=self.processor, prompt=prompt, image=None, max_tokens=max_tokens, temperature=temperature, top_p=top_p, verbose=False)
        return getattr(output, "text", str(output))

    def apply_chat_template(self, messages, tools=None, **kwargs) -> str:
        if tools:
            gemma_tools = self._convert_tools(tools)
            try:
                return self.processor.tokenizer.apply_chat_template(messages, tools=gemma_tools, tokenize=False, add_generation_prompt=True)
            except TypeError: pass
        if tools:
            messages = build_tool_prompt_prefix(messages, tools)
        return self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def parse_tool_calls(self, text: str) -> list[dict]:
        calls = []
        gemma_pattern = r"<\|tool_call\>call:(.+?)\{(.+?)\}<tool_call\|/"
        for name, args_text in re.findall(gemma_pattern, text):
            def cast(v):
                v = v.strip()
                try: return int(v)
                except ValueError:
                    try: return float(v)
                    except ValueError: return {'true':True,'false':False}.get(v.lower(), v.strip("'"))
            args = {}
            for k, v1, v2 in re.findall(r'(\w+):(?:<">(.*?)<">|([^,}]*))', args_text):
                args[k] = cast(v1 or v2)
            calls.append({"id": f"call_{len(calls)}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}})
        if calls: return calls
        json_pattern = r"(?:```(?:json)?\s*)?(\{[\s\S]*?\})(?:\s*```)?"
        for match in re.finditer(json_pattern, text):
            try:
                obj = json.loads(match.group(1))
                if "name" in obj and "arguments" in obj:
                    calls.append({"id": f"call_{len(calls)}", "type": "function", "function": {"name": obj["name"], "arguments": json.dumps(obj["arguments"])}})
            except json.JSONDecodeError: continue
        return calls

    @staticmethod
    def _convert_tools(tools):
        return [{'type':'function','function':{'name':t.function.name,'description':t.function.description or '','parameters':t.function.parameters or {'type':'object','properties':{}}}} for t in tools]


class MiniCPMBackend(ModelBackend):
    def __init__(self, model_id):
        super().__init__(model_id)
        self.model = None
        self.tokenizer = None

    def load(self):
        from mlx_lm import load as lm_load
        self.model, self.tokenizer = lm_load(self.model_id)
        self._loaded = True

    def unload(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False
        import gc; gc.collect()

    def generate(self, prompt, max_tokens, temperature, top_p) -> str:
        from mlx_lm import generate as lm_generate
        # NOTE: mlx_lm.generate() signature verified in Task 4 Step 3. Adapt if needed.
        return lm_generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    def apply_chat_template(self, messages, tools=None, **kwargs) -> str:
        enable_thinking = kwargs.get('enable_thinking', False)
        kw = dict(tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
        if tools:
            return self.tokenizer.apply_chat_template(messages, tools=tools, **kw)
        return self.tokenizer.apply_chat_template(messages, **kw)

    def parse_tool_calls(self, text):
        return parse_mcp_tool_calls(text)


class ModelRegistry:
    _BACKENDS = {'gemma': GemmaBackend, 'minicpm': MiniCPMBackend}

    def __init__(self):
        self._models = {}
        self._active = None
        self._active_name = None
        self._default = None
        self._lock = asyncio.Lock()

    def register(self, name, config):
        self._models[name] = config

    def set_default(self, name):
        self._default = name

    def list_models(self):
        return [{'id': n, 'backend': c['backend']} for n, c in self._models.items()]

    async def get_or_load(self, name):
        if not name:
            name = self._default or os.environ.get('DEFAULT_MODEL', 'gemma-4-e2b-it-4bit')
        if name not in self._models:
            available = ', '.join(self._models.keys())
            raise ValueError(f"Unknown model: {name}. Available: {available}")
        async with self._lock:
            if self._active_name == name and self._active and self._active._loaded:
                return self._active
            if self._active:
                print(f"[REGISTRY] Unloading {self._active_name}...")
                self._active.unload()
            cfg = self._models[name]
            backend_cls = self._BACKENDS[cfg['backend']]
            backend = backend_cls(cfg['model_id'])
            print(f"[REGISTRY] Loading {name}...")
            backend.load()
            print(f"[REGISTRY] Warming up {name}...")
            backend.warmup()
            self._active = backend
            self._active_name = name
            return self._active
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest model/backends_test.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```
git add model/backends.py model/backends_test.py
git commit -m "feat: add ModelBackend, GemmaBackend, MiniCPMBackend, ModelRegistry"
```

---

### Task 5: Update model/__init__.py

**Files:**
- Modify: `model/__init__.py`

- [ ] **Step 1: Add re-exports**

```python
from model.backends import ModelBackend, GemmaBackend, MiniCPMBackend, ModelRegistry
__all__ = ['ModelBackend', 'GemmaBackend', 'MiniCPMBackend', 'ModelRegistry']
```

- [ ] **Step 2: Verify imports work**

Run: `python -c 'from model import ModelRegistry; print("OK")'`
Expected: `OK`

- [ ] **Step 3: Commit**

```
git add model/__init__.py
git commit -m "feat: export backend classes from model package"
```

---

### Task 6: Refactor main.py to use ModelRegistry

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Replace main.py content**

Replace the entire content of `main.py`. Key changes:
- Remove hardcoded model loading (lines 19-25)
- Remove convert_openai_tools_to_gemma (lines 58-72, moved to GemmaBackend)
- Remove parse_tool_calls (lines 74-124, moved to backends)
- Remove lifespan warmup (lines 127-138, now lazy)
- Remove /generate endpoint (lines 244-255)
- Add ModelRegistry initialization
- Add GET /v1/models endpoint
- Route /v1/chat/completions through registry.get_or_load()
- Pass enable_thinking to backend.apply_chat_template()
- Use backend.parse_tool_calls() instead of global function

The new main.py structure:
- Pydantic request types (ToolFunction, Tool, ChatCompletionRequest) for FastAPI validation
- ModelRegistry init from DEFAULT_MODELS
- GET /v1/models
- POST /v1/chat/completions with registry routing
- Streaming logic (unchanged, stays in main.py)

Full implementation code is in the design doc and Task 4 shows the backend API. The main.py changes are a reorganization of existing code, not new logic.

- [ ] **Step 2: Run existing Python tests**

Run: `python -m pytest server_message_adapter_test.py -v`
Expected: All tests PASS (backward compatibility)

- [ ] **Step 3: Commit**

```
git add main.py
git commit -m "feat: refactor main.py to use ModelRegistry with multi-model routing"
```

---

### Task 7: Update JS clients

**Files:**
- Modify: `minimal_pi_session.js`
- Modify: `multiturn_replay.js`

- [ ] **Step 1: Update minimal_pi_session.js**

Add optional `model` parameter to `createLocalPiSession`:

```javascript
// In createLocalPiSession function signature, add model param:
export async function createLocalPiSession({
  apiBase = DEFAULT_API_BASE,
  cwd = process.cwd(),
  systemPrompt = SYSTEM_PROMPT,
  sessionManager = SessionManager.inMemory(cwd),
  model = LOCAL_MODEL_ID,  // NEW: optional model override
} = {}) {
  // Use `model` instead of hardcoded LOCAL_MODEL_ID:
  // 1. In registerProvider: id: model, name: model
  // 2. In find: modelRegistry.find('local', model)
```

- [ ] **Step 2: Update multiturn_replay.js**

Add optional `model` parameter to `chat()`:

```javascript
// In chat() function signature, add model param:
async function chat(messages, tools, apiBase = DEFAULT_API_BASE, model = LOCAL_MODEL_ID) {
  // Use `model` in fetch body instead of LOCAL_MODEL_ID
```

- [ ] **Step 3: Run JS tests**

Run: `npm test`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```
git add minimal_pi_session.js multiturn_replay.js
git commit -m "feat: add optional model parameter to JS clients"
```

---

### Task 8: Final verification

- [ ] **Step 1: Run all Python tests**

Run: `python -m pytest server_message_adapter_test.py model/backends_test.py model/minicpm_tool_parser_test.py -v`
Expected: All tests PASS

- [ ] **Step 2: Run all JS tests**

Run: `npm test`
Expected: All tests PASS

- [ ] **Step 3: Verify model listing endpoint**

Start the server: `python main.py`
Test: `curl http://localhost:8000/v1/models`
Expected: JSON with both gemma-4-e2b-it-4bit and minicpm5-1b-mlx