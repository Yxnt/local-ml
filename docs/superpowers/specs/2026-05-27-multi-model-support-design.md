# Multi-Model Support Design

## Goal

Extend the local ML inference stack to support multiple models (Gemma 4 + MiniCPM5-1B-MLX) with request-parameter routing, while keeping memory usage low via single-model hot-switching.

## Architecture

```
main.py (FastAPI routing layer)
  ├── ModelRegistry
  │     ├── models: dict[str, ModelMeta]       # registered models (metadata only)
  │     ├── active: ModelBackend | None         # currently loaded model
  │     ├── get_or_load(name) -> ModelBackend   # switch if needed
  │     └── list_models() -> list[str]
  │
  ├── ModelBackend (abstract base)
  │     ├── load()
  │     ├── unload()
  │     ├── generate(messages, tools, **kwargs) -> str
  │     ├── apply_chat_template(messages, tools, **kwargs) -> str
  │     └── parse_tool_calls(text) -> list[dict]
  │
  ├── GemmaBackend(mlx_vlm)
  │     └── migrates existing main.py logic
  │
  └── MiniCPMBackend(mlx_lm)
        └── new implementation
```

## Model Loading Strategy

- **Lazy load**: Models register metadata at startup; weights load on first request
- **Single-model constraint**: Only one model in memory at a time (Mac memory pressure)
- **Switch flow**: `unload()` old model -> `load()` new model (~5-10s one-time cost)
- **Lock**: asyncio.Lock prevents concurrent switch races

## Endpoint Design

### POST /v1/chat/completions

Request adds `enable_thinking` field (used by MiniCPM5, ignored by Gemma):

```json
{
  "model": "minicpm5-1b-mlx",
  "messages": [],
  "tools": [],
  "enable_thinking": false,
  "stream": false,
  "max_tokens": 2048
}
```

Routing: `registry.get_or_load(body["model"]).generate(...)`.

### GET /v1/models

Returns all registered models:

```json
{
  "data": [
    {"id": "gemma-4-e2b-it-4bit", "backend": "mlx_vlm"},
    {"id": "minicpm5-1b-mlx", "backend": "mlx_lm"}
  ]
}
```

### POST /generate (legacy)

Kept for backward compatibility. Uses default model.

## Backend Implementations

### GemmaBackend

Migrates existing logic from main.py:
- Loader: `mlx_vlm.load()`
- Chat template: `processor.tokenizer.apply_chat_template(messages, tools=...)`
- Tool call parsing: `<|tool_call>` regex + JSON fallback
- Config: `mlx_vlm.utils.load_config()`

### MiniCPMBackend

New implementation:
- Loader: `mlx_lm.load()`
- Chat template: `tokenizer.apply_chat_template(messages, tools=..., enable_thinking=...)`
- Tool call parsing: XML-style `<function>` blocks (MiniCPM5 native format)
- Generation: `mlx_lm.generate()`
- Model ID: `openbmb/MiniCPM5-1B-MLX`

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `model/backends.py` | **New** | ModelBackend base, GemmaBackend, MiniCPMBackend, ModelRegistry |
| `model/minicpm_tool_parser.py` | **New** | MiniCPM5 XML tool call parser |
| `main.py` | **Modify** | Remove hardcoded model, use registry for routing |
| `model/ml.py` | **Modify** | Add `enable_thinking` to ChatCompletionRequest |

## Configuration

Environment variables:
- `DEFAULT_MODEL`: model to load on startup (default: `gemma-4-e2b-it-4bit`)
- `MODELS_CONFIG`: optional JSON override for model registry entries

Registry defaults (hardcoded, overridable):

```python
DEFAULT_MODELS = {
    "gemma-4-e2b-it-4bit": {
        "backend": "gemma",
        "model_id": "mlx-community/gemma-4-e2b-it-4bit",
    },
    "minicpm5-1b-mlx": {
        "backend": "minicpm",
        "model_id": "openbmb/MiniCPM5-1B-MLX",
    },
}
```

## Tool Call Formats

### Gemma 4 (existing)

```
<|tool_call>call:function_name{arg1:value1, arg2:value2}<tool_call|>
```

### MiniCPM5 (new)

XML-style output, parsed by regex:

```xml
<function=get_weather>
<parameter=city>北京