# Multi-Model Support Design

## Goal

Extend the local ML inference stack to support multiple models (Gemma 4 + MiniCPM5-1B-MLX) with request-parameter routing, while keeping memory usage low via single-model hot-switching.

## Architecture

```
main.py (FastAPI routing layer)
  +-- ModelRegistry
  |     +-- models: dict[str, ModelMeta]       # registered models (metadata only)
  |     +-- active: ModelBackend | None         # currently loaded model
  |     +-- get_or_load(name) -> ModelBackend   # switch if needed
  |     +-- list_models() -> list[str]
  |
  +-- ModelBackend (abstract base)
  |     +-- load()
  |     +-- unload()
  |     +-- generate(messages, tools, **kwargs) -> str
  |     +-- apply_chat_template(messages, tools, **kwargs) -> str
  |     +-- parse_tool_calls(text) -> list[dict]
  |
  +-- GemmaBackend(mlx_vlm)
  |     +-- migrates existing main.py logic
  |
  +-- MiniCPMBackend(mlx_lm)
        +-- new implementation
```

## Model Loading Strategy

- **Lazy load**: Models register metadata at startup; weights load on first request
- **Single-model constraint**: Only one model in memory at a time (Mac memory pressure)
- **Switch flow**: unload() old model -> load() new model (~5-10s one-time cost)
- **Lock**: asyncio.Lock prevents concurrent switch races

## Endpoint Design

### POST /v1/chat/completions

Request adds enable_thinking field (used by MiniCPM5, ignored by Gemma):

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

Routing: registry.get_or_load(body["model"]).generate(...).

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

## Backend Implementations

### GemmaBackend

Migrates existing logic from main.py:
- Loader: mlx_vlm.load()
- Chat template: processor.tokenizer.apply_chat_template(messages, tools=...)
- Tool call parsing: <|tool_call|> regex + JSON fallback
- Config: mlx_vlm.utils.load_config()

### MiniCPMBackend

New implementation:
- Loader: mlx_lm.load()
- Chat template: tokenizer.apply_chat_template(messages, tools=..., enable_thinking=...)
- Tool call parsing: XML-style function blocks (MiniCPM5 native format)
- Generation: mlx_lm.generate()
- Model ID: openbmb/MiniCPM5-1B-MLX

## server_message_adapter 归属

当前 server_message_adapter.py 中的函数命名带 "gemma"，但逻辑是通用的（OpenAI format -> pure role/content）。重构方案:
- normalize_messages_for_gemma -> normalize_messages (通用，两个 backend 共用)
- build_tool_prompt_prefix, format_tools_for_prompt, extract_content 保持不变（已是通用名)
- server_message_adapter.py 文件名不变，避免不必要的文件重命名
- 两个 backend 都调用 normalize_messages 处理输入，再各自调用自己的 apply_chat_template

## Streaming 策略

两个 backend 统一使用 same streaming approach:
- mlx_vlm (Gemma) 和 mlx_lm (MiniCPM5) 都不支持真 token-level streaming
- 策略: 先完整生成 output_text，再分段推送 SSE events
- 流式 tool_calls: 先推 tool call header，再分段推 arguments
- 流式 text: 按固定 chunk size (3 chars) 分段推送
- 流式逻辑在 main.py 路由层实现，不在 backend 内部

## JS 客户端兼容

### minimal_pi_session.js
- DEFAULT_API_BASE 和 LOCAL_MODEL_ID 保持为默认值
- createLocalPiSession() 增加可选 model 参数，默认使用 LOCAL_MODEL_ID
- 调用方可传入不同 model name 来使用不同模型

### multiturn_replay.js
- 增加可选 model 参数，默认使用现有 LOCAL_MODEL_ID
- 发送请求时使用传入的 model name

### main.js
- 简单 demo，不需要改动（使用默认模型即可）

## File Changes

| File | Action | Description |
|------|--------|-------------|
| model/backends.py | **New** | ModelBackend base, GemmaBackend, MiniCPMBackend, ModelRegistry |
| model/minicpm_tool_parser.py | **New** | MiniCPM5 XML tool call parser |
| main.py | **Modify** | Remove hardcoded model + /generate route, use registry routing |
| model/ml.py | **Modify** | Add enable_thinking to ChatCompletionRequest |
| server_message_adapter.py | **Modify** | Rename normalize_messages_for_gemma -> normalize_messages |
| minimal_pi_session.js | **Modify** | Add optional model parameter to createLocalPiSession |
| multiturn_replay.js | **Modify** | Add optional model parameter |

## Configuration

Environment variables:
- DEFAULT_MODEL: model to load on startup (default: gemma-4-e2b-it-4bit)
- MODELS_CONFIG: optional JSON override for model registry entries

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
  <parameter=city>Beijing</parameter>
</function>
```

Parser extracts function name and parameter key-value pairs from XML tags.

## Error Handling

- **Unknown model**: Return 400 with list of available models
- **Model load failure**: Return 503 with error details, do not crash server
- **Concurrent switch**: asyncio.Lock queues requests during model switch
- **Parse failure**: If tool call parsing fails, return raw text as content (graceful fallback)

## Testing Strategy

1. **Unit tests**: Test each backend independently (mock MLX loaders)
2. **Integration tests**: Test model switching flow with real models
3. **Tool call parsing**: Test XML parser with various MiniCPM5 output formats
4. **Backward compatibility**: Existing tests (server_message_adapter_test.py, minimal_pi_session.test.js) must still pass