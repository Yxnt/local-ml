# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local ML inference stack running on Apple Silicon: a Gemma 4 VLM served via MLX with an OpenAI-compatible API, a Jina embedding service, and a SQLite+vector memory system. JavaScript clients use the `@earendil-works/pi-coding-agent` SDK to drive multi-turn tool-calling conversations against the local model.

## Commands

### Python Services

```bash
# Gemma 4 inference server (port 8000)
python main.py

# Jina embedding service (port 8001)
python embedding_service.py

# Initialize memory DB (creates memory/assistant.db)
python memory/db.py
```

### Tests

```bash
# Python unit tests
python -m pytest server_message_adapter_test.py -v

# JavaScript tests (Node.js built-in test runner)
npm test
# or directly:
node --test minimal_pi_session.test.js
```

## Architecture

### Python Layer (Inference)

- [main.py](main.py) — FastAPI server exposing `/v1/chat/completions` (OpenAI-compatible) and `/generate`. Loads `mlx-community/gemma-4-e2b-it-4bit` via `mlx_vlm`. Streaming is simulated (full generation, then chunked delivery). Tool calls are parsed from model output in two formats: Gemma 4 native `<|tool_call>` tags and JSON fallback.
- [server_message_adapter.py](server_message_adapter.py) — Converts OpenAI-format messages (with `tool_calls` and `tool` role) into plain `{role, content}` pairs that Gemma's `apply_chat_template` accepts. `build_tool_prompt_prefix` injects tool definitions into the system prompt when native tool templating fails.
- [embedding_service.py](embedding_service.py) — Standalone FastAPI server using `jinaai/jina-embeddings-v5-omni-nano` with `transformers`. Exposes `/embed` (768-dim vectors) and `/health`.
- [model/ml.py](model/ml.py) — Shared Pydantic models (`ChatCompletionRequest`, `Tool`, etc.) for request validation.

### JavaScript Layer (Agent Client)

- [minimal_pi_session.js](minimal_pi_session.js) — Creates a `pi-coding-agent` session with a single `bash` tool, pointing at the local model's OpenAI-compatible endpoint. Exports `createLocalPiSession()`, `DEFAULT_API_BASE`, `LOCAL_MODEL_ID`, `SYSTEM_PROMPT`.
- [multiturn_replay.js](multiturn_replay.js) — Manual multi-turn conversation driver: sends a prompt, receives tool calls, executes them via `execSync`, feeds results back. Used for testing tool-calling flows without the full agent framework.
- [main.js](main.js) — Simple demo that creates a session, sends a prompt, and prints the conversation history.

### Memory System

- [memory/db.py](memory/db.py) — SQLite database with `sqlite-vec` extension for vector similarity search. Tables: `conversations` (session history), `memories` (long-term memory), `memories_vec` (768-dim embeddings for semantic retrieval).

## Key Patterns

- The Python server does NOT stream tokens from MLX — it generates the full response then chunks it into SSE events. This is a `mlx_vlm` limitation.
- Tool call parsing in `main.py` has two strategies: regex for Gemma 4's native `<|tool_call>call:name{args}<tool_call|>` format, then JSON extraction as fallback.
- `server_message_adapter.py` converts `tool`/`toolResult` role messages into `user`-role messages with structured text, since Gemma's chat template only supports `user`/`model` roles.
- The JS client registers a "terminating" bash tool (`terminate: true`) so the agent stops after executing a command rather than looping.

<!-- pensieve:instructions:start -->
## How To Use Pensieve

Use `.pensieve/` as the first source of architectural intent.

- `maxims/` are active engineering rules.
- `decisions/` are active project decisions.
- `knowledge/` explains boundary maps and debugging paths.
- `pipelines/` gives executable workflows.

Use these project pipelines directly when trigger words match; do not rediscover them through skills first.

- Commit requests (`commit`, `提交`, `git commit`): use `.pensieve/pipelines/run-when-committing.md`. Check staged diff, decide whether reusable insight should be captured, then make atomic commits.
- Refactor requests (`重构`, `refactor`, `大改`, `拆代码`): use `.pensieve/pipelines/run-when-refactoring.md`. Confirm the real problem, fix upstream data authority first, split large work into 2-3 user-visible steps, delete old paths when new paths work, and avoid compatibility/fallback branches.
- Review requests (`review`, `代码审查`, `检查代码`): use `.pensieve/pipelines/run-when-reviewing-code.md`. Start from git history and changed hot spots, verify candidate issues, and report only high-signal findings with evidence and file locations.
<!-- pensieve:instructions:end -->
