import json
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Literal

from server_message_adapter import normalize_messages
from model.backends import ModelRegistry

# ====== Pydantic request types (for FastAPI validation) ======

class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = ""
    parameters: Optional[dict] = None

class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunction

# ====== Registry setup ======

registry = ModelRegistry()
registry.register_defaults()
default_model = os.environ.get("DEFAULT_MODEL", "gemma-4-e2b-it-4bit")


# ====== FastAPI ======

app = FastAPI(title="Local ML Service")


@app.get("/v1/models")
async def list_models():
    return {"data": registry.list_models()}


@app.post("/v1/chat/completions")
async def chat_completions(raw_req: Request):
    body = await raw_req.json()
    print(f"[RAW_REQ] {json.dumps(body, ensure_ascii=False)}")

    model_name = body.get("model") or default_model
    messages_raw = body.get("messages", [])
    tools_raw = body.get("tools")
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens") or 2048
    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 0.9)
    enable_thinking = body.get("enable_thinking") or False

    tools = None
    if tools_raw:
        tools = [Tool(type=t.get("type", "function"), function=ToolFunction(**t.get("function", {}))) for t in tools_raw]

    print(f"[REQ] model={model_name} stream={stream} messages={len(messages_raw)} tools={len(tools) if tools else 0}")

    # Get backend via registry (lazy load + hot-switch)
    try:
        backend = await registry.get_or_load(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model load failed: {e}")

    # Normalize messages and build prompt
    messages_dict = normalize_messages(messages_raw)

    prompt = backend.apply_chat_template(
        messages_dict,
        tools=tools,
        enable_thinking=enable_thinking,
    )
    print(f"[PROMPT] {prompt[:300]}...")

    # Generate
    print("[GEN] generating...")
    output_text = backend.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    print(f"[OUTPUT] {output_text[:200]}...")

    tool_calls = backend.parse_tool_calls(output_text)

    if stream:
        async def stream_response():
            if tool_calls:
                for i, tc in enumerate(tool_calls):
                    yield f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant', 'tool_calls': [{'index': i, 'id': tc['id'], 'type': 'function', 'function': {'name': tc['function']['name'], 'arguments': ''}}]}, 'finish_reason': None}]})}\n\n"
                    args = tc['function']['arguments']
                    chunk_size = max(1, len(args) // 4)
                    for j in range(0, len(args), chunk_size):
                        yield f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [{'index': i, 'function': {'arguments': args[j:j+chunk_size]}}]}, 'finish_reason': None}]})}\n\n"
                yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'tool_calls'}]})}\n\n"
            else:
                for i in range(0, len(output_text), 3):
                    chunk = output_text[i:i+3]
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"
                yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    # Non-streaming
    if tool_calls:
        return {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "choices": [{
                "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
                "finish_reason": "tool_calls"
            }]
        }
    else:
        return {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "choices": [{
                "message": {"role": "assistant", "content": output_text},
                "finish_reason": "stop"
            }]
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
