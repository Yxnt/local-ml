from contextlib import asynccontextmanager
import json
import re
import mlx.core as mx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from mlx_vlm import generate as vlm_generate
from mlx_vlm import load
from mlx_vlm.utils import load_config
from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Literal

from server_message_adapter import (
    build_tool_prompt_prefix,
    normalize_messages,
)

MODEL_NAME = "mlx-community/gemma-4-e2b-it-4bit"

model, processor = load(MODEL_NAME)
config = load_config(MODEL_NAME)
mx.eval(model.parameters())
mx.set_cache_limit(8 * 1024 * 1024 * 1024)
print("Model loaded.")


# ====== OpenAI 兼容的请求类型 ======

class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = ""
    parameters: Optional[dict] = None

class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunction

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str
    messages: List[dict]
    tools: Optional[List[Tool]] = None
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


# 旧版 generate 请求（兼容你现有代码）
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9

    
def convert_openai_tools_to_gemma(tools: List[Tool]) -> List[dict]:
    """把 OpenAI 格式的 tools 转成 Gemma 4 apply_chat_template 期望的 JSON schema"""
    schemas = []
    for t in tools:
        fn = t.function
        schema = {
            "type": "function",
            "function": {
                "name": fn.name,
                "description": fn.description or "",
                "parameters": fn.parameters or {"type": "object", "properties": {}}
            }
        }
        schemas.append(schema)
    return schemas

def parse_tool_calls(text: str) -> List[dict]:
    """解析模型输出中的 tool call — 支持 Gemma 4 的 <|tool_call|> 格式和 JSON 格式"""
    calls = []

    # 1. 先尝试 Gemma 4 原生格式: <|tool_call>call:name{args}<tool_call|>
    # 注意结束标记是 <tool_call|> 不是 <|tool_call|>
    gemma_pattern = r'<\|tool_call\>call:(.+?)\{(.+?)\}<tool_call\|>'
    for name, args_text in re.findall(gemma_pattern, text):
        def cast(v):
            v = v.strip()
            try:
                return int(v)
            except ValueError:
                try:
                    return float(v)
                except ValueError:
                    return {"true": True, "false": False}.get(v.lower(), v.strip("'\""))

        args = {}
        for k, v1, v2 in re.findall(r'(\w+):(?:<\|"\|>(.*?)<\|"\|>|([^,}]*))', args_text):
            args[k] = cast(v1 or v2)

        calls.append({
            "id": f"call_{len(calls)}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args)
            }
        })

    if calls:
        return calls

    # 2. 回退到 JSON 格式匹配
    json_pattern = r'(?:```(?:json)?\s*)?(\{[\s\S]*?\})(?:\s*```)?'
    for match in re.finditer(json_pattern, text):
        try:
            obj = json.loads(match.group(1))
            if "name" in obj and "arguments" in obj:
                calls.append({
                    "id": f"call_{len(calls)}",
                    "type": "function",
                    "function": {
                        "name": obj["name"],
                        "arguments": json.dumps(obj["arguments"])
                    }
                })
        except json.JSONDecodeError:
            continue
    return calls
# ====== FastAPI ======

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Warming up model...")
    dummy = processor.tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True
    )
    vlm_generate(
        model=model, processor=processor, prompt=dummy, image=None,
        max_tokens=5, verbose=False,
    )
    print("Warmup done.")
    yield


app = FastAPI(title="Gemma4 Local Service", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(raw_req: Request):
    body = await raw_req.json()
    print(f"[RAW_REQ] {json.dumps(body, ensure_ascii=False)}")

    # 手动解析，跳过 Pydantic 校验
    messages_raw = body.get("messages", [])
    tools_raw = body.get("tools")
    tools = None
    if tools_raw:
        tools = [Tool(type=t.get("type", "function"), function=ToolFunction(**t.get("function", {}))) for t in tools_raw]

    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens") or 2048
    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 0.9)

    print(f"[REQ] stream={stream} messages={len(messages_raw)} tools={len(tools) if tools else 0}")

    # 尝试用 Gemma 4 原生的 apply_chat_template(tools=...) 方式
    gemma_tools = convert_openai_tools_to_gemma(tools) if tools else None

    messages_dict = normalize_messages(messages_raw)

    prompt = None
    if gemma_tools:
        try:
            prompt = processor.tokenizer.apply_chat_template(
                messages_dict, tools=gemma_tools, tokenize=False, add_generation_prompt=True
            )
            print("[APPLY_CHAT_TEMPLATE] Native tools support OK")
        except TypeError as e:
            print(f"[APPLY_CHAT_TEMPLATE] Native tools failed: {e}, falling back to manual prompt")
            prompt = None

    if prompt is None:
        if tools:
            messages_dict = build_tool_prompt_prefix(messages_dict, tools_raw)

        prompt = processor.tokenizer.apply_chat_template(
            messages_dict, tokenize=False, add_generation_prompt=True
        )
    print(f"[PROMPT] {prompt[:300]}...")

    # mlx_vlm 不支持真 streaming，先完整生成再分段推送
    print("[GEN] generating...")
    output = vlm_generate(
        model=model, processor=processor, prompt=prompt, image=None,
        max_tokens=max_tokens, temperature=temperature,
        top_p=top_p, verbose=False,
    )
    # vlm_generate 返回 GenerationResult 对象，取 .text 属性
    output_text = getattr(output, "text", str(output))
    print(f"[OUTPUT] {output_text[:200]}...")

    tool_calls = parse_tool_calls(output_text)

    if stream:
        async def stream_response():
            if tool_calls:
                # 先推 tool call 头部
                for i, tc in enumerate(tool_calls):
                    yield f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant', 'tool_calls': [{'index': i, 'id': tc['id'], 'type': 'function', 'function': {'name': tc['function']['name'], 'arguments': ''}}]}, 'finish_reason': None}]})}\n\n"
                    # 再推 arguments
                    args = tc['function']['arguments']
                    chunk_size = max(1, len(args) // 4)
                    for j in range(0, len(args), chunk_size):
                        yield f"data: {json.dumps({'choices': [{'delta': {'tool_calls': [{'index': i, 'function': {'arguments': args[j:j+chunk_size]}}]}, 'finish_reason': None}]})}\n\n"
                yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'tool_calls'}]})}\n\n"
            else:
                # 普通文本，按字分块
                for i in range(0, len(output_text), 3):
                    chunk = output_text[i:i+3]
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"
                yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    # 非 streaming
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


@app.post("/generate")
async def generate(req: GenerateRequest):
    messages = [{"role": "user", "content": req.prompt}]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    output = vlm_generate(
        model=model, processor=processor, prompt=prompt, image=None,
        max_tokens=req.max_new_tokens, temperature=req.temperature,
        top_p=req.top_p, verbose=False,
    )
    return {"response": getattr(output, "text", str(output))}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
