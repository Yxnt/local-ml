"""Remote LLM backend - 支持远程 API（OpenAI 兼容）

支持:
- Xiaomi MiMo (小米大模型)
- OpenAI
- 任何 OpenAI 兼容 API
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from backends.base import ModelBackend

logger = logging.getLogger(__name__)

# 预定义的远程模型配置
PREDEFINED_MODELS: dict[str, dict[str, str]] = {
    "mimo-v2.5-pro": {
        "provider": "xiaomi",
        "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
        "model_id": "mimo-v2.5-pro",
        "description": "小米 MiMo V2.5 Pro",
    },
    "mimo-v2.5-flash": {
        "provider": "xiaomi",
        "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
        "model_id": "mimo-v2.5-flash",
        "description": "小米 MiMo V2.5 Flash (快速)",
    },
    "gpt-4o": {
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "model_id": "gpt-4o",
        "description": "OpenAI GPT-4o",
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "model_id": "gpt-4o-mini",
        "description": "OpenAI GPT-4o Mini",
    },
}


class RemoteBackend(ModelBackend):
    """远程 LLM 后端（OpenAI 兼容协议）"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_id: str,
        provider: str = "openai",
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model_id = model_id
        self._provider = provider
        self._client: httpx.AsyncClient | None = None

    @classmethod
    def from_predefined(cls, model_name: str, api_key: str) -> RemoteBackend:
        """从预定义模型创建"""
        if model_name not in PREDEFINED_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(PREDEFINED_MODELS.keys())}")

        config = PREDEFINED_MODELS[model_name]
        return cls(
            base_url=config["base_url"],
            api_key=api_key,
            model_id=config["model_id"],
            provider=config["provider"],
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """获取 HTTP 客户端"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "api-key": self._api_key,
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )
        return self._client

    # -- lifecycle --

    def load(self, model_id: str) -> None:
        """远程模型不需要加载"""
        self._loaded = True
        logger.info("Remote model ready: %s", model_id)

    def unload(self) -> None:
        """远程模型不需要卸载"""
        self._loaded = True  # 保持可用

    # -- inference --

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """同步生成（用于简单调用）"""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.generate_async(prompt, max_tokens, temperature, top_p)
        )

    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """异步生成"""
        client = await self._get_client()

        response = await client.post(
            "/chat/completions",
            json={
                "model": self._model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        )

        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"]

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        enable_thinking: bool = False,
    ) -> str:
        """远程模型不需要模板（API 直接处理）"""
        # 返回消息的 JSON 表示（供 generate 使用）
        return json.dumps(messages, ensure_ascii=False)

    def parse_tool_calls(self, text: str) -> list[dict]:
        """解析远程模型返回的工具调用"""
        calls = []

        # 尝试解析 JSON 格式的工具调用
        try:
            data = json.loads(text)
            if "tool_calls" in data:
                for tc in data["tool_calls"]:
                    calls.append({
                        "id": tc.get("id", f"call_{len(calls)}"),
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": json.dumps(tc["function"]["arguments"]),
                        },
                    })
        except (json.JSONDecodeError, KeyError):
            pass

        # 尝试解析文本中的 JSON 块
        if not calls:
            import re
            json_pattern = r'(?:```(?:json)?\s*)?(\{(?:[^{}]|\{[^{}]*\})*\})(?:\s*```)?'
            for match in re.finditer(json_pattern, text):
                try:
                    obj = json.loads(match.group(1))
                    if "name" in obj and "arguments" in obj:
                        calls.append({
                            "id": f"call_{len(calls)}",
                            "type": "function",
                            "function": {
                                "name": obj["name"],
                                "arguments": json.dumps(obj["arguments"]),
                            },
                        })
                except json.JSONDecodeError:
                    continue

        return calls

    def warmup(self) -> None:
        """远程模型不需要预热"""
        pass

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> dict[str, Any]:
        """完整的聊天补全 API（支持工具调用）"""
        client = await self._get_client()

        payload = {
            "model": self._model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        if tools:
            payload["tools"] = tools

        response = await client.post("/chat/completions", json=payload)

        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")

        return response.json()

    async def close(self) -> None:
        """关闭客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None


def create_remote_backend(
    model_name: str,
    api_key: str,
    base_url: str | None = None,
) -> RemoteBackend:
    """工厂函数：创建远程后端"""
    if base_url:
        # 自定义配置
        return RemoteBackend(
            base_url=base_url,
            api_key=api_key,
            model_id=model_name,
        )
    else:
        # 使用预定义配置
        return RemoteBackend.from_predefined(model_name, api_key)
