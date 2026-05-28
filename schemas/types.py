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
