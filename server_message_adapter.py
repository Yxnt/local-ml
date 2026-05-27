import json
from typing import Any


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image":
                    parts.append("[image]")
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def format_tools_for_prompt(tools: list[Any]) -> str:
    parts = []
    for tool in tools:
        fn = _get(tool, "function", {})
        name = _get(fn, "name", "")
        description = _get(fn, "description", "") or "No description"
        parameters = _get(fn, "parameters", None) or {}
        parts.append(
            f"Tool: {name}\n"
            f"Description: {description}\n"
            f"Parameters: {json.dumps(parameters, indent=2)}"
        )
    return "\n\n".join(parts)


def _format_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    rendered = []
    for tool_call in tool_calls:
        fn = tool_call.get("function", {})
        arguments = fn.get("arguments", "{}")
        try:
            parsed_arguments = json.loads(arguments)
        except (TypeError, json.JSONDecodeError):
            parsed_arguments = arguments
        rendered.append(
            json.dumps(
                {
                    "id": tool_call.get("id"),
                    "name": fn.get("name"),
                    "arguments": parsed_arguments,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    joined = "\n\n".join(rendered)
    return (
        "Assistant previously called the following tool(s):\n"
        f"```json\n{joined}\n```"
    )


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized = []
    for raw_message in messages:
        role = raw_message.get("role", "")
        content = extract_content(raw_message.get("content"))
        tool_calls = raw_message.get("tool_calls") or []

        if role == "assistant" and tool_calls:
            tool_call_summary = _format_tool_calls(tool_calls)
            if content:
                content = f"{content}\n\n{tool_call_summary}"
            else:
                content = tool_call_summary
            normalized.append({"role": "assistant", "content": content})
            continue

        if role in ("tool", "function", "toolResult"):
            tool_name = raw_message.get("name") or "unknown_tool"
            tool_call_id = raw_message.get("tool_call_id") or raw_message.get("toolCallId") or "unknown_call"
            tool_result = content or "[empty tool result]"
            normalized.append(
                {
                    "role": "user",
                    "content": (
                        "Tool result for the previous assistant tool call:\n"
                        f"- tool name: {tool_name}\n"
                        f"- tool call id: {tool_call_id}\n"
                        f"{tool_result}"
                    ),
                }
            )
            continue

        normalized.append({"role": role, "content": content})

    return normalized


def build_tool_prompt_prefix(messages: list[dict[str, str]], tools: list[Any]) -> list[dict[str, str]]:
    if not tools:
        return list(messages)

    tool_text = format_tools_for_prompt(tools)
    tool_instructions = (
        "You can use tools by outputting a JSON object exactly like this:\n"
        "```json\n"
        '{"name": "tool_name", "arguments": {"key": "value"}}\n'
        "```\n\n"
        f"Available tools:\n\n{tool_text}\n\n"
        "If you need a tool, output ONLY the JSON block. "
        "If a tool result is already present in the conversation, use it instead of calling the same tool again unless new information is required."
    )

    prefixed = [dict(message) for message in messages]
    system_msg = next((message for message in prefixed if message["role"] == "system"), None)
    if system_msg:
        system_msg["content"] = f"{tool_instructions}\n\n{system_msg['content']}"
    else:
        prefixed.insert(0, {"role": "system", "content": tool_instructions})
    return prefixed
