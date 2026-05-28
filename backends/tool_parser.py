"""Parser for MiniCPM5 XML-style tool call format.

MiniCPM5 emits tool calls in an XML format:
    <function=function_name>
      <parameter=param_name>param_value</parameter>
    </function>

This module extracts those into OpenAI-compatible tool_call dicts.
"""

import json
import re
from typing import List


def parse_mcp_tool_calls(text: str) -> List[dict]:
    """Parse MiniCPM5 XML tool calls from model output.

    Args:
        text: Raw model output that may contain XML tool calls.

    Returns:
        List of OpenAI-compatible tool_call dicts, each with keys:
        - id: "call_{index}"
        - type: "function"
        - function: {"name": str, "arguments": json str}
    """
    calls = []

    func_pattern = re.compile(
        r'<function\s*=\s*(\w+)\s*>(.*?)</function>',
        re.DOTALL,
    )

    for func_match in func_pattern.finditer(text):
        name = func_match.group(1)
        body = func_match.group(2)

        param_pattern = re.compile(
            r'<parameter\s*=\s*(\w+)\s*>(.*?)</parameter>',
            re.DOTALL,
        )

        args = {}
        for param_match in param_pattern.finditer(body):
            key = param_match.group(1)
            value = param_match.group(2).strip()
            args[key] = value

        calls.append({
            "id": f"call_{len(calls)}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })

    return calls
