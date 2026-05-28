import unittest

from server.message_adapter import (
    build_tool_prompt_prefix,
    normalize_messages,
)


class NormalizeMessagesTests(unittest.TestCase):
    def test_preserves_existing_system_prompt(self):
        messages = [
            {"role": "system", "content": "Custom client prompt."},
            {"role": "user", "content": "Run ls -F and output the result verbatim"},
        ]

        normalized = normalize_messages(messages)

        self.assertEqual(normalized[0]["role"], "system")
        self.assertEqual(normalized[0]["content"], "Custom client prompt.")

    def test_keeps_assistant_tool_call_history_and_tool_result(self):
        messages = [
            {"role": "user", "content": "Run ls -F and output the result verbatim"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": "{\"command\":\"ls -F\"}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0",
                "name": "bash",
                "content": "123\ndirect_test.js\nmain.js\n",
            },
        ]

        normalized = normalize_messages(messages)

        self.assertEqual(normalized[1]["role"], "assistant")
        self.assertIn('"name": "bash"', normalized[1]["content"])
        self.assertIn('"command": "ls -F"', normalized[1]["content"])

        self.assertEqual(normalized[2]["role"], "user")
        self.assertIn("Tool result for the previous assistant tool call", normalized[2]["content"])
        self.assertIn("123\ndirect_test.js\nmain.js\n", normalized[2]["content"])

    def test_tool_prefix_prepends_without_overwriting_system_prompt(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Execute bash commands",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]
        messages = [
            {"role": "system", "content": "Custom client prompt."},
            {"role": "user", "content": "Run ls -F"},
        ]

        normalized = normalize_messages(messages)
        prefixed = build_tool_prompt_prefix(normalized, tools)

        self.assertTrue(prefixed[0]["content"].startswith("You can use tools by outputting"))
        self.assertIn("Custom client prompt.", prefixed[0]["content"])


if __name__ == "__main__":
    unittest.main()
