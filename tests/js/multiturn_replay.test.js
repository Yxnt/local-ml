import assert from "node:assert/strict";
import test from "node:test";

import { buildSecondTurnMessages } from "./multiturn_replay.js";

test("buildSecondTurnMessages preserves assistant tool calls and tool result metadata", () => {
  const initialMessages = [
    {
      role: "system",
      content: "You are a helpful coding assistant.",
    },
    {
      role: "user",
      content: "What files are in the current directory?",
    },
  ];

  const toolCall = {
    id: "call_0",
    type: "function",
    function: {
      name: "bash",
      arguments: "{\"command\":\"ls -F\"}",
    },
  };

  const nextMessages = buildSecondTurnMessages(initialMessages, toolCall, "bash", "main.js\nserver.py\n");

  assert.deepEqual(nextMessages, [
    ...initialMessages,
    {
      role: "assistant",
      content: null,
      tool_calls: [toolCall],
    },
    {
      role: "tool",
      name: "bash",
      tool_call_id: "call_0",
      content: "main.js\nserver.py\n",
    },
  ]);
});
