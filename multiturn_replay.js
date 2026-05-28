import { execSync } from "node:child_process";

import { DEFAULT_API_BASE, LOCAL_MODEL_ID, SYSTEM_PROMPT } from "./minimal_pi_session.js";

export const BASH_TOOL = {
  type: "function",
  function: {
    name: "bash",
    description: "Execute a shell command and return its stdout or stderr verbatim.",
    parameters: {
      type: "object",
      required: ["command"],
      properties: {
        command: {
          type: "string",
          description: "The shell command to run.",
        },
      },
    },
    strict: false,
  },
};

export function buildSecondTurnMessages(initialMessages, toolCall, toolName, toolResult) {
  return [
    ...initialMessages,
    {
      role: "assistant",
      content: null,
      tool_calls: [toolCall],
    },
    {
      role: "tool",
      name: toolName,
      tool_call_id: toolCall.id,
      content: toolResult,
    },
  ];
}

async function chat(messages, tools, apiBase = DEFAULT_API_BASE, model = LOCAL_MODEL_ID) {
  const response = await fetch(`${apiBase}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: "Bearer dummy",
    },
    body: JSON.stringify({
      model,
      messages,
      tools,
      stream: false,
      max_tokens: 2048,
      temperature: 0.0,
    }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  }

  return response.json();
}

function executeToolCall(toolCall) {
  const args = JSON.parse(toolCall.function.arguments);
  if (toolCall.function.name !== "bash") {
    throw new Error(`Unsupported tool: ${toolCall.function.name}`);
  }

  try {
    return execSync(args.command, { encoding: "utf8", cwd: process.cwd() });
  } catch (error) {
    return typeof error?.stderr === "string" && error.stderr.length > 0
      ? error.stderr
      : error instanceof Error
        ? error.message
        : String(error);
  }
}

async function main() {
  const initialMessages = [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: "What files are in the current directory?" },
  ];

  console.log("===== Turn 1 =====");
  const first = await chat(initialMessages, [BASH_TOOL]);
  console.log(JSON.stringify(first, null, 2));

  const firstChoice = first.choices?.[0];
  const toolCall = firstChoice?.message?.tool_calls?.[0];
  if (!toolCall) {
    throw new Error("Expected first turn to produce a tool call.");
  }

  const toolResult = executeToolCall(toolCall);
  console.log("\n===== Tool Result =====");
  console.log(toolResult);

  const secondTurnMessages = buildSecondTurnMessages(
    initialMessages,
    toolCall,
    toolCall.function.name,
    toolResult,
  );

  console.log("\n===== Turn 2 =====");
  const second = await chat(secondTurnMessages, [BASH_TOOL]);
  console.log(JSON.stringify(second, null, 2));
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
