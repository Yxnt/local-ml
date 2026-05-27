import { createLocalPiSession } from "./minimal_pi_session.js";

const { session } = await createLocalPiSession();

// 打印实际使用的 system prompt（调试用）
console.log("System prompt:", session.systemPrompt);
console.log("");

session.subscribe((event) => {
  if (event.type === "message_update") {
    const ev = event.assistantMessageEvent;
    if (ev.type === "text_delta") {
      process.stdout.write(ev.delta);
    }
  }
});

console.log("Sending prompt...\n");
await session.prompt("What files are in the current directory?");

// 打印完整对话历史
console.log("\n\n===== Full conversation history =====\n");
for (const msg of session.messages) {
  if (msg.role === "user") {
    const text = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content);
    console.log(`[USER] ${text.substring(0, 500)}`);
  } else if (msg.role === "assistant") {
    const parts = [];
    for (const c of msg.content) {
      if (c.type === "text") parts.push(c.text);
      else if (c.type === "toolCall") parts.push(`[TOOL_CALL: ${c.name}(${JSON.stringify(c.arguments)})]`);
      else if (c.type === "thinking") parts.push(`[THINKING: ${c.thinking}]`);
    }
    console.log(`[ASSISTANT] ${parts.join(" ").substring(0, 500)}`);
  } else if (msg.role === "toolResult") {
    const text = msg.content.map(c => c.type === "text" ? c.text : "[image]").join("");
    console.log(`[TOOL_RESULT: ${msg.toolName}] ${text.substring(0, 500)}`);
  }
  console.log("");
}
