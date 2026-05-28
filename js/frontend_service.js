/**
 * Frontend service example - model listing and switching.
 *
 * Usage:
 *   node frontend_service.js
 */

const API_BASE = process.env.API_BASE || "http://localhost:8000/v1";

/**
 * List all available models from the server.
 */
async function listModels() {
  const response = await fetch(`${API_BASE}/models`);
  if (!response.ok) {
    throw new Error(`Failed to list models: ${response.status}`);
  }
  return response.json();
}

/**
 * Send a chat completion request with a specific model.
 */
async function chatCompletion(messages, { model = "gemma-4-e2b-it-4bit", tools = null, stream = false } = {}) {
  const body = {
    model,
    messages,
    stream,
    max_tokens: 2048,
  };
  if (tools) {
    body.tools = tools;
  }

  const response = await fetch(`${API_BASE}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: "Bearer dummy",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`HTTP ${response.status}: ${text}`);
  }

  return response.json();
}

// Demo
async function main() {
  console.log("=== Available Models ===");
  const models = await listModels();
  for (const m of models.data) {
    console.log(`  ${m.id} (${m.backend})`);
  }

  console.log("\n=== Chat with MiniCPM-V 4.6 ===");
  const result = await chatCompletion(
    [{ role: "user", content: "Say hello in one word." }],
    { model: "minicpm-v-4_6" }
  );
  console.log("Response:", result.choices?.[0]?.message?.content);
}

main().catch(console.error);
