import { execSync } from "node:child_process";

import {
  AuthStorage,
  createAgentSession,
  createExtensionRuntime,
  defineTool,
  ModelRegistry,
  SessionManager,
  SettingsManager,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

export const DEFAULT_API_BASE = process.env.API_BASE || "http://localhost:8000/v1";
export const LOCAL_MODEL_ID = "gemma-4-e2b-it-4bit";
export const SYSTEM_PROMPT =
  "You are a helpful coding assistant. Follow user instructions precisely. Use the bash tool when shell output is needed.";

const terminatingBashTool = defineTool({
  name: "bash",
  label: "Bash",
  description: "Execute a shell command and return its stdout or stderr verbatim.",
  parameters: Type.Object({
    command: Type.String({ description: "The shell command to run." }),
  }),
  async execute(_toolCallId, params) {
    try {
      const output = execSync(params.command, {
        cwd: process.cwd(),
        encoding: "utf8",
      });

      return {
        content: [{ type: "text", text: output }],
        details: { command: params.command, ok: true },
        terminate: true,
      };
    } catch (error) {
      const text =
        typeof error?.stderr === "string" && error.stderr.length > 0
          ? error.stderr
          : error instanceof Error
            ? error.message
            : String(error);

      return {
        content: [{ type: "text", text }],
        details: { command: params.command, ok: false },
        terminate: true,
      };
    }
  },
});

function createMinimalResourceLoader(systemPrompt = SYSTEM_PROMPT) {
  return {
    getExtensions: () => ({ extensions: [], errors: [], runtime: createExtensionRuntime() }),
    getSkills: () => ({ skills: [], diagnostics: [] }),
    getPrompts: () => ({ prompts: [], diagnostics: [] }),
    getThemes: () => ({ themes: [], diagnostics: [] }),
    getAgentsFiles: () => ({ agentsFiles: [] }),
    getSystemPrompt: () => systemPrompt,
    getAppendSystemPrompt: () => [],
    extendResources: () => {},
    reload: async () => {},
  };
}

export async function createLocalPiSession({
  apiBase = DEFAULT_API_BASE,
  cwd = process.cwd(),
  systemPrompt = SYSTEM_PROMPT,
  sessionManager = SessionManager.inMemory(cwd),
  model: modelId = LOCAL_MODEL_ID,
} = {}) {
  const authStorage = AuthStorage.create();
  const modelRegistry = ModelRegistry.inMemory(authStorage);

  modelRegistry.registerProvider("local", {
    baseUrl: apiBase,
    apiKey: "dummy",
    models: [
      {
        id: modelId,
        name: modelId,
        api: "openai-completions",
        reasoning: false,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 128000,
        maxTokens: 8192,
      },
    ],
  });

  const model = modelRegistry.find("local", modelId);
  if (!model) {
    throw new Error(`Model not found: ${modelId}`);
  }

  const { session } = await createAgentSession({
    cwd,
    model,
    thinkingLevel: "off",
    authStorage,
    modelRegistry,
    resourceLoader: createMinimalResourceLoader(systemPrompt),
    tools: ["bash"],
    customTools: [terminatingBashTool],
    noTools: "builtin",
    sessionManager,
    settingsManager: SettingsManager.inMemory({
      compaction: { enabled: false },
      retry: { enabled: false, maxRetries: 0 },
    }),
  });

  return { session, model, modelRegistry, authStorage };
}
