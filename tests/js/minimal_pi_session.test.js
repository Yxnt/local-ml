import assert from "node:assert/strict";
import test from "node:test";

import { SYSTEM_PROMPT, createLocalPiSession } from "../../js/minimal_pi_session.js";

test("createLocalPiSession uses a minimal prompt and a single bash tool", async () => {
  const { session } = await createLocalPiSession({
    apiBase: "http://127.0.0.1:9/v1",
  });

  try {
    assert.equal(session.agent.state.tools.length, 1);
    assert.equal(session.agent.state.tools[0].name, "bash");
    assert.ok(session.systemPrompt.startsWith(SYSTEM_PROMPT));
    assert.ok(!session.systemPrompt.includes("<available_skills>"));
    assert.ok(session.systemPrompt.includes("Current date:"));
    assert.ok(session.systemPrompt.includes(`Current working directory: ${process.cwd()}`));
  } finally {
    session.dispose();
  }
});
