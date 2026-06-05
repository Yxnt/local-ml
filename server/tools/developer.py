"""ToolDeveloper — generate Python tool code from a ToolRequest.

Uses a remote LLM (MiMo) to produce a self-contained Python module that
conforms to the sandbox contract:

    from pydantic import BaseModel

    class InputModel(BaseModel):
        ...

    class OutputModel(BaseModel):
        ...

    def run(input: InputModel) -> OutputModel:
        ...

Only L0/L1 tools are eligible for auto-generation.  The generated code is
written to ``sandbox_dir/<tool_name>.py`` and then verified by ToolVerifier
before registration.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from server.optimization.prompt_store import PromptStore
from server.tools.spec import RiskLevel, ToolRequest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a tool developer for a local AI assistant.  Generate a single, self-contained Python module that implements the requested tool.

STRICT RULES:
1. The module MUST define exactly these three things:
   - `class InputModel(BaseModel)` — input parameters
   - `class OutputModel(BaseModel)` — return values
   - `def run(input: InputModel) -> OutputModel` — the entry point
2. Use ONLY these imports: `pydantic`, `json`, `re`, `datetime`, `math`, `pathlib.Path`, `typing`, `collections`, `itertools`, `functools`, `hashlib`, `base64`, `urllib.parse`, `csv`, `io`, `textwrap`, `unicodedata`.
3. FORBIDDEN imports: `os`, `sys`, `subprocess`, `shutil`, `socket`, `http`, `requests`, `httpx`, `ctypes`, `signal`, `multiprocessing`, `threading`, `asyncio`, `importlib`, `__import__`, `eval`, `exec`, `compile`, `globals`, `locals`, `vars`, `getattr`, `setattr`, `delattr`.
4. NO file I/O outside of `Path("/tmp/tool_sandbox")`.
5. NO network access.
6. NO shell commands.
7. Keep the code under 80 lines.  If the task is too complex, return an error JSON instead.
8. Handle edge cases gracefully — return meaningful error messages in OutputModel, don't raise exceptions.
9. Include type hints on all fields and the run function.
10. Output ONLY the Python source code, no markdown fences, no explanation.

Example output:
```python
from pydantic import BaseModel

class InputModel(BaseModel):
    text: str

class OutputModel(BaseModel):
    result: str
    word_count: int

def run(input: InputModel) -> OutputModel:
    words = input.text.split()
    return OutputModel(result=" ".join(words), word_count=len(words))
```
"""

USER_PROMPT_TEMPLATE = """Generate a Python tool module for the following requirement:

Tool name: {candidate_name}
Description: {candidate_description}
Missing capability: {missing_capability}
Input schema: {input_schema}
Output schema: {output_schema}
Risk level: {risk_level}

Return ONLY the Python source code."""


# ---------------------------------------------------------------------------
# Developer
# ---------------------------------------------------------------------------


class ToolDeveloper:
    """Generate tool code from a ToolRequest using a remote LLM.

    Args:
        base_url: Remote LLM API base URL.
        api_key: API key for the remote LLM.
        model_id: Model identifier.
        sandbox_dir: Directory for generated tool files.
    """

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        model_id: str = "mimo-v2.5-pro",
        sandbox_dir: str = "server/tools/sandbox",
        prompt_store: PromptStore | None = None,
    ) -> None:
        self._base_url = (base_url or os.environ.get("MIMO_BASE_URL", "")).rstrip("/")
        self._api_key = api_key or os.environ.get("MIMO_API_KEY", "")
        self._model_id = model_id
        self._sandbox_dir = Path(sandbox_dir)
        self._sandbox_dir.mkdir(parents=True, exist_ok=True)
        self._prompt_store = prompt_store

    # -- public API ----------------------------------------------------------

    async def generate(self, request: ToolRequest) -> dict[str, Any]:
        """Generate a tool module from a ToolRequest.

        Returns::

            {
                "success": bool,
                "tool_name": str,
                "source_code": str | None,
                "file_path": str | None,
                "error": str | None,
            }
        """
        # Gate: only L0/L1
        if request.risk_level not in (RiskLevel.L0, RiskLevel.L1):
            return {
                "success": False,
                "tool_name": request.candidate_name,
                "source_code": None,
                "file_path": None,
                "error": f"Risk level {request.risk_level.value} exceeds L0/L1 for auto-generation",
            }

        if not self._base_url or not self._api_key:
            return {
                "success": False,
                "tool_name": request.candidate_name,
                "source_code": None,
                "file_path": None,
                "error": "Remote LLM not configured (set MIMO_BASE_URL and MIMO_API_KEY)",
            }

        # Build prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            candidate_name=request.candidate_name,
            candidate_description=request.candidate_description,
            missing_capability=request.missing_capability,
            input_schema=json.dumps(request.candidate_input_schema, ensure_ascii=False, indent=2),
            output_schema=json.dumps(request.candidate_output_schema, ensure_ascii=False, indent=2),
            risk_level=request.risk_level.value,
        )

        # Call remote LLM
        try:
            source_code = await self._call_llm(user_prompt)
        except Exception as e:
            return {
                "success": False,
                "tool_name": request.candidate_name,
                "source_code": None,
                "file_path": None,
                "error": f"LLM call failed: {e}",
            }

        # Clean up response (strip markdown fences if present)
        source_code = self._clean_source(source_code)

        # Write to sandbox
        file_path = self._sandbox_dir / f"{request.candidate_name}.py"
        file_path.write_text(source_code, encoding="utf-8")

        return {
            "success": True,
            "tool_name": request.candidate_name,
            "source_code": source_code,
            "file_path": str(file_path),
            "error": None,
        }

    # -- internals -----------------------------------------------------------

    async def _call_llm(self, user_prompt: str) -> str:
        """Call the remote LLM and return the generated text."""
        system_prompt = SYSTEM_PROMPT  # builtin default
        if self._prompt_store:
            try:
                active = self._prompt_store.get_active("tool_developer_prompt")
                if active:
                    system_prompt = active.content
            except Exception:
                logger.warning("Failed to read active developer prompt", exc_info=True)

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model_id,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 2048,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    @staticmethod
    def _clean_source(raw: str) -> str:
        """Strip markdown fences and leading/trailing whitespace."""
        # Remove ```python ... ``` wrappers
        match = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw.strip()
