# Tool Evolution Follow-Ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the paper-aligned self-evolving tool loop from `2601.18226v2.pdf`: fixed workflow/context, query-local tool accumulation, batch absorption into a global tool pool, and convergence-oriented eval metrics.

**Architecture:** Keep Agent workflow `W` and context policy `C` fixed. Evolve only toolset `T` by adding a conservative in-situ synthesis path, a query-scoped `LocalToolPool` overlay, absorber replay checks for local-to-global consolidation, and eval signals that distinguish tool creation pressure from generated-tool reuse.

**Tech Stack:** Python, pytest, existing `server.tools` registry/router/developer/verifier/absorber stack, `evals/local_ml_eval`

---

## Scope Check

This plan follows `docs/superpowers/specs/2026-06-09-tool-evolution-followups-design.md` and is organized as three sequential PR slices:

1. PR A: In-situ tool accumulation.
2. PR B: Batch absorption and behavioral consolidation.
3. PR C: Evolution monitoring and self-evolution eval.

The plan intentionally avoids changing prompt topology, workflow graphs, memory policy, or external-account benchmark coverage. Those would evolve `W` or `C`, while this follow-up is about evolving `T`.

## File Structure

| File | Role |
|------|------|
| `server/tools/risk_classifier.py` | Classifies missing capabilities before any in-situ generation attempt |
| `tests/tools/test_risk_classifier.py` | Unit coverage for safe L0/L1 generation gates and blocked unknown/high-risk requests |
| `server/tools/local_pool.py` | Query-scoped generated tool overlay with local-first lookup, OpenAI schema listing, sandbox materialization, and dispatch |
| `tests/tools/test_local_pool.py` | Unit coverage for local registration, deduped tool listing, and dispatch via the existing router contract |
| `server/agent.py` | Wires optional in-situ generation into the existing tool loop with legacy fallback preserved |
| `tests/server/test_agent_in_situ_generation.py` | Agent-level tests for same-run generated tool use, blocked risk, and bounded retries |
| `server/tools/absorber.py` | Extends parent test checks to replay `ToolSpec.metadata["replay_cases"]` against merged tools |
| `tests/tools/test_absorber_e2e.py` | Adds replay success and replay-failure merge-blocking coverage |
| `evals/local_ml_eval/metrics.py` | Splits creation-pressure EGL from generated-tool use and adds self-evolution rates |
| `evals/local_ml_eval/runner.py` | Emits per-result fields needed for zero-start and warm-start evolution metrics |
| `evals/local_ml_eval/tasks.jsonl` | Adds deterministic zero-start, synthesis-pressure, and warm-start reuse tasks |
| `evals/local_ml_eval/fixtures.py` | Adds deterministic generated fixtures for warm-start and reuse scenarios |
| `tests/evals/test_local_ml_eval.py` | Updates metric/report assertions and adds self-evolution behavior coverage |
| `evals/local_ml_eval/README.md` | Documents zero-start, warm-start, and metric interpretation |
| `docs/tool_evolution.md` | Aligns docs with the `M_t = <W0, C0, T_t>` state transition |

---

## PR A: In-Situ Tool Accumulation

### Task 1: Add Explicit Capability Risk Classification

**Files:**
- Create: `server/tools/risk_classifier.py`
- Create: `tests/tools/test_risk_classifier.py`
- Modify: `server/tools/__init__.py`

- [ ] **Step 1: Write failing risk-classifier tests**

Create `tests/tools/test_risk_classifier.py`:

```python
from server.tools.risk_classifier import CapabilityRiskClassifier
from server.tools.spec import RiskLevel, ToolRequest


def _request(name: str, capability: str, schema: dict | None = None) -> ToolRequest:
    return ToolRequest(
        candidate_name=name,
        candidate_description=capability,
        missing_capability=capability,
        candidate_input_schema=schema or {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )


def test_allows_pure_text_processing_as_l0():
    decision = CapabilityRiskClassifier().classify(
        _request("text_count_words", "Count words in a provided text string")
    )

    assert decision.auto_generatable is True
    assert decision.risk_level == RiskLevel.L0
    assert decision.reason == "pure_local_computation"


def test_blocks_empty_or_ambiguous_missing_capability_even_with_default_l0():
    decision = CapabilityRiskClassifier().classify(
        _request("do_the_thing", "")
    )

    assert decision.auto_generatable is False
    assert decision.risk_level is None
    assert decision.reason == "ambiguous_capability"


def test_blocks_network_file_shell_and_integration_requests():
    classifier = CapabilityRiskClassifier()

    blocked = [
        _request("download_url", "Download a URL from the internet"),
        _request("write_file", "Write a file to the project directory"),
        _request("run_shell", "Run a shell command"),
        _request("calendar_read", "Read the user's calendar"),
        _request("desktop_click", "Click a button on the desktop"),
        _request("token_lookup", "Use an API key to call a service"),
    ]

    for req in blocked:
        decision = classifier.classify(req)
        assert decision.auto_generatable is False
        assert decision.risk_level is None
        assert decision.reason.startswith("blocked_")


def test_annotate_request_records_classifier_metadata_without_trusting_default_risk():
    req = _request("calendar_read", "Read the user's calendar")

    decision = CapabilityRiskClassifier().annotate_request(req)

    assert decision.auto_generatable is False
    assert req.metadata["risk_classifier"]["auto_generatable"] is False
    assert req.metadata["risk_classifier"]["reason"] == "blocked_integration_or_private_data"
    assert req.risk_level == RiskLevel.L0
```

- [ ] **Step 2: Run the test and verify it fails because the module is absent**

Run: `python -m pytest tests/tools/test_risk_classifier.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'server.tools.risk_classifier'`.

- [ ] **Step 3: Implement the classifier**

Create `server/tools/risk_classifier.py`:

```python
"""Conservative risk classification for in-situ tool generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from server.tools.spec import RiskLevel, ToolRequest


@dataclass(frozen=True)
class CapabilityRiskDecision:
    auto_generatable: bool
    risk_level: RiskLevel | None
    reason: str
    blocked_terms: tuple[str, ...] = ()

    def to_metadata(self) -> dict[str, Any]:
        return {
            "auto_generatable": self.auto_generatable,
            "risk_level": self.risk_level.value if self.risk_level else None,
            "reason": self.reason,
            "blocked_terms": list(self.blocked_terms),
        }


class CapabilityRiskClassifier:
    """Allow only clearly local L0/L1 generated tools.

    Model-originated ToolRequests default to L0 today, so the decision must be
    derived from the requested capability text and schema instead of trusting
    ToolRequest.risk_level.
    """

    _BLOCKED_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("blocked_shell_or_process", ("shell", "terminal", "subprocess", "command", "bash", "zsh", "python script")),
        ("blocked_network", ("http", "https", "url", "download", "upload", "network", "internet", "webhook", "api call")),
        ("blocked_file_mutation", ("write file", "modify file", "delete file", "rename file", "move file", "project directory")),
        ("blocked_desktop_control", ("desktop", "mouse", "keyboard", "click", "screenshot", "window")),
        ("blocked_credentials", ("password", "credential", "token", "api key", "secret")),
        ("blocked_integration_or_private_data", ("calendar", "mailbox", "gmail", "imap", "smtp", "obsidian", "contacts")),
    )
    _SAFE_HINTS: tuple[str, ...] = (
        "text",
        "string",
        "json",
        "markdown",
        "format",
        "convert",
        "count",
        "extract",
        "parse",
        "date",
        "math",
        "calculate",
        "normalize",
        "sort",
        "dedupe",
        "slug",
        "hash",
        "base64",
    )

    def classify(self, request: ToolRequest) -> CapabilityRiskDecision:
        text = self._request_text(request)
        if not text.strip() or not request.candidate_name.strip():
            return CapabilityRiskDecision(False, None, "ambiguous_capability")

        for reason, terms in self._BLOCKED_GROUPS:
            hits = tuple(term for term in terms if term in text)
            if hits:
                return CapabilityRiskDecision(False, None, reason, hits)

        if any(hint in text for hint in self._SAFE_HINTS):
            return CapabilityRiskDecision(True, RiskLevel.L0, "pure_local_computation")

        return CapabilityRiskDecision(False, None, "ambiguous_capability")

    def annotate_request(self, request: ToolRequest) -> CapabilityRiskDecision:
        decision = self.classify(request)
        request.metadata = dict(request.metadata)
        request.metadata["risk_classifier"] = decision.to_metadata()
        if decision.risk_level is not None:
            request.risk_level = decision.risk_level
        return decision

    def _request_text(self, request: ToolRequest) -> str:
        schema_text = json.dumps(request.candidate_input_schema, ensure_ascii=False, sort_keys=True)
        return " ".join(
            [
                request.candidate_name,
                request.candidate_description,
                request.missing_capability,
                schema_text,
            ]
        ).lower()
```

- [ ] **Step 4: Export classifier types**

Modify `server/tools/__init__.py` to export:

```python
from server.tools.risk_classifier import CapabilityRiskClassifier, CapabilityRiskDecision
```

and add both names to `__all__`.

- [ ] **Step 5: Run classifier tests**

Run: `python -m pytest tests/tools/test_risk_classifier.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add server/tools/risk_classifier.py tests/tools/test_risk_classifier.py server/tools/__init__.py
git commit -m "feat: add capability risk classifier"
```

---

### Task 2: Add Query-Scoped Local Tool Pool

**Files:**
- Create: `server/tools/local_pool.py`
- Create: `tests/tools/test_local_pool.py`
- Modify: `server/tools/__init__.py`

- [ ] **Step 1: Write failing local-pool tests**

Create `tests/tools/test_local_pool.py`:

```python
from pathlib import Path

import pytest

from server.tools.local_pool import LocalToolPool
from server.tools.spec import RiskLevel, ToolContext, ToolResult, ToolRuntime, ToolSpec


class FakeRouter:
    def __init__(self) -> None:
        self.calls = []

    async def dispatch(self, spec: ToolSpec, arguments: dict, ctx: ToolContext) -> ToolResult:
        self.calls.append((spec, arguments, ctx))
        return ToolResult(content=f"local:{spec.name}:{arguments['text']}")


def _spec(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Reverse provided text",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        output_schema={
            "type": "object",
            "properties": {"result": {"type": "string"}},
        },
        runtime=ToolRuntime.PYTHON_GENERATED,
        provider="generated",
        risk_level=RiskLevel.L0,
    )


def test_register_materializes_source_and_returns_local_tool(tmp_path: Path):
    pool = LocalToolPool(router=FakeRouter(), sandbox_dir=str(tmp_path))
    source = "from pydantic import BaseModel\n"

    registered = pool.register(_spec("text_reverse"), source)

    assert registered.name == "text_reverse"
    assert registered.provider == "query_local"
    assert registered.metadata["local_pool"] is True
    assert (tmp_path / "text_reverse.py").read_text(encoding="utf-8") == source
    assert pool.get_tool("text_reverse") is registered


def test_list_openai_tools_is_local_first_and_dedupes_global_names(tmp_path: Path):
    pool = LocalToolPool(router=FakeRouter(), sandbox_dir=str(tmp_path))
    pool.register(_spec("text_reverse"), "source")

    global_tools = [
        {"type": "function", "function": {"name": "text_reverse", "description": "global", "parameters": {}}},
        {"type": "function", "function": {"name": "memory_recall", "description": "memory", "parameters": {}}},
    ]

    tools = pool.list_openai_tools(global_tools)

    assert [tool["function"]["name"] for tool in tools] == ["text_reverse", "memory_recall"]
    assert tools[0]["function"]["description"] == "Reverse provided text"


@pytest.mark.asyncio
async def test_dispatch_uses_router_with_local_spec(tmp_path: Path):
    router = FakeRouter()
    pool = LocalToolPool(router=router, sandbox_dir=str(tmp_path))
    pool.register(_spec("text_reverse"), "source")

    result = await pool.dispatch("text_reverse", {"text": "abc"}, ToolContext(session_id="s1"))

    assert result.success is True
    assert result.content == "local:text_reverse:abc"
    assert router.calls[0][0].name == "text_reverse"


@pytest.mark.asyncio
async def test_dispatch_returns_tool_not_found_for_missing_local_tool(tmp_path: Path):
    pool = LocalToolPool(router=FakeRouter(), sandbox_dir=str(tmp_path))

    result = await pool.dispatch("missing_tool", {}, ToolContext(session_id="s1"))

    assert result.success is False
    assert result.error_type == "tool_not_found"
```

- [ ] **Step 2: Run the test and verify it fails because the module is absent**

Run: `python -m pytest tests/tools/test_local_pool.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'server.tools.local_pool'`.

- [ ] **Step 3: Implement `LocalToolPool`**

Create `server/tools/local_pool.py`:

```python
"""Query-local tool overlay for in-situ generated tools."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from server.tools.spec import ToolContext, ToolResult, ToolRuntime, ToolSpec, ToolStatus
from server.tools.router import ToolRuntimeRouter


class LocalToolPool:
    """Holds generated tools for one Agent.run invocation.

    The durable registry remains the global pool. This overlay is query-local:
    lookup and listing are local-first, and generated source is materialized
    into the sandbox layout expected by GeneratedPythonExecutor.
    """

    def __init__(self, router: ToolRuntimeRouter, sandbox_dir: str) -> None:
        self._router = router
        self._sandbox_dir = Path(sandbox_dir)
        self._tools: dict[str, ToolSpec] = {}

    @property
    def sandbox_dir(self) -> Path:
        return self._sandbox_dir

    def register(self, spec: ToolSpec, source_code: str) -> ToolSpec:
        self._sandbox_dir.mkdir(parents=True, exist_ok=True)
        local_spec = replace(
            spec,
            runtime=ToolRuntime.PYTHON_GENERATED,
            provider="query_local",
            status=ToolStatus.ACTIVE,
            metadata={
                **spec.metadata,
                "local_pool": True,
                "source_file": str(self._sandbox_dir / f"{spec.name}.py"),
            },
        )
        (self._sandbox_dir / f"{local_spec.name}.py").write_text(source_code, encoding="utf-8")
        self._tools[local_spec.name] = local_spec
        return local_spec

    def get_tool(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def list_openai_tools(self, global_tools: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        tools = [spec.to_openai_schema() for spec in self._tools.values()]
        seen = {tool["function"]["name"] for tool in tools}
        for tool in global_tools or []:
            name = tool.get("function", {}).get("name")
            if name and name not in seen:
                tools.append(tool)
                seen.add(name)
        return tools

    async def dispatch(self, name: str, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        spec = self.get_tool(name)
        if spec is None:
            return ToolResult(
                content=f"未知工具: {name}",
                success=False,
                error_type="tool_not_found",
            )
        return await self._router.dispatch(spec, arguments, ctx)
```

- [ ] **Step 4: Export `LocalToolPool`**

Modify `server/tools/__init__.py`:

```python
from server.tools.local_pool import LocalToolPool
```

and add `"LocalToolPool"` to `__all__`.

- [ ] **Step 5: Run local-pool tests**

Run: `python -m pytest tests/tools/test_local_pool.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add server/tools/local_pool.py tests/tools/test_local_pool.py server/tools/__init__.py
git commit -m "feat: add query local tool pool"
```

---

### Task 3: Wire Agent for Optional In-Situ Generation

**Files:**
- Modify: `server/agent.py`
- Create: `tests/server/test_agent_in_situ_generation.py`

- [ ] **Step 1: Write failing Agent tests**

Create `tests/server/test_agent_in_situ_generation.py`:

```python
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from server.agent import Agent
from server.tools.risk_classifier import CapabilityRiskClassifier
from server.tools.spec import ToolResult


GENERATED_REVERSE_SOURCE = """
from pydantic import BaseModel

class InputModel(BaseModel):
    text: str

class OutputModel(BaseModel):
    result: str

def run(input: InputModel) -> OutputModel:
    return OutputModel(result=input.text[::-1])
""".strip()


class FakeDeveloper:
    def __init__(self) -> None:
        self.calls = []

    async def generate(self, request):
        self.calls.append(request)
        return {
            "success": True,
            "tool_name": request.candidate_name,
            "source_code": GENERATED_REVERSE_SOURCE,
            "file_path": None,
            "error": None,
        }


class FakeVerifyResult:
    passed = True
    errors = []


class FakeVerifier:
    def verify(self, tool_name: str, source_code: str):
        return FakeVerifyResult()


@pytest.fixture
def agent(tmp_path: Path):
    a = Agent(
        data_dir=str(tmp_path / "data"),
        use_tool_registry=True,
        enable_in_situ_tool_generation=True,
        tool_sandbox_dir=str(tmp_path / "sandbox"),
        tool_developer=FakeDeveloper(),
        tool_verifier=FakeVerifier(),
        tool_risk_classifier=CapabilityRiskClassifier(),
    )
    return a


@pytest.mark.asyncio
async def test_dispatch_generates_safe_missing_tool_and_retries_locally(agent: Agent):
    ctx = SimpleNamespace(
        task_id="task1",
        user_input="Reverse abc",
        local_tool_pool=agent._create_local_tool_pool(),
        attempted_generation=set(),
    )

    async def missing_dispatch(name, args, tool_ctx):
        return ToolResult(content=f"未知工具: {name}", success=False, error_type="tool_not_found")

    agent._tool_registry.dispatch = missing_dispatch

    content = await agent._dispatch_tool("text_reverse", {"text": "abc"}, ctx)

    assert json.loads(content) == {"result": "cba"}
    assert (Path(agent._tool_sandbox_dir) / "text_reverse.py").exists()
    assert "text_reverse" in ctx.attempted_generation


@pytest.mark.asyncio
async def test_dispatch_records_request_only_when_classifier_blocks(agent: Agent):
    ctx = SimpleNamespace(
        task_id="task1",
        user_input="Read calendar",
        local_tool_pool=agent._create_local_tool_pool(),
        attempted_generation=set(),
    )
    recorded = []

    async def missing_dispatch(name, args, tool_ctx):
        return ToolResult(content=f"未知工具: {name}", success=False, error_type="tool_not_found")

    agent._tool_registry.dispatch = missing_dispatch
    agent._tool_telemetry.record_tool_request = recorded.append

    content = await agent._dispatch_tool("calendar_read", {"query": "today"}, ctx)

    assert content == "未知工具: calendar_read"
    assert recorded[0].metadata["risk_classifier"]["auto_generatable"] is False
    assert agent._tool_developer.calls == []


@pytest.mark.asyncio
async def test_in_situ_generation_attempt_is_bounded_per_tool(agent: Agent):
    ctx = SimpleNamespace(
        task_id="task1",
        user_input="Reverse abc",
        local_tool_pool=agent._create_local_tool_pool(),
        attempted_generation={"text_reverse"},
    )

    async def missing_dispatch(name, args, tool_ctx):
        return ToolResult(content=f"未知工具: {name}", success=False, error_type="tool_not_found")

    agent._tool_registry.dispatch = missing_dispatch

    content = await agent._dispatch_tool("text_reverse", {"text": "abc"}, ctx)

    assert content == "未知工具: text_reverse"
    assert agent._tool_developer.calls == []
```

- [ ] **Step 2: Run the new Agent tests and verify constructor/signature failures**

Run: `python -m pytest tests/server/test_agent_in_situ_generation.py -q`

Expected: FAIL with `TypeError: Agent.__init__() got an unexpected keyword argument 'enable_in_situ_tool_generation'`.

- [ ] **Step 3: Extend `Agent.__init__` with optional generation dependencies**

Modify `server/agent.py` constructor signature:

```python
def __init__(
    self,
    data_dir: str = "memory/data",
    default_model: str = "gemma-4-e2b-it-4bit",
    max_context_turns: int = 20,
    use_tool_registry: bool = True,
    tool_retrieval_mode: str = "auto",
    tool_retrieval_top_k: int = 12,
    tool_db_path: str | None = None,
    tool_telemetry_db_path: str | None = None,
    enable_in_situ_tool_generation: bool = False,
    tool_sandbox_dir: str | None = None,
    tool_developer: Any = None,
    tool_verifier: Any = None,
    tool_risk_classifier: Any = None,
) -> None:
```

Add assignments after tool DB paths:

```python
self._enable_in_situ_tool_generation = enable_in_situ_tool_generation
self._tool_sandbox_dir = tool_sandbox_dir or os.path.join(data_dir, "generated_tools")
self._tool_developer = tool_developer
self._tool_verifier = tool_verifier
self._tool_risk_classifier = tool_risk_classifier
```

- [ ] **Step 4: Initialize default developer/verifier/classifier only when enabled**

At the end of `_init_tool_system()`, after the router exists, add:

```python
if self._enable_in_situ_tool_generation:
    from server.tools.developer import ToolDeveloper
    from server.tools.risk_classifier import CapabilityRiskClassifier
    from server.tools.verifier import ToolVerifier

    self._tool_developer = self._tool_developer or ToolDeveloper(sandbox_dir=self._tool_sandbox_dir)
    self._tool_verifier = self._tool_verifier or ToolVerifier(sandbox_dir=self._tool_sandbox_dir)
    self._tool_risk_classifier = self._tool_risk_classifier or CapabilityRiskClassifier()
```

Keep this inside the existing try block so init failure preserves legacy fallback.

- [ ] **Step 5: Add query-local dispatch context helpers**

Add near the tool dispatch section in `server/agent.py`:

```python
from dataclasses import dataclass, field


@dataclass
class _ToolDispatchContext:
    task_id: str
    user_input: str
    local_tool_pool: Any = None
    attempted_generation: set[str] = field(default_factory=set)
```

Add this Agent helper:

```python
def _create_local_tool_pool(self) -> Any:
    if self._tool_router is None:
        return None
    from server.tools.local_pool import LocalToolPool

    return LocalToolPool(router=self._tool_router, sandbox_dir=self._tool_sandbox_dir)
```

- [ ] **Step 6: Pass local pool through the run loop**

In `run()`, after `max_rounds = 5`, add:

```python
dispatch_ctx = _ToolDispatchContext(
    task_id=task_id,
    user_input=user_input,
    local_tool_pool=self._create_local_tool_pool(),
)
```

Change tool listing and execution:

```python
tools = await self.get_tools_async(user_input, local_tool_pool=dispatch_ctx.local_tool_pool)
```

```python
tool_results = await self.execute_tools(tool_calls, dispatch_ctx=dispatch_ctx)
```

- [ ] **Step 7: Add local tool schemas to `get_tools_async`**

Change the signature:

```python
async def get_tools_async(
    self,
    query: str | None = None,
    local_tool_pool: Any = None,
) -> list[dict[str, Any]]:
```

Before each return of registry-provided tools, wrap the result:

```python
if local_tool_pool is not None:
    return local_tool_pool.list_openai_tools(global_tools)
return global_tools
```

Apply that wrapping to `all`, `top_k`, `hybrid`, and `auto` registry returns. Leave the legacy return path unchanged because local generated tools require the registry/router execution path.

- [ ] **Step 8: Pass dispatch context through `execute_tools`**

Change the signature:

```python
async def execute_tools(
    self,
    tool_calls: list[dict[str, Any]],
    dispatch_ctx: _ToolDispatchContext | None = None,
) -> list[dict[str, Any]]:
```

Change dispatch call:

```python
content = await self._dispatch_tool(name, args, dispatch_ctx)
```

- [ ] **Step 9: Update `_dispatch_tool` for local-first lookup and in-situ generation**

Change the signature:

```python
async def _dispatch_tool(
    self,
    name: str,
    args: dict[str, Any],
    dispatch_ctx: _ToolDispatchContext | None = None,
) -> str:
```

At the start of the registry branch, build `ToolContext` with task/user data:

```python
from server.tools.spec import ToolContext, ToolRequest

ctx = ToolContext(
    session_id=self._session_id,
    task_id=dispatch_ctx.task_id if dispatch_ctx else "",
    user_input=dispatch_ctx.user_input if dispatch_ctx else "",
    model_name=self._current_model,
)
```

Before global registry dispatch, check the query-local pool:

```python
if dispatch_ctx and dispatch_ctx.local_tool_pool is not None:
    local_result = await dispatch_ctx.local_tool_pool.dispatch(name, args, ctx)
    if local_result.error_type != "tool_not_found":
        return local_result.content
```

Replace the current `ToolRequest` block inside `result.error_type == "tool_not_found"` with:

```python
req = self._build_tool_request(name, args, dispatch_ctx)

generated = await self._try_in_situ_generation(req, args, ctx, dispatch_ctx)
if generated is not None:
    return generated.content

if self._tool_telemetry is not None:
    try:
        self._tool_telemetry.record_tool_request(req)
    except Exception:
        logger.warning("Failed to record tool request for '%s'", name, exc_info=True)

return self._integration_error_for_tool(name)
```

- [ ] **Step 10: Add request building and in-situ generation helpers**

Add helper methods to `Agent`:

```python
def _build_tool_request(
    self,
    name: str,
    args: dict[str, Any],
    dispatch_ctx: _ToolDispatchContext | None,
) -> Any:
    from server.tools.spec import ToolRequest

    return ToolRequest(
        task_id=dispatch_ctx.task_id if dispatch_ctx else "",
        session_id=self._session_id,
        reason=f"Model requested tool '{name}' which does not exist",
        missing_capability=f"Tool '{name}' with args {json.dumps(args, ensure_ascii=False)[:200]}",
        candidate_name=name,
        candidate_description=f"Auto-detected need for tool: {name}",
        candidate_input_schema={
            "type": "object",
            "properties": {k: {"type": "string"} for k in args},
            "required": list(args.keys()),
        },
        candidate_output_schema={"type": "object", "properties": {}},
    )


async def _try_in_situ_generation(
    self,
    request: Any,
    args: dict[str, Any],
    ctx: Any,
    dispatch_ctx: _ToolDispatchContext | None,
) -> Any | None:
    if not self._enable_in_situ_tool_generation:
        return None
    if dispatch_ctx is None or dispatch_ctx.local_tool_pool is None:
        return None
    if request.candidate_name in dispatch_ctx.attempted_generation:
        return None
    if self._tool_developer is None or self._tool_verifier is None or self._tool_risk_classifier is None:
        return None

    dispatch_ctx.attempted_generation.add(request.candidate_name)
    decision = self._tool_risk_classifier.annotate_request(request)
    if not decision.auto_generatable:
        return None

    generation = await self._tool_developer.generate(request)
    if not generation.get("success") or not generation.get("source_code"):
        request.metadata["in_situ_generation"] = {
            "success": False,
            "error": generation.get("error"),
        }
        return None

    source_code = generation["source_code"]
    verify_result = self._tool_verifier.verify(request.candidate_name, source_code)
    if not verify_result.passed:
        request.metadata["in_situ_generation"] = {
            "success": False,
            "error": "; ".join(verify_result.errors),
        }
        return None

    spec = self._build_generated_tool_spec(request)
    dispatch_ctx.local_tool_pool.register(spec, source_code)
    request.metadata["in_situ_generation"] = {"success": True}
    return await dispatch_ctx.local_tool_pool.dispatch(request.candidate_name, args, ctx)


def _build_generated_tool_spec(self, request: Any) -> Any:
    from server.tools.spec import ToolSpec, ToolRuntime, ToolStatus

    return ToolSpec(
        name=request.candidate_name,
        description=request.candidate_description or request.missing_capability,
        input_schema=request.candidate_input_schema,
        output_schema=request.candidate_output_schema,
        runtime=ToolRuntime.PYTHON_GENERATED,
        provider="query_local",
        risk_level=request.risk_level,
        status=ToolStatus.ACTIVE,
        metadata={
            "generated_from_tool_request": True,
            "task_id": request.task_id,
            "session_id": request.session_id,
            **request.metadata,
        },
    )
```

- [ ] **Step 11: Ensure blocked/successful requests are visible in telemetry**

Inside `_try_in_situ_generation`, after classifier annotation and before returning for blocked requests, do not record telemetry there. Keep the single `record_tool_request(req)` call in `_dispatch_tool` after `_try_in_situ_generation()` returns `None`; this ensures both blocked and failed generation attempts retain classifier metadata. Successful local generation can remain query-local and does not need durable registry mutation in PR A.

- [ ] **Step 12: Run focused Agent tests**

Run: `python -m pytest tests/server/test_agent_in_situ_generation.py -q`

Expected: PASS.

- [ ] **Step 13: Run existing eval live test for dispatch regression**

Run: `python -m pytest tests/evals/test_local_ml_eval.py::TestLiveRun::test_run_live_executes_agent_and_collects_results -q`

Expected: PASS.

- [ ] **Step 14: Commit**

```bash
git add server/agent.py tests/server/test_agent_in_situ_generation.py
git commit -m "feat: enable optional in-situ tool generation"
```

---

## PR B: Batch Absorption and Behavioral Consolidation

### Task 4: Add Replay Cases to Absorber Parent Tests

**Files:**
- Modify: `server/tools/absorber.py`
- Modify: `tests/tools/test_absorber_e2e.py`

- [ ] **Step 1: Add replay success and failure tests**

Append tests to `tests/tools/test_absorber_e2e.py`. Use the existing fixtures and fake remote generator patterns in that file:

```python
@pytest.mark.asyncio
async def test_absorber_replay_cases_pass_for_compatible_merge(tmp_path):
    registry, telemetry, verifier = _make_absorber_test_stack(tmp_path)
    registry.register(_char_counter_tool("char_counter_a", replay_cases=[
        {
            "name": "counts_ascii_chars",
            "arguments": {"text": "abc"},
            "expect": {"count": 3},
            "match": "subset",
        }
    ]))
    registry.register(_char_counter_tool("char_counter_b", replay_cases=[
        {
            "name": "counts_empty_string",
            "arguments": {"text": ""},
            "expect": {"count": 0},
            "match": "subset",
        }
    ]))

    absorber = ToolAbsorber(
        registry=registry,
        telemetry=telemetry,
        verifier=verifier,
        remote_generate=_fake_char_counter_merge,
        sandbox_dir=str(tmp_path / "sandbox"),
    )

    result = await absorber.run(dry_run=False)

    assert result["clusters_merged"] == 1
    assert registry.get_tool("char_counter_merged") is not None


@pytest.mark.asyncio
async def test_absorber_replay_cases_block_incompatible_merge(tmp_path):
    registry, telemetry, verifier = _make_absorber_test_stack(tmp_path)
    registry.register(_char_counter_tool("char_counter_a", replay_cases=[
        {
            "name": "protects_character_count",
            "arguments": {"text": "abc"},
            "expect": {"count": 3},
            "match": "subset",
        }
    ]))
    registry.register(_char_counter_tool("char_counter_b", replay_cases=[
        {
            "name": "protects_empty_string",
            "arguments": {"text": ""},
            "expect": {"count": 0},
            "match": "subset",
        }
    ]))

    absorber = ToolAbsorber(
        registry=registry,
        telemetry=telemetry,
        verifier=verifier,
        remote_generate=_fake_broken_char_counter_merge,
        sandbox_dir=str(tmp_path / "sandbox"),
    )

    result = await absorber.run(dry_run=False)

    assert result["clusters_merged"] == 0
    assert any(detail["action"] == "parent_tests_failed" for detail in result["details"])
    assert registry.get_tool("char_counter_merged") is None
```

Add helper updates in the same test file:

```python
def _char_counter_tool(name: str, replay_cases: list[dict] | None = None) -> ToolSpec:
    spec = ToolSpec(
        name=name,
        description="Count characters in text",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        output_schema={
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        },
        runtime=ToolRuntime.PYTHON_GENERATED,
        provider="generated_fixture",
        risk_level=RiskLevel.L0,
        status=ToolStatus.ACTIVE,
        metadata={},
        embedding=[0.1] * 8,
    )
    if replay_cases:
        spec.metadata["replay_cases"] = replay_cases
    return spec
```

- [ ] **Step 2: Run the new absorber tests and verify replay is not enforced**

Run: `python -m pytest tests/tools/test_absorber_e2e.py -q`

Expected: FAIL on `test_absorber_replay_cases_block_incompatible_merge` because `_run_parent_tests()` currently accepts structurally valid broken merges.

- [ ] **Step 3: Implement replay collection and matching**

Modify `server/tools/absorber.py` imports:

```python
import json
from pathlib import Path
from server.tools.router import GeneratedPythonExecutor
from server.tools.spec import ToolContext, ToolResult
```

Add helper methods inside `ToolAbsorber`:

```python
def _collect_replay_cases(self, cluster: list[ToolSpec]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for parent in cluster:
        for case in parent.metadata.get("replay_cases", []):
            enriched = dict(case)
            enriched["parent_tool"] = parent.name
            cases.append(enriched)
    return cases


async def _run_replay_cases(
    self,
    merged_name: str,
    source_code: str,
    replay_cases: list[dict[str, Any]],
) -> bool:
    sandbox = Path(self._sandbox_dir)
    sandbox.mkdir(parents=True, exist_ok=True)
    (sandbox / f"{merged_name}.py").write_text(source_code, encoding="utf-8")

    executor = GeneratedPythonExecutor(sandbox_dir=str(sandbox))
    spec = ToolSpec(
        name=merged_name,
        description="Merged replay target",
        input_schema={"type": "object", "properties": {}},
        runtime=ToolRuntime.PYTHON_GENERATED,
        provider="merged",
        risk_level=RiskLevel.L0,
    )
    ctx = ToolContext()

    for case in replay_cases:
        result = await executor.execute(spec, case.get("arguments", {}), ctx)
        if not result.success:
            return False
        if not self._matches_replay_expectation(result, case):
            return False
    return True


def _matches_replay_expectation(self, result: ToolResult, case: dict[str, Any]) -> bool:
    try:
        actual = json.loads(result.content)
    except json.JSONDecodeError:
        return False

    expected = case.get("expect", {})
    match = case.get("match", "subset")
    if match == "exact":
        return actual == expected
    if match == "subset":
        return all(actual.get(key) == value for key, value in expected.items())
    return False
```

- [ ] **Step 4: Enforce replay from `_run_parent_tests()`**

Replace `_run_parent_tests()` with:

```python
async def _run_parent_tests(
    self, cluster: list[ToolSpec], merged_name: str, source_code: str
) -> bool:
    """Verify merged tools and replay parent behavior contracts when present."""
    result = self._verifier.verify(merged_name, source_code)
    if not result.passed:
        return False

    replay_cases = self._collect_replay_cases(cluster)
    if not replay_cases:
        return True

    return await self._run_replay_cases(merged_name, source_code, replay_cases)
```

- [ ] **Step 5: Preserve replay provenance on merged specs**

When constructing `merged_spec.metadata`, add:

```python
"replay_cases": self._collect_replay_cases(cluster),
"replay_case_count": len(self._collect_replay_cases(cluster)),
```

If this repeats collection in the code, assign `replay_cases = self._collect_replay_cases(cluster)` before `merged_spec = ToolSpec(...)` and reuse that variable.

- [ ] **Step 6: Run absorber tests**

Run: `python -m pytest tests/tools/test_absorber_e2e.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add server/tools/absorber.py tests/tools/test_absorber_e2e.py
git commit -m "feat: enforce absorber replay cases"
```

---

## PR C: Evolution Monitoring and Self-Evolution Eval

### Task 5: Split EGL and Generated-Tool Use Metrics

**Files:**
- Modify: `evals/local_ml_eval/metrics.py`
- Modify: `evals/local_ml_eval/report.py`
- Modify: `tests/evals/test_local_ml_eval.py`

- [ ] **Step 1: Update metrics tests first**

Modify `tests/evals/test_local_ml_eval.py` metric expectations:

```python
def test_compute_metrics_with_self_evolution_fields():
    from evals.local_ml_eval.metrics import compute_metrics

    results = [
        {
            "task_success": True,
            "expected_tool_hit": True,
            "forbidden_tool_called": False,
            "tool_failed": False,
            "tool_requested": True,
            "tool_called": True,
            "tool_invocation_count": 1,
            "generated_tools_created": 1,
            "generated_tool_success": True,
            "in_situ_generation_attempted": True,
            "in_situ_generation_success": True,
            "mode": "zero-start",
            "category": "generated_tool",
            "latency_ms": 100,
        },
        {
            "task_success": True,
            "expected_tool_hit": True,
            "forbidden_tool_called": False,
            "tool_failed": False,
            "tool_requested": False,
            "tool_called": True,
            "tool_invocation_count": 1,
            "generated_tools_created": 0,
            "generated_tool_success": True,
            "in_situ_generation_attempted": False,
            "in_situ_generation_success": False,
            "mode": "warm-start",
            "warm_start_reused_generated_tool": True,
            "category": "generated_tool",
            "latency_ms": 50,
        },
    ]

    m = compute_metrics(results)

    assert m["evolution_growth_level"] == 0.5
    assert m["egl"] == 0.5
    assert m["generated_tool_use_rate"] == 1.0
    assert m["in_situ_generation_success_rate"] == 1.0
    assert m["warm_start_reuse_rate"] == 1.0
    assert m["tool_request_rate"] == 0.5
```

Update the existing "all fields" assertions to require:

```python
required = {
    "total_tasks",
    "task_success_rate",
    "expected_tool_hit_rate",
    "forbidden_tool_call_rate",
    "tool_failure_rate",
    "tool_request_rate",
    "generated_tool_success_rate",
    "evolution_growth_level",
    "egl",
    "generated_tool_use_rate",
    "in_situ_generation_success_rate",
    "warm_start_reuse_rate",
    "avg_latency_ms",
}
```

- [ ] **Step 2: Run metrics tests and verify missing fields**

Run: `python -m pytest tests/evals/test_local_ml_eval.py::TestMetrics -q`

Expected: FAIL because `evolution_growth_level`, `generated_tool_use_rate`, `in_situ_generation_success_rate`, and `warm_start_reuse_rate` are absent.

- [ ] **Step 3: Update `compute_metrics()`**

Modify `evals/local_ml_eval/metrics.py`:

```python
def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return _empty_metrics()

    task_successes = sum(1 for r in results if r.get("task_success"))
    expected_hits = sum(1 for r in results if r.get("expected_tool_hit"))
    forbidden_calls = sum(1 for r in results if r.get("forbidden_tool_called"))
    tool_failures = sum(1 for r in results if r.get("tool_failed"))
    tool_requests = sum(1 for r in results if r.get("tool_requested"))
    generated_successes = sum(1 for r in results if r.get("generated_tool_success"))
    generated_total = sum(1 for r in results if r.get("category") == "generated_tool")
    latencies = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]

    evolution_growth_level = _compute_evolution_growth_level(results)

    return {
        "total_tasks": total,
        "task_success_rate": round(task_successes / total, 4),
        "expected_tool_hit_rate": round(expected_hits / total, 4),
        "forbidden_tool_call_rate": round(forbidden_calls / total, 4),
        "tool_failure_rate": round(tool_failures / total, 4),
        "tool_request_rate": round(tool_requests / total, 4),
        "generated_tool_success_rate": round(generated_successes / generated_total, 4)
        if generated_total > 0
        else None,
        "evolution_growth_level": evolution_growth_level,
        "egl": evolution_growth_level,
        "generated_tool_use_rate": _compute_generated_tool_use_rate(results),
        "in_situ_generation_success_rate": _compute_in_situ_generation_success_rate(results),
        "warm_start_reuse_rate": _compute_warm_start_reuse_rate(results),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
    }
```

Add helpers:

```python
def _tool_invocations(results: list[dict[str, Any]]) -> int:
    return sum(int(r.get("tool_invocation_count") or (1 if r.get("tool_called") else 0)) for r in results)


def _compute_evolution_growth_level(results: list[dict[str, Any]]) -> float | None:
    invocations = _tool_invocations(results)
    if invocations == 0:
        return None
    created = sum(int(r.get("generated_tools_created") or 0) for r in results)
    return round(created / invocations, 4)


def _compute_generated_tool_use_rate(results: list[dict[str, Any]]) -> float | None:
    invocations = _tool_invocations(results)
    if invocations == 0:
        return None
    generated_used = sum(1 for r in results if r.get("generated_tool_success"))
    return round(generated_used / invocations, 4)


def _compute_in_situ_generation_success_rate(results: list[dict[str, Any]]) -> float | None:
    attempts = sum(1 for r in results if r.get("in_situ_generation_attempted"))
    if attempts == 0:
        return None
    successes = sum(1 for r in results if r.get("in_situ_generation_success"))
    return round(successes / attempts, 4)


def _compute_warm_start_reuse_rate(results: list[dict[str, Any]]) -> float | None:
    warm_tasks = [r for r in results if r.get("mode") == "warm-start"]
    if not warm_tasks:
        return None
    reused = sum(1 for r in warm_tasks if r.get("warm_start_reused_generated_tool"))
    return round(reused / len(warm_tasks), 4)
```

Update `_empty_metrics()` and the dry-run metrics function to include all new keys with `None` where execution did not happen.

- [ ] **Step 4: Update report labels**

Modify `evals/local_ml_eval/report.py` markdown rows:

```python
f"| Evolution growth level (EGL) | {metrics.get('evolution_growth_level', 'N/A')} |",
f"| Generated tool use rate | {metrics.get('generated_tool_use_rate', 'N/A')} |",
f"| In-situ generation success rate | {metrics.get('in_situ_generation_success_rate', 'N/A')} |",
f"| Warm-start reuse rate | {metrics.get('warm_start_reuse_rate', 'N/A')} |",
```

Keep `egl` in JSON for backward compatibility, but treat it as an alias for `evolution_growth_level`.

- [ ] **Step 5: Run eval metrics/report tests**

Run: `python -m pytest tests/evals/test_local_ml_eval.py::TestMetrics tests/evals/test_local_ml_eval.py::TestReport -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add evals/local_ml_eval/metrics.py evals/local_ml_eval/report.py tests/evals/test_local_ml_eval.py
git commit -m "feat: split self evolution eval metrics"
```

---

### Task 6: Add Zero-Start and Warm-Start Eval Observability

**Files:**
- Modify: `evals/local_ml_eval/runner.py`
- Modify: `evals/local_ml_eval/tasks.jsonl`
- Modify: `evals/local_ml_eval/fixtures.py`
- Modify: `tests/evals/test_local_ml_eval.py`

- [ ] **Step 1: Add task schema tests for mode values**

Modify `tests/evals/test_local_ml_eval.py` task validation coverage:

```python
def test_tasks_include_self_evolution_modes(self):
    tasks = load_tasks("evals/local_ml_eval/tasks.jsonl")
    modes = {task.get("mode") for task in tasks if task.get("mode")}

    assert "zero-start" in modes
    assert "warm-start" in modes
```

Update task validation so `mode`, when present, must be one of:

```python
{"zero-start", "warm-start"}
```

- [ ] **Step 2: Add deterministic self-evolution task rows**

Append rows to `evals/local_ml_eval/tasks.jsonl`:

```json
{"id": "evolve_zero_001", "category": "generated_tool", "mode": "zero-start", "input": "把文本 Codex Rocks 反转", "expected_tools": ["text_reverse"], "forbidden_tools": ["computer_action"], "success_check": "tool_called", "risk_level": "L0"}
{"id": "evolve_zero_002", "category": "generated_tool", "mode": "zero-start", "input": "把 hello-world 转成 slug 风格", "expected_tools": ["slugify_text"], "forbidden_tools": ["computer_action"], "success_check": "tool_called", "risk_level": "L0"}
{"id": "evolve_warm_001", "category": "generated_tool", "mode": "warm-start", "input": "再次把文本 abc 反转", "expected_tools": ["text_reverse"], "forbidden_tools": ["computer_action"], "success_check": "tool_called", "risk_level": "L0"}
```

- [ ] **Step 3: Extend fixture generated tools**

Add `text_reverse` and `slugify_text` entries to the generated fixture map in `evals/local_ml_eval/fixtures.py`:

```python
"text_reverse": {
    "description": "Reverse provided text",
    "input_schema": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    "output_schema": {
        "type": "object",
        "properties": {"result": {"type": "string"}},
    },
    "source": """from pydantic import BaseModel

class InputModel(BaseModel):
    text: str

class OutputModel(BaseModel):
    result: str

def run(input: InputModel) -> OutputModel:
    return OutputModel(result=input.text[::-1])
""",
},
"slugify_text": {
    "description": "Convert text to a lowercase slug",
    "input_schema": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    "output_schema": {
        "type": "object",
        "properties": {"slug": {"type": "string"}},
    },
    "source": """import re
from pydantic import BaseModel

class InputModel(BaseModel):
    text: str

class OutputModel(BaseModel):
    slug: str

def run(input: InputModel) -> OutputModel:
    slug = re.sub(r"[^a-z0-9]+", "-", input.text.lower()).strip("-")
    return OutputModel(slug=slug)
""",
},
```

- [ ] **Step 4: Emit per-result evolution fields from runner**

In `evals/local_ml_eval/runner.py`, extend `_execute_task()` result construction:

```python
tool_invocation_count = len(tool_rows)
created_rows = _rows_since(conn, "tool_events", before_tool_event_id)
generated_tools_created = sum(1 for row in created_rows if row.get("event_type") == "tool_created")
in_situ_generation_attempted = any(
    row.get("metadata", {}).get("risk_classifier", {}).get("auto_generatable") is not None
    for row in request_rows
)
in_situ_generation_success = generated_tools_created > 0 and in_situ_generation_attempted
warm_start_reused_generated_tool = (
    task.get("mode") == "warm-start"
    and generated_tool_success
    and generated_tools_created == 0
)
```

If telemetry metadata is stored as JSON text in `request_rows`, parse it with `json.loads()` before accessing nested fields.

Add fields to the result dict:

```python
"mode": task.get("mode"),
"tool_invocation_count": tool_invocation_count,
"generated_tools_created": generated_tools_created,
"in_situ_generation_attempted": in_situ_generation_attempted,
"in_situ_generation_success": in_situ_generation_success,
"warm_start_reused_generated_tool": warm_start_reused_generated_tool,
```

- [ ] **Step 5: Ensure fixture mode produces stable warm-start reuse**

When `generated_tool_fixtures=True`, fixture tools are already installed before the run. For zero-start testing with no preinstalled generated tools, the user can run with `--no-generated-tool-fixtures` after PR A is complete. Do not mutate global registry mid-eval in this task; just expose fields that allow both modes to be measured.

- [ ] **Step 6: Run task and live eval tests**

Run: `python -m pytest tests/evals/test_local_ml_eval.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add evals/local_ml_eval/runner.py evals/local_ml_eval/tasks.jsonl evals/local_ml_eval/fixtures.py tests/evals/test_local_ml_eval.py
git commit -m "feat: add self evolution eval modes"
```

---

### Task 7: Update Tool Evolution Docs and Verification Script

**Files:**
- Modify: `docs/tool_evolution.md`
- Modify: `evals/local_ml_eval/README.md`
- Modify: `scripts/verify_tool_evolution.sh` if its metric checks mention only the old EGL interpretation

- [ ] **Step 1: Update `docs/tool_evolution.md` state model**

Add a section:

```markdown
## Self-Evolution State Model

This implementation treats the agent state as:

`M_t = <W0, C0, T_t>`

`W0` is the fixed Agent.run workflow, `C0` is the fixed context policy, and
`T_t` is the evolving toolset. Missing capabilities can create query-local
tools during a run. Batch absorption later consolidates those local tools into
the next global pool.
```

- [ ] **Step 2: Document metric meanings in `evals/local_ml_eval/README.md`**

Replace the old EGL row with:

```markdown
| Metric | Meaning |
|--------|---------|
| evolution_growth_level / egl | Generated tools created or registered per tool invocation |
| generated_tool_use_rate | Generated tools successfully used per tool invocation |
| tool_request_rate | Tasks that observed a missing capability |
| in_situ_generation_success_rate | In-situ generation attempts that produced verified local tools |
| warm_start_reuse_rate | Warm-start tasks solved by existing generated tools without new synthesis |
```

Add interpretation:

```markdown
In zero-start runs, high `evolution_growth_level` can be healthy because the
tool pool is growing. As the pool stabilizes, creation pressure should fall
while `generated_tool_use_rate` and `warm_start_reuse_rate` rise.
```

- [ ] **Step 3: Align verification script metric names if needed**

Run: `rg -n "egl|generated_tool_success_rate|generated_tool_use_rate|evolution_growth_level" scripts/verify_tool_evolution.sh docs/tool_evolution.md evals/local_ml_eval/README.md`

If `scripts/verify_tool_evolution.sh` asserts only the old eval meaning of `egl`, change it to check `evolution_growth_level` and `generated_tool_use_rate` separately.

- [ ] **Step 4: Run docs and verification checks**

Run: `bash scripts/verify_tool_evolution.sh`

Expected: PASS.

Run: `python -m pytest tests/evals/test_local_ml_eval.py tests/tools/test_risk_classifier.py tests/tools/test_local_pool.py tests/server/test_agent_in_situ_generation.py tests/tools/test_absorber_e2e.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/tool_evolution.md evals/local_ml_eval/README.md scripts/verify_tool_evolution.sh
git commit -m "docs: align tool evolution self evolution model"
```

---

## Final Verification

- [ ] **Step 1: Run focused test suite**

Run:

```bash
python -m pytest tests/tools/test_risk_classifier.py tests/tools/test_local_pool.py tests/server/test_agent_in_situ_generation.py tests/tools/test_absorber_e2e.py tests/evals/test_local_ml_eval.py -q
```

Expected: PASS.

- [ ] **Step 2: Run tool evolution verification script**

Run:

```bash
bash scripts/verify_tool_evolution.sh
```

Expected: PASS.

- [ ] **Step 3: Run formatting checks available in repo**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 4: Inspect branch diff**

Run:

```bash
git status --short
git log --oneline -7
```

Expected: working tree is clean after the task commits, and the recent commits match the seven task-level commits above.

---

## Execution Order

1. Complete PR A tasks first. The local pool and in-situ path are prerequisites for same-run self-evolution.
2. Complete PR B next. Replay-backed absorber consolidation depends on generated local tools being a real source of future global tools.
3. Complete PR C last. Metrics and reports should measure the behavior introduced by PR A and PR B.

This sequence keeps the implementation close to the paper's transition:

`M_(t-1) = <W0, C0, T_(t-1)> -> M_t = <W0, C0, T_t>`
