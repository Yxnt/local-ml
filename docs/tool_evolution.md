# Tool Evolution System

## Overview

The tool evolution system enables local-ml to automatically detect missing capabilities, generate new tools, verify them safely, and evolve the tool set over time. It follows a "background evolution" philosophy — evolution never blocks the user's request path.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Agent.run() [user path]             │
│  get_tools() → dispatch() → telemetry               │
│  tool_not_found → record ToolRequest                │
└────────────────────┬────────────────────────────────┘
                     │ (async, non-blocking)
┌────────────────────▼────────────────────────────────┐
│           ToolEvolutionOrchestrator [admin/CLI]      │
│  process_pending_requests() → Developer → Verifier   │
│  run_absorber() → Discover → Aggregate → Merge       │
│  promote_candidates() → CANDIDATE → ACTIVE           │
│  compute_metrics() → EGL, success rates              │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│              DSPy / GEPA Optimization                │
│  TraceDatasetBuilder → train/val sets                │
│  GEPAEvolutionOptimizer → prompt candidates          │
│  PromptStore → versioning + rollback                 │
└────────────────────┬────────────────────────────────┘
                     │ (optional, external CLI)
┌────────────────────▼────────────────────────────────┐
│         Darwinian Evolver Adapter                    │
│  Export candidate → CLI → Read evolved → Verify      │
│  Default: disabled                                   │
└─────────────────────────────────────────────────────┘
```

## Components

### ToolRegistry (`server/tools/registry.py`)
Central registry for all tools. SQLite-backed with in-memory cache.
- `register(spec)` / `unregister(name)` — lifecycle management
- `list_openai_tools()` — used by Agent.run() each turn
- `dispatch(name, args, ctx)` — routes to correct executor

### ToolRetriever (`server/tools/retriever.py`)
Semantic + keyword search for relevant tools.
- `retrieve(query, limit)` — returns top-k tools for a task
- Falls back to keyword matching when sqlite-vec is unavailable
- Used by Agent.get_tools() in `top_k` or `auto` mode

### TelemetryService (`server/tools/telemetry.py`)
Append-only event log for tool lifecycle tracking.
- Records: tool_invoked, tool_succeeded, tool_failed, tool_request, tool_created, tool_registered, tool_deprecated, tool_merged, tool_promoted
- Storage: `memory/usage.db` (tool_events + tool_requests tables)

### ToolDeveloper (`server/tools/developer.py`)
Generates Python tool code from ToolRequests using a remote LLM.
- Risk gate: only L0/L1 tools
- Prompt enforces strict sandbox contract (InputModel/OutputModel/run)
- Forbids: os, subprocess, eval, shell, network

### ToolVerifier (`server/tools/verifier.py`)
4-stage verification of generated code:
1. **Static scan** (AST) — forbidden imports/calls, in-process
2. **Import test** — subprocess sandbox_runner.py
3. **Structure check** — InputModel/OutputModel/run existence
4. **Schema test** — instantiate + call run()

Stages 2-4 run in a **subprocess** (`sandbox_runner.py`) for isolation.

### GeneratedPythonExecutor (`server/tools/router.py`)
Executes generated tools via subprocess by default.
- `execution_mode="subprocess"` (default) — isolated execution
- `execution_mode="in_process_dev_only"` — development only

### ToolAbsorber (`server/tools/absorber.py`)
Finds and merges similar tools.
- **Discover**: cosine similarity (brute-force or sqlite-vec)
- **Aggregate**: LLM or heuristic (same provider + runtime)
- **Merge**: LLM or heuristic (pick broadest schema)
- Records lineage in `tool_lineage` table

### ToolEvolutionOrchestrator (`server/tools/orchestrator.py`)
Coordinates all evolution steps.
- `process_pending_requests(limit, dry_run)` — consume ToolRequests
- `run_absorber(dry_run, max_clusters)` — merge similar tools
- `promote_candidates(min_success_count, min_success_rate)` — CANDIDATE → ACTIVE
- `compute_metrics()` — EGL and other stats

### ToolMetrics (`server/tools/metrics.py`)
Computes evolution metrics from telemetry data.
- `get_egl(window)` — Evolution Gap Level
- `get_tool_success_rate(tool_name, window)`
- `get_tool_request_count(window)`
- `get_all_metrics(window)` — all metrics in one dict

## CLI Usage

```bash
# Process pending tool requests (dry-run first)
python -m server.tools.evolve_cli requests --limit 5 --dry-run

# Apply tool generation
python -m server.tools.evolve_cli requests --limit 5 --apply

# Run absorber (dry-run)
python -m server.tools.evolve_cli absorber --dry-run

# View metrics
python -m server.tools.evolve_cli metrics

# Promote candidates
python -m server.tools.evolve_cli promote --min-success-count 3 --min-success-rate 0.8

# Run all steps
python -m server.tools.evolve_cli run-once --dry-run
```

## Configuration

### Agent tool_retrieval_mode
Controls how tools are provided to the model each turn:
- `"all"` — every active tool (default, fine for <20 tools)
- `"top_k"` — semantic search for most relevant tools
- `"auto"` — use top_k when tool count exceeds `tool_retrieval_top_k`

### ToolDeveloper environment
- `MIMO_BASE_URL` — remote LLM endpoint
- `MIMO_API_KEY` — API key

### Darwinian Evolver
- `tool_evolution.darwinian.enabled: false` — default disabled
- `tool_evolution.darwinian.cli_command` — external CLI path
- `tool_evolution.darwinian.timeout_sec: 300`

## EGL Metric

EGL = cumulative_created_or_registered_generated_tools / cumulative_tool_invocations

- **Numerator**: tools with `runtime=PYTHON_GENERATED` or `provider in (generated, merged)`
- **Denominator**: `tool_invoked` events
- **Interpretation**: decreasing EGL means the tool pool is stabilizing

## Safety

### Risk Levels
- **L0/L1**: Auto-generatable (pure computation, read temp dirs)
- **L2+**: Requires manual approval (network, integrations)
- **L4+**: Never auto-generate (computer use, smart home)

### Subprocess Isolation
- Generated code runs in `sandbox_runner.py` subprocess
- Environment filtered: no KEY/SECRET/TOKEN/PASSWORD vars
- 15-second timeout
- AST blacklist: os, subprocess, eval, exec, __import__, etc.

### Prompt Safety
- DSPy/GEPA can only optimize prompts, never safety rules
- Optimized prompts are always `status=candidate`, require explicit promotion
- Rollback available via PromptStore

## Rollback

### Prompt rollback
```python
store = PromptStore(db_path="memory/usage.db")
store.rollback("manager_tool_selection_prompt")
```

### Tool deprecation
```python
registry.unregister("tool_name")
```

### Lineage tracking
```python
absorber.get_lineage("merged_tool_name")
# Returns list of {parent_tool_name, relation, created_at}
```

## Verification & Evaluation

### Verify tool evolution infrastructure
```bash
bash scripts/verify_tool_evolution.sh
```

### Run E2E smoke test
```bash
pytest tests/e2e/test_tool_evolution_smoke.py -q
```
Tests the full pipeline: ToolRequest → FakeDeveloper → Verifier → Registry → Execute → Promote → EGL.

### Run eval harness
```bash
# Dry-run (validate tasks, output skeleton report)
python -m evals.local_ml_eval.runner \
  --tasks evals/local_ml_eval/tasks.jsonl \
  --output /tmp/local_ml_eval_report.json \
  --dry-run

# View report
cat /tmp/local_ml_eval_report.json
```

Tasks are in `evals/local_ml_eval/tasks.jsonl` (20 tasks: 5 memory, 5 retrieval, 5 missing, 5 generated).

## Known Limitations

1. **sqlite-vec dependency**: Semantic search requires sqlite-vec. Without it, falls back to keyword matching.
2. **LLM dependency**: ToolDeveloper and absorber merge require a remote LLM. Without it, heuristic fallbacks are used.
3. **No OS-level sandbox**: subprocess isolation is not OS-enforced (no seccomp/chroot). AST blacklist provides defense-in-depth.
4. **No real regression tests**: `_run_parent_tests()` in absorber is a simplified schema check, not a full replay of historical invocations.
5. **Darwinian Evolver**: external CLI must be installed separately. Only synthetic test data is passed.
6. **EGL time windows**: 24h/7d windows depend on `created_at` timestamps being accurate.
7. **Agent.run does NOT auto-generate tools**: `tool_not_found` only records a ToolRequest. Tool generation is explicit via CLI/admin.
8. **Agent.run does NOT auto-run absorber**: Absorber is only triggered via `ToolEvolutionOrchestrator` / CLI.
9. **Darwinian Evolver is disabled by default**: Must be explicitly enabled in config.
10. **GEPA/BootstrapFewShot only via CLI/admin**: Not triggered from Agent.run().
