# Tool Evolution Follow-Ups Design

## Goal

Align the next round of `local-ml` work with the core logic of `2601.18226v2.pdf`: an in-situ self-evolving agent whose fixed workflow and context evolve primarily through tool accumulation, tool absorption, and convergence toward a stable reusable tool pool.

In this framing, the important question is not "how do we add more engineering guardrails around the current tool system?" It is "how do we move the implementation closer to a true self-evolving tool loop?"

## Paper Alignment

The paper makes four claims that matter directly for this repo:

1. The evolving state is primarily the toolset `T`, while workflow `W` and context `C` stay fixed.
2. Missing capabilities should trigger tool synthesis in situ during task execution, not only as an offline admin action.
3. Parallel batch evolution should allow multiple queries to synthesize local tools, then absorb them into a consolidated global pool.
4. Evolution should be monitored with a convergence-oriented metric, analogous to a training-loss curve.

This design therefore centers the follow-up work on:

- in-situ tool accumulation
- absorbing and consolidating local tool pools
- evolution monitoring and evaluation

## Scope

This design covers:

- Runtime missing-capability handling that can synthesize and use tools during execution
- A local-tool-pool plus global-tool-pool model that matches the paper's batch evolution story
- Absorber safety that checks behavioral compatibility, not only structural validity
- Evaluation/reporting that reflects self-evolution progress rather than only static smoke success

This design does not cover:

- CI automation as a primary deliverable
- Evolving workflow graphs or context/memory policy
- Production-grade historical replay infrastructure mined from real traces
- External-account or network-heavy benchmarks that break determinism

CI can still be added later, but it is not the center of this follow-up because it is not the center of the paper's self-evolution logic.

## Current State

The repo already contains many of the right building blocks, but the control loop is still incomplete relative to the paper:

- `Agent.run()` can retrieve tools and dispatch them through the registry/router path.
- When a requested tool is missing, the system records a `ToolRequest`.
- Tool development, verification, registry insertion, absorber merging, and metrics exist as separate components.
- `local_ml_eval` can exercise the real `Agent.run -> ToolRegistry -> ToolRuntimeRouter -> Telemetry` path.

But three gaps remain:

1. `Agent.run()` does not yet perform in-situ synthesis and resume execution.
2. Tool absorption is still modeled mainly as an explicit offline step, not as the consolidation of per-query local pools into the next global state.
3. Evaluation focuses on harness correctness and smoke success more than on self-evolution dynamics and convergence.

## Delivery Strategy

Use three sequential PRs, but organize them around the paper's self-evolution loop rather than general infra polish.

### PR A: In-Situ Tool Accumulation

Purpose: make missing capabilities synthesizable and usable during execution, with workflow/context fixed and only the toolset evolving.

### PR B: Batch Absorption and Behavioral Consolidation

Purpose: consolidate local tool pools into a global canonical pool while preserving behavior and reducing redundancy.

### PR C: Evolution Monitoring and Self-Evolution Eval

Purpose: evaluate the system as a self-evolving agent, not only as a static tool runtime.

This order is intentional:

- PR A creates the in-situ evolution loop
- PR B makes the resulting tool growth sustainable and canonical
- PR C measures whether the system is actually evolving in the intended direction

## Architecture

```
Fixed Agent Workflow / Context
  +-- Agent.run()
        +-- Manager-like capability check
        +-- retrieve global tools T_(t-1)
        +-- execute with local tool pool P_(t,i) + retrieved subset
        +-- if capability gap:
              +-- synthesize tool in situ
              +-- verify tool
              +-- attach to local pool
              +-- resume execution

Batch Evolution
  +-- queries q_(t,1) ... q_(t,B)
        +-- each query builds local tool pool P_(t,i)
        +-- after batch completion:
              +-- union(T_(t-1), P_(t,1)...P_(t,B))
              +-- aggregator clusters semantically similar tools
              +-- merger produces canonical tools
              +-- next global pool becomes T_t

Evolution Monitoring
  +-- telemetry
  +-- absorber outcomes
  +-- local_ml_eval batch runs
  +-- convergence-oriented metrics (including EGL-style signals)
```

## State Model

Follow the paper's simplification:

- workflow `W` is fixed
- context `C` is fixed
- toolset `T` is the evolving component

So the practical state transition in this repo should be treated as:

`M_(t-1) = <W0, C0, T_(t-1)> -> M_t = <W0, C0, T_t>`

The implementation should therefore avoid broad redesigns of prompts, orchestration topology, or memory policy in this follow-up. The focus is on making `T_t` evolve correctly.

## PR A Design: In-Situ Tool Accumulation

### Problem

The paper's agent can discover a capability gap during execution, synthesize a tool, attach it to the local context, and continue the same task. The current repo stops at `ToolRequest` recording.

That means the repo supports observation of missing capability, but not yet in-situ self-evolution.

### Design

Introduce a query-scoped local tool pool that exists alongside the global registry snapshot used for the current task.

When the Executor path encounters a missing capability:

1. Record the `ToolRequest`
2. Classify the requested capability with an explicit risk classifier
3. Invoke the existing developer/verifier pipeline
4. If verification passes, register the generated tool into a local pool for the current query/session
5. Resume the interrupted execution with the augmented local pool

The key idea is that new tools should become immediately usable for the current task without first being promoted into the durable global pool.

### Risk Classification Contract

The current `ToolRequest` default risk level is not enough for in-situ generation because missing-tool requests originate from model output and default to `L0` unless a caller sets a stricter value. PR A must therefore add an explicit capability risk classifier before any developer/verifier call.

The classifier should:

- treat unknown or ambiguous capabilities as not auto-generatable
- allow only clearly L0/L1 capabilities through the in-situ path
- block requests involving network access, integrations, file mutation, desktop control, credentials, shell execution, or privileged operations
- record the classifier decision in `ToolRequest.metadata` so eval reports can distinguish "missing capability observed" from "safe to synthesize"

If classification is unavailable or inconclusive, the system must keep the current behavior: record the request and return the normal missing-tool error path.

### Local Tool Pool Contract

PR A should make the query-local pool a concrete overlay instead of an informal concept.

A `LocalToolPool` should provide:

- `register(spec, source_code)` for verified query-local generated tools
- `get_tool(name)` for local lookup
- `list_openai_tools()` so `Agent.get_tools_async()` can expose local tools to the model on retry
- `dispatch(name, args, ctx)` or an equivalent helper that executes local specs through the existing `ToolRuntimeRouter`

Lookup order should be local-first, then global registry. Tool listing should return the global tool subset plus local query tools, deduped by name.

Generated source for local tools must be materialized in the sandbox used by `GeneratedPythonExecutor`, because the current executor resolves generated tools as `<sandbox_dir>/<tool_name>.py`. The spec may still carry `entrypoint`, but same-run local execution cannot rely on `entrypoint` alone until the executor supports it.

Same-run retry must be bounded. A single missing tool should not trigger unbounded generate/retry loops; the first implementation should allow at most one in-situ synthesis attempt per missing tool name per task.

### Component Changes

- `server/agent.py`
  - introduce a query-scoped local tool pool
  - allow missing-capability handling to trigger synthesis and retry/resume
- local tool pool helper under `server/tools/`
  - support query-local lookup, listing, and dispatch layered over the global registry
- tool developer / verifier path
  - reuse existing generation and verification contracts instead of inventing a new generation stack

### Constraints

- Keep workflow and prompts fixed
- Respect current risk gating; do not allow high-risk auto-generation
- Do not trust a `ToolRequest` default risk level as sufficient evidence for auto-generation
- If synthesis or verification fails, fall back to the current behavior of recording the request and returning the existing error path

### Acceptance Criteria

- A missing-capability query can generate a safe tool and finish within the same run
- Generated tools become available immediately in the query-local pool
- Unknown-risk requests remain request-only and are not auto-generated
- Failure still degrades gracefully to recorded request plus normal failure path

## PR B Design: Batch Absorption and Behavioral Consolidation

### Problem

The paper's batch evolution story is not just "generate tools repeatedly." It is "let each query create local tools, then absorb redundant local results into a compact global repository."

The repo already has an absorber, but two issues remain:

- its mental model is not yet explicitly tied to local-pool to global-pool consolidation
- its safety check is still structural, not behavioral

### Design

Treat the output of PR A as local tool pools `P_(t,i)`. After a batch of queries completes, the system should consolidate:

`T_t = Φ({T_(t-1), P_(t,1), ..., P_(t,B)})`

where `Φ` is the absorber pipeline:

- aggregate similar tools into clusters
- merge compatible clusters into canonical tools
- preserve or reject tools based on behavioral compatibility

### Replay/Regression Contract

To make absorption trustworthy, extend `_run_parent_tests()` beyond verifier-only checks.

Use `ToolSpec.metadata["replay_cases"]` as the first implementation vehicle for behavior-preserving checks. Each replay case is a JSON-serializable contract describing:

- parent behavior being protected
- input arguments
- expected output subset or exact output
- matching mode

This keeps the first implementation lightweight and directly attached to tool specs, while still enforcing that merged tools preserve parent semantics.

### Component Changes

- `server/tools/absorber.py`
  - interpret absorption as consolidation of global + local tool pools
  - run replay cases after structural verification
- `tests/tools/test_absorber_e2e.py`
  - seed replay cases through `ToolSpec.metadata`
  - prove that incompatible merged behavior blocks consolidation

### Constraints

- No full historical trace ingestion in this iteration
- Replay should block merges on incompatibility, not merely log warnings
- If no replay cases exist, verifier-only behavior may remain as the fallback path

### Acceptance Criteria

- Batch-produced local tools can be absorbed into a canonical next-step global pool
- Behavior-breaking merges are rejected
- Absorber semantics are explicitly framed as part of self-evolution, not just offline cleanup

## PR C Design: Evolution Monitoring and Self-Evolution Eval

### Problem

The paper emphasizes convergence monitoring, zero-start evolution, and accumulation of reusable general capabilities. The current eval harness proves correctness of pieces, but it does not yet tell us whether the system is evolving the way the paper describes.

### Design

Refocus `local_ml_eval` from "smoke harness with live execution support" to "self-evolution observation harness."

This does not require turning it into a giant benchmark suite. It means the tasks and reports should make visible:

- missing-capability discovery
- in-situ generation success
- reuse of previously generated tools
- absorption/consolidation effects
- convergence-oriented trends such as decreasing need for new tool synthesis

### Evaluation Modes

The harness should support at least two conceptual modes:

- `zero-start`: start from an empty or minimal toolset and observe capability growth
- `warm-start`: start from an already accumulated pool and observe transfer/reuse

The first implementation can simulate these modes with offline fixtures and controlled registries rather than requiring a production-grade persistent corpus.

### Metrics

Preserve and refine the repo's EGL-style signals because they are already the closest local analogue to the paper's convergence metric, but first separate the existing metric names.

Current repo code uses two meanings under the same `egl` label:

- server/docs: `created_generated_tools / tool_invocations`
- eval harness: `successfully_used_generated_tools / tool_invocations`

PR C should standardize these as distinct metrics:

- `evolution_growth_level` or `egl`: generated tools created or registered per tool invocation. This is the convergence-pressure metric and should match `server/tools/metrics.py` and `docs/tool_evolution.md`.
- `generated_tool_use_rate`: generated tools successfully used per tool invocation. This measures reuse/utility, not creation pressure.
- `tool_request_rate`: missing-capability pressure per task.
- `in_situ_generation_success_rate`: safe missing capabilities that became verified local tools in the same run.
- `warm_start_reuse_rate`: warm-start tasks solved by existing generated tools without new synthesis.

The important interpretation is:

- high creation pressure early in zero-start is expected
- as the pool stabilizes, the need for new tools should drop relative to usage
- convergence is not "no tools are ever created"; it is "new synthesis becomes increasingly exceptional"
- reuse metrics should rise as growth pressure falls; they should not be collapsed into the same EGL value

### Task Expansion

Extend `evals/local_ml_eval/tasks.jsonl` and fixture coverage with scenarios that specifically probe self-evolution dynamics:

- missing capability discovered, generated, and reused
- semantically similar generated tools across batch-like scenarios
- warm-start reuse without regeneration

General-purpose deterministic text/computation tasks are still preferred because they keep the loop reproducible.

### Component Changes

- `evals/local_ml_eval/tasks.jsonl`
  - add scenarios tied to zero-start, warm-start, and reuse
- `evals/local_ml_eval/fixtures.py`
  - add deterministic generated-tool fixtures that make reuse observable
- `tests/evals/test_local_ml_eval.py`
  - assert self-evolution behavior, not just one-shot execution success
- `evals/local_ml_eval/metrics.py`
  - split creation-pressure and generated-tool-use metrics into explicit names
  - ensure reports surface convergence-relevant signals clearly

### Acceptance Criteria

- The harness can distinguish zero-start from warm-start behavior
- Reports expose whether the system is still creating tools aggressively or beginning to reuse accumulated capability
- Eval scenarios explicitly exercise self-evolution rather than only static tool execution

## File-Level Change Plan

### PR A

- Modify: `server/agent.py`
- Modify: tool-registry lookup path to support query-local augmentation
- Reuse: existing developer / verifier / generated execution stack

### PR B

- Modify: `server/tools/absorber.py`
- Modify: `tests/tools/test_absorber_e2e.py`
- Reuse: `ToolSpec.metadata` for replay cases

### PR C

- Modify: `evals/local_ml_eval/tasks.jsonl`
- Modify: `evals/local_ml_eval/fixtures.py`
- Modify: `evals/local_ml_eval/metrics.py`
- Modify: `tests/evals/test_local_ml_eval.py`
- Modify: `evals/local_ml_eval/README.md`
- Modify: `docs/tool_evolution.md`

## Risks and Mitigations

### Runtime synthesis makes execution brittle

Mitigation:

- keep risk gates intact
- fall back cleanly when synthesis/verification fails
- start with deterministic offline-friendly cases

### Local/global pool layering becomes confusing

Mitigation:

- make the distinction explicit in code and docs
- avoid mutating the durable global pool mid-query
- promote only after query or batch completion

### Absorber safety remains too weak

Mitigation:

- require replay compatibility when replay cases are present
- treat replay failures as merge blockers

### Eval becomes a static smoke suite again

Mitigation:

- define tasks around zero-start, warm-start, reuse, and synthesis pressure
- keep convergence-oriented metrics visible in reports

## Testing Strategy

### PR A

- targeted tests for missing-capability synthesis and resume
- regression coverage for existing missing-tool failure path

### PR B

- absorber replay/regression tests
- existing absorber end-to-end tests

### PR C

- `python -m pytest -q tests/evals/test_local_ml_eval.py tests/e2e/test_tool_evolution_smoke.py`
- controlled zero-start and warm-start eval runs
- report inspection for convergence-relevant metrics

## Acceptance Criteria

This follow-up track is complete when all of the following are true:

- missing capabilities can trigger in-situ tool synthesis and same-run reuse under the current risk policy
- local tool pools can be consolidated into a next-step global pool through the absorber
- merged tools are checked for behavioral compatibility, not only structural validity
- `local_ml_eval` can report self-evolution behavior such as zero-start growth, warm-start reuse, and convergence-oriented trends
- the resulting system is materially closer to the paper's `M_t = <W0, C0, T_t>` evolution loop
