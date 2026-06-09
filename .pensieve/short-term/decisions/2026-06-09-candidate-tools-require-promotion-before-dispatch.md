---
id: candidate-tools-require-promotion-before-dispatch
type: decision
title: Candidate tools require promotion before dispatch
status: active
created: 2026-06-09
updated: 2026-06-09
tags: [tool-evolution, registry, candidate, testing]
---

# Candidate tools require promotion before dispatch

## 一句话结论
> Treat `ToolStatus.CANDIDATE` tools as absorber/promotion inputs, not normal registry dispatch targets.

## 上下文链接
- 基于：[[decisions/2026-06-07-tool-evolution-integration-pattern]]
- 相关：[[maxims/preserve-user-visible-behavior-as-a-hard-rule]]

## Context

The tool evolution follow-up introduced query-local generated tools that are exported into the durable registry as `CANDIDATE` absorber inputs. This creates a lifecycle distinction between "generated and available for validation/consolidation" and "active global tool available to the agent."

## Problem

Older end-to-end tests executed newly registered `CANDIDATE` tools through `ToolRegistry.dispatch()` to collect success telemetry before promotion. That made candidates behave like active tools and contradicted the local-pool to batch-absorption model.

## Alternatives Considered

- Allow `CANDIDATE` dispatch and rely on retrieval filtering: rejected because direct registry dispatch would still expose unabsorbed tools as callable.
- Promote immediately after generation: rejected because it bypasses absorber replay and batch consolidation.
- Validate candidates through the runtime router, then promote based on telemetry: chosen because it preserves behavioral validation without making candidates ordinary agent-callable tools.

## Decision

`ToolRegistry.dispatch()` must reject `ToolStatus.CANDIDATE` with `error_type="tool_candidate"`. Candidate behavior checks should use explicit internal validation paths, such as `ToolRuntimeRouter.dispatch(spec, args, ctx)`, then promote to `ACTIVE` before normal registry dispatch.

## Consequence

Tests for generated tool lifecycle should follow: request -> generate -> verify -> register as candidate -> candidate dispatch rejected -> internal validation records telemetry -> promote -> normal dispatch succeeds.

## 探索减负
- 下次可以少问什么：CANDIDATE 是否能通过普通 registry dispatch 调用。
- 下次可以少查什么：旧 e2e 测试为什么要先走 router validation 再 promote。
- 失效条件：如果 registry gains a separate authenticated validation API that can execute candidates without using private router access, update tests to use that API.
