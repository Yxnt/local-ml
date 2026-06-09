# local_ml_eval

Minimal evaluation harness for local-ml tool evolution.

## Quick start

```bash
# Dry-run (validate tasks, output skeleton report with placeholder metrics)
python -m evals.local_ml_eval.runner \
  --tasks evals/local_ml_eval/tasks.jsonl \
  --output /tmp/local_ml_eval_report.json \
  --dry-run

# View report
cat /tmp/local_ml_eval_report.json
```

```bash
# Live execution (real Agent.run path, offline fixture integrations by default)
python -m evals.local_ml_eval.runner \
  --tasks evals/local_ml_eval/tasks.jsonl \
  --output /tmp/local_ml_eval_report.json \
  --markdown /tmp/local_ml_eval_report.md \
  --model gemma-4-e2b-it-4bit
```

Live mode runs through the real `Agent.run -> ToolRegistry -> ToolRuntimeRouter ->
Telemetry` flow. By default it connects offline fixture integrations for
Obsidian / Calendar / Email and pre-registers generated tool fixtures so the
harness can execute end-to-end without external accounts.

## Task format (JSONL)

```json
{
  "id": "memory_001",
  "category": "memory",
  "input": "记住我喜欢简洁回答",
  "expected_tools": ["memory_remember"],
  "forbidden_tools": ["computer_action"],
  "success_check": "tool_called",
  "risk_level": "L0",
  "mode": "zero-start"
}
```

Fields:
- `id`: unique task id
- `category`: memory | tool_retrieval | missing_tool | generated_tool
- `input`: user input text
- `expected_tools`: tools that should be called
- `forbidden_tools`: tools that must NOT be called
- `success_check`: "tool_called" | "tool_request_recorded"
- `risk_level`: L0-L5
- `mode`: optional self-evolution mode, "zero-start" | "warm-start"

## Metrics

| Metric | Description |
|--------|-------------|
| task_success_rate | Fraction of tasks completed successfully |
| expected_tool_hit_rate | Fraction where expected tool was called |
| forbidden_tool_call_rate | Fraction where a forbidden tool was called |
| tool_failure_rate | Fraction where tool execution failed |
| tool_request_rate | Fraction where a ToolRequest was recorded |
| generated_tool_success_rate | Fraction of generated tools that executed successfully |
| evolution_growth_level | Generated tools created per tool invocation |
| egl | Alias for `evolution_growth_level` |
| generated_tool_use_rate | Generated tools successfully used per tool invocation |
| in_situ_generation_success_rate | In-situ generated tools created per generation attempt |
| warm_start_reuse_rate | Warm-start tasks solved by existing generated tools without new synthesis |
| avg_latency_ms | Average task latency |

## Modes

- `--dry-run`: validate task structure only; metrics are placeholders (`null`)
- default live mode: execute tasks through `Agent.run`
- `--evolution-mode zero-start`: run only zero-start tasks without pre-registering generated fixtures
- `--evolution-mode warm-start`: run only warm-start tasks with generated fixtures pre-registered
- `--integration-fixtures none`: do not connect offline fixture integrations
- `--no-generated-tool-fixtures`: do not pre-register generated eval tools

Interpretation: early zero-start runs may show high `evolution_growth_level`.
As the generated tool pool stabilizes, creation pressure should fall while
`generated_tool_use_rate` and `warm_start_reuse_rate` rise.
