# local_ml_eval

Minimal evaluation harness for local-ml tool evolution.

## Quick start

```bash
# Dry-run (validate tasks, output skeleton report)
python -m evals.local_ml_eval.runner \
  --tasks evals/local_ml_eval/tasks.jsonl \
  --output /tmp/local_ml_eval_report.json \
  --dry-run

# View report
cat /tmp/local_ml_eval_report.json
```

## Task format (JSONL)

```json
{
  "id": "memory_001",
  "category": "memory",
  "input": "记住我喜欢简洁回答",
  "expected_tools": ["memory_remember"],
  "forbidden_tools": ["computer_action"],
  "success_check": "tool_called",
  "risk_level": "L0"
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

## Metrics

| Metric | Description |
|--------|-------------|
| task_success_rate | Fraction of tasks completed successfully |
| expected_tool_hit_rate | Fraction where expected tool was called |
| forbidden_tool_call_rate | Fraction where a forbidden tool was called |
| tool_failure_rate | Fraction where tool execution failed |
| tool_request_rate | Fraction where a ToolRequest was recorded |
| generated_tool_success_rate | Fraction of generated tools that executed successfully |
| egl | Generated tools used / total tool invocations |
| avg_latency_ms | Average task latency |
