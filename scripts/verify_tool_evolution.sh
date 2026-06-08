set -e

python -c "import server.agent; print('OK')"

pytest tests/python/test_agent.py -q
pytest tests/python/test_agent_tool_system.py -q
pytest tests/tools tests/optimization tests/python memory/test_memory.py -q

grep -R --exclude-dir=__pycache__ -n "await self.get_tools_async" server/agent.py
grep -R --exclude-dir=__pycache__ -n "retriever.retrieve" server/agent.py
grep -R --exclude-dir=__pycache__ -n "_tool_registry.dispatch" server/agent.py
grep -R --exclude-dir=__pycache__ -n "record_tool_request" server/agent.py server/tools
grep -R --exclude-dir=__pycache__ -n "GEPA(" server/optimization
grep -R --exclude-dir=__pycache__ -n "os.environ.clear" server/tools/sandbox_runner.py

python -m server.tools.evolve_cli metrics
python -m server.tools.evolve_cli requests --limit 1 --dry-run
python -m server.tools.evolve_cli absorber --dry-run

pytest tests/e2e/test_tool_evolution_smoke.py tests/evals/test_local_ml_eval.py -q
python -m evals.local_ml_eval.runner \
  --tasks evals/local_ml_eval/tasks.jsonl \
  --output /tmp/local_ml_eval_report.json \
  --markdown /tmp/local_ml_eval_report.md \
  --dry-run
