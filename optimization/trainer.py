"""Auto Trainer - automatic optimization pipeline.

Continuously collects usage data and optimizes the agent.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from optimization.collector import UsageCollector, Outcome
from optimization.optimizer import PromptOptimizer

logger = logging.getLogger(__name__)


class AutoTrainer:
    """Automatic training pipeline for the agent.

    Collects real usage data and periodically optimizes:
    - System prompts
    - Tool descriptions
    - Response strategies

    Usage:
        trainer = AutoTrainer()
        trainer.start()

        # In your agent loop:
        trainer.record_interaction(user_input, response, tool_calls, outcome)

        # Periodically:
        trainer.train()
    """

    def __init__(
        self,
        data_dir: str = "memory/data",
        lm_model: str = "openai/gpt-4o-mini",
        min_interactions: int = 20,
        auto_train_threshold: int = 50,
    ):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._collector = UsageCollector(str(self._data_dir / "usage.db"))
        self._lm_model = lm_model
        self._optimizer = None  # Lazy init, requires DSPy

        self._min_interactions = min_interactions
        self._auto_train_threshold = auto_train_threshold

        self._optimized_dir = self._data_dir / "optimized"
        self._optimized_dir.mkdir(exist_ok=True)

        self._session_id: str | None = None
        self._interaction_count: int = 0

    def start(self) -> None:
        """Start the trainer."""
        self._collector.connect()
        self._session_id = self._collector.start_session()
        self._interaction_count = 0
        logger.info("Trainer started, session: %s", self._session_id)

    def stop(self) -> None:
        """Stop the trainer."""
        if self._session_id:
            self._collector.end_session()
        self._collector.disconnect()
        logger.info("Trainer stopped")

    def record_interaction(
        self,
        user_input: str,
        agent_response: str,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        outcome: Outcome = Outcome.UNKNOWN,
        feedback_score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Record an interaction for training."""
        interaction_id = self._collector.record_interaction(
            user_input=user_input,
            agent_response=agent_response,
            tool_calls=tool_calls,
            tool_results=tool_results,
            outcome=outcome,
            feedback_score=feedback_score,
            metadata=metadata,
        )

        self._interaction_count += 1

        # Auto-train if threshold reached
        if self._interaction_count >= self._auto_train_threshold:
            logger.info("Auto-training threshold reached, starting training...")
            self.train()

        return interaction_id

    def train(self, force: bool = False) -> dict[str, Any]:
        """Run optimization pipeline.

        Returns:
            Training results with optimized prompts.
        """
        stats = self._collector.get_stats()
        total = stats["total_interactions"]

        if total < self._min_interactions and not force:
            return {
                "status": "skipped",
                "reason": f"Not enough data ({total}/{self._min_interactions})",
            }

        logger.info("Starting training with %d interactions", total)

        # Get training examples
        examples = self._collector.get_training_examples(
            min_score=0.3,
            limit=200,
        )

        if not examples:
            return {"status": "skipped", "reason": "No valid examples"}

        # Lazy init optimizer (requires DSPy)
        if self._optimizer is None:
            try:
                self._optimizer = PromptOptimizer(self._lm_model)
            except ImportError:
                return {"status": "skipped", "reason": "DSPy not installed (pip install dspy)"}

        # Configure optimizer
        self._optimizer.configure()

        # Load examples
        self._optimizer.load_examples(examples)

        results = {"status": "success", "timestamp": datetime.now().isoformat()}

        # Optimize system prompt
        try:
            current_prompt = self._load_current_prompt()
            optimized = self._optimizer.optimize_system_prompt(current_prompt)

            # Save optimized prompt
            output_path = self._optimized_dir / "system_prompt.json"
            self._optimizer.save_optimizations(output_path)

            results["system_prompt"] = {
                "status": "optimized",
                "path": str(output_path),
                "improvement": self._calculate_improvement(current_prompt, optimized),
            }
        except Exception as e:
            results["system_prompt"] = {"status": "error", "error": str(e)}
            logger.error("Failed to optimize system prompt: %s", e)

        # Analyze failures
        failures = self._collector.get_failed_interactions(limit=20)
        if failures:
            results["failure_analysis"] = self._analyze_failures(failures)

        # Reset counter
        self._interaction_count = 0

        # Save training report
        report_path = self._data_dir / f"training_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        results["report_path"] = str(report_path)

        logger.info("Training complete: %s", results.get("status"))
        return results

    def get_stats(self) -> dict[str, Any]:
        """Get training statistics."""
        stats = self._collector.get_stats()
        stats["current_session"] = self._session_id
        stats["interactions_this_session"] = self._interaction_count
        return stats

    def get_optimized_prompt(self) -> str | None:
        """Get the latest optimized system prompt."""
        path = self._optimized_dir / "system_prompt.json"
        if path.exists():
            data = json.loads(path.read_text())
            return data.get("system")
        return None

    def _load_current_prompt(self) -> str:
        """Load the current system prompt."""
        # Try to load from optimized dir first
        optimized = self.get_optimized_prompt()
        if optimized:
            return optimized

        # Fall back to soul
        from memory.soul import Soul
        soul_path = self._data_dir / "soul.json"
        if soul_path.exists():
            soul = Soul.load(soul_path)
            return soul.get_system_prompt()

        return "You are a helpful assistant."

    def _calculate_improvement(self, old: str, new: str) -> dict[str, Any]:
        """Calculate improvement metrics."""
        return {
            "old_length": len(old),
            "new_length": len(new),
            "added_examples": new.count("### 示例"),
        }

    def _analyze_failures(self, failures: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze failure patterns."""
        patterns = {}
        for failure in failures:
            # Extract failure pattern
            tool_calls = failure.get("tool_calls", [])
            for tc in tool_calls:
                tool_name = tc.get("name", "unknown")
                if tool_name not in patterns:
                    patterns[tool_name] = {"count": 0, "examples": []}
                patterns[tool_name]["count"] += 1
                if len(patterns[tool_name]["examples"]) < 3:
                    patterns[tool_name]["examples"].append(failure["user_input"])

        return {
            "total_failures": len(failures),
            "patterns": patterns,
        }
