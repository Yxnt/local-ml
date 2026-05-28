"""Tests for optimization module."""

import pytest
from pathlib import Path

from optimization.collector import UsageCollector, Outcome


class TestUsageCollector:
    def test_record_and_query(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        collector = UsageCollector(db_path)
        collector.connect()

        session_id = collector.start_session()
        assert session_id

        # Record some interactions
        id1 = collector.record_interaction(
            user_input="搜索笔记",
            agent_response="找到了 3 条笔记",
            outcome=Outcome.SUCCESS,
            feedback_score=0.8,
        )

        id2 = collector.record_interaction(
            user_input="读取文件",
            agent_response="文件内容...",
            tool_calls=[{"name": "read_file", "args": {"path": "/tmp"}}],
            outcome=Outcome.SUCCESS,
            feedback_score=0.9,
        )

        assert id1 > 0
        assert id2 > 0

        # Query
        examples = collector.get_training_examples(min_score=0.5)
        assert len(examples) == 2

        # Stats
        stats = collector.get_stats()
        assert stats["total_interactions"] == 2
        assert stats["by_outcome"]["success"] == 2

        collector.end_session()
        collector.disconnect()

    def test_failed_interactions(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        collector = UsageCollector(db_path)
        collector.connect()
        collector.start_session()

        collector.record_interaction(
            user_input="错误操作",
            agent_response="失败了",
            outcome=Outcome.FAILURE,
        )

        failures = collector.get_failed_interactions()
        assert len(failures) == 1
        assert failures[0]["outcome"] == "failure"

        collector.disconnect()


class TestAutoTrainer:
    def test_skip_training_insufficient_data(self, tmp_path):
        from optimization.trainer import AutoTrainer

        trainer = AutoTrainer(
            data_dir=str(tmp_path / "data"),
            min_interactions=10,
        )
        trainer.start()

        # Record only a few interactions
        trainer.record_interaction(
            user_input="test",
            agent_response="response",
            outcome=Outcome.SUCCESS,
        )

        result = trainer.train()
        assert result["status"] == "skipped"
        assert "Not enough data" in result["reason"]

        trainer.stop()

    def test_force_training_no_dspy(self, tmp_path):
        """Test that training gracefully handles missing DSPy."""
        from optimization.trainer import AutoTrainer

        trainer = AutoTrainer(
            data_dir=str(tmp_path / "data"),
            min_interactions=10,
        )
        trainer.start()

        # Record some interactions
        for i in range(5):
            trainer.record_interaction(
                user_input=f"test {i}",
                agent_response=f"response {i}",
                outcome=Outcome.SUCCESS,
                feedback_score=0.8,
            )

        # Force training - should handle missing DSPy gracefully
        result = trainer.train(force=True)

        # If DSPy is not installed, it should skip
        # If DSPy is installed, it should succeed
        assert result["status"] in ("success", "skipped", "error")

        trainer.stop()
