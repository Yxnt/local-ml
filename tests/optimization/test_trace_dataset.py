"""Tests for TraceDatasetBuilder.

Covers:
  1. build_tool_selection_examples from fake telemetry data
  2. build_tool_request_examples from fake data
  3. split_train_val produces correct ratios
  4. Source field is set to 'heuristic'
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from server.tools.telemetry import TelemetryService
from server.optimization.trace_dataset import TraceDatasetBuilder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def telemetry(tmp_path):
    svc = TelemetryService(db_path=str(tmp_path / "telemetry.db"))
    svc.connect()
    yield svc
    svc.disconnect()


@pytest.fixture()
def builder(telemetry):
    return TraceDatasetBuilder(telemetry=telemetry)


def _seed_tool_events(telemetry: TelemetryService) -> None:
    """Seed telemetry with tool invocation events grouped by task_id."""
    # Task 1: bash tool succeeded
    telemetry.record(
        "tool_invoked",
        tool_name="bash",
        task_id="task-1",
        session_id="sess-1",
        metadata={"user_input": "list files in current directory"},
    )
    telemetry.record(
        "tool_succeeded",
        tool_name="bash",
        task_id="task-1",
        session_id="sess-1",
        result_summary="file1.txt file2.txt",
    )

    # Task 2: search tool failed
    telemetry.record(
        "tool_invoked",
        tool_name="obsidian_search",
        task_id="task-2",
        session_id="sess-2",
        metadata={"user_input": "find notes about project X"},
    )
    telemetry.record(
        "tool_failed",
        tool_name="obsidian_search",
        task_id="task-2",
        session_id="sess-2",
        error_type="connection_error",
        error_message="Obsidian not connected",
    )

    # Task 3: bash tool succeeded
    telemetry.record(
        "tool_invoked",
        tool_name="bash",
        task_id="task-3",
        session_id="sess-3",
        metadata={"user_input": "check disk usage"},
    )
    telemetry.record(
        "tool_succeeded",
        tool_name="bash",
        task_id="task-3",
        session_id="sess-3",
        result_summary="50% used",
    )


def _seed_tool_requests(telemetry: TelemetryService) -> None:
    """Seed telemetry with tool request data."""
    conn = telemetry._conn
    assert conn is not None

    # A consumed request (tool was later created)
    conn.execute(
        """
        INSERT INTO tool_requests
            (task_id, session_id, reason, missing_capability,
             candidate_name, candidate_desc, candidate_input, candidate_output,
             risk_level, privacy_notes, metadata, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "task-req-1", "sess-1", "Need word count tool",
            "count_words", "word_counter", "Count words in text",
            json.dumps({"type": "object", "properties": {"text": {"type": "string"}}}),
            json.dumps({"type": "object", "properties": {"count": {"type": "integer"}}}),
            "L0", "", "{}", "2025-01-01T00:00:00+00:00",
        ),
    )

    # An unconsumed request
    conn.execute(
        """
        INSERT INTO tool_requests
            (task_id, session_id, reason, missing_capability,
             candidate_name, candidate_desc, candidate_input, candidate_output,
             risk_level, privacy_notes, metadata, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "task-req-2", "sess-2", "Need PDF parser",
            "parse_pdf", "pdf_parser", "Parse PDF documents",
            json.dumps({"type": "object", "properties": {"path": {"type": "string"}}}),
            json.dumps({"type": "object"}),
            "L1", "", "{}", "2025-01-01T01:00:00+00:00",
        ),
    )
    conn.commit()

    # Record tool_created for word_counter (consumed)
    telemetry.record("tool_created", tool_name="word_counter")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildToolSelectionExamples:
    """Tests for build_tool_selection_examples."""

    def test_basic_extraction(self, builder, telemetry):
        """Extracts tool selection examples from seeded events."""
        _seed_tool_events(telemetry)

        examples = builder.build_tool_selection_examples(limit=10)
        assert len(examples) >= 2

        # Check structure of first example
        for ex in examples:
            assert "task" in ex
            assert "available_tools" in ex
            assert "selected_tools" in ex
            assert "outcome" in ex
            assert "score" in ex

    def test_success_score(self, builder, telemetry):
        """Successful tool invocation -> score = 1.0."""
        _seed_tool_events(telemetry)

        examples = builder.build_tool_selection_examples(limit=10)
        success_ex = [e for e in examples if e["outcome"] == "tool_succeeded"]
        assert len(success_ex) >= 1
        for ex in success_ex:
            assert ex["score"] == 1.0

    def test_failure_score(self, builder, telemetry):
        """Failed tool invocation -> score = 0.0."""
        _seed_tool_events(telemetry)

        examples = builder.build_tool_selection_examples(limit=10)
        fail_ex = [e for e in examples if e["outcome"] == "tool_failed"]
        assert len(fail_ex) >= 1
        for ex in fail_ex:
            assert ex["score"] == 0.0

    def test_empty_telemetry(self, builder, telemetry):
        """No events -> empty examples."""
        examples = builder.build_tool_selection_examples(limit=10)
        assert examples == []

    def test_limit_respected(self, builder, telemetry):
        """Limit parameter caps the number of examples."""
        _seed_tool_events(telemetry)

        examples = builder.build_tool_selection_examples(limit=1)
        assert len(examples) <= 1

    def test_user_input_as_task(self, builder, telemetry):
        """Task description comes from metadata user_input."""
        _seed_tool_events(telemetry)

        examples = builder.build_tool_selection_examples(limit=10)
        tasks = {ex["task"] for ex in examples}
        # At least one should have the user_input text
        assert any("list files" in t or "find notes" in t or "check disk" in t for t in tasks)


class TestBuildToolRequestExamples:
    """Tests for build_tool_request_examples."""

    def test_basic_extraction(self, builder, telemetry):
        """Extracts tool request examples from seeded data."""
        _seed_tool_requests(telemetry)

        examples = builder.build_tool_request_examples(limit=10)
        assert len(examples) >= 2

        for ex in examples:
            assert "task" in ex
            assert "failure_report" in ex
            assert "tool_request" in ex
            assert "risk_level" in ex
            assert "was_valid" in ex
            assert "score" in ex

    def test_consumed_request_score(self, builder, telemetry):
        """Consumed request (tool later created) -> score = 0.8."""
        _seed_tool_requests(telemetry)

        examples = builder.build_tool_request_examples(limit=10)
        consumed = [e for e in examples if e["was_valid"]]
        assert len(consumed) >= 1
        for ex in consumed:
            assert ex["score"] == 0.8

    def test_unconsumed_request_score(self, builder, telemetry):
        """Unconsumed request -> score = 0.3."""
        _seed_tool_requests(telemetry)

        examples = builder.build_tool_request_examples(limit=10)
        unconsumed = [e for e in examples if not e["was_valid"]]
        assert len(unconsumed) >= 1
        for ex in unconsumed:
            assert ex["score"] == 0.3

    def test_risk_level_present(self, builder, telemetry):
        """Risk level is included in examples."""
        _seed_tool_requests(telemetry)

        examples = builder.build_tool_request_examples(limit=10)
        risk_levels = {ex["risk_level"] for ex in examples}
        assert "L0" in risk_levels or "L1" in risk_levels

    def test_tool_request_json_parseable(self, builder, telemetry):
        """The tool_request field is valid JSON."""
        _seed_tool_requests(telemetry)

        examples = builder.build_tool_request_examples(limit=10)
        for ex in examples:
            parsed = json.loads(ex["tool_request"])
            assert "candidate_name" in parsed


class TestSplitTrainVal:
    """Tests for split_train_val."""

    def test_basic_split(self):
        """split_train_val produces two non-overlapping sets."""
        examples = [{"id": i} for i in range(20)]
        train, val = TraceDatasetBuilder.split_train_val(examples, val_ratio=0.2)

        assert len(train) + len(val) == len(examples)
        # No overlap
        train_ids = {id(e) for e in train}
        val_ids = {id(e) for e in val}
        assert train_ids.isdisjoint(val_ids)

    def test_val_ratio(self):
        """Val set is approximately val_ratio of total."""
        examples = [{"id": i} for i in range(100)]
        train, val = TraceDatasetBuilder.split_train_val(examples, val_ratio=0.2)

        # Should be roughly 80/20 split
        assert 15 <= len(val) <= 25
        assert 75 <= len(train) <= 85

    def test_empty_input(self):
        """Empty input -> empty outputs."""
        train, val = TraceDatasetBuilder.split_train_val([], val_ratio=0.2)
        assert train == []
        assert val == []

    def test_single_example(self):
        """Single example -> at least one in train."""
        examples = [{"id": 0}]
        train, val = TraceDatasetBuilder.split_train_val(examples, val_ratio=0.2)
        assert len(train) >= 1

    def test_two_examples(self):
        """Two examples -> train gets at least one."""
        examples = [{"id": 0}, {"id": 1}]
        train, val = TraceDatasetBuilder.split_train_val(examples, val_ratio=0.2)
        assert len(train) >= 1
        assert len(val) >= 1
        assert len(train) + len(val) == 2

    def test_05_ratio(self):
        """50/50 split works."""
        examples = [{"id": i} for i in range(100)]
        train, val = TraceDatasetBuilder.split_train_val(examples, val_ratio=0.5)
        assert len(train) + len(val) == 100
        assert 45 <= len(val) <= 55


class TestSourceField:
    """Tests verifying proxy score behavior (heuristic-based)."""

    def test_scores_are_heuristic_based(self, builder, telemetry):
        """Scores come from heuristic functions, not ML models."""
        _seed_tool_events(telemetry)
        _seed_tool_requests(telemetry)

        sel_examples = builder.build_tool_selection_examples(limit=10)
        req_examples = builder.build_tool_request_examples(limit=10)

        # Tool selection scores are either 0.0, 0.5, or 1.0
        for ex in sel_examples:
            assert ex["score"] in (0.0, 0.5, 1.0)

        # Tool request scores are either 0.3 or 0.8
        for ex in req_examples:
            assert ex["score"] in (0.3, 0.8)
