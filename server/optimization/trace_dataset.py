"""Convert telemetry data into DSPy-compatible training examples.

Reads from the ``tool_events`` and ``tool_requests`` tables (via
TelemetryService) and optionally from UsageCollector's ``interactions``
table to produce labeled examples for optimization.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from server.tools.telemetry import TelemetryService

logger = logging.getLogger(__name__)

# Lazy-import UsageCollector to avoid hard dependency at module level.
_UsageCollector = None  # resolved on first use


def _get_collector_cls():
    global _UsageCollector
    if _UsageCollector is None:
        try:
            from optimization.collector import UsageCollector
            _UsageCollector = UsageCollector
        except ImportError:
            _UsageCollector = type(None)  # sentinel: unavailable
    return _UsageCollector


# ---------------------------------------------------------------------------
# Score heuristics (proxy labels)
# ---------------------------------------------------------------------------

def _score_for_tool_event(outcome: str) -> float:
    """Return a proxy score based on the event outcome."""
    if outcome == "tool_succeeded":
        return 1.0
    if outcome == "tool_failed":
        return 0.0
    return 0.5  # no data / unknown


def _score_for_request(event_type: str, metadata: dict[str, Any]) -> float:
    """Return a proxy score for a tool-request event.

    - tool_request consumed (tool_registered/tool_created afterwards) -> 0.8
    - tool_request unconsumed -> 0.3
    - no data -> 0.5
    """
    if event_type == "tool_request":
        # Check metadata for a "consumed" flag if present
        if metadata.get("consumed"):
            return 0.8
        return 0.3
    return 0.5


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class TraceDatasetBuilder:
    """Convert telemetry data into DSPy-compatible training examples.

    Args:
        telemetry: TelemetryService instance (must be connected).
        collector: Optional UsageCollector for richer interaction context.
    """

    def __init__(
        self,
        telemetry: TelemetryService,
        collector: Any | None = None,
    ) -> None:
        self._telemetry = telemetry
        self._collector = collector

    # ------------------------------------------------------------------
    # Tool selection examples
    # ------------------------------------------------------------------

    def build_tool_selection_examples(self, limit: int = 100) -> list[dict[str, Any]]:
        """Build examples for tool selection optimization.

        Each example::

            {
                "task": str,             # task description / user input
                "available_tools": str,  # comma-separated tool names
                "selected_tools": str,   # tool(s) actually invoked
                "outcome": str,          # "tool_succeeded" | "tool_failed"
                "score": float,          # proxy label 0.0-1.0
            }

        Groups events by ``task_id`` and extracts the tool invocation +
        outcome pairs.  Returns at most *limit* examples.
        """
        try:
            raw = self._query_tool_selection_events(limit * 3)  # over-fetch, then group
        except Exception as exc:
            logger.warning("build_tool_selection_examples query failed: %s", exc)
            return []

        # Group by task_id
        by_task: dict[str, list[dict[str, Any]]] = {}
        for row in raw:
            tid = row.get("task_id") or "__no_task__"
            by_task.setdefault(tid, []).append(row)

        examples: list[dict[str, Any]] = []
        for task_id, events in by_task.items():
            if len(examples) >= limit:
                break

            # Find tool invocations and their outcomes
            invocations = [e for e in events if e["event_type"] == "tool_invoked"]
            outcomes = [e for e in events if e["event_type"] in ("tool_succeeded", "tool_failed")]
            outcome_map: dict[str, str] = {}
            for o in outcomes:
                tname = o.get("tool_name", "")
                if tname and tname not in outcome_map:
                    outcome_map[tname] = o["event_type"]

            if not invocations:
                continue

            # Derive task description from the first event's metadata or task_id
            task_desc = task_id
            for e in events:
                meta = self._parse_json(e.get("metadata", "{}"))
                if meta.get("user_input"):
                    task_desc = meta["user_input"]
                    break

            selected = list({e.get("tool_name", "") for e in invocations if e.get("tool_name")})
            if not selected:
                continue

            # Determine overall outcome for this task
            tool_outcomes = [outcome_map.get(t, "") for t in selected]
            if "tool_failed" in tool_outcomes:
                overall = "tool_failed"
            elif "tool_succeeded" in tool_outcomes:
                overall = "tool_succeeded"
            else:
                overall = "unknown"

            # Infer available tools from all tool_names seen in session
            session_tools = list({e.get("tool_name", "") for e in raw if e.get("tool_name") and e.get("session_id") == events[0].get("session_id")})
            if not session_tools:
                session_tools = selected

            examples.append({
                "task": task_desc,
                "available_tools": ", ".join(sorted(session_tools)),
                "selected_tools": ", ".join(sorted(selected)),
                "outcome": overall,
                "score": _score_for_tool_event(overall),
            })

        return examples

    # ------------------------------------------------------------------
    # Tool request quality examples
    # ------------------------------------------------------------------

    def build_tool_request_examples(self, limit: int = 50) -> list[dict[str, Any]]:
        """Build examples for tool request quality.

        Each example::

            {
                "task": str,              # task description
                "failure_report": str,    # error/reason from the request
                "tool_request": str,      # candidate tool request JSON
                "risk_level": str,        # "L0"-"L5"
                "was_valid": bool,        # True if later consumed
                "score": float,           # proxy label 0.0-1.0
            }
        """
        try:
            requests = self._telemetry.get_tool_requests(limit=limit)
        except Exception as exc:
            logger.warning("build_tool_request_examples query failed: %s", exc)
            return []

        # Collect set of tool_names that were later registered/created
        consumed_names: set[str] = set()
        try:
            for ev in self._telemetry.get_recent_events(limit=500):
                if ev["event_type"] in ("tool_registered", "tool_created"):
                    consumed_names.add(ev.get("tool_name", ""))
        except Exception:
            pass

        examples: list[dict[str, Any]] = []
        for req in requests:
            candidate_name = req.get("candidate_name", "")
            was_valid = candidate_name in consumed_names

            tool_request_json = json.dumps({
                "candidate_name": candidate_name,
                "candidate_desc": req.get("candidate_desc", ""),
                "candidate_input": req.get("candidate_input", ""),
                "candidate_output": req.get("candidate_output", ""),
            }, ensure_ascii=False)

            score = 0.8 if was_valid else 0.3

            examples.append({
                "task": req.get("task_id", ""),
                "failure_report": req.get("reason", "") or req.get("missing_capability", ""),
                "tool_request": tool_request_json,
                "risk_level": req.get("risk_level", "L0"),
                "was_valid": was_valid,
                "score": score,
            })

        return examples

    # ------------------------------------------------------------------
    # Train / val split
    # ------------------------------------------------------------------

    @staticmethod
    def split_train_val(
        examples: list[dict[str, Any]],
        val_ratio: float = 0.2,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split examples into train and validation sets.

        Shuffles before splitting so the result is stochastic.
        """
        if not examples:
            return [], []
        shuffled = list(examples)
        random.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        if len(shuffled) <= n_val:
            # Too few examples — put at least one in train
            return shuffled[:1], shuffled[1:]
        return shuffled[n_val:], shuffled[:n_val]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_tool_selection_events(self, limit: int) -> list[dict[str, Any]]:
        """Fetch tool_invoked + tool_succeeded/failed events."""
        conn = self._telemetry._conn
        if conn is None:
            logger.warning("TelemetryService not connected")
            return []
        rows = conn.execute(
            """
            SELECT event_type, tool_name, task_id, session_id,
                   result_summary, error_type, error_message, metadata
            FROM tool_events
            WHERE event_type IN ('tool_invoked', 'tool_succeeded', 'tool_failed')
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
