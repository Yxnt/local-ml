"""Unified telemetry service — all tool & task lifecycle events flow here.

Storage: SQLite ``tool_events`` table in the same database used by
UsageCollector (``memory/usage.db`` by default).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

EVENT_TYPES = (
    "task_started",
    "task_finished",
    "model_call",
    "tool_invoked",
    "tool_succeeded",
    "tool_failed",
    "tool_created",
    "tool_verified",
    "tool_registered",
    "tool_deprecated",
    "tool_merged",
    "tool_request",
    "remote_escalation",
    "learning_event",
)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class TelemetryService:
    """Append-only event log for tool and task lifecycle tracking."""

    def __init__(self, db_path: str = "memory/usage.db") -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _init_tables(self) -> None:
        assert self._conn is not None
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_events (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type      TEXT NOT NULL,
                tool_name       TEXT,
                tool_version    TEXT,
                task_id         TEXT,
                session_id      TEXT,
                interaction_id  INTEGER,
                args_json       TEXT,
                result_summary  TEXT,
                error_type      TEXT,
                error_message   TEXT,
                latency_ms      INTEGER,
                metadata        TEXT DEFAULT '{}',
                created_at      TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tool_events_tool ON tool_events(tool_name)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tool_events_task ON tool_events(task_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tool_events_type ON tool_events(event_type)"
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_requests (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id             TEXT,
                session_id          TEXT,
                reason              TEXT,
                missing_capability  TEXT,
                candidate_name      TEXT,
                candidate_desc      TEXT,
                candidate_input     TEXT,
                candidate_output    TEXT,
                risk_level          TEXT DEFAULT 'L0',
                privacy_notes       TEXT,
                metadata            TEXT DEFAULT '{}',
                created_at          TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    # -- public API ----------------------------------------------------------

    def record(self, event_type: str, **kwargs: Any) -> int:
        """Record a single event.  Returns the new row id."""
        assert self._conn is not None, "Not connected"
        now = datetime.now(timezone.utc).isoformat()
        metadata = kwargs.pop("metadata", None)
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else "{}"
        row = {
            "event_type": event_type,
            "tool_name": kwargs.get("tool_name"),
            "tool_version": kwargs.get("tool_version"),
            "task_id": kwargs.get("task_id"),
            "session_id": kwargs.get("session_id"),
            "interaction_id": kwargs.get("interaction_id"),
            "args_json": json.dumps(kwargs.get("args"), ensure_ascii=False) if kwargs.get("args") else None,
            "result_summary": kwargs.get("result_summary"),
            "error_type": kwargs.get("error_type"),
            "error_message": kwargs.get("error_message"),
            "latency_ms": kwargs.get("latency_ms"),
            "metadata": metadata_json,
            "created_at": now,
        }
        cols = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row)
        cur = self._conn.execute(f"INSERT INTO tool_events ({cols}) VALUES ({placeholders})", row)
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    # -- convenience helpers -------------------------------------------------

    def record_tool_invoked(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        *,
        task_id: str = "",
        session_id: str = "",
        tool_version: str = "",
    ) -> int:
        return self.record(
            "tool_invoked",
            tool_name=tool_name,
            tool_version=tool_version,
            task_id=task_id,
            session_id=session_id,
            args=args,
        )

    def record_tool_succeeded(
        self,
        tool_name: str,
        result_summary: str = "",
        latency_ms: int | None = None,
        *,
        task_id: str = "",
        session_id: str = "",
        tool_version: str = "",
    ) -> int:
        return self.record(
            "tool_succeeded",
            tool_name=tool_name,
            tool_version=tool_version,
            task_id=task_id,
            session_id=session_id,
            result_summary=result_summary[:500] if result_summary else "",
            latency_ms=latency_ms,
        )

    def record_tool_failed(
        self,
        tool_name: str,
        error_type: str = "",
        error_message: str = "",
        latency_ms: int | None = None,
        *,
        task_id: str = "",
        session_id: str = "",
        tool_version: str = "",
    ) -> int:
        return self.record(
            "tool_failed",
            tool_name=tool_name,
            tool_version=tool_version,
            task_id=task_id,
            session_id=session_id,
            error_type=error_type,
            error_message=error_message[:1000] if error_message else "",
            latency_ms=latency_ms,
        )

    def record_tool_registered(self, tool_name: str, tool_version: str = "", **kw: Any) -> int:
        return self.record("tool_registered", tool_name=tool_name, tool_version=tool_version, **kw)

    def record_tool_created(self, tool_name: str, tool_version: str = "", **kw: Any) -> int:
        return self.record("tool_created", tool_name=tool_name, tool_version=tool_version, **kw)

    def record_tool_deprecated(self, tool_name: str, tool_version: str = "", **kw: Any) -> int:
        return self.record("tool_deprecated", tool_name=tool_name, tool_version=tool_version, **kw)

    def record_task_started(self, task_id: str, session_id: str = "", **kw: Any) -> int:
        return self.record("task_started", task_id=task_id, session_id=session_id, **kw)

    def record_task_finished(self, task_id: str, session_id: str = "", **kw: Any) -> int:
        return self.record("task_finished", task_id=task_id, session_id=session_id, **kw)

    def record_remote_escalation(self, task_id: str = "", session_id: str = "", **kw: Any) -> int:
        return self.record("remote_escalation", task_id=task_id, session_id=session_id, **kw)

    def record_learning_event(self, task_id: str = "", session_id: str = "", **kw: Any) -> int:
        return self.record("learning_event", task_id=task_id, session_id=session_id, **kw)

    def record_tool_request(self, request: Any) -> int:
        """Record a ToolRequest to the dedicated tool_requests table.

        Also emits a 'tool_request' event to tool_events for unified querying.
        ``request`` should be a ToolSpec.ToolRequest or any object with a
        ``to_dict()`` method.
        """
        assert self._conn is not None, "Not connected"
        d = request.to_dict() if hasattr(request, "to_dict") else dict(request)
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO tool_requests
                (task_id, session_id, reason, missing_capability,
                 candidate_name, candidate_desc, candidate_input, candidate_output,
                 risk_level, privacy_notes, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                d.get("task_id", ""),
                d.get("session_id", ""),
                d.get("reason", ""),
                d.get("missing_capability", ""),
                d.get("candidate_name", ""),
                d.get("candidate_description", ""),
                json.dumps(d.get("candidate_input_schema", {}), ensure_ascii=False),
                json.dumps(d.get("candidate_output_schema", {}), ensure_ascii=False),
                d.get("risk_level", "L0"),
                d.get("privacy_notes", ""),
                json.dumps(d.get("metadata", {}), ensure_ascii=False),
                d.get("created_at", now),
            ),
        )
        self._conn.commit()

        # Also emit to unified event log
        return self.record(
            "tool_request",
            task_id=d.get("task_id", ""),
            session_id=d.get("session_id", ""),
            result_summary=d.get("missing_capability", "")[:200],
            metadata={"candidate_name": d.get("candidate_name", ""), "risk_level": d.get("risk_level", "L0")},
        )

    # -- queries -------------------------------------------------------------

    def get_tool_stats(self, tool_name: str) -> dict[str, int]:
        """Return invocation/success/failure counts for a tool."""
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT event_type, COUNT(*) as cnt FROM tool_events WHERE tool_name = ? GROUP BY event_type",
            (tool_name,),
        ).fetchall()
        return {row["event_type"]: row["cnt"] for row in rows}

    def get_recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM tool_events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_tool_requests(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent ToolRequests from the dedicated table."""
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM tool_requests ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_tool_request_stats(self) -> dict[str, int]:
        """Return summary stats about tool requests."""
        assert self._conn is not None
        total = self._conn.execute("SELECT COUNT(*) FROM tool_requests").fetchone()[0]
        by_risk = self._conn.execute(
            "SELECT risk_level, COUNT(*) as cnt FROM tool_requests GROUP BY risk_level"
        ).fetchall()
        result: dict[str, int] = {"total": total}
        for row in by_risk:
            result[f"risk_{row[0]}"] = row[1]
        return result
