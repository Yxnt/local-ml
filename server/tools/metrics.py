"""Tool evolution metrics — queries TelemetryService and ToolRegistry to compute
EGL (Evolution Growth Level) and related tool lifecycle statistics."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from server.tools.registry import ToolRegistry
from server.tools.telemetry import TelemetryService

logger = logging.getLogger(__name__)


class ToolMetrics:
    """Compute tool evolution metrics by combining registry and telemetry data.

    All methods return safe defaults (0 or None) when data is missing.  Time
    windows filter on ``created_at`` stored as ISO-8601 UTC strings.
    """

    def __init__(self, registry: ToolRegistry, telemetry: TelemetryService) -> None:
        self._registry = registry
        self._telemetry = telemetry

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _telemetry_conn(self):
        """Return the raw sqlite3 connection from the telemetry service."""
        return self._telemetry._conn

    def _registry_conn(self):
        """Return the raw sqlite3 connection from the registry."""
        return self._registry._conn

    @staticmethod
    def _window_filter_sql(column: str, window: str) -> tuple[str, list[Any]]:
        """Return a (SQL fragment, params) tuple for time-window filtering.

        ``column`` is the datetime column name (e.g. ``created_at``).
        ``window`` is one of ``"all_time"``, ``"24h"``, ``"7d"``.

        Returns an empty string and empty list for ``"all_time"``.
        """
        if window == "24h":
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
            return f"AND {column} >= ?", [cutoff]
        if window == "7d":
            cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            return f"AND {column} >= ?", [cutoff]
        return "", []

    def _count_events(self, event_type: str, window: str = "all_time") -> int:
        """Count rows in tool_events matching *event_type* within *window*."""
        conn = self._telemetry_conn()
        if conn is None:
            return 0
        filter_sql, params = self._window_filter_sql("created_at", window)
        sql = (
            f"SELECT COUNT(*) FROM tool_events "
            f"WHERE event_type = ? {filter_sql}"
        )
        try:
            row = conn.execute(sql, [event_type, *params]).fetchone()
            return row[0] if row else 0
        except Exception:
            logger.debug("Failed to count %s events", event_type, exc_info=True)
            return 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tool_invocation_count(self, window: str = "all_time") -> int:
        """Count ``tool_invoked`` events from ``tool_events``."""
        return self._count_events("tool_invoked", window)

    def get_created_tool_count(self, window: str = "all_time") -> int:
        """Count ``tool_created`` events from ``tool_events``."""
        return self._count_events("tool_created", window)

    def get_registered_tool_count(self) -> int:
        """Count tools in the registry with status ACTIVE or CANDIDATE."""
        conn = self._registry_conn()
        if conn is None:
            return 0
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM tools WHERE status IN ('active', 'candidate')"
            ).fetchone()
            return row[0] if row else 0
        except Exception:
            logger.debug("Failed to count registered tools", exc_info=True)
            return 0

    def get_candidate_count(self) -> int:
        """Count tools with status CANDIDATE in the registry."""
        conn = self._registry_conn()
        if conn is None:
            return 0
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM tools WHERE status = 'candidate'"
            ).fetchone()
            return row[0] if row else 0
        except Exception:
            logger.debug("Failed to count candidate tools", exc_info=True)
            return 0

    def get_active_generated_tool_count(self) -> int:
        """Count ACTIVE tools whose runtime is ``python_generated``."""
        conn = self._registry_conn()
        if conn is None:
            return 0
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM tools WHERE status = 'active' AND runtime = 'python_generated'"
            ).fetchone()
            return row[0] if row else 0
        except Exception:
            logger.debug("Failed to count active generated tools", exc_info=True)
            return 0

    def get_tool_request_count(self, window: str = "all_time") -> int:
        """Count entries in the ``tool_requests`` table."""
        conn = self._telemetry_conn()
        if conn is None:
            return 0
        filter_sql, params = self._window_filter_sql("created_at", window)
        sql = f"SELECT COUNT(*) FROM tool_requests WHERE 1=1 {filter_sql}"
        try:
            row = conn.execute(sql, params).fetchone()
            return row[0] if row else 0
        except Exception:
            logger.debug("Failed to count tool requests", exc_info=True)
            return 0

    def get_tool_request_consumed_count(self, window: str = "all_time") -> int:
        """Count tool_requests that led to a ``tool_created`` event.

        Matching is done by joining ``tool_requests.candidate_name`` to
        ``tool_events.tool_name`` where ``event_type = 'tool_created'``.
        """
        conn = self._telemetry_conn()
        if conn is None:
            return 0
        filter_sql, params = self._window_filter_sql("tr.created_at", window)
        sql = (
            "SELECT COUNT(DISTINCT tr.id) FROM tool_requests tr "
            "INNER JOIN tool_events te "
            "  ON te.event_type = 'tool_created' AND te.tool_name = tr.candidate_name "
            f"WHERE tr.candidate_name != '' {filter_sql}"
        )
        try:
            row = conn.execute(sql, params).fetchone()
            return row[0] if row else 0
        except Exception:
            logger.debug("Failed to count consumed tool requests", exc_info=True)
            return 0

    def get_tool_success_rate(
        self, tool_name: str | None = None, window: str = "all_time"
    ) -> float | None:
        """Return ``succeeded / (succeeded + failed)`` or ``None`` if no data."""
        conn = self._telemetry_conn()
        if conn is None:
            return None
        filter_sql, params = self._window_filter_sql("created_at", window)
        name_filter = "AND tool_name = ?" if tool_name else ""
        name_params: list[Any] = [tool_name] if tool_name else []

        sql = (
            "SELECT event_type, COUNT(*) FROM tool_events "
            f"WHERE event_type IN ('tool_succeeded', 'tool_failed') "
            f"{name_filter} {filter_sql} "
            "GROUP BY event_type"
        )
        try:
            rows = conn.execute(sql, [*name_params, *params]).fetchall()
        except Exception:
            logger.debug("Failed to compute success rate", exc_info=True)
            return None

        counts: dict[str, int] = {row[0]: row[1] for row in rows}
        succeeded = counts.get("tool_succeeded", 0)
        failed = counts.get("tool_failed", 0)
        total = succeeded + failed
        if total == 0:
            return None
        return succeeded / total

    def get_remote_escalation_rate(self, window: str = "all_time") -> float | None:
        """Return ``remote_escalation / task_started`` or ``None`` if no data."""
        conn = self._telemetry_conn()
        if conn is None:
            return None
        filter_sql, params = self._window_filter_sql("created_at", window)
        sql = (
            "SELECT event_type, COUNT(*) FROM tool_events "
            f"WHERE event_type IN ('remote_escalation', 'task_started') {filter_sql} "
            "GROUP BY event_type"
        )
        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            logger.debug("Failed to compute remote escalation rate", exc_info=True)
            return None

        counts: dict[str, int] = {row[0]: row[1] for row in rows}
        escalations = counts.get("remote_escalation", 0)
        tasks = counts.get("task_started", 0)
        if tasks == 0:
            return None
        return escalations / tasks

    def get_egl(self, window: str = "all_time") -> float | None:
        """Compute EGL (Evolution Growth Level).

        ``EGL = created_generated_tools / tool_invocations``

        The numerator counts ``tool_created`` events whose metadata indicates
        the tool was auto-generated: ``runtime = 'python_generated'`` **or**
        ``provider IN ('generated', 'merged', 'evolved')``.  The denominator
        is the total number of ``tool_invoked`` events.

        Falls back to checking the ``tools`` table in the registry when
        ``tool_events.metadata`` is empty for a given event.

        Returns ``None`` when the denominator is zero.
        """
        conn = self._telemetry_conn()
        if conn is None:
            return None
        filter_sql, params = self._window_filter_sql("created_at", window)

        # Denominator: total tool invocations
        invocations = self.get_tool_invocation_count(window)

        # Numerator: tool_created events for generated tools.
        # tool_events.metadata is a JSON string; we parse it in Python to
        # check runtime/provider since JSON1 extension may not be available.
        sql_created = (
            "SELECT tool_name, metadata FROM tool_events "
            f"WHERE event_type = 'tool_created' {filter_sql}"
        )
        try:
            rows = conn.execute(sql_created, params).fetchall()
        except Exception:
            logger.debug("Failed to query tool_created for EGL", exc_info=True)
            return None

        # Build a lookup of tool_name -> (runtime, provider) from the
        # registry's tools table for fallback when metadata is empty.
        registry_lookup: dict[str, dict[str, str]] = {}
        reg_conn = self._registry_conn()
        if reg_conn is not None:
            try:
                reg_rows = reg_conn.execute(
                    "SELECT name, runtime, provider FROM tools"
                ).fetchall()
                for rr in reg_rows:
                    registry_lookup[rr[0]] = {"runtime": rr[1] or "", "provider": rr[2] or ""}
            except Exception:
                logger.debug("Failed to query tools table for EGL fallback", exc_info=True)

        created_generated = 0
        for row in rows:
            tool_name = row[0]
            try:
                meta = json.loads(row[1]) if row[1] else {}
            except (json.JSONDecodeError, TypeError):
                meta = {}

            runtime = meta.get("runtime", "")
            provider = meta.get("provider", "")

            # Fallback: check registry tools table when metadata is empty
            if not runtime and not provider and tool_name in registry_lookup:
                reg_info = registry_lookup[tool_name]
                runtime = reg_info.get("runtime", "")
                provider = reg_info.get("provider", "")

            if runtime == "python_generated" or provider in ("generated", "merged", "evolved"):
                created_generated += 1

        if invocations == 0:
            return None
        return created_generated / invocations

    def get_all_metrics(self, window: str = "all_time") -> dict[str, Any]:
        """Return all metrics in a single dictionary."""
        return {
            "tool_invocation_count": self.get_tool_invocation_count(window),
            "created_tool_count": self.get_created_tool_count(window),
            "registered_tool_count": self.get_registered_tool_count(),
            "candidate_count": self.get_candidate_count(),
            "active_generated_tool_count": self.get_active_generated_tool_count(),
            "tool_request_count": self.get_tool_request_count(window),
            "tool_request_consumed_count": self.get_tool_request_consumed_count(window),
            "tool_success_rate": self.get_tool_success_rate(window=window),
            "remote_escalation_rate": self.get_remote_escalation_rate(window),
            "egl": self.get_egl(window),
            "window": window,
        }
