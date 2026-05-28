"""Usage data collector for optimization.

Collects real agent interactions including:
- User inputs
- Agent responses
- Tool calls and results
- User feedback (implicit/explicit)
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class Outcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


@dataclass
class Interaction:
    """A single agent interaction."""
    id: int | None = None
    session_id: str = ""
    user_input: str = ""
    agent_response: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    outcome: Outcome = Outcome.UNKNOWN
    feedback_score: float | None = None  # -1.0 to 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "user_input": self.user_input,
            "agent_response": self.agent_response,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "outcome": self.outcome.value,
            "feedback_score": self.feedback_score,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


class UsageCollector:
    """Collects and stores agent usage data for optimization.

    Usage:
        collector = UsageCollector()
        collector.start_session()

        # After each interaction
        collector.record_interaction(
            user_input="搜索 Obsidian 笔记",
            agent_response="找到了 3 条相关笔记...",
            tool_calls=[{"name": "obsidian_search", "args": {"query": "笔记"}}],
            outcome=Outcome.SUCCESS
        )

        # Get training data
        examples = collector.get_training_examples(min_score=0.5)
    """

    def __init__(self, db_path: str = "memory/usage.db"):
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._current_session: str | None = None

    def connect(self) -> None:
        """Connect to the database."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    def disconnect(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _init_tables(self) -> None:
        """Initialize database tables."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS interactions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      TEXT NOT NULL,
                user_input      TEXT NOT NULL,
                agent_response  TEXT NOT NULL,
                tool_calls      TEXT DEFAULT '[]',
                tool_results    TEXT DEFAULT '[]',
                outcome         TEXT DEFAULT 'unknown',
                feedback_score  REAL,
                metadata        TEXT DEFAULT '{}',
                created_at      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id              TEXT PRIMARY KEY,
                started_at      TEXT NOT NULL,
                ended_at        TEXT,
                interaction_count INTEGER DEFAULT 0,
                avg_score       REAL
            );

            CREATE INDEX IF NOT EXISTS idx_interactions_session
                ON interactions(session_id);
            CREATE INDEX IF NOT EXISTS idx_interactions_outcome
                ON interactions(outcome);
        """)
        self._conn.commit()

    def start_session(self) -> str:
        """Start a new collection session."""
        import uuid
        session_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        self._conn.execute(
            "INSERT INTO sessions (id, started_at) VALUES (?, ?)",
            (session_id, now)
        )
        self._conn.commit()
        self._current_session = session_id
        return session_id

    def end_session(self) -> None:
        """End the current session."""
        if not self._current_session:
            return

        now = datetime.now().isoformat()

        # Calculate session stats
        stats = self._conn.execute(
            """SELECT COUNT(*) as count, AVG(feedback_score) as avg_score
               FROM interactions WHERE session_id = ?""",
            (self._current_session,)
        ).fetchone()

        self._conn.execute(
            """UPDATE sessions SET ended_at = ?, interaction_count = ?, avg_score = ?
               WHERE id = ?""",
            (now, stats["count"], stats["avg_score"], self._current_session)
        )
        self._conn.commit()
        self._current_session = None

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
        """Record a single interaction."""
        if not self._conn:
            raise RuntimeError("Not connected")

        now = datetime.now().isoformat()
        cursor = self._conn.execute(
            """INSERT INTO interactions
               (session_id, user_input, agent_response, tool_calls, tool_results,
                outcome, feedback_score, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self._current_session or "unknown",
                user_input,
                agent_response,
                json.dumps(tool_calls or []),
                json.dumps(tool_results or []),
                outcome.value,
                feedback_score,
                json.dumps(metadata or {}),
                now,
            )
        )
        self._conn.commit()
        return cursor.lastrowid

    def update_feedback(self, interaction_id: int, score: float) -> None:
        """Update feedback score for an interaction."""
        if not self._conn:
            raise RuntimeError("Not connected")

        self._conn.execute(
            "UPDATE interactions SET feedback_score = ? WHERE id = ?",
            (score, interaction_id)
        )
        self._conn.commit()

    def get_training_examples(
        self,
        min_score: float | None = None,
        outcome: Outcome | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get training examples for optimization."""
        if not self._conn:
            raise RuntimeError("Not connected")

        sql = "SELECT * FROM interactions WHERE 1=1"
        params: list[Any] = []

        if min_score is not None:
            sql += " AND feedback_score >= ?"
            params.append(min_score)

        if outcome:
            sql += " AND outcome = ?"
            params.append(outcome.value)

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_failed_interactions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get failed interactions for analysis."""
        return self.get_training_examples(outcome=Outcome.FAILURE, limit=limit)

    def get_stats(self) -> dict[str, Any]:
        """Get collection statistics."""
        if not self._conn:
            raise RuntimeError("Not connected")

        total = self._conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        by_outcome = {}
        for outcome in Outcome:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM interactions WHERE outcome = ?",
                (outcome.value,)
            ).fetchone()[0]
            by_outcome[outcome.value] = count

        avg_score = self._conn.execute(
            "SELECT AVG(feedback_score) FROM interactions WHERE feedback_score IS NOT NULL"
        ).fetchone()[0]

        sessions = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

        return {
            "total_interactions": total,
            "by_outcome": by_outcome,
            "avg_feedback_score": avg_score,
            "total_sessions": sessions,
        }

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        """Convert a database row to a dictionary."""
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "user_input": row["user_input"],
            "agent_response": row["agent_response"],
            "tool_calls": json.loads(row["tool_calls"]),
            "tool_results": json.loads(row["tool_results"]),
            "outcome": row["outcome"],
            "feedback_score": row["feedback_score"],
            "metadata": json.loads(row["metadata"]),
            "created_at": row["created_at"],
        }
