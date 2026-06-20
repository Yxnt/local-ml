"""SQLite-backed persistence for lifelog memory."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from memory.memory_schema import EventMemory, FeedbackMemory, PreferenceMemory


class MemoryStore:
    """Persist lifelog events, journals, preferences, and feedback locally."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path).expanduser() if db_path is not None else self.default_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @staticmethod
    def default_path() -> Path:
        return Path("~/local-ml-journal/memory.sqlite3").expanduser()

    def save_event(self, event: EventMemory) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO event_memories
                    (event_id, timestamp, summary, people, location, embeddings)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.timestamp.isoformat(),
                    event.summary,
                    json.dumps(event.people),
                    event.location,
                    json.dumps(event.embeddings) if event.embeddings is not None else None,
                ),
            )

    def list_events(self, *, limit: int | None = None) -> list[EventMemory]:
        sql = "SELECT * FROM event_memories ORDER BY timestamp DESC"
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_event(row) for row in rows]

    def upsert_preference(self, preference: PreferenceMemory) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO preference_memories
                    (category, preference, confidence, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(category, preference) DO UPDATE SET
                    confidence = excluded.confidence,
                    updated_at = excluded.updated_at
                """,
                (
                    preference.category,
                    preference.preference,
                    preference.confidence,
                    preference.updated_at.isoformat(),
                ),
            )

    def list_preferences(self) -> list[PreferenceMemory]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM preference_memories ORDER BY confidence DESC, updated_at DESC"
            ).fetchall()
        return [
            PreferenceMemory(
                category=row["category"],
                preference=row["preference"],
                confidence=float(row["confidence"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            for row in rows
        ]

    def save_journal(
        self,
        *,
        journal_date: str,
        markdown: str,
        event_ids: list[str],
        prompt: str,
    ) -> str:
        journal_id = f"journal-{journal_date}-{uuid.uuid4().hex[:8]}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO journal_history
                    (journal_id, journal_date, markdown, event_ids, prompt, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    journal_id,
                    journal_date,
                    markdown,
                    json.dumps(event_ids),
                    prompt,
                    datetime.now().isoformat(),
                ),
            )
        return journal_id

    def list_journals(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM journal_history ORDER BY journal_date DESC, created_at DESC"
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "journal_id": row["journal_id"],
                "journal_date": row["journal_date"],
                "markdown": row["markdown"],
                "event_ids": json.loads(row["event_ids"]),
                "prompt": row["prompt"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def save_feedback(
        self,
        *,
        journal_id: str,
        event_id: str | None = None,
        rating: str | None = None,
        note: str | None = None,
        edited_text: str | None = None,
    ) -> str:
        feedback_id = f"feedback-{uuid.uuid4().hex[:12]}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback_logs
                    (feedback_id, journal_id, event_id, rating, note, edited_text, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feedback_id,
                    journal_id,
                    event_id,
                    rating,
                    note,
                    edited_text,
                    datetime.now().isoformat(),
                ),
            )
        return feedback_id

    def list_feedback(self, *, limit: int | None = None) -> list[FeedbackMemory]:
        sql = "SELECT * FROM feedback_logs ORDER BY timestamp DESC"
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            FeedbackMemory(
                feedback_id=row["feedback_id"],
                journal_id=row["journal_id"],
                event_id=row["event_id"],
                rating=row["rating"],
                note=row["note"],
                edited_text=row["edited_text"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
            )
            for row in rows
        ]

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS event_memories (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    people TEXT NOT NULL,
                    location TEXT,
                    embeddings TEXT
                );

                CREATE TABLE IF NOT EXISTS preference_memories (
                    category TEXT NOT NULL,
                    preference TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (category, preference)
                );

                CREATE TABLE IF NOT EXISTS journal_history (
                    journal_id TEXT PRIMARY KEY,
                    journal_date TEXT NOT NULL,
                    markdown TEXT NOT NULL,
                    event_ids TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS feedback_logs (
                    feedback_id TEXT PRIMARY KEY,
                    journal_id TEXT NOT NULL,
                    event_id TEXT,
                    rating TEXT,
                    note TEXT,
                    edited_text TEXT,
                    timestamp TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_event_memories_timestamp
                    ON event_memories(timestamp);
                CREATE INDEX IF NOT EXISTS idx_journal_history_date
                    ON journal_history(journal_date);
                CREATE INDEX IF NOT EXISTS idx_feedback_logs_journal
                    ON feedback_logs(journal_id);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> EventMemory:
        embeddings_raw = row["embeddings"]
        return EventMemory(
            event_id=row["event_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            summary=row["summary"],
            people=json.loads(row["people"]),
            location=row["location"],
            embeddings=json.loads(embeddings_raw) if embeddings_raw else None,
        )
