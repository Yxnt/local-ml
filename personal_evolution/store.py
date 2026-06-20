from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from personal_evolution.models import (
    ApprovedMemory,
    AuditEvent,
    CandidateMemory,
    Evidence,
    MemoryStatus,
    ObservedEvent,
)


class PersonalEvolutionStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def save_evidence(self, evidence: Evidence) -> None:
        data = evidence.to_dict()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO evidence (
                    evidence_id, source_type, source_ref, observed_at, summary,
                    sensitivity, content_hash, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["evidence_id"],
                    data["source_type"],
                    data["source_ref"],
                    data["observed_at"],
                    data["summary"],
                    data["sensitivity"],
                    data["content_hash"],
                    _dumps(data["metadata"]),
                    data["created_at"],
                ),
            )

    def list_evidence(self) -> list[Evidence]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM evidence ORDER BY observed_at, evidence_id"
            ).fetchall()
        return [Evidence.from_dict(_evidence_from_row(row)) for row in rows]

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM evidence WHERE evidence_id = ?",
                (evidence_id,),
            ).fetchone()
        return Evidence.from_dict(_evidence_from_row(row)) if row else None

    def save_observed_event(self, event: ObservedEvent) -> None:
        data = event.to_dict()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO observed_events (
                    event_id, start_at, end_at, title, summary, evidence_ids_json,
                    confidence, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["event_id"],
                    data["start_at"],
                    data["end_at"],
                    data["title"],
                    data["summary"],
                    _dumps(data["evidence_ids"]),
                    data["confidence"],
                    data["created_at"],
                ),
            )

    def list_observed_events(self) -> list[ObservedEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM observed_events ORDER BY start_at, event_id"
            ).fetchall()
        return [ObservedEvent.from_dict(_observed_event_from_row(row)) for row in rows]

    def save_candidate(self, candidate: CandidateMemory) -> None:
        data = candidate.to_dict()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO candidates (
                    candidate_id, memory_type, claim, rationale, evidence_ids_json,
                    status, confidence, source_model, remote_assisted, created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["candidate_id"],
                    data["memory_type"],
                    data["claim"],
                    data["rationale"],
                    _dumps(data["evidence_ids"]),
                    data["status"],
                    data["confidence"],
                    data["source_model"],
                    int(data["remote_assisted"]),
                    data["created_at"],
                    data["updated_at"],
                ),
            )

    def get_candidate(self, candidate_id: str) -> CandidateMemory | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return CandidateMemory.from_dict(_candidate_from_row(row)) if row else None

    def list_candidates(
        self, status: MemoryStatus | None = None
    ) -> list[CandidateMemory]:
        query = "SELECT * FROM candidates"
        params: tuple[str, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status.value,)
        query += " ORDER BY created_at, candidate_id"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [CandidateMemory.from_dict(_candidate_from_row(row)) for row in rows]

    def save_approved_memory(self, memory: ApprovedMemory) -> None:
        data = memory.to_dict()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO approved_memories (
                    memory_id, memory_type, content, evidence_ids_json,
                    candidate_id, version, confidence, status, approved_at,
                    revoked_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["memory_id"],
                    data["memory_type"],
                    data["content"],
                    _dumps(data["evidence_ids"]),
                    data["candidate_id"],
                    data["version"],
                    data["confidence"],
                    data["status"],
                    data["approved_at"],
                    data["revoked_at"],
                ),
            )

    def get_approved_memory(self, memory_id: str) -> ApprovedMemory | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM approved_memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        return ApprovedMemory.from_dict(_approved_memory_from_row(row)) if row else None

    def list_approved_memories(self) -> list[ApprovedMemory]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM approved_memories ORDER BY approved_at, memory_id"
            ).fetchall()
        return [ApprovedMemory.from_dict(_approved_memory_from_row(row)) for row in rows]

    def append_audit(self, event: AuditEvent) -> None:
        data = event.to_dict()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_events (
                    audit_id, entity_type, entity_id, action, actor, before_json,
                    after_json, reason, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["audit_id"],
                    data["entity_type"],
                    data["entity_id"],
                    data["action"],
                    data["actor"],
                    _dumps(data["before"]),
                    _dumps(data["after"]),
                    data["reason"],
                    data["created_at"],
                ),
            )

    def list_audit_events(self, entity_id: str | None = None) -> list[AuditEvent]:
        query = "SELECT rowid, * FROM audit_events"
        params: tuple[str, ...] = ()
        if entity_id is not None:
            query += " WHERE entity_id = ?"
            params = (entity_id,)
        query += " ORDER BY created_at, rowid"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [AuditEvent.from_dict(_audit_event_from_row(row)) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    sensitivity TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_evidence_observed_at
                    ON evidence (observed_at);

                CREATE TABLE IF NOT EXISTS observed_events (
                    event_id TEXT PRIMARY KEY,
                    start_at TEXT NOT NULL,
                    end_at TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_observed_events_start_at
                    ON observed_events (start_at);

                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    claim TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source_model TEXT NOT NULL,
                    remote_assisted INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_candidates_status
                    ON candidates (status);
                CREATE INDEX IF NOT EXISTS idx_candidates_created_at
                    ON candidates (created_at);

                CREATE TABLE IF NOT EXISTS approved_memories (
                    memory_id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    approved_at TEXT NOT NULL,
                    revoked_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_approved_memories_candidate_id
                    ON approved_memories (candidate_id);
                CREATE INDEX IF NOT EXISTS idx_approved_memories_status
                    ON approved_memories (status);

                CREATE TABLE IF NOT EXISTS audit_events (
                    audit_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    before_json TEXT,
                    after_json TEXT,
                    reason TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_audit_events_entity_id
                    ON audit_events (entity_id);
                CREATE INDEX IF NOT EXISTS idx_audit_events_created_at
                    ON audit_events (created_at);
                """
            )


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: str | None) -> Any:
    return json.loads(value) if value is not None else None


def _evidence_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "evidence_id": row["evidence_id"],
        "source_type": row["source_type"],
        "source_ref": row["source_ref"],
        "observed_at": row["observed_at"],
        "summary": row["summary"],
        "sensitivity": row["sensitivity"],
        "content_hash": row["content_hash"],
        "metadata": _loads(row["metadata_json"]),
        "created_at": row["created_at"],
    }


def _observed_event_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "event_id": row["event_id"],
        "start_at": row["start_at"],
        "end_at": row["end_at"],
        "title": row["title"],
        "summary": row["summary"],
        "evidence_ids": _loads(row["evidence_ids_json"]),
        "confidence": row["confidence"],
        "created_at": row["created_at"],
    }


def _candidate_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "candidate_id": row["candidate_id"],
        "memory_type": row["memory_type"],
        "claim": row["claim"],
        "rationale": row["rationale"],
        "evidence_ids": _loads(row["evidence_ids_json"]),
        "status": row["status"],
        "confidence": row["confidence"],
        "source_model": row["source_model"],
        "remote_assisted": bool(row["remote_assisted"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _approved_memory_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "memory_id": row["memory_id"],
        "memory_type": row["memory_type"],
        "content": row["content"],
        "evidence_ids": _loads(row["evidence_ids_json"]),
        "candidate_id": row["candidate_id"],
        "version": row["version"],
        "confidence": row["confidence"],
        "status": row["status"],
        "approved_at": row["approved_at"],
        "revoked_at": row["revoked_at"],
    }


def _audit_event_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "audit_id": row["audit_id"],
        "entity_type": row["entity_type"],
        "entity_id": row["entity_id"],
        "action": row["action"],
        "actor": row["actor"],
        "before": _loads(row["before_json"]),
        "after": _loads(row["after_json"]),
        "reason": row["reason"],
        "created_at": row["created_at"],
    }
