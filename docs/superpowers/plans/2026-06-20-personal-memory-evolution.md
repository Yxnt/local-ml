# Personal Memory Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an audit-first personal memory evolution MVP with mock-friendly multi-source ingestion, review workflow APIs, and a responsive review console.

**Architecture:** Add a new `personal_evolution/` package that owns evidence, observed events, candidate memories, approved memories, and append-only audit history. Reuse existing `lifelog/`, `integrations/obsidian/`, and `integrations/photos/` patterns where helpful, but keep the first implementation mock-friendly and focused on the full review loop rather than real device sync depth.

**Tech Stack:** Python 3, dataclasses, sqlite3, FastAPI, pytest, FastAPI TestClient, vanilla HTML/CSS/JS served as static assets.

---

## Scope And First Release Boundary

This plan implements the first working slice of the approved design:

- local data model for evidence, observed events, candidate memories, approved memories, and audit events
- SQLite store with append-only audit behavior
- mock-friendly ingestors for Photos/lifelog, Obsidian, and Health/Fitness daily summaries
- deterministic candidate generation for the MVP
- review workflow service for approve, edit-and-approve, reject, revoke, and audit lookup
- FastAPI routes under `/personal-evolution/*`
- responsive static review console under `/personal-evolution/app`
- end-to-end tests for `evidence -> observed event -> candidate -> approval -> revocation -> audit lookup`

This plan does not implement real Apple Health permissions, full Calendar ingestion, model fine-tuning, or automatic write policies.

## File Structure

Create:

- `personal_evolution/__init__.py`
  Public package exports.
- `personal_evolution/models.py`
  Dataclasses, enums, JSON serialization helpers, and validation for personal evolution records.
- `personal_evolution/store.py`
  SQLite persistence for evidence, observed events, candidates, approved memories, and audit events.
- `personal_evolution/ingestors.py`
  Mock-friendly source ingestors and normalized source summary types.
- `personal_evolution/generator.py`
  Deterministic observed-event and candidate-memory generation from source summaries.
- `personal_evolution/review.py`
  Review workflow service that enforces state transitions and audit logging.
- `server/personal_evolution_api.py`
  FastAPI router factory for review period, candidate queue, approve/edit/reject/revoke, memory ledger, evidence, and audit endpoints.
- `web/personal-evolution/index.html`
  Static review console shell.
- `web/personal-evolution/styles.css`
  Responsive review console styling.
- `web/personal-evolution/app.js`
  Browser-side API calls and UI state handling.
- `tests/personal_evolution/test_models.py`
- `tests/personal_evolution/test_store.py`
- `tests/personal_evolution/test_ingestors.py`
- `tests/personal_evolution/test_generator.py`
- `tests/personal_evolution/test_review.py`
- `tests/personal_evolution/test_api.py`
- `tests/personal_evolution/test_e2e.py`
- `tests/personal_evolution/test_static_app.py`

Modify:

- `server/main.py`
  Include the personal evolution API router and mount the static app.

## Task 1: Models And Serialization

**Files:**
- Create: `personal_evolution/__init__.py`
- Create: `personal_evolution/models.py`
- Test: `tests/personal_evolution/test_models.py`

- [ ] **Step 1: Write failing model tests**

Create `tests/personal_evolution/test_models.py`:

```python
from __future__ import annotations

from datetime import datetime

from personal_evolution.models import (
    ApprovedMemory,
    AuditAction,
    AuditEvent,
    CandidateMemory,
    Evidence,
    MemoryStatus,
    MemoryType,
    ObservedEvent,
    SourceType,
    utc_now_iso,
)


def test_models_round_trip_to_json_safe_dicts() -> None:
    evidence = Evidence(
        evidence_id="ev-photo-1",
        source_type=SourceType.PHOTO,
        source_ref="photos://uuid-1",
        observed_at="2026-06-20T09:00:00",
        summary="Coffee photo cluster near home",
        sensitivity="low",
        content_hash="hash-photo-1",
        metadata={"tags": ["coffee"], "latitude": 37.7},
        created_at="2026-06-20T09:01:00",
    )
    event = ObservedEvent(
        event_id="obs-1",
        start_at="2026-06-20T09:00:00",
        end_at="2026-06-20T09:10:00",
        title="Coffee moment",
        summary="A short coffee moment.",
        evidence_ids=["ev-photo-1"],
        confidence=0.8,
        created_at="2026-06-20T09:02:00",
    )
    candidate = CandidateMemory(
        candidate_id="cand-1",
        memory_type=MemoryType.EVENT,
        claim="Morning coffee is part of the user's routine.",
        rationale="Repeated morning coffee evidence.",
        evidence_ids=["ev-photo-1"],
        status=MemoryStatus.PENDING,
        confidence=0.7,
        source_model="local-rules",
        remote_assisted=False,
        created_at="2026-06-20T09:03:00",
        updated_at="2026-06-20T09:03:00",
    )
    approved = ApprovedMemory(
        memory_id="mem-1",
        memory_type=MemoryType.EVENT,
        content="Morning coffee is part of the user's routine.",
        evidence_ids=["ev-photo-1"],
        candidate_id="cand-1",
        version=1,
        confidence=0.7,
        status=MemoryStatus.APPROVED,
        approved_at="2026-06-20T09:04:00",
        revoked_at=None,
    )
    audit = AuditEvent(
        audit_id="audit-1",
        entity_type="candidate",
        entity_id="cand-1",
        action=AuditAction.APPROVED,
        actor="user",
        before={"status": "pending"},
        after={"status": "approved"},
        reason="Looks right",
        created_at="2026-06-20T09:04:00",
    )

    assert Evidence.from_dict(evidence.to_dict()) == evidence
    assert ObservedEvent.from_dict(event.to_dict()) == event
    assert CandidateMemory.from_dict(candidate.to_dict()) == candidate
    assert ApprovedMemory.from_dict(approved.to_dict()) == approved
    assert AuditEvent.from_dict(audit.to_dict()) == audit


def test_utc_now_iso_returns_parseable_timestamp() -> None:
    value = utc_now_iso()

    parsed = datetime.fromisoformat(value)

    assert parsed.tzinfo is not None
```

- [ ] **Step 2: Run model tests and verify they fail**

Run:

```bash
python -m pytest tests/personal_evolution/test_models.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'personal_evolution'`.

- [ ] **Step 3: Implement models**

Create `personal_evolution/__init__.py`:

```python
"""Audit-first personal memory evolution package."""
```

Create `personal_evolution/models.py`:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, TypeVar


class SourceType(str, Enum):
    PHOTO = "photo"
    OBSIDIAN = "obsidian"
    HEALTH = "health"


class MemoryType(str, Enum):
    EVENT = "event"
    PREFERENCE = "preference"
    PATTERN = "pattern"
    INSIGHT = "insight"
    HEALTH_CORRELATION = "health_correlation"
    STYLE = "style"


class MemoryStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVOKED = "revoked"


class AuditAction(str, Enum):
    EVIDENCE_CREATED = "evidence_created"
    OBSERVED_EVENT_CREATED = "observed_event_created"
    CANDIDATE_CREATED = "candidate_created"
    CANDIDATE_EDITED = "candidate_edited"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVOKED = "revoked"
    AUTO_WRITTEN = "auto_written"
    BATCH_IMPORTED = "batch_imported"


T = TypeVar("T")


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _json_safe_dict(obj: Any) -> dict[str, Any]:
    data = asdict(obj)
    return {key: _enum_value(value) for key, value in data.items()}


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source_type: SourceType
    source_ref: str
    observed_at: str
    summary: str
    sensitivity: str
    content_hash: str
    metadata: dict[str, Any]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Evidence":
        return cls(
            evidence_id=data["evidence_id"],
            source_type=SourceType(data["source_type"]),
            source_ref=data["source_ref"],
            observed_at=data["observed_at"],
            summary=data["summary"],
            sensitivity=data["sensitivity"],
            content_hash=data["content_hash"],
            metadata=dict(data.get("metadata", {})),
            created_at=data["created_at"],
        )


@dataclass(frozen=True)
class ObservedEvent:
    event_id: str
    start_at: str
    end_at: str
    title: str
    summary: str
    evidence_ids: list[str]
    confidence: float
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservedEvent":
        return cls(
            event_id=data["event_id"],
            start_at=data["start_at"],
            end_at=data["end_at"],
            title=data["title"],
            summary=data["summary"],
            evidence_ids=list(data.get("evidence_ids", [])),
            confidence=float(data["confidence"]),
            created_at=data["created_at"],
        )


@dataclass(frozen=True)
class CandidateMemory:
    candidate_id: str
    memory_type: MemoryType
    claim: str
    rationale: str
    evidence_ids: list[str]
    status: MemoryStatus
    confidence: float
    source_model: str
    remote_assisted: bool
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateMemory":
        return cls(
            candidate_id=data["candidate_id"],
            memory_type=MemoryType(data["memory_type"]),
            claim=data["claim"],
            rationale=data["rationale"],
            evidence_ids=list(data.get("evidence_ids", [])),
            status=MemoryStatus(data["status"]),
            confidence=float(data["confidence"]),
            source_model=data["source_model"],
            remote_assisted=bool(data["remote_assisted"]),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )


@dataclass(frozen=True)
class ApprovedMemory:
    memory_id: str
    memory_type: MemoryType
    content: str
    evidence_ids: list[str]
    candidate_id: str
    version: int
    confidence: float
    status: MemoryStatus
    approved_at: str
    revoked_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApprovedMemory":
        return cls(
            memory_id=data["memory_id"],
            memory_type=MemoryType(data["memory_type"]),
            content=data["content"],
            evidence_ids=list(data.get("evidence_ids", [])),
            candidate_id=data["candidate_id"],
            version=int(data["version"]),
            confidence=float(data["confidence"]),
            status=MemoryStatus(data["status"]),
            approved_at=data["approved_at"],
            revoked_at=data.get("revoked_at"),
        )


@dataclass(frozen=True)
class AuditEvent:
    audit_id: str
    entity_type: str
    entity_id: str
    action: AuditAction
    actor: str
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    reason: str | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEvent":
        return cls(
            audit_id=data["audit_id"],
            entity_type=data["entity_type"],
            entity_id=data["entity_id"],
            action=AuditAction(data["action"]),
            actor=data["actor"],
            before=data.get("before"),
            after=data.get("after"),
            reason=data.get("reason"),
            created_at=data["created_at"],
        )
```

- [ ] **Step 4: Run model tests**

Run:

```bash
python -m pytest tests/personal_evolution/test_models.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add personal_evolution/__init__.py personal_evolution/models.py tests/personal_evolution/test_models.py
git commit -m "Add personal evolution models"
```

## Task 2: SQLite Store With Audit History

**Files:**
- Create: `personal_evolution/store.py`
- Test: `tests/personal_evolution/test_store.py`

- [ ] **Step 1: Write failing store tests**

Create `tests/personal_evolution/test_store.py`:

```python
from __future__ import annotations

from personal_evolution.models import (
    AuditAction,
    AuditEvent,
    CandidateMemory,
    Evidence,
    MemoryStatus,
    MemoryType,
    ObservedEvent,
    SourceType,
)
from personal_evolution.store import PersonalEvolutionStore


def _evidence() -> Evidence:
    return Evidence(
        evidence_id="ev-1",
        source_type=SourceType.PHOTO,
        source_ref="photos://uuid-1",
        observed_at="2026-06-20T09:00:00",
        summary="Coffee photo",
        sensitivity="low",
        content_hash="hash-1",
        metadata={"tags": ["coffee"]},
        created_at="2026-06-20T09:01:00",
    )


def _event() -> ObservedEvent:
    return ObservedEvent(
        event_id="obs-1",
        start_at="2026-06-20T09:00:00",
        end_at="2026-06-20T09:10:00",
        title="Coffee",
        summary="Coffee photo event",
        evidence_ids=["ev-1"],
        confidence=0.8,
        created_at="2026-06-20T09:02:00",
    )


def _candidate() -> CandidateMemory:
    return CandidateMemory(
        candidate_id="cand-1",
        memory_type=MemoryType.EVENT,
        claim="Morning coffee appears in the user's routine.",
        rationale="Coffee evidence appeared in a morning event.",
        evidence_ids=["ev-1"],
        status=MemoryStatus.PENDING,
        confidence=0.7,
        source_model="local-rules",
        remote_assisted=False,
        created_at="2026-06-20T09:03:00",
        updated_at="2026-06-20T09:03:00",
    )


def test_store_persists_records_and_audit_events(tmp_path) -> None:
    store = PersonalEvolutionStore(tmp_path / "personal.sqlite3")

    store.save_evidence(_evidence())
    store.save_observed_event(_event())
    store.save_candidate(_candidate())
    store.append_audit(
        AuditEvent(
            audit_id="audit-1",
            entity_type="candidate",
            entity_id="cand-1",
            action=AuditAction.CANDIDATE_CREATED,
            actor="system",
            before=None,
            after={"status": "pending"},
            reason=None,
            created_at="2026-06-20T09:03:00",
        )
    )

    reopened = PersonalEvolutionStore(tmp_path / "personal.sqlite3")

    assert reopened.list_evidence()[0].evidence_id == "ev-1"
    assert reopened.list_observed_events()[0].event_id == "obs-1"
    assert reopened.list_candidates()[0].candidate_id == "cand-1"
    assert reopened.list_audit_events(entity_id="cand-1")[0].action == AuditAction.CANDIDATE_CREATED


def test_store_updates_candidate_status_without_deleting_audit(tmp_path) -> None:
    store = PersonalEvolutionStore(tmp_path / "personal.sqlite3")
    store.save_candidate(_candidate())
    store.append_audit(
        AuditEvent(
            audit_id="audit-1",
            entity_type="candidate",
            entity_id="cand-1",
            action=AuditAction.CANDIDATE_CREATED,
            actor="system",
            before=None,
            after={"status": "pending"},
            reason=None,
            created_at="2026-06-20T09:03:00",
        )
    )

    updated = _candidate().__class__(
        **{
            **_candidate().to_dict(),
            "status": MemoryStatus.REJECTED,
            "updated_at": "2026-06-20T09:04:00",
        }
    )
    store.save_candidate(updated)
    store.append_audit(
        AuditEvent(
            audit_id="audit-2",
            entity_type="candidate",
            entity_id="cand-1",
            action=AuditAction.REJECTED,
            actor="user",
            before={"status": "pending"},
            after={"status": "rejected"},
            reason="Wrong inference",
            created_at="2026-06-20T09:04:00",
        )
    )

    assert store.get_candidate("cand-1").status == MemoryStatus.REJECTED
    assert [event.audit_id for event in store.list_audit_events(entity_id="cand-1")] == [
        "audit-1",
        "audit-2",
    ]
```

- [ ] **Step 2: Run store tests and verify they fail**

Run:

```bash
python -m pytest tests/personal_evolution/test_store.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'personal_evolution.store'`.

- [ ] **Step 3: Implement SQLite store**

Create `personal_evolution/store.py`:

```python
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
    """SQLite persistence for audit-first personal memory evolution."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def save_evidence(self, evidence: Evidence) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO evidence
                    (evidence_id, source_type, source_ref, observed_at, summary,
                     sensitivity, content_hash, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence.evidence_id,
                    evidence.source_type.value,
                    evidence.source_ref,
                    evidence.observed_at,
                    evidence.summary,
                    evidence.sensitivity,
                    evidence.content_hash,
                    json.dumps(evidence.metadata, ensure_ascii=False, sort_keys=True),
                    evidence.created_at,
                ),
            )

    def list_evidence(self) -> list[Evidence]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM evidence ORDER BY observed_at, evidence_id").fetchall()
        return [self._row_to_evidence(row) for row in rows]

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM evidence WHERE evidence_id = ?", (evidence_id,)).fetchone()
        return self._row_to_evidence(row) if row else None

    def save_observed_event(self, event: ObservedEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO observed_events
                    (event_id, start_at, end_at, title, summary, evidence_ids_json,
                     confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.start_at,
                    event.end_at,
                    event.title,
                    event.summary,
                    json.dumps(event.evidence_ids, ensure_ascii=False),
                    event.confidence,
                    event.created_at,
                ),
            )

    def list_observed_events(self) -> list[ObservedEvent]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM observed_events ORDER BY start_at, event_id").fetchall()
        return [self._row_to_observed_event(row) for row in rows]

    def save_candidate(self, candidate: CandidateMemory) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO candidate_memories
                    (candidate_id, memory_type, claim, rationale, evidence_ids_json,
                     status, confidence, source_model, remote_assisted, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.candidate_id,
                    candidate.memory_type.value,
                    candidate.claim,
                    candidate.rationale,
                    json.dumps(candidate.evidence_ids, ensure_ascii=False),
                    candidate.status.value,
                    candidate.confidence,
                    candidate.source_model,
                    int(candidate.remote_assisted),
                    candidate.created_at,
                    candidate.updated_at,
                ),
            )

    def get_candidate(self, candidate_id: str) -> CandidateMemory | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM candidate_memories WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
        return self._row_to_candidate(row) if row else None

    def list_candidates(self, status: MemoryStatus | None = None) -> list[CandidateMemory]:
        with self._connect() as conn:
            if status is None:
                rows = conn.execute(
                    "SELECT * FROM candidate_memories ORDER BY created_at, candidate_id"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM candidate_memories WHERE status = ? ORDER BY created_at, candidate_id",
                    (status.value,),
                ).fetchall()
        return [self._row_to_candidate(row) for row in rows]

    def save_approved_memory(self, memory: ApprovedMemory) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO approved_memories
                    (memory_id, memory_type, content, evidence_ids_json, candidate_id,
                     version, confidence, status, approved_at, revoked_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.memory_id,
                    memory.memory_type.value,
                    memory.content,
                    json.dumps(memory.evidence_ids, ensure_ascii=False),
                    memory.candidate_id,
                    memory.version,
                    memory.confidence,
                    memory.status.value,
                    memory.approved_at,
                    memory.revoked_at,
                ),
            )

    def get_approved_memory(self, memory_id: str) -> ApprovedMemory | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM approved_memories WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        return self._row_to_approved_memory(row) if row else None

    def list_approved_memories(self) -> list[ApprovedMemory]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM approved_memories ORDER BY approved_at, memory_id"
            ).fetchall()
        return [self._row_to_approved_memory(row) for row in rows]

    def append_audit(self, event: AuditEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_events
                    (audit_id, entity_type, entity_id, action, actor, before_json,
                     after_json, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.audit_id,
                    event.entity_type,
                    event.entity_id,
                    event.action.value,
                    event.actor,
                    json.dumps(event.before, ensure_ascii=False, sort_keys=True)
                    if event.before is not None
                    else None,
                    json.dumps(event.after, ensure_ascii=False, sort_keys=True)
                    if event.after is not None
                    else None,
                    event.reason,
                    event.created_at,
                ),
            )

    def list_audit_events(self, entity_id: str | None = None) -> list[AuditEvent]:
        with self._connect() as conn:
            if entity_id is None:
                rows = conn.execute(
                    "SELECT * FROM audit_events ORDER BY created_at, audit_id"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM audit_events WHERE entity_id = ? ORDER BY created_at, audit_id",
                    (entity_id,),
                ).fetchall()
        return [self._row_to_audit_event(row) for row in rows]

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

                CREATE TABLE IF NOT EXISTS candidate_memories (
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

                CREATE INDEX IF NOT EXISTS idx_evidence_observed_at ON evidence(observed_at);
                CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidate_memories(status);
                CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_events(entity_id);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _loads(value: str | None, default: Any) -> Any:
        return json.loads(value) if value else default

    @classmethod
    def _row_to_evidence(cls, row: sqlite3.Row) -> Evidence:
        return Evidence.from_dict(
            {
                "evidence_id": row["evidence_id"],
                "source_type": row["source_type"],
                "source_ref": row["source_ref"],
                "observed_at": row["observed_at"],
                "summary": row["summary"],
                "sensitivity": row["sensitivity"],
                "content_hash": row["content_hash"],
                "metadata": cls._loads(row["metadata_json"], {}),
                "created_at": row["created_at"],
            }
        )

    @classmethod
    def _row_to_observed_event(cls, row: sqlite3.Row) -> ObservedEvent:
        return ObservedEvent.from_dict(
            {
                "event_id": row["event_id"],
                "start_at": row["start_at"],
                "end_at": row["end_at"],
                "title": row["title"],
                "summary": row["summary"],
                "evidence_ids": cls._loads(row["evidence_ids_json"], []),
                "confidence": row["confidence"],
                "created_at": row["created_at"],
            }
        )

    @classmethod
    def _row_to_candidate(cls, row: sqlite3.Row) -> CandidateMemory:
        return CandidateMemory.from_dict(
            {
                "candidate_id": row["candidate_id"],
                "memory_type": row["memory_type"],
                "claim": row["claim"],
                "rationale": row["rationale"],
                "evidence_ids": cls._loads(row["evidence_ids_json"], []),
                "status": row["status"],
                "confidence": row["confidence"],
                "source_model": row["source_model"],
                "remote_assisted": bool(row["remote_assisted"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )

    @classmethod
    def _row_to_approved_memory(cls, row: sqlite3.Row) -> ApprovedMemory:
        return ApprovedMemory.from_dict(
            {
                "memory_id": row["memory_id"],
                "memory_type": row["memory_type"],
                "content": row["content"],
                "evidence_ids": cls._loads(row["evidence_ids_json"], []),
                "candidate_id": row["candidate_id"],
                "version": row["version"],
                "confidence": row["confidence"],
                "status": row["status"],
                "approved_at": row["approved_at"],
                "revoked_at": row["revoked_at"],
            }
        )

    @classmethod
    def _row_to_audit_event(cls, row: sqlite3.Row) -> AuditEvent:
        return AuditEvent.from_dict(
            {
                "audit_id": row["audit_id"],
                "entity_type": row["entity_type"],
                "entity_id": row["entity_id"],
                "action": row["action"],
                "actor": row["actor"],
                "before": cls._loads(row["before_json"], None),
                "after": cls._loads(row["after_json"], None),
                "reason": row["reason"],
                "created_at": row["created_at"],
            }
        )
```

- [ ] **Step 4: Run store tests**

Run:

```bash
python -m pytest tests/personal_evolution/test_store.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add personal_evolution/store.py tests/personal_evolution/test_store.py
git commit -m "Add personal evolution SQLite store"
```

## Task 3: Source Ingestors And Redacted Summaries

**Files:**
- Create: `personal_evolution/ingestors.py`
- Test: `tests/personal_evolution/test_ingestors.py`

- [ ] **Step 1: Write failing ingestor tests**

Create `tests/personal_evolution/test_ingestors.py`:

```python
from __future__ import annotations

from pathlib import Path

from personal_evolution.ingestors import (
    HealthDailySummary,
    HealthSummaryIngestor,
    ObsidianVaultIngestor,
    PhotoSummary,
    PhotoSummaryIngestor,
)
from personal_evolution.models import SourceType


def test_photo_summary_ingestor_redacts_paths_and_keeps_metadata() -> None:
    ingestor = PhotoSummaryIngestor(
        [
            PhotoSummary(
                photo_id="photo-1",
                taken_at="2026-06-20T09:00:00",
                summary="Coffee and laptop on a desk",
                tags=["coffee", "desk"],
                latitude=37.7,
                longitude=-122.4,
                local_path="/Users/example/Pictures/private.jpg",
            )
        ]
    )

    evidence = ingestor.ingest()[0]

    assert evidence.source_type == SourceType.PHOTO
    assert evidence.source_ref == "photos://photo-1"
    assert "private.jpg" not in evidence.summary
    assert evidence.metadata["tags"] == ["coffee", "desk"]
    assert evidence.metadata["has_local_path"] is True
    assert "local_path" not in evidence.metadata


def test_obsidian_ingestor_extracts_title_and_hash_without_full_body(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "Projects.md"
    note.write_text("# Project Memory\n\nThe user prefers audit-first systems.\n", encoding="utf-8")

    evidence = ObsidianVaultIngestor(vault).ingest()[0]

    assert evidence.source_type == SourceType.OBSIDIAN
    assert evidence.source_ref == "obsidian://Projects.md"
    assert evidence.summary == "Project Memory: The user prefers audit-first systems."
    assert evidence.metadata["path"] == "Projects.md"
    assert "audit-first" not in evidence.metadata


def test_health_ingestor_uses_daily_aggregate_summary() -> None:
    ingestor = HealthSummaryIngestor(
        [
            HealthDailySummary(
                date="2026-06-20",
                steps=8600,
                workouts=1,
                sleep_minutes=430,
                notes="Higher activity than usual",
            )
        ]
    )

    evidence = ingestor.ingest()[0]

    assert evidence.source_type == SourceType.HEALTH
    assert evidence.source_ref == "health://2026-06-20"
    assert "8600 steps" in evidence.summary
    assert evidence.metadata["sleep_minutes"] == 430
```

- [ ] **Step 2: Run ingestor tests and verify they fail**

Run:

```bash
python -m pytest tests/personal_evolution/test_ingestors.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'personal_evolution.ingestors'`.

- [ ] **Step 3: Implement ingestors**

Create `personal_evolution/ingestors.py`:

```python
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from personal_evolution.models import Evidence, SourceType, utc_now_iso


@dataclass(frozen=True)
class PhotoSummary:
    photo_id: str
    taken_at: str
    summary: str
    tags: list[str]
    latitude: float | None = None
    longitude: float | None = None
    local_path: str | None = None


@dataclass(frozen=True)
class HealthDailySummary:
    date: str
    steps: int | None
    workouts: int | None
    sleep_minutes: int | None
    notes: str | None = None


class PhotoSummaryIngestor:
    def __init__(self, photos: list[PhotoSummary]) -> None:
        self.photos = photos

    def ingest(self) -> list[Evidence]:
        evidence: list[Evidence] = []
        for photo in self.photos:
            content_hash = _hash_text(
                "|".join([photo.photo_id, photo.taken_at, photo.summary, ",".join(photo.tags)])
            )
            metadata = {
                "tags": photo.tags,
                "latitude": photo.latitude,
                "longitude": photo.longitude,
                "has_local_path": photo.local_path is not None,
            }
            evidence.append(
                Evidence(
                    evidence_id=f"photo-{content_hash[:16]}",
                    source_type=SourceType.PHOTO,
                    source_ref=f"photos://{photo.photo_id}",
                    observed_at=photo.taken_at,
                    summary=photo.summary,
                    sensitivity="low",
                    content_hash=content_hash,
                    metadata=metadata,
                    created_at=utc_now_iso(),
                )
            )
        return evidence


class ObsidianVaultIngestor:
    def __init__(self, vault_path: str | Path) -> None:
        self.vault_path = Path(vault_path).expanduser()

    def ingest(self) -> list[Evidence]:
        evidence: list[Evidence] = []
        for path in sorted(self.vault_path.rglob("*.md")):
            if any(part.startswith(".") for part in path.relative_to(self.vault_path).parts):
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            rel_path = str(path.relative_to(self.vault_path))
            title = _extract_title(path, content)
            snippet = _first_body_sentence(content)
            summary = f"{title}: {snippet}" if snippet else title
            content_hash = _hash_text(content)
            evidence.append(
                Evidence(
                    evidence_id=f"obsidian-{content_hash[:16]}",
                    source_type=SourceType.OBSIDIAN,
                    source_ref=f"obsidian://{rel_path}",
                    observed_at=utc_now_iso(),
                    summary=summary,
                    sensitivity="medium",
                    content_hash=content_hash,
                    metadata={"path": rel_path, "title": title, "size": len(content)},
                    created_at=utc_now_iso(),
                )
            )
        return evidence


class HealthSummaryIngestor:
    def __init__(self, days: list[HealthDailySummary]) -> None:
        self.days = days

    def ingest(self) -> list[Evidence]:
        evidence: list[Evidence] = []
        for day in self.days:
            parts: list[str] = []
            if day.steps is not None:
                parts.append(f"{day.steps} steps")
            if day.workouts is not None:
                parts.append(f"{day.workouts} workout(s)")
            if day.sleep_minutes is not None:
                parts.append(f"{day.sleep_minutes} minutes sleep")
            if day.notes:
                parts.append(day.notes)
            summary = f"Health summary for {day.date}: " + ", ".join(parts)
            content_hash = _hash_text(summary)
            evidence.append(
                Evidence(
                    evidence_id=f"health-{day.date}-{content_hash[:8]}",
                    source_type=SourceType.HEALTH,
                    source_ref=f"health://{day.date}",
                    observed_at=f"{day.date}T00:00:00",
                    summary=summary,
                    sensitivity="medium",
                    content_hash=content_hash,
                    metadata={
                        "date": day.date,
                        "steps": day.steps,
                        "workouts": day.workouts,
                        "sleep_minutes": day.sleep_minutes,
                    },
                    created_at=utc_now_iso(),
                )
            )
        return evidence


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _extract_title(path: Path, content: str) -> str:
    for line in content.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def _first_body_sentence(content: str) -> str:
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line == "---":
            continue
        if len(line) > 160:
            return line[:157].rstrip() + "..."
        return line
    return ""
```

- [ ] **Step 4: Run ingestor tests**

Run:

```bash
python -m pytest tests/personal_evolution/test_ingestors.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add personal_evolution/ingestors.py tests/personal_evolution/test_ingestors.py
git commit -m "Add personal evolution source ingestors"
```

## Task 4: Deterministic Event And Candidate Generation

**Files:**
- Create: `personal_evolution/generator.py`
- Test: `tests/personal_evolution/test_generator.py`

- [ ] **Step 1: Write failing generator tests**

Create `tests/personal_evolution/test_generator.py`:

```python
from __future__ import annotations

from personal_evolution.generator import CandidateMemoryGenerator, ObservedEventBuilder
from personal_evolution.ingestors import HealthDailySummary, HealthSummaryIngestor, PhotoSummary, PhotoSummaryIngestor
from personal_evolution.models import MemoryStatus, MemoryType


def test_observed_event_builder_groups_evidence_by_day() -> None:
    evidence = []
    evidence.extend(
        PhotoSummaryIngestor(
            [
                PhotoSummary(
                    photo_id="photo-1",
                    taken_at="2026-06-20T09:00:00",
                    summary="Coffee and laptop",
                    tags=["coffee"],
                )
            ]
        ).ingest()
    )
    evidence.extend(
        HealthSummaryIngestor(
            [HealthDailySummary(date="2026-06-20", steps=8600, workouts=1, sleep_minutes=430)]
        ).ingest()
    )

    events = ObservedEventBuilder().build(evidence)

    assert len(events) == 1
    assert events[0].title == "Personal signals for 2026-06-20"
    assert len(events[0].evidence_ids) == 2


def test_candidate_generator_creates_pending_review_memory() -> None:
    evidence = PhotoSummaryIngestor(
        [
            PhotoSummary(
                photo_id="photo-1",
                taken_at="2026-06-20T09:00:00",
                summary="Coffee and laptop",
                tags=["coffee", "work"],
            )
        ]
    ).ingest()
    events = ObservedEventBuilder().build(evidence)

    candidates = CandidateMemoryGenerator().generate(events, evidence)

    assert len(candidates) == 1
    assert candidates[0].status == MemoryStatus.PENDING
    assert candidates[0].memory_type == MemoryType.EVENT
    assert candidates[0].remote_assisted is False
    assert "2026-06-20" in candidates[0].claim
```

- [ ] **Step 2: Run generator tests and verify they fail**

Run:

```bash
python -m pytest tests/personal_evolution/test_generator.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'personal_evolution.generator'`.

- [ ] **Step 3: Implement generator**

Create `personal_evolution/generator.py`:

```python
from __future__ import annotations

import hashlib
from collections import defaultdict

from personal_evolution.models import (
    CandidateMemory,
    Evidence,
    MemoryStatus,
    MemoryType,
    ObservedEvent,
    utc_now_iso,
)


class ObservedEventBuilder:
    """Build simple daily observed events from source evidence."""

    def build(self, evidence: list[Evidence]) -> list[ObservedEvent]:
        by_day: dict[str, list[Evidence]] = defaultdict(list)
        for item in evidence:
            day = item.observed_at[:10]
            by_day[day].append(item)

        events: list[ObservedEvent] = []
        for day, items in sorted(by_day.items()):
            ordered = sorted(items, key=lambda item: item.observed_at)
            evidence_ids = [item.evidence_id for item in ordered]
            summaries = "; ".join(item.summary for item in ordered)
            event_id = f"observed-{day}-{_hash_text('|'.join(evidence_ids))[:10]}"
            events.append(
                ObservedEvent(
                    event_id=event_id,
                    start_at=min(item.observed_at for item in ordered),
                    end_at=max(item.observed_at for item in ordered),
                    title=f"Personal signals for {day}",
                    summary=summaries,
                    evidence_ids=evidence_ids,
                    confidence=0.65,
                    created_at=utc_now_iso(),
                )
            )
        return events


class CandidateMemoryGenerator:
    """Generate deterministic pending memories for the MVP review loop."""

    def generate(
        self,
        events: list[ObservedEvent],
        evidence: list[Evidence],
    ) -> list[CandidateMemory]:
        evidence_by_id = {item.evidence_id: item for item in evidence}
        candidates: list[CandidateMemory] = []
        for event in events:
            day = event.start_at[:10]
            source_types = sorted(
                {evidence_by_id[eid].source_type.value for eid in event.evidence_ids if eid in evidence_by_id}
            )
            source_text = ", ".join(source_types) if source_types else "local signals"
            claim = f"On {day}, the user had personal signals from {source_text}."
            rationale = f"Built from {len(event.evidence_ids)} evidence item(s): {event.summary}"
            candidate_id = f"candidate-{event.event_id}-{_hash_text(claim)[:8]}"
            now = utc_now_iso()
            candidates.append(
                CandidateMemory(
                    candidate_id=candidate_id,
                    memory_type=MemoryType.EVENT,
                    claim=claim,
                    rationale=rationale,
                    evidence_ids=event.evidence_ids,
                    status=MemoryStatus.PENDING,
                    confidence=event.confidence,
                    source_model="local-rules",
                    remote_assisted=False,
                    created_at=now,
                    updated_at=now,
                )
            )
        return candidates


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run generator tests**

Run:

```bash
python -m pytest tests/personal_evolution/test_generator.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add personal_evolution/generator.py tests/personal_evolution/test_generator.py
git commit -m "Add personal memory candidate generation"
```

## Task 5: Review Workflow Service

**Files:**
- Create: `personal_evolution/review.py`
- Test: `tests/personal_evolution/test_review.py`

- [ ] **Step 1: Write failing review service tests**

Create `tests/personal_evolution/test_review.py`:

```python
from __future__ import annotations

import pytest

from personal_evolution.ingestors import PhotoSummary, PhotoSummaryIngestor
from personal_evolution.generator import CandidateMemoryGenerator, ObservedEventBuilder
from personal_evolution.models import AuditAction, MemoryStatus
from personal_evolution.review import ReviewWorkflow, ReviewWorkflowError
from personal_evolution.store import PersonalEvolutionStore


def _seed_pending_candidate(store: PersonalEvolutionStore) -> str:
    evidence = PhotoSummaryIngestor(
        [PhotoSummary("photo-1", "2026-06-20T09:00:00", "Coffee and laptop", ["coffee"])]
    ).ingest()
    for item in evidence:
        store.save_evidence(item)
    events = ObservedEventBuilder().build(evidence)
    for event in events:
        store.save_observed_event(event)
    candidate = CandidateMemoryGenerator().generate(events, evidence)[0]
    store.save_candidate(candidate)
    return candidate.candidate_id


def test_approve_candidate_creates_approved_memory_and_audit(tmp_path) -> None:
    store = PersonalEvolutionStore(tmp_path / "personal.sqlite3")
    candidate_id = _seed_pending_candidate(store)
    workflow = ReviewWorkflow(store)

    memory = workflow.approve_candidate(candidate_id, actor="user", reason="Correct")

    assert memory.status == MemoryStatus.APPROVED
    assert store.get_candidate(candidate_id).status == MemoryStatus.APPROVED
    actions = [event.action for event in store.list_audit_events(entity_id=candidate_id)]
    assert AuditAction.APPROVED in actions


def test_edit_and_approve_uses_edited_content(tmp_path) -> None:
    store = PersonalEvolutionStore(tmp_path / "personal.sqlite3")
    candidate_id = _seed_pending_candidate(store)
    workflow = ReviewWorkflow(store)

    memory = workflow.edit_and_approve_candidate(
        candidate_id,
        content="The user often pairs coffee with focused work.",
        actor="user",
        reason="More precise",
    )

    assert memory.content == "The user often pairs coffee with focused work."
    actions = [event.action for event in store.list_audit_events(entity_id=candidate_id)]
    assert AuditAction.CANDIDATE_EDITED in actions
    assert AuditAction.APPROVED in actions


def test_reject_candidate_blocks_later_approval(tmp_path) -> None:
    store = PersonalEvolutionStore(tmp_path / "personal.sqlite3")
    candidate_id = _seed_pending_candidate(store)
    workflow = ReviewWorkflow(store)

    workflow.reject_candidate(candidate_id, actor="user", reason="Not enough evidence")

    assert store.get_candidate(candidate_id).status == MemoryStatus.REJECTED
    with pytest.raises(ReviewWorkflowError, match="not pending"):
        workflow.approve_candidate(candidate_id, actor="user")


def test_revoke_memory_preserves_approved_record_and_adds_audit(tmp_path) -> None:
    store = PersonalEvolutionStore(tmp_path / "personal.sqlite3")
    candidate_id = _seed_pending_candidate(store)
    workflow = ReviewWorkflow(store)
    memory = workflow.approve_candidate(candidate_id, actor="user")

    revoked = workflow.revoke_memory(memory.memory_id, actor="user", reason="Changed mind")

    assert revoked.status == MemoryStatus.REVOKED
    assert store.get_approved_memory(memory.memory_id).status == MemoryStatus.REVOKED
    assert store.get_approved_memory(memory.memory_id).revoked_at is not None
    actions = [event.action for event in store.list_audit_events(entity_id=memory.memory_id)]
    assert AuditAction.REVOKED in actions
```

- [ ] **Step 2: Run review tests and verify they fail**

Run:

```bash
python -m pytest tests/personal_evolution/test_review.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'personal_evolution.review'`.

- [ ] **Step 3: Implement review service**

Create `personal_evolution/review.py`:

```python
from __future__ import annotations

import hashlib

from personal_evolution.models import (
    ApprovedMemory,
    AuditAction,
    AuditEvent,
    CandidateMemory,
    MemoryStatus,
    utc_now_iso,
)
from personal_evolution.store import PersonalEvolutionStore


class ReviewWorkflowError(ValueError):
    """Raised when a review workflow transition is invalid."""


class ReviewWorkflow:
    def __init__(self, store: PersonalEvolutionStore) -> None:
        self.store = store

    def approve_candidate(
        self,
        candidate_id: str,
        *,
        actor: str,
        reason: str | None = None,
    ) -> ApprovedMemory:
        candidate = self._pending_candidate(candidate_id)
        return self._approve(candidate, candidate.claim, actor=actor, reason=reason)

    def edit_and_approve_candidate(
        self,
        candidate_id: str,
        *,
        content: str,
        actor: str,
        reason: str | None = None,
    ) -> ApprovedMemory:
        candidate = self._pending_candidate(candidate_id)
        edited = self._replace_candidate(candidate, claim=content)
        self.store.save_candidate(edited)
        self.store.append_audit(
            self._audit(
                entity_type="candidate",
                entity_id=candidate_id,
                action=AuditAction.CANDIDATE_EDITED,
                actor=actor,
                before={"claim": candidate.claim},
                after={"claim": content},
                reason=reason,
            )
        )
        return self._approve(edited, content, actor=actor, reason=reason)

    def reject_candidate(
        self,
        candidate_id: str,
        *,
        actor: str,
        reason: str | None = None,
    ) -> CandidateMemory:
        candidate = self._pending_candidate(candidate_id)
        rejected = self._replace_candidate(candidate, status=MemoryStatus.REJECTED)
        self.store.save_candidate(rejected)
        self.store.append_audit(
            self._audit(
                entity_type="candidate",
                entity_id=candidate_id,
                action=AuditAction.REJECTED,
                actor=actor,
                before={"status": candidate.status.value},
                after={"status": rejected.status.value},
                reason=reason,
            )
        )
        return rejected

    def revoke_memory(
        self,
        memory_id: str,
        *,
        actor: str,
        reason: str | None = None,
    ) -> ApprovedMemory:
        memory = self.store.get_approved_memory(memory_id)
        if memory is None:
            raise ReviewWorkflowError(f"Approved memory not found: {memory_id}")
        if memory.status == MemoryStatus.REVOKED:
            raise ReviewWorkflowError(f"Approved memory already revoked: {memory_id}")

        revoked = ApprovedMemory(
            memory_id=memory.memory_id,
            memory_type=memory.memory_type,
            content=memory.content,
            evidence_ids=memory.evidence_ids,
            candidate_id=memory.candidate_id,
            version=memory.version,
            confidence=memory.confidence,
            status=MemoryStatus.REVOKED,
            approved_at=memory.approved_at,
            revoked_at=utc_now_iso(),
        )
        self.store.save_approved_memory(revoked)
        self.store.append_audit(
            self._audit(
                entity_type="approved_memory",
                entity_id=memory_id,
                action=AuditAction.REVOKED,
                actor=actor,
                before={"status": memory.status.value, "revoked_at": memory.revoked_at},
                after={"status": revoked.status.value, "revoked_at": revoked.revoked_at},
                reason=reason,
            )
        )
        return revoked

    def _approve(
        self,
        candidate: CandidateMemory,
        content: str,
        *,
        actor: str,
        reason: str | None,
    ) -> ApprovedMemory:
        now = utc_now_iso()
        approved_candidate = self._replace_candidate(candidate, status=MemoryStatus.APPROVED)
        memory = ApprovedMemory(
            memory_id=f"memory-{_hash_text(candidate.candidate_id + content)[:16]}",
            memory_type=candidate.memory_type,
            content=content,
            evidence_ids=candidate.evidence_ids,
            candidate_id=candidate.candidate_id,
            version=1,
            confidence=candidate.confidence,
            status=MemoryStatus.APPROVED,
            approved_at=now,
            revoked_at=None,
        )
        self.store.save_candidate(approved_candidate)
        self.store.save_approved_memory(memory)
        self.store.append_audit(
            self._audit(
                entity_type="candidate",
                entity_id=candidate.candidate_id,
                action=AuditAction.APPROVED,
                actor=actor,
                before={"status": candidate.status.value},
                after={"status": approved_candidate.status.value, "memory_id": memory.memory_id},
                reason=reason,
            )
        )
        return memory

    def _pending_candidate(self, candidate_id: str) -> CandidateMemory:
        candidate = self.store.get_candidate(candidate_id)
        if candidate is None:
            raise ReviewWorkflowError(f"Candidate not found: {candidate_id}")
        if candidate.status != MemoryStatus.PENDING:
            raise ReviewWorkflowError(
                f"Candidate is not pending: {candidate_id} ({candidate.status.value})"
            )
        return candidate

    @staticmethod
    def _replace_candidate(
        candidate: CandidateMemory,
        *,
        claim: str | None = None,
        status: MemoryStatus | None = None,
    ) -> CandidateMemory:
        return CandidateMemory(
            candidate_id=candidate.candidate_id,
            memory_type=candidate.memory_type,
            claim=claim if claim is not None else candidate.claim,
            rationale=candidate.rationale,
            evidence_ids=candidate.evidence_ids,
            status=status if status is not None else candidate.status,
            confidence=candidate.confidence,
            source_model=candidate.source_model,
            remote_assisted=candidate.remote_assisted,
            created_at=candidate.created_at,
            updated_at=utc_now_iso(),
        )

    @staticmethod
    def _audit(
        *,
        entity_type: str,
        entity_id: str,
        action: AuditAction,
        actor: str,
        before: dict | None,
        after: dict | None,
        reason: str | None,
    ) -> AuditEvent:
        now = utc_now_iso()
        audit_id = f"audit-{_hash_text('|'.join([entity_type, entity_id, action.value, now]))[:16]}"
        return AuditEvent(
            audit_id=audit_id,
            entity_type=entity_type,
            entity_id=entity_id,
            action=action,
            actor=actor,
            before=before,
            after=after,
            reason=reason,
            created_at=now,
        )


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run review tests**

Run:

```bash
python -m pytest tests/personal_evolution/test_review.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add personal_evolution/review.py tests/personal_evolution/test_review.py
git commit -m "Add personal memory review workflow"
```

## Task 6: Review API Router

**Files:**
- Create: `server/personal_evolution_api.py`
- Modify: `server/main.py`
- Test: `tests/personal_evolution/test_api.py`

- [ ] **Step 1: Write failing API tests**

Create `tests/personal_evolution/test_api.py`:

```python
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from personal_evolution.ingestors import PhotoSummary, PhotoSummaryIngestor
from personal_evolution.generator import CandidateMemoryGenerator, ObservedEventBuilder
from personal_evolution.store import PersonalEvolutionStore
from server.personal_evolution_api import create_personal_evolution_router


def _seed_store(store: PersonalEvolutionStore) -> str:
    evidence = PhotoSummaryIngestor(
        [PhotoSummary("photo-1", "2026-06-20T09:00:00", "Coffee and laptop", ["coffee"])]
    ).ingest()
    for item in evidence:
        store.save_evidence(item)
    events = ObservedEventBuilder().build(evidence)
    for event in events:
        store.save_observed_event(event)
    candidate = CandidateMemoryGenerator().generate(events, evidence)[0]
    store.save_candidate(candidate)
    return candidate.candidate_id


def _client(tmp_path) -> tuple[TestClient, str]:
    store = PersonalEvolutionStore(tmp_path / "personal.sqlite3")
    candidate_id = _seed_store(store)
    app = FastAPI()
    app.include_router(create_personal_evolution_router(store), prefix="/personal-evolution")
    return TestClient(app), candidate_id


def test_api_lists_review_and_candidates(tmp_path) -> None:
    client, candidate_id = _client(tmp_path)

    review = client.get("/personal-evolution/review/2026-06-20")
    queue = client.get("/personal-evolution/candidates")

    assert review.status_code == 200
    assert review.json()["date"] == "2026-06-20"
    assert review.json()["events"][0]["title"] == "Personal signals for 2026-06-20"
    assert queue.status_code == 200
    assert queue.json()["candidates"][0]["candidate_id"] == candidate_id


def test_api_approve_reject_revoke_and_audit(tmp_path) -> None:
    client, candidate_id = _client(tmp_path)

    approve = client.post(
        f"/personal-evolution/candidates/{candidate_id}/approve",
        json={"actor": "user", "reason": "Correct"},
    )
    memory_id = approve.json()["memory"]["memory_id"]
    revoke = client.post(
        f"/personal-evolution/memories/{memory_id}/revoke",
        json={"actor": "user", "reason": "Changed mind"},
    )
    audit = client.get(f"/personal-evolution/audit?entity_id={memory_id}")

    assert approve.status_code == 200
    assert revoke.status_code == 200
    assert revoke.json()["memory"]["status"] == "revoked"
    assert audit.status_code == 200
    assert audit.json()["events"][0]["action"] == "revoked"
```

- [ ] **Step 2: Run API tests and verify they fail**

Run:

```bash
python -m pytest tests/personal_evolution/test_api.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'server.personal_evolution_api'`.

- [ ] **Step 3: Implement API router**

Create `server/personal_evolution_api.py`:

```python
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from personal_evolution.models import MemoryStatus
from personal_evolution.review import ReviewWorkflow, ReviewWorkflowError
from personal_evolution.store import PersonalEvolutionStore


class ReviewActionRequest(BaseModel):
    actor: str = "user"
    reason: str | None = None


class EditApproveRequest(ReviewActionRequest):
    content: str


def create_personal_evolution_router(store: PersonalEvolutionStore) -> APIRouter:
    router = APIRouter()
    workflow = ReviewWorkflow(store)

    @router.get("/review/{date}")
    def review_period(date: str) -> dict:
        events = [
            event.to_dict()
            for event in store.list_observed_events()
            if event.start_at.startswith(date)
        ]
        evidence_ids = {eid for event in events for eid in event["evidence_ids"]}
        evidence = [
            item.to_dict()
            for item in store.list_evidence()
            if item.evidence_id in evidence_ids
        ]
        candidates = [
            candidate.to_dict()
            for candidate in store.list_candidates()
            if any(eid in evidence_ids for eid in candidate.evidence_ids)
        ]
        return {
            "date": date,
            "events": events,
            "evidence": evidence,
            "candidates": candidates,
        }

    @router.get("/candidates")
    def list_candidates(status: str = "pending") -> dict:
        parsed_status = MemoryStatus(status) if status else None
        return {"candidates": [candidate.to_dict() for candidate in store.list_candidates(parsed_status)]}

    @router.post("/candidates/{candidate_id}/approve")
    def approve_candidate(candidate_id: str, request: ReviewActionRequest) -> dict:
        try:
            memory = workflow.approve_candidate(
                candidate_id,
                actor=request.actor,
                reason=request.reason,
            )
        except ReviewWorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"memory": memory.to_dict()}

    @router.post("/candidates/{candidate_id}/edit-approve")
    def edit_and_approve_candidate(candidate_id: str, request: EditApproveRequest) -> dict:
        try:
            memory = workflow.edit_and_approve_candidate(
                candidate_id,
                content=request.content,
                actor=request.actor,
                reason=request.reason,
            )
        except ReviewWorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"memory": memory.to_dict()}

    @router.post("/candidates/{candidate_id}/reject")
    def reject_candidate(candidate_id: str, request: ReviewActionRequest) -> dict:
        try:
            candidate = workflow.reject_candidate(
                candidate_id,
                actor=request.actor,
                reason=request.reason,
            )
        except ReviewWorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"candidate": candidate.to_dict()}

    @router.get("/memories")
    def list_memories() -> dict:
        return {"memories": [memory.to_dict() for memory in store.list_approved_memories()]}

    @router.post("/memories/{memory_id}/revoke")
    def revoke_memory(memory_id: str, request: ReviewActionRequest) -> dict:
        try:
            memory = workflow.revoke_memory(
                memory_id,
                actor=request.actor,
                reason=request.reason,
            )
        except ReviewWorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"memory": memory.to_dict()}

    @router.get("/evidence/{evidence_id}")
    def get_evidence(evidence_id: str) -> dict:
        evidence = store.get_evidence(evidence_id)
        if evidence is None:
            raise HTTPException(status_code=404, detail=f"Evidence not found: {evidence_id}")
        return {"evidence": evidence.to_dict()}

    @router.get("/audit")
    def list_audit(entity_id: str | None = None) -> dict:
        return {"events": [event.to_dict() for event in store.list_audit_events(entity_id)]}

    return router


def mount_personal_evolution_app(app, *, store: PersonalEvolutionStore) -> None:
    app.include_router(create_personal_evolution_router(store), prefix="/personal-evolution")
    static_dir = Path(__file__).resolve().parents[1] / "web" / "personal-evolution"
    if static_dir.exists():
        app.mount(
            "/personal-evolution/app",
            StaticFiles(directory=str(static_dir), html=True),
            name="personal-evolution-app",
        )
```

Modify `server/main.py`:

```python
from personal_evolution.store import PersonalEvolutionStore
from server.personal_evolution_api import mount_personal_evolution_app
```

Add after `handle_websocket(app, registry)`:

```python
personal_evolution_store = PersonalEvolutionStore(
    os.environ.get("PERSONAL_EVOLUTION_DB", "memory/personal_evolution.sqlite3")
)
mount_personal_evolution_app(app, store=personal_evolution_store)
```

- [ ] **Step 4: Run API tests**

Run:

```bash
python -m pytest tests/personal_evolution/test_api.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add server/personal_evolution_api.py server/main.py tests/personal_evolution/test_api.py
git commit -m "Add personal evolution review API"
```

## Task 7: Static Responsive Review Console

**Files:**
- Create: `web/personal-evolution/index.html`
- Create: `web/personal-evolution/styles.css`
- Create: `web/personal-evolution/app.js`
- Test: `tests/personal_evolution/test_static_app.py`

- [ ] **Step 1: Write failing static app tests**

Create `tests/personal_evolution/test_static_app.py`:

```python
from __future__ import annotations

from pathlib import Path


APP_DIR = Path("web/personal-evolution")


def test_static_app_files_exist_and_wire_expected_api_routes() -> None:
    html = (APP_DIR / "index.html").read_text(encoding="utf-8")
    css = (APP_DIR / "styles.css").read_text(encoding="utf-8")
    js = (APP_DIR / "app.js").read_text(encoding="utf-8")

    assert "Review Today/Week" in html
    assert "Learning Queue" in html
    assert "Memory Ledger" in html
    assert "@media" in css
    assert "/personal-evolution/candidates" in js
    assert "/personal-evolution/memories" in js
    assert "approveCandidate" in js
    assert "revokeMemory" in js
```

- [ ] **Step 2: Run static app test and verify it fails**

Run:

```bash
python -m pytest tests/personal_evolution/test_static_app.py -q
```

Expected: FAIL with `FileNotFoundError` for `web/personal-evolution/index.html`.

- [ ] **Step 3: Create review console HTML**

Create `web/personal-evolution/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Personal Memory Review</title>
    <link rel="stylesheet" href="./styles.css" />
  </head>
  <body>
    <header class="topbar">
      <div>
        <h1>Personal Memory Review</h1>
        <p id="status">Loading local review queue...</p>
      </div>
      <input id="review-date" type="date" />
    </header>

    <main class="layout">
      <section class="panel" aria-labelledby="review-heading">
        <h2 id="review-heading">Review Today/Week</h2>
        <div id="review-list" class="stack"></div>
      </section>

      <section class="panel" aria-labelledby="queue-heading">
        <h2 id="queue-heading">Learning Queue</h2>
        <div id="candidate-list" class="stack"></div>
      </section>

      <section class="panel" aria-labelledby="ledger-heading">
        <h2 id="ledger-heading">Memory Ledger</h2>
        <div id="memory-list" class="stack"></div>
      </section>
    </main>

    <script src="./app.js" type="module"></script>
  </body>
</html>
```

- [ ] **Step 4: Create responsive CSS**

Create `web/personal-evolution/styles.css`:

```css
:root {
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f7f7f4;
  color: #1e2528;
}

body {
  margin: 0;
}

.topbar {
  align-items: center;
  background: #ffffff;
  border-bottom: 1px solid #d8ddd9;
  display: flex;
  gap: 16px;
  justify-content: space-between;
  padding: 18px 24px;
}

h1,
h2,
p {
  margin: 0;
}

h1 {
  font-size: 22px;
}

h2 {
  font-size: 16px;
  margin-bottom: 12px;
}

#status {
  color: #607078;
  font-size: 14px;
  margin-top: 4px;
}

#review-date {
  border: 1px solid #b8c2bd;
  border-radius: 6px;
  font: inherit;
  padding: 8px 10px;
}

.layout {
  display: grid;
  gap: 16px;
  grid-template-columns: 1fr 1fr 1fr;
  padding: 16px;
}

.panel {
  background: #ffffff;
  border: 1px solid #d8ddd9;
  border-radius: 8px;
  min-width: 0;
  padding: 16px;
}

.stack {
  display: grid;
  gap: 10px;
}

.item {
  border: 1px solid #e0e5e1;
  border-radius: 8px;
  padding: 12px;
}

.item h3 {
  font-size: 14px;
  margin: 0 0 6px;
}

.item p {
  color: #344247;
  font-size: 14px;
  line-height: 1.45;
  overflow-wrap: anywhere;
}

.meta {
  color: #6a777d;
  font-size: 12px;
  margin-top: 8px;
}

.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}

button {
  background: #1f6f5b;
  border: 0;
  border-radius: 6px;
  color: #ffffff;
  cursor: pointer;
  font: inherit;
  min-height: 36px;
  padding: 8px 10px;
}

button.secondary {
  background: #edf1ee;
  color: #1e2528;
}

button.danger {
  background: #a33b3b;
}

@media (max-width: 920px) {
  .layout {
    grid-template-columns: 1fr;
  }

  .topbar {
    align-items: stretch;
    flex-direction: column;
  }
}
```

- [ ] **Step 5: Create browser JavaScript**

Create `web/personal-evolution/app.js`:

```javascript
const statusEl = document.querySelector("#status");
const dateInput = document.querySelector("#review-date");
const reviewList = document.querySelector("#review-list");
const candidateList = document.querySelector("#candidate-list");
const memoryList = document.querySelector("#memory-list");

dateInput.value = new Date().toISOString().slice(0, 10);
dateInput.addEventListener("change", refresh);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status}: ${text}`);
  }
  return response.json();
}

function item(title, body, meta = "", actions = []) {
  const node = document.createElement("article");
  node.className = "item";
  node.innerHTML = `
    <h3></h3>
    <p></p>
    <div class="meta"></div>
    <div class="actions"></div>
  `;
  node.querySelector("h3").textContent = title;
  node.querySelector("p").textContent = body;
  node.querySelector(".meta").textContent = meta;
  const actionsEl = node.querySelector(".actions");
  for (const action of actions) actionsEl.append(action);
  return node;
}

function button(label, className, onClick) {
  const node = document.createElement("button");
  node.textContent = label;
  node.className = className;
  node.addEventListener("click", onClick);
  return node;
}

async function approveCandidate(candidateId) {
  await api(`/personal-evolution/candidates/${candidateId}/approve`, {
    method: "POST",
    body: JSON.stringify({ actor: "user", reason: "Approved in review console" }),
  });
  await refresh();
}

async function rejectCandidate(candidateId) {
  await api(`/personal-evolution/candidates/${candidateId}/reject`, {
    method: "POST",
    body: JSON.stringify({ actor: "user", reason: "Rejected in review console" }),
  });
  await refresh();
}

async function revokeMemory(memoryId) {
  await api(`/personal-evolution/memories/${memoryId}/revoke`, {
    method: "POST",
    body: JSON.stringify({ actor: "user", reason: "Revoked in review console" }),
  });
  await refresh();
}

async function refresh() {
  statusEl.textContent = "Loading...";
  reviewList.replaceChildren();
  candidateList.replaceChildren();
  memoryList.replaceChildren();

  const [review, queue, ledger] = await Promise.all([
    api(`/personal-evolution/review/${dateInput.value}`),
    api("/personal-evolution/candidates"),
    api("/personal-evolution/memories"),
  ]);

  for (const event of review.events) {
    reviewList.append(
      item(event.title, event.summary, `${event.start_at} · ${event.evidence_ids.length} evidence item(s)`)
    );
  }

  for (const candidate of queue.candidates) {
    const actions = [
      button("Approve", "", () => approveCandidate(candidate.candidate_id)),
      button("Reject", "secondary", () => rejectCandidate(candidate.candidate_id)),
    ];
    candidateList.append(
      item(
        candidate.claim,
        candidate.rationale,
        `${candidate.memory_type} · ${candidate.remote_assisted ? "remote-assisted" : "local-only"}`,
        actions
      )
    );
  }

  for (const memory of ledger.memories) {
    const actions =
      memory.status === "approved"
        ? [button("Revoke", "danger", () => revokeMemory(memory.memory_id))]
        : [];
    memoryList.append(
      item(memory.content, `Status: ${memory.status}`, `${memory.memory_type} · v${memory.version}`, actions)
    );
  }

  statusEl.textContent = `Loaded ${review.events.length} event(s), ${queue.candidates.length} pending candidate(s), ${ledger.memories.length} memory record(s).`;
}

refresh().catch((error) => {
  statusEl.textContent = `Error: ${error.message}`;
});
```

- [ ] **Step 6: Run static app tests**

Run:

```bash
python -m pytest tests/personal_evolution/test_static_app.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add web/personal-evolution/index.html web/personal-evolution/styles.css web/personal-evolution/app.js tests/personal_evolution/test_static_app.py
git commit -m "Add personal memory review console"
```

## Task 8: End-To-End Personal Evolution Flow

**Files:**
- Test: `tests/personal_evolution/test_e2e.py`
- Modify: files from previous tasks only if the E2E test exposes a real integration bug.

- [ ] **Step 1: Write E2E test**

Create `tests/personal_evolution/test_e2e.py`:

```python
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from personal_evolution.generator import CandidateMemoryGenerator, ObservedEventBuilder
from personal_evolution.ingestors import (
    HealthDailySummary,
    HealthSummaryIngestor,
    PhotoSummary,
    PhotoSummaryIngestor,
)
from personal_evolution.store import PersonalEvolutionStore
from server.personal_evolution_api import create_personal_evolution_router


def test_evidence_to_candidate_to_approval_to_revocation_audit_flow(tmp_path) -> None:
    store = PersonalEvolutionStore(tmp_path / "personal.sqlite3")
    evidence = []
    evidence.extend(
        PhotoSummaryIngestor(
            [PhotoSummary("photo-1", "2026-06-20T09:00:00", "Coffee and laptop", ["coffee", "work"])]
        ).ingest()
    )
    evidence.extend(
        HealthSummaryIngestor(
            [HealthDailySummary("2026-06-20", steps=8600, workouts=1, sleep_minutes=430)]
        ).ingest()
    )
    for item in evidence:
        store.save_evidence(item)
    events = ObservedEventBuilder().build(evidence)
    for event in events:
        store.save_observed_event(event)
    candidates = CandidateMemoryGenerator().generate(events, evidence)
    for candidate in candidates:
        store.save_candidate(candidate)

    app = FastAPI()
    app.include_router(create_personal_evolution_router(store), prefix="/personal-evolution")
    client = TestClient(app)

    review = client.get("/personal-evolution/review/2026-06-20").json()
    candidate_id = client.get("/personal-evolution/candidates").json()["candidates"][0]["candidate_id"]
    approved = client.post(
        f"/personal-evolution/candidates/{candidate_id}/edit-approve",
        json={
            "actor": "user",
            "reason": "Clearer wording",
            "content": "Coffee and focused work appeared together on 2026-06-20.",
        },
    ).json()
    memory_id = approved["memory"]["memory_id"]
    revoked = client.post(
        f"/personal-evolution/memories/{memory_id}/revoke",
        json={"actor": "user", "reason": "Testing revocation"},
    ).json()
    candidate_audit = client.get(f"/personal-evolution/audit?entity_id={candidate_id}").json()
    memory_audit = client.get(f"/personal-evolution/audit?entity_id={memory_id}").json()

    assert len(review["evidence"]) == 2
    assert len(review["events"]) == 1
    assert approved["memory"]["content"] == "Coffee and focused work appeared together on 2026-06-20."
    assert revoked["memory"]["status"] == "revoked"
    assert [event["action"] for event in candidate_audit["events"]] == [
        "candidate_edited",
        "approved",
    ]
    assert [event["action"] for event in memory_audit["events"]] == ["revoked"]
```

- [ ] **Step 2: Run E2E test**

Run:

```bash
python -m pytest tests/personal_evolution/test_e2e.py -q
```

Expected: PASS.

- [ ] **Step 3: Run focused personal evolution suite**

Run:

```bash
python -m pytest tests/personal_evolution -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/personal_evolution/test_e2e.py personal_evolution server web
git commit -m "Verify personal memory evolution flow"
```

## Task 9: Documentation And Final Verification

**Files:**
- Modify: `docs/superpowers/specs/2026-06-20-personal-memory-evolution-design.md` only if implementation reveals a necessary clarification.
- Modify: `docs/tool_evolution.md` only if a future task crosses into tool evolution; this first implementation should not need it.

- [ ] **Step 1: Run all focused Python tests**

Run:

```bash
python -m pytest tests/personal_evolution integrations/obsidian/test_obsidian.py integrations/photos/test_integration.py tests/lifelog/test_memory_store.py -q
```

Expected: PASS. If an existing integration test fails because an optional local dependency is missing, record the exact missing dependency and run the narrower personal evolution suite as the hard gate.

- [ ] **Step 2: Run API import smoke**

Run:

```bash
python - <<'PY'
from server.main import app
routes = sorted(route.path for route in app.routes)
assert "/personal-evolution/review/{date}" in routes
assert "/personal-evolution/candidates" in routes
print("personal evolution routes registered")
PY
```

Expected output:

```text
personal evolution routes registered
```

- [ ] **Step 3: Check git diff**

Run:

```bash
git diff --check
git status --short
```

Expected: `git diff --check` exits 0. `git status --short` shows only intentional implementation files before the final commit, or clean after commit.

- [ ] **Step 4: Commit any documentation clarifications**

Only run this if Step 1 or Step 2 required a spec clarification:

```bash
git add docs/superpowers/specs/2026-06-20-personal-memory-evolution-design.md
git commit -m "Clarify personal memory evolution implementation notes"
```

## Implementation Notes

- Keep raw source data out of API responses. API responses should use evidence summaries and metadata only.
- Do not add remote model calls in this implementation. Use the `remote_assisted` field and local deterministic generation to keep the first flow testable.
- Do not add automatic approval. All candidates begin as `pending`.
- Treat `revoked` approved memories as preserved history, not deleted records.
- Keep the PWA static and dependency-free unless the user explicitly asks for a richer frontend stack.
- Use `PERSONAL_EVOLUTION_DB` to point the service at a different SQLite database during manual testing.

## Self-Review Checklist

- Spec coverage:
  - Evidence, observed events, candidates, approved memories, and audit events are covered in Tasks 1-2.
  - Photos/lifelog, Obsidian, and Health/Fitness mock-friendly inputs are covered in Task 3.
  - Candidate generation and daily review material are covered in Task 4.
  - Approve, edit, reject, revoke, and audit history are covered in Task 5.
  - API routes are covered in Task 6.
  - Responsive review console is covered in Task 7.
  - End-to-end acceptance flow is covered in Task 8.
  - Verification and docs are covered in Task 9.
- Red-flag scan: no unresolved planning markers or vague implementation instructions remain.
- Type consistency:
  - `Evidence`, `ObservedEvent`, `CandidateMemory`, `ApprovedMemory`, and `AuditEvent` field names match across model, store, API, and tests.
  - Status values use `pending`, `approved`, `rejected`, and `revoked`.
  - API routes use `/personal-evolution/*` consistently.
