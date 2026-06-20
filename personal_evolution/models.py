from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
from typing import Any


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


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _freeze(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return value
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _json_safe(value: Any) -> Any:
    value = _enum_value(value)
    if isinstance(value, MappingProxyType | dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    return value


def _json_safe_dict(obj: Any) -> dict[str, Any]:
    return {field.name: _json_safe(getattr(obj, field.name)) for field in fields(obj)}


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return bool(value)


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Evidence:
        return cls(
            evidence_id=data["evidence_id"],
            source_type=SourceType(data["source_type"]),
            source_ref=data["source_ref"],
            observed_at=data["observed_at"],
            summary=data["summary"],
            sensitivity=data["sensitivity"],
            content_hash=data["content_hash"],
            metadata=data["metadata"],
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", _freeze(self.evidence_ids))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservedEvent:
        return cls(
            event_id=data["event_id"],
            start_at=data["start_at"],
            end_at=data["end_at"],
            title=data["title"],
            summary=data["summary"],
            evidence_ids=data["evidence_ids"],
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", _freeze(self.evidence_ids))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateMemory:
        return cls(
            candidate_id=data["candidate_id"],
            memory_type=MemoryType(data["memory_type"]),
            claim=data["claim"],
            rationale=data["rationale"],
            evidence_ids=data["evidence_ids"],
            status=MemoryStatus(data["status"]),
            confidence=float(data["confidence"]),
            source_model=data["source_model"],
            remote_assisted=_parse_bool(data["remote_assisted"]),
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", _freeze(self.evidence_ids))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApprovedMemory:
        return cls(
            memory_id=data["memory_id"],
            memory_type=MemoryType(data["memory_type"]),
            content=data["content"],
            evidence_ids=data["evidence_ids"],
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
    def from_dict(cls, data: dict[str, Any]) -> AuditEvent:
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
