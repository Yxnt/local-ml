from __future__ import annotations

from datetime import datetime

import pytest

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


def test_to_dict_emits_plain_string_enum_values() -> None:
    evidence = Evidence(
        evidence_id="ev-photo-1",
        source_type=SourceType.PHOTO,
        source_ref="photos://uuid-1",
        observed_at="2026-06-20T09:00:00",
        summary="Coffee photo cluster near home",
        sensitivity="low",
        content_hash="hash-photo-1",
        metadata={"tags": ["coffee"]},
        created_at="2026-06-20T09:01:00",
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

    assert evidence.to_dict()["source_type"] == "photo"
    assert type(evidence.to_dict()["source_type"]) is str
    assert candidate.to_dict()["memory_type"] == "event"
    assert type(candidate.to_dict()["memory_type"]) is str
    assert candidate.to_dict()["status"] == "pending"
    assert type(candidate.to_dict()["status"]) is str
    assert audit.to_dict()["action"] == "approved"
    assert type(audit.to_dict()["action"]) is str


def test_from_dict_requires_metadata_and_evidence_ids() -> None:
    evidence_data = {
        "evidence_id": "ev-photo-1",
        "source_type": "photo",
        "source_ref": "photos://uuid-1",
        "observed_at": "2026-06-20T09:00:00",
        "summary": "Coffee photo cluster near home",
        "sensitivity": "low",
        "content_hash": "hash-photo-1",
        "created_at": "2026-06-20T09:01:00",
    }
    event_data = {
        "event_id": "obs-1",
        "start_at": "2026-06-20T09:00:00",
        "end_at": "2026-06-20T09:10:00",
        "title": "Coffee moment",
        "summary": "A short coffee moment.",
        "confidence": 0.8,
        "created_at": "2026-06-20T09:02:00",
    }
    candidate_data = {
        "candidate_id": "cand-1",
        "memory_type": "event",
        "claim": "Morning coffee is part of the user's routine.",
        "rationale": "Repeated morning coffee evidence.",
        "status": "pending",
        "confidence": 0.7,
        "source_model": "local-rules",
        "remote_assisted": False,
        "created_at": "2026-06-20T09:03:00",
        "updated_at": "2026-06-20T09:03:00",
    }
    approved_data = {
        "memory_id": "mem-1",
        "memory_type": "event",
        "content": "Morning coffee is part of the user's routine.",
        "candidate_id": "cand-1",
        "version": 1,
        "confidence": 0.7,
        "status": "approved",
        "approved_at": "2026-06-20T09:04:00",
        "revoked_at": None,
    }

    with pytest.raises(KeyError, match="metadata"):
        Evidence.from_dict(evidence_data)
    with pytest.raises(KeyError, match="evidence_ids"):
        ObservedEvent.from_dict(event_data)
    with pytest.raises(KeyError, match="evidence_ids"):
        CandidateMemory.from_dict(candidate_data)
    with pytest.raises(KeyError, match="evidence_ids"):
        ApprovedMemory.from_dict(approved_data)


def test_models_defensively_copy_mutable_inputs() -> None:
    metadata = {"tags": ["coffee"]}
    evidence_ids = ["ev-photo-1"]

    evidence = Evidence(
        evidence_id="ev-photo-1",
        source_type=SourceType.PHOTO,
        source_ref="photos://uuid-1",
        observed_at="2026-06-20T09:00:00",
        summary="Coffee photo cluster near home",
        sensitivity="low",
        content_hash="hash-photo-1",
        metadata=metadata,
        created_at="2026-06-20T09:01:00",
    )
    event = ObservedEvent(
        event_id="obs-1",
        start_at="2026-06-20T09:00:00",
        end_at="2026-06-20T09:10:00",
        title="Coffee moment",
        summary="A short coffee moment.",
        evidence_ids=evidence_ids,
        confidence=0.8,
        created_at="2026-06-20T09:02:00",
    )

    metadata["tags"].append("mutated")
    evidence_ids.append("ev-photo-2")

    assert evidence.to_dict()["metadata"] == {"tags": ["coffee"]}
    assert event.to_dict()["evidence_ids"] == ["ev-photo-1"]
    with pytest.raises(TypeError):
        evidence.metadata["tags"] += ("mutated",)
    with pytest.raises(AttributeError):
        event.evidence_ids.append("ev-photo-2")


def test_candidate_from_dict_parses_remote_assisted_false_string() -> None:
    candidate = CandidateMemory.from_dict(
        {
            "candidate_id": "cand-1",
            "memory_type": "event",
            "claim": "Morning coffee is part of the user's routine.",
            "rationale": "Repeated morning coffee evidence.",
            "evidence_ids": ["ev-photo-1"],
            "status": "pending",
            "confidence": 0.7,
            "source_model": "local-rules",
            "remote_assisted": "false",
            "created_at": "2026-06-20T09:03:00",
            "updated_at": "2026-06-20T09:03:00",
        }
    )

    assert candidate.remote_assisted is False
