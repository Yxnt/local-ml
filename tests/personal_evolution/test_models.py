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
