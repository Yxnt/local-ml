from __future__ import annotations

from pathlib import Path

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
)
from personal_evolution.store import PersonalEvolutionStore


def make_evidence(evidence_id: str = "ev-1") -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        source_type=SourceType.PHOTO,
        source_ref="photos://uuid-1",
        observed_at="2026-06-20T09:00:00+00:00",
        summary="Coffee photo cluster near home",
        sensitivity="low",
        content_hash="hash-photo-1",
        metadata={"tags": ["coffee"], "location": {"city": "SF"}},
        created_at="2026-06-20T09:01:00+00:00",
    )


def make_observed_event(event_id: str = "obs-1") -> ObservedEvent:
    return ObservedEvent(
        event_id=event_id,
        start_at="2026-06-20T09:00:00+00:00",
        end_at="2026-06-20T09:10:00+00:00",
        title="Coffee moment",
        summary="A short coffee moment.",
        evidence_ids=["ev-1"],
        confidence=0.8,
        created_at="2026-06-20T09:02:00+00:00",
    )


def make_candidate(
    candidate_id: str = "cand-1",
    status: MemoryStatus = MemoryStatus.PENDING,
    updated_at: str = "2026-06-20T09:03:00+00:00",
) -> CandidateMemory:
    return CandidateMemory(
        candidate_id=candidate_id,
        memory_type=MemoryType.EVENT,
        claim="Morning coffee is part of the user's routine.",
        rationale="Repeated morning coffee evidence.",
        evidence_ids=["ev-1"],
        status=status,
        confidence=0.7,
        source_model="local-rules",
        remote_assisted=False,
        created_at="2026-06-20T09:03:00+00:00",
        updated_at=updated_at,
    )


def make_approved_memory(memory_id: str = "mem-1") -> ApprovedMemory:
    return ApprovedMemory(
        memory_id=memory_id,
        memory_type=MemoryType.EVENT,
        content="Morning coffee is part of the user's routine.",
        evidence_ids=["ev-1"],
        candidate_id="cand-1",
        version=1,
        confidence=0.7,
        status=MemoryStatus.APPROVED,
        approved_at="2026-06-20T09:04:00+00:00",
        revoked_at=None,
    )


def make_audit_event(
    audit_id: str = "audit-1",
    action: AuditAction = AuditAction.CANDIDATE_CREATED,
    created_at: str = "2026-06-20T09:04:00+00:00",
) -> AuditEvent:
    return AuditEvent(
        audit_id=audit_id,
        entity_type="candidate",
        entity_id="cand-1",
        action=action,
        actor="user",
        before=None,
        after={"status": action.value},
        reason="Looks right",
        created_at=created_at,
    )


def test_store_persists_records_and_audit_events(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "personal.sqlite3"
    store = PersonalEvolutionStore(db_path)

    evidence = make_evidence()
    observed_event = make_observed_event()
    candidate = make_candidate()
    audit = make_audit_event()

    store.save_evidence(evidence)
    store.save_observed_event(observed_event)
    store.save_candidate(candidate)
    store.append_audit(audit)

    reopened = PersonalEvolutionStore(db_path)

    assert db_path.exists()
    assert reopened.list_evidence() == [evidence]
    assert reopened.get_evidence("ev-1") == evidence
    assert reopened.list_observed_events() == [observed_event]
    assert reopened.list_candidates() == [candidate]
    assert reopened.get_candidate("cand-1") == candidate
    assert reopened.list_audit_events() == [audit]


def test_store_updates_candidate_status_without_deleting_audit(tmp_path: Path) -> None:
    store = PersonalEvolutionStore(tmp_path / "personal.sqlite3")
    pending_candidate = make_candidate()
    rejected_candidate = make_candidate(
        status=MemoryStatus.REJECTED,
        updated_at="2026-06-20T09:05:00+00:00",
    )
    created_audit = make_audit_event(
        audit_id="audit-created",
        action=AuditAction.CANDIDATE_CREATED,
        created_at="2026-06-20T09:04:00+00:00",
    )
    rejected_audit = make_audit_event(
        audit_id="audit-rejected",
        action=AuditAction.REJECTED,
        created_at="2026-06-20T09:05:00+00:00",
    )

    store.save_candidate(pending_candidate)
    store.append_audit(created_audit)
    store.save_candidate(rejected_candidate)
    store.append_audit(rejected_audit)

    assert store.get_candidate("cand-1") == rejected_candidate
    assert store.list_candidates(status=MemoryStatus.PENDING) == []
    assert store.list_candidates(status=MemoryStatus.REJECTED) == [rejected_candidate]
    assert store.list_audit_events(entity_id="cand-1") == [
        created_audit,
        rejected_audit,
    ]


def test_store_saves_and_lists_approved_memories(tmp_path: Path) -> None:
    store = PersonalEvolutionStore(tmp_path / "personal.sqlite3")
    memory = make_approved_memory()

    store.save_approved_memory(memory)

    assert store.get_approved_memory("mem-1") == memory
    assert store.list_approved_memories() == [memory]
    assert store.get_approved_memory("missing") is None


def test_get_methods_return_none_for_missing_records(tmp_path: Path) -> None:
    store = PersonalEvolutionStore(tmp_path / "personal.sqlite3")

    assert store.get_evidence("missing") is None
    assert store.get_candidate("missing") is None
    assert store.get_approved_memory("missing") is None
