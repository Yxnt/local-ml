from __future__ import annotations

from pathlib import Path

import pytest

import personal_evolution.review as review_module
from personal_evolution.generator import CandidateMemoryGenerator, ObservedEventBuilder
from personal_evolution.ingestors import PhotoSummary, PhotoSummaryIngestor
from personal_evolution.models import AuditAction, AuditEvent, CandidateMemory, MemoryStatus
from personal_evolution.review import ReviewWorkflow, ReviewWorkflowError
from personal_evolution.store import PersonalEvolutionStore


def seed_pending_candidate(
    tmp_path: Path,
) -> tuple[PersonalEvolutionStore, CandidateMemory]:
    store = PersonalEvolutionStore(tmp_path / "personal.sqlite3")
    evidence = PhotoSummaryIngestor(
        [
            PhotoSummary(
                photo_id="photo-1",
                taken_at="2026-06-20T09:00:00",
                summary="Coffee and laptop on a desk",
                tags=["coffee", "desk"],
            )
        ]
    ).ingest()
    events = ObservedEventBuilder().build(evidence)
    candidate = CandidateMemoryGenerator().generate(events, evidence)[0]

    for item in evidence:
        store.save_evidence(item)
    for event in events:
        store.save_observed_event(event)
    store.save_candidate(candidate)

    return store, candidate


def test_approve_candidate_creates_approved_memory_and_audit(tmp_path: Path) -> None:
    store, candidate = seed_pending_candidate(tmp_path)

    memory = ReviewWorkflow(store).approve_candidate(
        candidate.candidate_id,
        actor="user",
        reason="Looks right",
    )

    assert memory.status == MemoryStatus.APPROVED
    assert memory.version == 1
    assert memory.candidate_id == candidate.candidate_id
    assert memory.content == candidate.claim
    assert memory.evidence_ids == candidate.evidence_ids
    assert memory.revoked_at is None
    assert store.get_candidate(candidate.candidate_id).status == MemoryStatus.APPROVED
    audit_events = store.list_audit_events(candidate.candidate_id)
    assert [event.action for event in audit_events] == [AuditAction.APPROVED]
    assert audit_events[0].actor == "user"
    assert audit_events[0].reason == "Looks right"
    assert audit_events[0].before["status"] == MemoryStatus.PENDING.value
    assert audit_events[0].after["status"] == MemoryStatus.APPROVED.value
    assert audit_events[0].before["claim"] == candidate.claim
    assert audit_events[0].after["claim"] == candidate.claim


def test_edit_and_approve_uses_edited_content(tmp_path: Path) -> None:
    store, candidate = seed_pending_candidate(tmp_path)
    edited_content = "The user often pairs morning coffee with laptop work."

    memory = ReviewWorkflow(store).edit_and_approve_candidate(
        candidate.candidate_id,
        content=edited_content,
        actor="user",
        reason="Tighter wording",
    )

    assert memory.content == edited_content
    assert store.get_candidate(candidate.candidate_id).claim == edited_content
    audit_events = store.list_audit_events(candidate.candidate_id)
    assert [event.action for event in audit_events] == [
        AuditAction.CANDIDATE_EDITED,
        AuditAction.APPROVED,
    ]
    assert audit_events[0].before["claim"] == candidate.claim
    assert audit_events[0].after["claim"] == edited_content
    assert audit_events[1].before["claim"] == edited_content
    assert audit_events[1].after["status"] == MemoryStatus.APPROVED.value


def test_reject_candidate_blocks_later_approval(tmp_path: Path) -> None:
    store, candidate = seed_pending_candidate(tmp_path)
    workflow = ReviewWorkflow(store)

    rejected = workflow.reject_candidate(
        candidate.candidate_id,
        actor="user",
        reason="Too speculative",
    )

    assert rejected.status == MemoryStatus.REJECTED
    assert store.get_candidate(candidate.candidate_id).status == MemoryStatus.REJECTED
    with pytest.raises(ReviewWorkflowError, match="not pending"):
        workflow.approve_candidate(candidate.candidate_id, actor="user")


def test_revoke_memory_preserves_approved_record_and_adds_audit(tmp_path: Path) -> None:
    store, candidate = seed_pending_candidate(tmp_path)
    workflow = ReviewWorkflow(store)
    memory = workflow.approve_candidate(candidate.candidate_id, actor="user")

    revoked = workflow.revoke_memory(
        memory.memory_id,
        actor="user",
        reason="No longer accurate",
    )

    assert revoked.status == MemoryStatus.REVOKED
    assert revoked.revoked_at is not None
    assert store.get_approved_memory(memory.memory_id) == revoked
    audit_events = store.list_audit_events(memory.memory_id)
    assert [event.action for event in audit_events] == [AuditAction.REVOKED]
    assert audit_events[0].before["status"] == MemoryStatus.APPROVED.value
    assert audit_events[0].before["revoked_at"] is None
    assert audit_events[0].after["status"] == MemoryStatus.REVOKED.value
    assert audit_events[0].after["revoked_at"] == revoked.revoked_at


def test_approve_rolls_back_candidate_and_memory_when_audit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, candidate = seed_pending_candidate(tmp_path)
    force_audit_id_collision(store, monkeypatch)

    with pytest.raises(Exception, match="UNIQUE constraint failed"):
        ReviewWorkflow(store).approve_candidate(candidate.candidate_id, actor="user")

    assert store.get_candidate(candidate.candidate_id) == candidate
    assert store.list_approved_memories() == []
    assert [
        event.audit_id for event in store.list_audit_events()
    ] == ["audit-collision"]


def test_revoke_rolls_back_memory_when_audit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, candidate = seed_pending_candidate(tmp_path)
    workflow = ReviewWorkflow(store)
    memory = workflow.approve_candidate(candidate.candidate_id, actor="user")
    force_audit_id_collision(store, monkeypatch)

    with pytest.raises(Exception, match="UNIQUE constraint failed"):
        workflow.revoke_memory(memory.memory_id, actor="user")

    assert store.get_approved_memory(memory.memory_id) == memory
    assert [
        event.action for event in store.list_audit_events(memory.memory_id)
    ] == []


def test_revoke_memory_blocks_double_revoke(tmp_path: Path) -> None:
    store, candidate = seed_pending_candidate(tmp_path)
    workflow = ReviewWorkflow(store)
    memory = workflow.approve_candidate(candidate.candidate_id, actor="user")
    workflow.revoke_memory(memory.memory_id, actor="user")

    with pytest.raises(ReviewWorkflowError, match="not approved"):
        workflow.revoke_memory(memory.memory_id, actor="user")


def test_missing_candidate_raises_review_workflow_error(tmp_path: Path) -> None:
    store = PersonalEvolutionStore(tmp_path / "personal.sqlite3")

    with pytest.raises(ReviewWorkflowError, match="Candidate missing not found"):
        ReviewWorkflow(store).approve_candidate("missing", actor="user")


def test_missing_memory_raises_review_workflow_error(tmp_path: Path) -> None:
    store = PersonalEvolutionStore(tmp_path / "personal.sqlite3")

    with pytest.raises(ReviewWorkflowError, match="Approved memory missing not found"):
        ReviewWorkflow(store).revoke_memory("missing", actor="user")


def force_audit_id_collision(
    store: PersonalEvolutionStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store.append_audit(
        AuditEvent(
            audit_id="audit-collision",
            entity_type="test",
            entity_id="collision-seed",
            action=AuditAction.CANDIDATE_CREATED,
            actor="test",
            before=None,
            after=None,
            reason=None,
            created_at="2026-06-20T00:00:00+00:00",
        )
    )
    original_stable_id = review_module._stable_id

    def stable_id_with_audit_collision(prefix: str, *parts: str) -> str:
        if prefix == "audit":
            return "audit-collision"
        return original_stable_id(prefix, *parts)

    monkeypatch.setattr(review_module, "_stable_id", stable_id_with_audit_collision)
