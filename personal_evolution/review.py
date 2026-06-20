from __future__ import annotations

import hashlib
import time

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
    pass


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
        now = utc_now_iso()
        approved_candidate = _candidate_with_status(candidate, MemoryStatus.APPROVED, now)
        memory = ApprovedMemory(
            memory_id=_stable_id("memory", candidate.candidate_id, candidate.claim),
            memory_type=candidate.memory_type,
            content=candidate.claim,
            evidence_ids=list(candidate.evidence_ids),
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
            _audit_event(
                entity_type="candidate",
                entity_id=candidate.candidate_id,
                action=AuditAction.APPROVED,
                actor=actor,
                before=candidate.to_dict(),
                after=approved_candidate.to_dict(),
                reason=reason,
            )
        )
        return memory

    def edit_and_approve_candidate(
        self,
        candidate_id: str,
        *,
        content: str,
        actor: str,
        reason: str | None = None,
    ) -> ApprovedMemory:
        candidate = self._pending_candidate(candidate_id)
        edited = CandidateMemory(
            candidate_id=candidate.candidate_id,
            memory_type=candidate.memory_type,
            claim=content,
            rationale=candidate.rationale,
            evidence_ids=list(candidate.evidence_ids),
            status=candidate.status,
            confidence=candidate.confidence,
            source_model=candidate.source_model,
            remote_assisted=candidate.remote_assisted,
            created_at=candidate.created_at,
            updated_at=utc_now_iso(),
        )

        self.store.save_candidate(edited)
        self.store.append_audit(
            _audit_event(
                entity_type="candidate",
                entity_id=candidate.candidate_id,
                action=AuditAction.CANDIDATE_EDITED,
                actor=actor,
                before=candidate.to_dict(),
                after=edited.to_dict(),
                reason=reason,
            )
        )
        return self.approve_candidate(candidate_id, actor=actor, reason=reason)

    def reject_candidate(
        self,
        candidate_id: str,
        *,
        actor: str,
        reason: str | None = None,
    ) -> CandidateMemory:
        candidate = self._pending_candidate(candidate_id)
        rejected = _candidate_with_status(candidate, MemoryStatus.REJECTED, utc_now_iso())

        self.store.save_candidate(rejected)
        self.store.append_audit(
            _audit_event(
                entity_type="candidate",
                entity_id=candidate.candidate_id,
                action=AuditAction.REJECTED,
                actor=actor,
                before=candidate.to_dict(),
                after=rejected.to_dict(),
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
            raise ReviewWorkflowError(f"Approved memory {memory_id} not found")
        if memory.status != MemoryStatus.APPROVED:
            raise ReviewWorkflowError(
                f"Approved memory {memory_id} is not approved: {memory.status.value}"
            )

        revoked = ApprovedMemory(
            memory_id=memory.memory_id,
            memory_type=memory.memory_type,
            content=memory.content,
            evidence_ids=list(memory.evidence_ids),
            candidate_id=memory.candidate_id,
            version=memory.version,
            confidence=memory.confidence,
            status=MemoryStatus.REVOKED,
            approved_at=memory.approved_at,
            revoked_at=utc_now_iso(),
        )

        self.store.save_approved_memory(revoked)
        self.store.append_audit(
            _audit_event(
                entity_type="approved_memory",
                entity_id=memory.memory_id,
                action=AuditAction.REVOKED,
                actor=actor,
                before=memory.to_dict(),
                after=revoked.to_dict(),
                reason=reason,
            )
        )
        return revoked

    def _pending_candidate(self, candidate_id: str) -> CandidateMemory:
        candidate = self.store.get_candidate(candidate_id)
        if candidate is None:
            raise ReviewWorkflowError(f"Candidate {candidate_id} not found")
        if candidate.status != MemoryStatus.PENDING:
            raise ReviewWorkflowError(
                f"Candidate {candidate_id} is not pending: {candidate.status.value}"
            )
        return candidate


def _candidate_with_status(
    candidate: CandidateMemory,
    status: MemoryStatus,
    updated_at: str,
) -> CandidateMemory:
    return CandidateMemory(
        candidate_id=candidate.candidate_id,
        memory_type=candidate.memory_type,
        claim=candidate.claim,
        rationale=candidate.rationale,
        evidence_ids=list(candidate.evidence_ids),
        status=status,
        confidence=candidate.confidence,
        source_model=candidate.source_model,
        remote_assisted=candidate.remote_assisted,
        created_at=candidate.created_at,
        updated_at=updated_at,
    )


def _audit_event(
    *,
    entity_type: str,
    entity_id: str,
    action: AuditAction,
    actor: str,
    before: dict[str, object] | None,
    after: dict[str, object] | None,
    reason: str | None,
) -> AuditEvent:
    created_at = utc_now_iso()
    return AuditEvent(
        audit_id=_stable_id(
            "audit",
            entity_type,
            entity_id,
            action.value,
            actor,
            reason or "",
            created_at,
            str(time.time_ns()),
        ),
        entity_type=entity_type,
        entity_id=entity_id,
        action=action,
        actor=actor,
        before=before,
        after=after,
        reason=reason,
        created_at=created_at,
    )


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:16]}"
