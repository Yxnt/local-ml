from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
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
    def review_date(date: str) -> dict[str, object]:
        events = [
            event
            for event in store.list_observed_events()
            if event.start_at.startswith(date)
        ]
        evidence_ids = {
            evidence_id for event in events for evidence_id in event.evidence_ids
        }
        evidence = [
            item for item in store.list_evidence() if item.evidence_id in evidence_ids
        ]
        candidates = [
            candidate
            for candidate in store.list_candidates()
            if any(
                evidence_id in evidence_ids for evidence_id in candidate.evidence_ids
            )
        ]
        return {
            "date": date,
            "events": [event.to_dict() for event in events],
            "evidence": [item.to_dict() for item in evidence],
            "candidates": [candidate.to_dict() for candidate in candidates],
        }

    @router.get("/candidates")
    def list_candidates(status: str | None = "pending") -> dict[str, object]:
        parsed_status = _parse_candidate_status(status)
        return {
            "candidates": [
                candidate.to_dict()
                for candidate in store.list_candidates(parsed_status)
            ]
        }

    @router.post("/candidates/{candidate_id}/approve")
    def approve_candidate(
        candidate_id: str,
        request: ReviewActionRequest,
    ) -> dict[str, object]:
        try:
            memory = workflow.approve_candidate(
                candidate_id,
                actor=request.actor,
                reason=request.reason,
            )
        except ReviewWorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"memory": memory.to_dict()}

    @router.post("/candidates/{candidate_id}/edit-approve")
    def edit_and_approve_candidate(
        candidate_id: str,
        request: EditApproveRequest,
    ) -> dict[str, object]:
        try:
            memory = workflow.edit_and_approve_candidate(
                candidate_id,
                content=request.content,
                actor=request.actor,
                reason=request.reason,
            )
        except ReviewWorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"memory": memory.to_dict()}

    @router.post("/candidates/{candidate_id}/reject")
    def reject_candidate(
        candidate_id: str,
        request: ReviewActionRequest,
    ) -> dict[str, object]:
        try:
            candidate = workflow.reject_candidate(
                candidate_id,
                actor=request.actor,
                reason=request.reason,
            )
        except ReviewWorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"candidate": candidate.to_dict()}

    @router.get("/memories")
    def list_memories() -> dict[str, object]:
        return {
            "memories": [
                memory.to_dict() for memory in store.list_approved_memories()
            ]
        }

    @router.post("/memories/{memory_id}/revoke")
    def revoke_memory(
        memory_id: str,
        request: ReviewActionRequest,
    ) -> dict[str, object]:
        try:
            memory = workflow.revoke_memory(
                memory_id,
                actor=request.actor,
                reason=request.reason,
            )
        except ReviewWorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"memory": memory.to_dict()}

    @router.get("/evidence/{evidence_id}")
    def get_evidence(evidence_id: str) -> dict[str, object]:
        evidence = store.get_evidence(evidence_id)
        if evidence is None:
            raise HTTPException(
                status_code=404,
                detail=f"Evidence not found: {evidence_id}",
            )
        return {"evidence": evidence.to_dict()}

    @router.get("/audit")
    def list_audit(entity_id: str | None = None) -> dict[str, object]:
        return {
            "events": [
                event.to_dict() for event in store.list_audit_events(entity_id)
            ]
        }

    return router


def mount_personal_evolution_app(
    app: FastAPI,
    *,
    store: PersonalEvolutionStore,
) -> None:
    app.include_router(
        create_personal_evolution_router(store),
        prefix="/personal-evolution",
    )
    static_dir = Path(__file__).resolve().parents[1] / "web" / "personal-evolution"
    if static_dir.exists():
        app.mount(
            "/personal-evolution/app",
            StaticFiles(directory=str(static_dir), html=True),
            name="personal-evolution-app",
        )


def _parse_candidate_status(status: str | None) -> MemoryStatus | None:
    if status is None or not status.strip():
        return None
    try:
        return MemoryStatus(status)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid candidate status: {status}",
        ) from exc
