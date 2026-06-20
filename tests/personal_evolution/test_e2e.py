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
            [
                PhotoSummary(
                    "photo-1",
                    "2026-06-20T09:00:00",
                    "Coffee and laptop",
                    ["coffee", "work"],
                )
            ]
        ).ingest()
    )
    evidence.extend(
        HealthSummaryIngestor(
            [
                HealthDailySummary(
                    "2026-06-20",
                    steps=8600,
                    workouts=1,
                    sleep_minutes=430,
                )
            ]
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
    app.include_router(
        create_personal_evolution_router(store),
        prefix="/personal-evolution",
    )
    client = TestClient(app)

    review_response = client.get("/personal-evolution/review/2026-06-20")
    candidates_response = client.get("/personal-evolution/candidates")
    assert review_response.status_code == 200
    assert candidates_response.status_code == 200

    review = review_response.json()
    candidate_id = candidates_response.json()["candidates"][0]["candidate_id"]
    approved_response = client.post(
        f"/personal-evolution/candidates/{candidate_id}/edit-approve",
        json={
            "actor": "user",
            "reason": "Clearer wording",
            "content": "Coffee and focused work appeared together on 2026-06-20.",
        },
    )
    assert approved_response.status_code == 200

    approved = approved_response.json()
    memory_id = approved["memory"]["memory_id"]
    revoked_response = client.post(
        f"/personal-evolution/memories/{memory_id}/revoke",
        json={"actor": "user", "reason": "Testing revocation"},
    )
    candidate_audit_response = client.get(
        f"/personal-evolution/audit?entity_id={candidate_id}"
    )
    memory_audit_response = client.get(
        f"/personal-evolution/audit?entity_id={memory_id}"
    )
    assert revoked_response.status_code == 200
    assert candidate_audit_response.status_code == 200
    assert memory_audit_response.status_code == 200

    revoked = revoked_response.json()
    candidate_audit = candidate_audit_response.json()
    memory_audit = memory_audit_response.json()

    assert len(review["evidence"]) == 2
    assert len(review["events"]) == 1
    assert approved["memory"]["content"] == (
        "Coffee and focused work appeared together on 2026-06-20."
    )
    assert revoked["memory"]["status"] == "revoked"
    assert [event["action"] for event in candidate_audit["events"]] == [
        "candidate_edited",
        "approved",
    ]
    assert [event["action"] for event in memory_audit["events"]] == ["revoked"]
