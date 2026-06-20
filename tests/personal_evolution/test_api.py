from __future__ import annotations

import importlib
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from personal_evolution.generator import CandidateMemoryGenerator, ObservedEventBuilder
from personal_evolution.ingestors import PhotoSummary, PhotoSummaryIngestor
from personal_evolution.models import MemoryStatus
from personal_evolution.store import PersonalEvolutionStore
from server.personal_evolution_api import create_personal_evolution_router


def _seed_store(store: PersonalEvolutionStore) -> str:
    evidence = PhotoSummaryIngestor(
        [
            PhotoSummary(
                photo_id="photo-1",
                taken_at="2026-06-20T09:00:00",
                summary="Coffee and laptop",
                tags=["coffee"],
            )
        ]
    ).ingest()
    events = ObservedEventBuilder().build(evidence)
    candidates = CandidateMemoryGenerator().generate(events, evidence)

    for item in evidence:
        store.save_evidence(item)
    for event in events:
        store.save_observed_event(event)
    store.save_candidate(candidates[0])

    return candidates[0].candidate_id


def _client(tmp_path: Path) -> tuple[TestClient, str]:
    store = PersonalEvolutionStore(tmp_path / "personal.sqlite3")
    candidate_id = _seed_store(store)
    app = FastAPI()
    app.include_router(
        create_personal_evolution_router(store),
        prefix="/personal-evolution",
    )
    return TestClient(app), candidate_id


def test_api_lists_review_and_candidates(tmp_path: Path) -> None:
    client, candidate_id = _client(tmp_path)

    review = client.get("/personal-evolution/review/2026-06-20")
    queue = client.get("/personal-evolution/candidates")

    assert review.status_code == 200
    assert review.json()["date"] == "2026-06-20"
    assert review.json()["events"][0]["title"] == "Personal signals for 2026-06-20"
    assert review.json()["evidence"][0]["summary"] == "Coffee and laptop"
    assert review.json()["candidates"][0]["candidate_id"] == candidate_id
    assert queue.status_code == 200
    assert queue.json()["candidates"][0]["candidate_id"] == candidate_id
    assert queue.json()["candidates"][0]["status"] == MemoryStatus.PENDING.value


def test_api_approve_revoke_and_audit(tmp_path: Path) -> None:
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
    assert approve.json()["memory"]["status"] == MemoryStatus.APPROVED.value
    assert revoke.status_code == 200
    assert revoke.json()["memory"]["status"] == MemoryStatus.REVOKED.value
    assert audit.status_code == 200
    assert audit.json()["events"][0]["action"] == "revoked"


def test_api_edit_approve_and_list_memories(tmp_path: Path) -> None:
    client, candidate_id = _client(tmp_path)

    approve = client.post(
        f"/personal-evolution/candidates/{candidate_id}/edit-approve",
        json={
            "actor": "user",
            "reason": "Tighter wording",
            "content": "Morning coffee often accompanies laptop work.",
        },
    )
    memories = client.get("/personal-evolution/memories")

    assert approve.status_code == 200
    assert approve.json()["memory"]["content"] == (
        "Morning coffee often accompanies laptop work."
    )
    assert memories.status_code == 200
    assert memories.json()["memories"] == [approve.json()["memory"]]


def test_api_rejects_candidate(tmp_path: Path) -> None:
    client, candidate_id = _client(tmp_path)

    reject = client.post(
        f"/personal-evolution/candidates/{candidate_id}/reject",
        json={"actor": "user", "reason": "Too speculative"},
    )
    approve = client.post(
        f"/personal-evolution/candidates/{candidate_id}/approve",
        json={"actor": "user", "reason": "Try again"},
    )

    assert reject.status_code == 200
    assert reject.json()["candidate"]["status"] == MemoryStatus.REJECTED.value
    assert approve.status_code == 400
    assert "not pending" in approve.json()["detail"]


def test_api_get_evidence_returns_404_for_missing_id(tmp_path: Path) -> None:
    client, _candidate_id = _client(tmp_path)

    response = client.get("/personal-evolution/evidence/missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "Evidence not found: missing"


def test_api_candidates_rejects_invalid_status(tmp_path: Path) -> None:
    client, _candidate_id = _client(tmp_path)

    response = client.get("/personal-evolution/candidates?status=unknown")

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid candidate status: unknown"


def test_main_import_does_not_create_default_db_under_process_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env_db_path = tmp_path / "env" / "personal.sqlite3"
    cwd_memory_db_path = tmp_path / "memory" / "personal_evolution.sqlite3"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PERSONAL_EVOLUTION_DB", str(env_db_path))

    main_module = importlib.import_module("server.main")
    main_module = importlib.reload(main_module)
    routes = sorted(route.path for route in main_module.app.routes)

    assert "/personal-evolution/review/{date}" in routes
    assert "/personal-evolution/candidates" in routes
    assert env_db_path.exists()
    assert not cwd_memory_db_path.exists()

    monkeypatch.delenv("PERSONAL_EVOLUTION_DB")
    sys.modules.pop("server.main", None)
    main_module = importlib.import_module("server.main")
    routes = sorted(route.path for route in main_module.app.routes)

    assert "/personal-evolution/review/{date}" in routes
    assert "/personal-evolution/candidates" in routes
    assert not cwd_memory_db_path.exists()
