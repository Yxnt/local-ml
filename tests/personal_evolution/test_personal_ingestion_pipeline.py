from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from lifelog.photos_reader import PhotoItem
from lifelog.vision_extractor import VisionResult
from personal_evolution.models import MemoryStatus
from personal_evolution.pipeline import ingest_personal_sources
from personal_evolution.store import PersonalEvolutionStore


def test_ingest_personal_sources_writes_obsidian_and_lifelog_photo_candidates(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Daily.md").write_text(
        "# Daily Reflection\n\nWorked on the local memory review flow.\n",
        encoding="utf-8",
    )
    photo_path = tmp_path / "coffee_work.jpg"
    photo_path.write_bytes(b"fake-image")
    store = PersonalEvolutionStore(tmp_path / "personal.sqlite3")

    result = ingest_personal_sources(
        store,
        obsidian_vaults=[vault],
        target_date=date(2026, 6, 20),
        read_photos=lambda day: [
            PhotoItem(
                path=str(photo_path),
                timestamp=datetime(2026, 6, 20, 9, 0, 0),
                local_identifier="photo-1",
            )
        ],
        extract_vision=lambda photos: {
            str(photo_path): VisionResult(
                photo_path=str(photo_path),
                objects=["laptop", "coffee"],
                scene="working session",
                people=[],
            )
        },
    )

    evidence = store.list_evidence()
    candidates = store.list_candidates(MemoryStatus.PENDING)

    assert result.evidence_saved == 2
    assert result.events_saved == 1
    assert result.candidates_saved == 1
    assert sorted(item.source_type.value for item in evidence) == ["obsidian", "photo"]
    assert all(str(tmp_path) not in item.summary for item in evidence)
    assert candidates[0].status == MemoryStatus.PENDING
    assert "obsidian, photo" in candidates[0].claim


def test_ingest_personal_sources_reports_recoverable_source_errors(
    tmp_path: Path,
) -> None:
    store = PersonalEvolutionStore(tmp_path / "personal.sqlite3")

    result = ingest_personal_sources(
        store,
        obsidian_vaults=[tmp_path / "missing-vault"],
        target_date=date(2026, 6, 20),
        read_photos=lambda day: (_ for _ in ()).throw(RuntimeError("Photos denied")),
    )

    assert result.evidence_saved == 0
    assert result.events_saved == 0
    assert result.candidates_saved == 0
    assert result.errors == [
        "obsidian:/missing-vault: vault not found",
        "photos: Photos denied",
    ]
