from __future__ import annotations

from pathlib import Path

from personal_evolution.ingestors import (
    HealthDailySummary,
    HealthSummaryIngestor,
    ObsidianVaultIngestor,
    PhotoSummary,
    PhotoSummaryIngestor,
)
from personal_evolution.models import SourceType


def test_photo_summary_ingestor_redacts_paths_and_keeps_metadata() -> None:
    ingestor = PhotoSummaryIngestor(
        [
            PhotoSummary(
                photo_id="photo-1",
                taken_at="2026-06-20T09:00:00",
                summary="Coffee and laptop on a desk at /Users/example/Pictures/private.jpg",
                tags=["coffee", "desk"],
                latitude=37.7,
                longitude=-122.4,
                local_path="/Users/example/Pictures/private.jpg",
            )
        ]
    )

    evidence = ingestor.ingest()[0]

    assert evidence.source_type == SourceType.PHOTO
    assert evidence.source_ref == "photos://photo-1"
    assert "private.jpg" not in evidence.summary
    assert "/Users/example/Pictures/private.jpg" not in evidence.summary
    assert evidence.metadata["tags"] == ("coffee", "desk")
    assert evidence.metadata["has_local_path"] is True
    assert "local_path" not in evidence.metadata


def test_obsidian_ingestor_extracts_title_and_hash_without_full_body(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "Projects.md"
    note.write_text(
        "# Project Memory\n\nThe user prefers audit-first systems.\n"
        "This second sentence should stay out of metadata.\n",
        encoding="utf-8",
    )

    evidence = ObsidianVaultIngestor(vault).ingest()[0]

    assert evidence.source_type == SourceType.OBSIDIAN
    assert evidence.source_ref == "obsidian://Projects.md"
    assert evidence.summary == "Project Memory: The user prefers audit-first systems."
    assert evidence.metadata["path"] == "Projects.md"
    assert evidence.metadata["title"] == "Project Memory"
    assert "raw_body" not in evidence.metadata
    assert "audit-first" not in evidence.metadata


def test_obsidian_ingestor_ignores_frontmatter_when_extracting_title(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    note = vault / "Daily.md"
    note.write_text(
        "---\n# Private frontmatter comment\nowner: user\n---\n\nVisible body.\n",
        encoding="utf-8",
    )

    evidence = ObsidianVaultIngestor(vault).ingest()[0]

    assert evidence.summary == "Daily: Visible body."
    assert evidence.metadata["title"] == "Daily"
    assert "Private frontmatter comment" not in evidence.summary
    assert "Private frontmatter comment" not in evidence.metadata["title"]


def test_health_ingestor_uses_daily_aggregate_summary() -> None:
    ingestor = HealthSummaryIngestor(
        [
            HealthDailySummary(
                date="2026-06-20",
                steps=8600,
                workouts=1,
                sleep_minutes=430,
                notes="Higher activity than usual",
            )
        ]
    )

    evidence = ingestor.ingest()[0]

    assert evidence.source_type == SourceType.HEALTH
    assert evidence.source_ref == "health://2026-06-20"
    assert "8600 steps" in evidence.summary
    assert evidence.metadata["sleep_minutes"] == 430


def test_obsidian_ingestor_skips_hidden_directories(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    hidden_dir = vault / ".archive"
    visible_dir = vault / "visible"
    hidden_dir.mkdir(parents=True)
    visible_dir.mkdir(parents=True)
    (hidden_dir / "Secret.md").write_text("# Secret\n\nHidden body.\n", encoding="utf-8")
    (visible_dir / "Note.md").write_text("# Note\n\nVisible body.\n", encoding="utf-8")

    evidence = ObsidianVaultIngestor(vault).ingest()

    assert [item.source_ref for item in evidence] == ["obsidian://visible/Note.md"]


def test_ingestors_create_deterministic_ids_for_same_content(tmp_path: Path) -> None:
    photo = PhotoSummary(
        photo_id="photo-1",
        taken_at="2026-06-20T09:00:00",
        summary="Coffee and laptop on a desk",
        tags=["coffee", "desk"],
    )
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Projects.md").write_text(
        "# Project Memory\n\nThe user prefers audit-first systems.\n",
        encoding="utf-8",
    )
    day = HealthDailySummary(
        date="2026-06-20",
        steps=8600,
        workouts=1,
        sleep_minutes=430,
    )

    first = [
        PhotoSummaryIngestor([photo]).ingest()[0].evidence_id,
        ObsidianVaultIngestor(vault).ingest()[0].evidence_id,
        HealthSummaryIngestor([day]).ingest()[0].evidence_id,
    ]
    second = [
        PhotoSummaryIngestor([photo]).ingest()[0].evidence_id,
        ObsidianVaultIngestor(vault).ingest()[0].evidence_id,
        HealthSummaryIngestor([day]).ingest()[0].evidence_id,
    ]

    assert second == first
