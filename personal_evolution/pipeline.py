from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from lifelog.photos_reader import PhotoItem, default_photos_export_dir, read_today_photos
from lifelog.vision_extractor import VisionResult, extract_all_vision
from personal_evolution.generator import CandidateMemoryGenerator, ObservedEventBuilder
from personal_evolution.ingestors import (
    ObsidianVaultIngestor,
    PhotoSummary,
    PhotoSummaryIngestor,
)
from personal_evolution.models import Evidence
from personal_evolution.store import PersonalEvolutionStore

ReadPhotos = Callable[[date], list[PhotoItem]]
ExtractVision = Callable[[list[PhotoItem]], dict[str, VisionResult]]


@dataclass(frozen=True)
class PersonalIngestionResult:
    evidence_saved: int
    events_saved: int
    candidates_saved: int
    errors: list[str]


def ingest_personal_sources(
    store: PersonalEvolutionStore,
    *,
    obsidian_vaults: list[str | Path] | None = None,
    target_date: date | None = None,
    read_photos: ReadPhotos | None = None,
    extract_vision: ExtractVision = extract_all_vision,
) -> PersonalIngestionResult:
    """Ingest configured personal sources into the review-first memory store."""
    errors: list[str] = []
    evidence: list[Evidence] = []

    for vault in obsidian_vaults or []:
        vault_path = Path(vault).expanduser()
        if not vault_path.exists():
            errors.append(f"obsidian:/{vault_path.name}: vault not found")
            continue
        try:
            evidence.extend(ObsidianVaultIngestor(vault_path).ingest())
        except Exception as exc:
            errors.append(f"obsidian:/{vault_path.name}: {exc}")

    photo_reader = read_photos or (
        lambda day: read_today_photos(day, export_dir=default_photos_export_dir())
    )
    day = target_date or date.today()
    try:
        photos = photo_reader(day)
        if photos:
            try:
                vision = extract_vision(photos)
            except Exception as exc:
                vision = {}
                errors.append(f"photos: vision extraction failed: {exc}")
            evidence.extend(PhotoSummaryIngestor(_photo_summaries(photos, vision)).ingest())
    except Exception as exc:
        errors.append(f"photos: {exc}")

    events = ObservedEventBuilder().build(evidence)
    candidates = CandidateMemoryGenerator().generate(events, evidence)

    for item in evidence:
        store.save_evidence(item)
    for event in events:
        store.save_observed_event(event)
    for candidate in candidates:
        store.save_candidate(candidate)

    return PersonalIngestionResult(
        evidence_saved=len(evidence),
        events_saved=len(events),
        candidates_saved=len(candidates),
        errors=errors,
    )


def _photo_summaries(
    photos: list[PhotoItem],
    vision: dict[str, VisionResult],
) -> list[PhotoSummary]:
    summaries: list[PhotoSummary] = []
    for photo in photos:
        result = vision.get(photo.path)
        tags: list[str] = []
        if result is not None:
            tags.extend(result.objects)
            if result.scene:
                tags.append(result.scene)
            tags.extend(result.people)
        clean_tags = _dedupe(tags)
        summary = _photo_summary_text(photo, clean_tags)
        summaries.append(
            PhotoSummary(
                photo_id=photo.local_identifier,
                taken_at=photo.timestamp.isoformat(),
                summary=summary,
                tags=clean_tags,
                local_path=photo.path,
            )
        )
    return summaries


def _photo_summary_text(photo: PhotoItem, tags: list[str]) -> str:
    if tags:
        return f"Photo at {photo.timestamp.isoformat()} with signals: {', '.join(tags)}"
    return f"Photo at {photo.timestamp.isoformat()}"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = value.strip().lower()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result
