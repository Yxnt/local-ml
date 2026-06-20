from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from personal_evolution.models import Evidence, SourceType, utc_now_iso


@dataclass(frozen=True)
class PhotoSummary:
    photo_id: str
    taken_at: str
    summary: str
    tags: list[str]
    latitude: float | None = None
    longitude: float | None = None
    local_path: str | None = None


@dataclass(frozen=True)
class HealthDailySummary:
    date: str
    steps: int | None
    workouts: int | None
    sleep_minutes: int | None
    notes: str | None = None


class PhotoSummaryIngestor:
    def __init__(self, photos: list[PhotoSummary]) -> None:
        self.photos = photos

    def ingest(self) -> list[Evidence]:
        evidence: list[Evidence] = []
        for photo in self.photos:
            safe_summary = _redact_local_path(photo.summary, photo.local_path)
            safe_metadata = {
                "tags": list(photo.tags),
                "latitude": photo.latitude,
                "longitude": photo.longitude,
                "has_local_path": photo.local_path is not None,
            }
            content_hash = _hash_text(
                "|".join(
                    [
                        photo.photo_id,
                        photo.taken_at,
                        safe_summary,
                        ",".join(photo.tags),
                        str(photo.latitude),
                        str(photo.longitude),
                    ]
                )
            )
            evidence.append(
                Evidence(
                    evidence_id=f"photo-{content_hash[:16]}",
                    source_type=SourceType.PHOTO,
                    source_ref=f"photos://{photo.photo_id}",
                    observed_at=photo.taken_at,
                    summary=safe_summary,
                    sensitivity="low",
                    content_hash=content_hash,
                    metadata=safe_metadata,
                    created_at=utc_now_iso(),
                )
            )
        return evidence


class ObsidianVaultIngestor:
    def __init__(self, vault_path: str | Path) -> None:
        self.vault_path = Path(vault_path).expanduser()

    def ingest(self) -> list[Evidence]:
        evidence: list[Evidence] = []
        for path in sorted(self.vault_path.rglob("*.md")):
            rel_path = path.relative_to(self.vault_path)
            if any(part.startswith(".") for part in rel_path.parts):
                continue

            content = path.read_text(encoding="utf-8", errors="ignore")
            rel_path_text = rel_path.as_posix()
            title = _extract_title(path, content)
            body_line = _first_body_sentence(content)
            summary = f"{title}: {body_line}" if body_line else title
            content_hash = _hash_text(content)
            now = utc_now_iso()

            evidence.append(
                Evidence(
                    evidence_id=f"obsidian-{content_hash[:16]}",
                    source_type=SourceType.OBSIDIAN,
                    source_ref=f"obsidian://{rel_path_text}",
                    observed_at=now,
                    summary=summary,
                    sensitivity="medium",
                    content_hash=content_hash,
                    metadata={
                        "path": rel_path_text,
                        "title": title,
                        "size": len(content.encode("utf-8")),
                    },
                    created_at=now,
                )
            )
        return evidence


class HealthSummaryIngestor:
    def __init__(self, days: list[HealthDailySummary]) -> None:
        self.days = days

    def ingest(self) -> list[Evidence]:
        evidence: list[Evidence] = []
        for day in self.days:
            parts: list[str] = []
            if day.steps is not None:
                parts.append(f"{day.steps} steps")
            if day.workouts is not None:
                parts.append(f"{day.workouts} workout(s)")
            if day.sleep_minutes is not None:
                parts.append(f"{day.sleep_minutes} minutes sleep")
            if day.notes:
                parts.append(day.notes)

            summary = f"Health summary for {day.date}"
            if parts:
                summary = f"{summary}: {', '.join(parts)}"
            content_hash = _hash_text(summary)

            evidence.append(
                Evidence(
                    evidence_id=f"health-{day.date}-{content_hash[:8]}",
                    source_type=SourceType.HEALTH,
                    source_ref=f"health://{day.date}",
                    observed_at=f"{day.date}T00:00:00",
                    summary=summary,
                    sensitivity="medium",
                    content_hash=content_hash,
                    metadata={
                        "date": day.date,
                        "steps": day.steps,
                        "workouts": day.workouts,
                        "sleep_minutes": day.sleep_minutes,
                        "notes": day.notes,
                    },
                    created_at=utc_now_iso(),
                )
            )
        return evidence


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _redact_local_path(summary: str, local_path: str | None) -> str:
    if not local_path:
        return summary
    return summary.replace(local_path, "[local photo path redacted]")


def _extract_title(path: Path, content: str) -> str:
    for stripped in _visible_markdown_lines(content):
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            if title:
                return title
    return path.stem


def _first_body_sentence(content: str) -> str:
    for stripped in _visible_markdown_lines(content):
        if stripped.startswith("#"):
            continue
        return stripped
    return ""


def _visible_markdown_lines(content: str) -> list[str]:
    lines = content.splitlines()
    start_index = 0
    if lines and lines[0].strip() == "---":
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                start_index = index + 1
                break

    return [line.strip() for line in lines[start_index:] if line.strip()]
