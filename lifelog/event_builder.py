"""Build lifelog events from photos and local vision signals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from lifelog.photos_reader import PhotoItem
from lifelog.vision_extractor import VisionResult


@dataclass(frozen=True)
class Event:
    start_time: datetime
    end_time: datetime
    photos: list[PhotoItem]
    summary_signals: list[str]


def build_events(
    photos: list[PhotoItem],
    vision_results: dict[str, VisionResult],
    *,
    gap_threshold: timedelta = timedelta(minutes=45),
) -> list[Event]:
    if not photos:
        return []

    ordered = sorted(photos, key=lambda photo: photo.timestamp)
    clusters: list[list[PhotoItem]] = [[ordered[0]]]

    for photo in ordered[1:]:
        previous = clusters[-1][-1]
        if photo.timestamp - previous.timestamp <= gap_threshold:
            clusters[-1].append(photo)
        else:
            clusters.append([photo])

    return [_cluster_to_event(cluster, vision_results) for cluster in clusters]


def _cluster_to_event(
    photos: list[PhotoItem],
    vision_results: dict[str, VisionResult],
) -> Event:
    signals: list[str] = []
    for photo in photos:
        result = vision_results.get(photo.path)
        if result is None:
            continue
        if result.scene:
            signals.append(result.scene)
        signals.extend(result.objects)
        if result.people:
            signals.append("people")

    return Event(
        start_time=photos[0].timestamp,
        end_time=photos[-1].timestamp,
        photos=photos,
        summary_signals=_dedupe(signals),
    )


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = value.strip().lower()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result
