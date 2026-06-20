from __future__ import annotations

from personal_evolution.generator import (
    CandidateMemoryGenerator,
    ObservedEventBuilder,
)
from personal_evolution.ingestors import (
    HealthDailySummary,
    HealthSummaryIngestor,
    PhotoSummary,
    PhotoSummaryIngestor,
)
from personal_evolution.models import MemoryStatus, MemoryType


def test_observed_event_builder_groups_evidence_by_day() -> None:
    evidence = [
        PhotoSummaryIngestor(
            [
                PhotoSummary(
                    photo_id="photo-1",
                    taken_at="2026-06-20T09:00:00",
                    summary="Coffee and laptop on a desk",
                    tags=["coffee", "desk"],
                )
            ]
        ).ingest()[0],
        HealthSummaryIngestor(
            [
                HealthDailySummary(
                    date="2026-06-20",
                    steps=8600,
                    workouts=1,
                    sleep_minutes=430,
                )
            ]
        ).ingest()[0],
    ]

    events = ObservedEventBuilder().build(evidence)

    assert len(events) == 1
    assert events[0].title == "Personal signals for 2026-06-20"
    assert len(events[0].evidence_ids) == 2


def test_candidate_generator_creates_pending_review_memory() -> None:
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

    candidates = CandidateMemoryGenerator().generate(events, evidence)

    assert len(candidates) == 1
    assert candidates[0].status == MemoryStatus.PENDING
    assert candidates[0].memory_type == MemoryType.EVENT
    assert candidates[0].remote_assisted is False
    assert "2026-06-20" in candidates[0].claim


def test_generator_ids_are_deterministic_independent_of_input_order() -> None:
    photo = PhotoSummaryIngestor(
        [
            PhotoSummary(
                photo_id="photo-1",
                taken_at="2026-06-20T09:00:00",
                summary="Coffee and laptop on a desk",
                tags=["coffee", "desk"],
            )
        ]
    ).ingest()[0]
    health = HealthSummaryIngestor(
        [
            HealthDailySummary(
                date="2026-06-20",
                steps=8600,
                workouts=1,
                sleep_minutes=430,
            )
        ]
    ).ingest()[0]

    first_events = ObservedEventBuilder().build([photo, health])
    second_events = ObservedEventBuilder().build([health, photo])
    first_candidates = CandidateMemoryGenerator().generate(first_events, [photo, health])
    second_candidates = CandidateMemoryGenerator().generate(second_events, [health, photo])

    assert second_events[0].event_id == first_events[0].event_id
    assert second_events[0].evidence_ids == first_events[0].evidence_ids
    assert second_candidates[0].candidate_id == first_candidates[0].candidate_id
    assert second_candidates[0].claim == first_candidates[0].claim


def test_observed_event_builder_creates_one_event_per_day() -> None:
    evidence = PhotoSummaryIngestor(
        [
            PhotoSummary(
                photo_id="photo-1",
                taken_at="2026-06-20T09:00:00",
                summary="Coffee and laptop on a desk",
                tags=["coffee", "desk"],
            ),
            PhotoSummary(
                photo_id="photo-2",
                taken_at="2026-06-21T11:00:00",
                summary="Walk through a park",
                tags=["walk", "park"],
            ),
        ]
    ).ingest()

    events = ObservedEventBuilder().build(evidence)

    assert [event.title for event in events] == [
        "Personal signals for 2026-06-20",
        "Personal signals for 2026-06-21",
    ]
