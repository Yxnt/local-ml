from __future__ import annotations

import hashlib
from collections import defaultdict

from personal_evolution.models import (
    CandidateMemory,
    Evidence,
    MemoryStatus,
    MemoryType,
    ObservedEvent,
    utc_now_iso,
)


class ObservedEventBuilder:
    def build(self, evidence: list[Evidence]) -> list[ObservedEvent]:
        evidence_by_day: dict[str, list[Evidence]] = defaultdict(list)
        for item in evidence:
            evidence_by_day[item.observed_at[:10]].append(item)

        events: list[ObservedEvent] = []
        for day in sorted(evidence_by_day):
            day_evidence = sorted(
                evidence_by_day[day],
                key=lambda item: (item.observed_at, item.evidence_id),
            )
            evidence_ids = [item.evidence_id for item in day_evidence]
            events.append(
                ObservedEvent(
                    event_id=_stable_id("event", day, *evidence_ids),
                    start_at=min(item.observed_at for item in day_evidence),
                    end_at=max(item.observed_at for item in day_evidence),
                    title=f"Personal signals for {day}",
                    summary=" ".join(item.summary for item in day_evidence),
                    evidence_ids=evidence_ids,
                    confidence=0.65,
                    created_at=utc_now_iso(),
                )
            )

        return events


class CandidateMemoryGenerator:
    def generate(
        self,
        events: list[ObservedEvent],
        evidence: list[Evidence],
    ) -> list[CandidateMemory]:
        evidence_by_id = {item.evidence_id: item for item in evidence}
        candidates: list[CandidateMemory] = []

        for event in sorted(events, key=lambda item: (item.start_at, item.event_id)):
            day = event.start_at[:10]
            source_types = _source_types_for_event(event, evidence_by_id)
            claim = (
                f"On {day}, personal signals from {source_types} indicated "
                f"{event.summary}"
            )
            now = utc_now_iso()
            candidates.append(
                CandidateMemory(
                    candidate_id=_stable_id("candidate", event.event_id, claim),
                    memory_type=MemoryType.EVENT,
                    claim=claim,
                    rationale=(
                        f"Generated from {len(event.evidence_ids)} evidence item(s): "
                        f"{event.summary}"
                    ),
                    evidence_ids=list(event.evidence_ids),
                    status=MemoryStatus.PENDING,
                    confidence=event.confidence,
                    source_model="local-rules",
                    remote_assisted=False,
                    created_at=now,
                    updated_at=now,
                )
            )

        return candidates


def _source_types_for_event(
    event: ObservedEvent,
    evidence_by_id: dict[str, Evidence],
) -> str:
    source_types = sorted(
        {
            evidence_by_id[evidence_id].source_type.value
            for evidence_id in event.evidence_ids
            if evidence_id in evidence_by_id
        }
    )
    return ", ".join(source_types) if source_types else "local signals"


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:16]}"
