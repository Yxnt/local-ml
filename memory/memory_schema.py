"""Schemas for the local lifelog memory system."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class EventMemory:
    event_id: str
    timestamp: datetime
    summary: str
    people: list[str]
    location: str | None
    embeddings: list[float] | None


@dataclass(frozen=True)
class PreferenceMemory:
    category: str
    preference: str
    confidence: float
    updated_at: datetime


@dataclass(frozen=True)
class FeedbackMemory:
    feedback_id: str
    journal_id: str
    event_id: str | None
    rating: str | None
    note: str | None
    edited_text: str | None
    timestamp: datetime
