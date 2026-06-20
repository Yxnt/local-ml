"""Infer durable journal preferences from local feedback history."""

from __future__ import annotations

from collections import Counter
from datetime import datetime

from memory.memory_schema import FeedbackMemory, PreferenceMemory


def learn_preferences(feedback: list[FeedbackMemory]) -> list[PreferenceMemory]:
    counts: Counter[str] = Counter()
    for item in feedback:
        text = " ".join(part for part in [item.note, item.edited_text, item.rating] if part).lower()
        if any(phrase in text for phrase in ["too verbose", "shorter", "concise", "less detail"]):
            counts["style|prefers concise summaries"] += 1
        if any(phrase in text for phrase in ["bullet", "bullets", "structured"]):
            counts["structure|likes structured bullet points"] += 1
        if any(phrase in text for phrase in ["productivity", "work", "focus", "office"]):
            counts["emphasis|productivity summaries are high importance"] += 1
        if any(phrase in text for phrase in ["gym", "fitness", "workout", "run", "health"]):
            counts["emphasis|fitness activities are high importance"] += 1
        if any(phrase in text for phrase in ["friend", "family", "social", "people"]):
            counts["emphasis|social context is high importance"] += 1

    now = datetime.now()
    preferences: list[PreferenceMemory] = []
    for key, count in counts.items():
        category, preference = key.split("|", 1)
        confidence = min(0.95, 0.45 + count * 0.2)
        preferences.append(
            PreferenceMemory(
                category=category,
                preference=preference,
                confidence=confidence,
                updated_at=now,
            )
        )
    preferences.sort(key=lambda pref: pref.confidence, reverse=True)
    return preferences


def update_preferences_from_feedback(store) -> list[PreferenceMemory]:  # type: ignore[no-untyped-def]
    preferences = learn_preferences(store.list_feedback())
    for preference in preferences:
        store.upsert_preference(preference)
    return preferences
