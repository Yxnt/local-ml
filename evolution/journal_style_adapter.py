"""Deterministic style adaptation for template journal generation."""

from __future__ import annotations

from dataclasses import dataclass

from memory.memory_schema import PreferenceMemory


@dataclass(frozen=True)
class JournalStyle:
    concise: bool = False
    bullet: bool = True
    emphasize_work: bool = False
    emphasize_health: bool = False
    emphasize_social: bool = False


def style_from_preferences(preferences: list[PreferenceMemory]) -> JournalStyle:
    text = " ".join(pref.preference.lower() for pref in preferences)
    return JournalStyle(
        concise="concise" in text or "short" in text,
        bullet="bullet" in text or "structured" in text or True,
        emphasize_work="productivity" in text or "work" in text,
        emphasize_health="fitness" in text or "health" in text,
        emphasize_social="social" in text,
    )
