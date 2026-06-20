"""Build adaptive journal prompts from memory and preferences."""

from __future__ import annotations

from datetime import date

from lifelog.event_builder import Event
from memory.memory_schema import EventMemory, PreferenceMemory


def build_journal_prompt(
    *,
    events: list[Event],
    journal_date: date,
    preferences: list[PreferenceMemory],
    similar_events: list[EventMemory],
) -> str:
    preference_lines = [
        f"- {pref.preference} (confidence {pref.confidence:.2f})"
        for pref in preferences
    ] or ["- No learned preferences yet. Use a concise, neutral style."]
    similar_lines = [
        f"- {event.timestamp.date().isoformat()}: {event.summary}"
        for event in similar_events
    ] or ["- No similar past events found."]
    event_lines = [
        f"- {event.start_time.strftime('%H:%M')}-{event.end_time.strftime('%H:%M')}: "
        f"{len(event.photos)} photo(s), signals: {', '.join(event.summary_signals) or 'photos'}"
        for event in events
    ] or ["- No photos were found today."]

    return (
        "You are a local-first personal memory assistant. Generate a daily journal only from local memories.\n"
        f"Date: {journal_date.isoformat()}\n\n"
        "Learned user preferences:\n"
        + "\n".join(preference_lines)
        + "\n\nSimilar past memories:\n"
        + "\n".join(similar_lines)
        + "\n\nToday's events:\n"
        + "\n".join(event_lines)
        + "\n\n"
        "Write markdown with exactly these headings: "
        f"# Daily Journal - {journal_date.isoformat()}, ## Morning, ## Afternoon, ## Evening.\n"
        "Adapt length, structure, and emphasis to the learned preferences. Do not invent names or places."
    )
