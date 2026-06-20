from __future__ import annotations

from datetime import datetime

from evolution.prompt_optimizer import build_journal_prompt
from lifelog.event_builder import Event
from lifelog.photos_reader import PhotoItem
from memory.memory_schema import EventMemory, PreferenceMemory


def test_prompt_optimizer_uses_preferences_and_similar_memories() -> None:
    event = Event(
        start_time=datetime(2026, 6, 20, 9, 0),
        end_time=datetime(2026, 6, 20, 9, 30),
        photos=[PhotoItem("/tmp/work.jpg", datetime(2026, 6, 20, 9, 0), "p1")],
        summary_signals=["working session"],
    )
    preferences = [
        PreferenceMemory("style", "prefers concise summaries", 0.8, datetime(2026, 6, 19, 9)),
        PreferenceMemory("emphasis", "productivity summaries are high importance", 0.6, datetime(2026, 6, 19, 9)),
    ]
    similar = [
        EventMemory("old", datetime(2026, 6, 18, 9), "office focus work session", [], None, None)
    ]

    prompt = build_journal_prompt(
        events=[event],
        journal_date=datetime(2026, 6, 20).date(),
        preferences=preferences,
        similar_events=similar,
    )

    assert "prefers concise summaries" in prompt
    assert "productivity summaries are high importance" in prompt
    assert "office focus work session" in prompt
    assert "working session" in prompt
