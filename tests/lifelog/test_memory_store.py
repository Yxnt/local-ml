from __future__ import annotations

from datetime import datetime

from memory.memory_schema import EventMemory, PreferenceMemory
from memory.memory_store import MemoryStore


def test_memory_store_persists_events_preferences_journals_and_feedback(tmp_path) -> None:
    db_path = tmp_path / "lifelog.sqlite3"
    store = MemoryStore(db_path)
    event = EventMemory(
        event_id="event-1",
        timestamp=datetime(2026, 6, 20, 9, 0),
        summary="morning coffee with people",
        people=["face"],
        location=None,
        embeddings=[0.1, 0.2],
    )
    preference = PreferenceMemory(
        category="style",
        preference="prefers concise summaries",
        confidence=0.7,
        updated_at=datetime(2026, 6, 20, 10, 0),
    )

    store.save_event(event)
    store.upsert_preference(preference)
    journal_id = store.save_journal(
        journal_date="2026-06-20",
        markdown="# Daily Journal - 2026-06-20",
        event_ids=["event-1"],
        prompt="Summarize today.",
    )
    feedback_id = store.save_feedback(
        journal_id=journal_id,
        event_id="event-1",
        rating="down",
        note="too verbose",
        edited_text="Shorter version",
    )

    reopened = MemoryStore(db_path)

    assert reopened.list_events()[0].summary == "morning coffee with people"
    assert reopened.list_preferences()[0].preference == "prefers concise summaries"
    assert reopened.list_journals()[0]["journal_id"] == journal_id
    assert reopened.list_feedback()[0].feedback_id == feedback_id


def test_memory_store_defaults_to_local_journal_database() -> None:
    path = MemoryStore.default_path()

    assert path.name == "memory.sqlite3"
    assert "local-ml-journal" in str(path)
