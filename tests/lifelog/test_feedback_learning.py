from __future__ import annotations

from datetime import datetime

from feedback.feedback_collector import collect_feedback
from feedback.preference_learner import learn_preferences
from memory.memory_schema import FeedbackMemory
from memory.memory_store import MemoryStore


def test_preference_learner_extracts_style_and_emphasis_patterns() -> None:
    feedback = [
        FeedbackMemory("f1", "j1", None, "down", "too verbose", None, datetime(2026, 6, 20, 9)),
        FeedbackMemory("f2", "j2", None, "down", "please make it shorter", None, datetime(2026, 6, 21, 9)),
        FeedbackMemory("f3", "j3", None, "down", "missed gym activity", None, datetime(2026, 6, 22, 9)),
    ]

    preferences = learn_preferences(feedback)

    assert any(pref.preference == "prefers concise summaries" for pref in preferences)
    assert any(pref.preference == "fitness activities are high importance" for pref in preferences)


def test_feedback_collector_accepts_file_notes(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    feedback_file = tmp_path / "feedback.txt"
    feedback_file.write_text("wrong interpretation", encoding="utf-8")

    feedback_id = collect_feedback(
        store,
        journal_id="journal-1",
        rating="down",
        feedback_file=feedback_file,
    )

    feedback = store.list_feedback()[0]
    assert feedback.feedback_id == feedback_id
    assert feedback.note == "wrong interpretation"
