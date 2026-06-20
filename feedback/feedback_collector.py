"""Collect feedback from CLI arguments or local files."""

from __future__ import annotations

from pathlib import Path

from memory.memory_store import MemoryStore


def collect_feedback(
    store: MemoryStore,
    *,
    journal_id: str,
    event_id: str | None = None,
    rating: str | None = None,
    note: str | None = None,
    edited_text: str | None = None,
    feedback_file: str | Path | None = None,
) -> str | None:
    if feedback_file is not None:
        file_text = Path(feedback_file).expanduser().read_text(encoding="utf-8").strip()
        if file_text:
            note = file_text if note is None else f"{note}\n{file_text}"

    if not any([rating, note, edited_text]):
        return None

    return store.save_feedback(
        journal_id=journal_id,
        event_id=event_id,
        rating=rating,
        note=note,
        edited_text=edited_text,
    )
