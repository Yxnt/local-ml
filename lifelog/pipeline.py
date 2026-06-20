"""End-to-end local lifelog pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

from feedback.feedback_collector import collect_feedback
from feedback.preference_learner import update_preferences_from_feedback
from lifelog.event_builder import build_events
from lifelog.journal_writer import LLMClient, prepare_journal, save_journal
from lifelog.photos_reader import PhotoItem, default_photos_export_dir, read_today_photos
from lifelog.vision_extractor import VisionResult, extract_all_vision
from memory.memory_schema import EventMemory
from memory.memory_store import MemoryStore
from memory.vector_store import VectorStore

ReadPhotos = Callable[[date], list[PhotoItem]]
ExtractVision = Callable[[list[PhotoItem]], dict[str, VisionResult]]


def run_pipeline(
    *,
    target_date: date | None = None,
    output_dir: str | Path = "~/local-ml-journal",
    memory_path: str | Path | None = None,
    read_photos: ReadPhotos | None = None,
    extract_vision: ExtractVision = extract_all_vision,
    llm_client: LLMClient | None = None,
    feedback_rating: str | None = None,
    feedback_text: str | None = None,
    edited_journal_text: str | None = None,
    feedback_file: str | Path | None = None,
) -> Path:
    journal_date = target_date or datetime.now().date()
    warnings: list[str] = []
    store = MemoryStore(memory_path)
    reader = read_photos or (
        lambda day: read_today_photos(day, export_dir=default_photos_export_dir())
    )

    try:
        photos = reader(journal_date)
    except Exception as exc:
        photos = []
        warnings.append(f"Photos reader failed: {exc}")

    try:
        vision = extract_vision(photos)
    except Exception as exc:
        vision = {}
        warnings.append(f"Vision extraction failed: {exc}")

    events = build_events(photos, vision)
    event_memories = [_event_to_memory(event) for event in events]
    query = " ".join(event.summary for event in event_memories)
    similar_matches = VectorStore().search(query, store.list_events(limit=200), limit=5)
    preferences = store.list_preferences()

    draft = prepare_journal(
        events,
        journal_date=journal_date,
        llm_client=llm_client,
        preferences=preferences,
        similar_events=[match.event for match in similar_matches if match.score > 0],
        warnings=warnings,
    )
    for event_memory in event_memories:
        store.save_event(event_memory)
    journal_id = store.save_journal(
        journal_date=journal_date.isoformat(),
        markdown=draft.markdown,
        event_ids=[event.event_id for event in event_memories],
        prompt=draft.prompt,
    )
    collect_feedback(
        store,
        journal_id=journal_id,
        event_id=event_memories[0].event_id if event_memories else None,
        rating=feedback_rating,
        note=feedback_text,
        edited_text=edited_journal_text,
        feedback_file=feedback_file,
    )
    update_preferences_from_feedback(store)
    return save_journal(draft.markdown, output_dir, journal_date)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local lifelog memory pipeline.")
    parser.add_argument("--date", help="Journal date in YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--output-dir", default="~/local-ml-journal")
    parser.add_argument("--memory-path", default=None)
    parser.add_argument("--rating", choices=["up", "down"], help="Optional feedback rating.")
    parser.add_argument("--feedback", help="Optional feedback note for the generated journal.")
    parser.add_argument("--edited-journal", help="Optional path to edited journal markdown.")
    parser.add_argument("--feedback-file", help="Optional path containing feedback notes.")
    args = parser.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else None
    edited_text = None
    if args.edited_journal:
        edited_text = Path(args.edited_journal).expanduser().read_text(encoding="utf-8")

    output_path = run_pipeline(
        target_date=target_date,
        output_dir=args.output_dir,
        memory_path=args.memory_path,
        feedback_rating=args.rating,
        feedback_text=args.feedback,
        edited_journal_text=edited_text,
        feedback_file=args.feedback_file,
    )
    print(f"Wrote daily journal: {output_path}")


def _event_to_memory(event) -> EventMemory:  # type: ignore[no-untyped-def]
    summary = _event_summary(event)
    event_id = f"{event.start_time.strftime('%Y%m%d%H%M%S')}-{abs(hash(summary)) % 100000:05d}"
    return EventMemory(
        event_id=event_id,
        timestamp=event.start_time,
        summary=summary,
        people=["people"] if "people" in event.summary_signals else [],
        location=None,
        embeddings=None,
    )


def _event_summary(event) -> str:  # type: ignore[no-untyped-def]
    signals = ", ".join(event.summary_signals) if event.summary_signals else "photos"
    return (
        f"{event.start_time.strftime('%H:%M')}-{event.end_time.strftime('%H:%M')}: "
        f"{len(event.photos)} photo(s), signals: {signals}"
    )


if __name__ == "__main__":
    main()
