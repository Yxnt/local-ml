from __future__ import annotations

from datetime import date, datetime

from lifelog.photos_reader import PhotoItem
from lifelog.pipeline import run_pipeline
from lifelog.vision_extractor import VisionResult
from memory.memory_store import MemoryStore


def test_pipeline_stores_events_and_journal_in_memory(tmp_path) -> None:
    memory_path = tmp_path / "memory.sqlite3"

    def read_photos(target_date: date) -> list[PhotoItem]:
        return [
            PhotoItem("/tmp/work.jpg", datetime(2026, 6, 20, 9, 0), "p1"),
        ]

    def extract_vision(photos: list[PhotoItem]) -> dict[str, VisionResult]:
        return {
            photos[0].path: VisionResult(
                photos[0].path,
                objects=["laptop"],
                scene="working session",
                people=[],
            )
        }

    output_path = run_pipeline(
        target_date=date(2026, 6, 20),
        output_dir=tmp_path,
        memory_path=memory_path,
        read_photos=read_photos,
        extract_vision=extract_vision,
        llm_client=None,
        feedback_text="too verbose",
        feedback_rating="down",
    )

    store = MemoryStore(memory_path)

    assert output_path.exists()
    assert len(store.list_events()) == 1
    assert len(store.list_journals()) == 1
    assert store.list_feedback()[0].note == "too verbose"
    assert any(pref.preference == "prefers concise summaries" for pref in store.list_preferences())
