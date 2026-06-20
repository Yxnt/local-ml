from __future__ import annotations

from datetime import date, datetime

from lifelog.event_builder import Event
from lifelog.journal_writer import generate_journal
from lifelog.photos_reader import PhotoItem


def test_generate_journal_uses_readable_template_when_llm_unavailable() -> None:
    event = Event(
        start_time=datetime(2026, 6, 20, 8, 30),
        end_time=datetime(2026, 6, 20, 9, 10),
        photos=[
            PhotoItem(
                path="/tmp/coffee.jpg",
                timestamp=datetime(2026, 6, 20, 8, 30),
                local_identifier="local-1",
            )
        ],
        summary_signals=["cafe", "coffee", "people"],
    )

    journal = generate_journal([event], journal_date=date(2026, 6, 20), llm_client=None)

    assert journal.startswith("# Daily Journal - 2026-06-20")
    assert "## Morning" in journal
    assert "8:30 AM-9:10 AM" in journal
    assert "cafe, coffee, people" in journal
    assert "## Afternoon" in journal
    assert "## Evening" in journal


def test_generate_journal_handles_empty_day() -> None:
    journal = generate_journal([], journal_date=date(2026, 6, 20), llm_client=None)

    assert "# Daily Journal - 2026-06-20" in journal
    assert "No photos were found for this part of the day." in journal
