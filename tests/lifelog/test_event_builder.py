from __future__ import annotations

from datetime import datetime, timedelta

from lifelog.event_builder import build_events
from lifelog.photos_reader import PhotoItem
from lifelog.vision_extractor import VisionResult


def test_build_events_groups_photos_by_default_time_gap() -> None:
    start = datetime(2026, 6, 20, 9, 0)
    photos = [
        PhotoItem("/tmp/morning-1.jpg", start, "a"),
        PhotoItem("/tmp/morning-2.jpg", start + timedelta(minutes=20), "b"),
        PhotoItem("/tmp/lunch.jpg", start + timedelta(hours=2), "c"),
    ]
    vision = {
        "/tmp/morning-1.jpg": VisionResult("/tmp/morning-1.jpg", ["coffee"], "cafe", []),
        "/tmp/morning-2.jpg": VisionResult("/tmp/morning-2.jpg", ["person"], None, ["face"]),
        "/tmp/lunch.jpg": VisionResult("/tmp/lunch.jpg", ["food"], "restaurant", []),
    }

    events = build_events(photos, vision)

    assert len(events) == 2
    assert [p.local_identifier for p in events[0].photos] == ["a", "b"]
    assert [p.local_identifier for p in events[1].photos] == ["c"]
    assert events[0].summary_signals == ["cafe", "coffee", "person", "people"]
    assert events[1].summary_signals == ["restaurant", "food"]


def test_build_events_returns_empty_list_for_no_photos() -> None:
    assert build_events([], {}) == []
