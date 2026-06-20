from __future__ import annotations

from datetime import datetime

from memory.memory_schema import EventMemory
from memory.vector_store import VectorStore


def test_vector_store_uses_simple_text_similarity_without_embeddings() -> None:
    store = VectorStore()
    events = [
        EventMemory("a", datetime(2026, 6, 18, 8), "morning gym workout", [], None, None),
        EventMemory("b", datetime(2026, 6, 19, 9), "office focus work session", [], None, None),
        EventMemory("c", datetime(2026, 6, 20, 21), "dinner with friends", [], None, None),
    ]

    matches = store.search("gym fitness workout", events, limit=2)

    assert matches[0].event.event_id == "a"
    assert matches[0].score > matches[1].score
