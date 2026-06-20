"""Lightweight vector memory with lexical fallback."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from memory.memory_schema import EventMemory


@dataclass(frozen=True)
class SimilarEvent:
    event: EventMemory
    score: float


class VectorStore:
    """Search event memory using embeddings or simple text similarity."""

    def search(
        self,
        query: str,
        events: list[EventMemory],
        *,
        query_embedding: list[float] | None = None,
        limit: int = 5,
    ) -> list[SimilarEvent]:
        scored: list[SimilarEvent] = []
        for event in events:
            if query_embedding is not None and event.embeddings is not None:
                score = _cosine(query_embedding, event.embeddings)
            else:
                score = _text_similarity(query, event.summary)
            scored.append(SimilarEvent(event=event, score=score))
        scored.sort(key=lambda match: match.score, reverse=True)
        return scored[:limit]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _text_similarity(query: str, text: str) -> float:
    query_terms = _terms(query)
    text_terms = _terms(text)
    if not query_terms or not text_terms:
        return 0.0
    overlap = query_terms & text_terms
    return len(overlap) / math.sqrt(len(query_terms) * len(text_terms))


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2
    }
