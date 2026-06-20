from __future__ import annotations

from pathlib import Path

from personal_evolution.context import build_approved_memory_context
from personal_evolution.models import ApprovedMemory, MemoryStatus, MemoryType
from personal_evolution.store import PersonalEvolutionStore


def test_build_approved_memory_context_includes_only_active_memories(
    tmp_path: Path,
) -> None:
    store = PersonalEvolutionStore(tmp_path / "personal.sqlite3")
    store.save_approved_memory(
        ApprovedMemory(
            memory_id="mem-active",
            memory_type=MemoryType.EVENT,
            content="The user prefers audit-first local learning.",
            evidence_ids=["ev-1"],
            candidate_id="cand-1",
            version=1,
            confidence=0.7,
            status=MemoryStatus.APPROVED,
            approved_at="2026-06-20T09:04:00+00:00",
            revoked_at=None,
        )
    )
    store.save_approved_memory(
        ApprovedMemory(
            memory_id="mem-revoked",
            memory_type=MemoryType.EVENT,
            content="This revoked item should not influence the model.",
            evidence_ids=["ev-2"],
            candidate_id="cand-2",
            version=1,
            confidence=0.7,
            status=MemoryStatus.REVOKED,
            approved_at="2026-06-20T09:05:00+00:00",
            revoked_at="2026-06-20T09:06:00+00:00",
        )
    )

    context = build_approved_memory_context(store, limit=10)

    assert "已确认的个人长期记忆" in context
    assert "audit-first local learning" in context
    assert "revoked item" not in context
    assert "mem-active" in context


def test_build_approved_memory_context_returns_empty_string_without_active_memories(
    tmp_path: Path,
) -> None:
    store = PersonalEvolutionStore(tmp_path / "personal.sqlite3")

    assert build_approved_memory_context(store) == ""
