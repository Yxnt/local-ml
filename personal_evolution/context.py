from __future__ import annotations

from personal_evolution.models import MemoryStatus
from personal_evolution.store import PersonalEvolutionStore


def build_approved_memory_context(
    store: PersonalEvolutionStore,
    *,
    limit: int = 20,
) -> str:
    memories = [
        memory
        for memory in store.list_approved_memories()
        if memory.status == MemoryStatus.APPROVED
    ][:limit]
    if not memories:
        return ""

    lines = ["已确认的个人长期记忆："]
    for memory in memories:
        lines.append(
            f"- [{memory.memory_id}] {memory.content} "
            f"(type={memory.memory_type.value}, confidence={memory.confidence:.2f})"
        )
    return "\n".join(lines)
