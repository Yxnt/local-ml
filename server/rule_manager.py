"""Rule Manager - learned rules with priority, conflict resolution, and lifecycle.

Stores rules as memories with MemoryType.RULE. Each rule carries metadata
for priority scoring, usage tracking, and soft-deletion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from memory.manager import MemoryManager
from memory.memory import MemoryType


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RuleExample:
    """A concrete input/output pair that demonstrates the rule."""
    input: str
    expected_output: str
    actual_output: str
    was_correct: bool


@dataclass
class LearningRule:
    """A learned rule stored as a memory with extended metadata."""
    id: int | None = None
    rule_type: str = ""
    pattern: str = ""
    logic: str = ""
    confidence: float = 0.5
    source: str = ""          # user_taught | remote_feedback | local_pattern
    created_at: str = ""
    updated_at: str = ""
    usage_count: int = 0
    success_count: int = 0
    status: str = "active"    # active | archived
    examples: list[RuleExample] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Priority map (source -> base priority)
# ---------------------------------------------------------------------------

SOURCE_PRIORITY: dict[str, int] = {
    "user_taught": 100,
    "remote_feedback": 80,
    "local_pattern": 60,
}


# ---------------------------------------------------------------------------
# RuleManager
# ---------------------------------------------------------------------------

class RuleManager:
    """Manages learned rules backed by the memory system.

    Usage:
        with MemoryManager() as mm:
            rm = RuleManager(mm)
            rm.add_rule("formatting", "Always use markdown headers", source="user_taught")
            rules = rm.get_relevant_rules("how to format output")
    """

    EXPIRY_DAYS = 90

    def __init__(self, memory_manager: MemoryManager):
        self._mm = memory_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_rule(
        self,
        rule_type: str,
        logic: str,
        confidence: float = 0.5,
        source: str = "local_pattern",
        example: RuleExample | None = None,
        pattern: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a new rule and return its memory id."""
        now = datetime.now().isoformat()
        rule = LearningRule(
            rule_type=rule_type,
            pattern=pattern,
            logic=logic,
            confidence=confidence,
            source=source,
            created_at=now,
            updated_at=now,
            usage_count=0,
            success_count=0,
            status="active",
            examples=[example] if example else [],
            metadata=metadata or {},
        )

        rule_meta = self._rule_to_metadata(rule)
        memory_id = self._mm.remember(
            content=logic,
            memory_type=MemoryType.RULE,
            importance=confidence,
            metadata=rule_meta,
        )
        return memory_id

    def get_relevant_rules(self, context: str, limit: int = 20) -> list[LearningRule]:
        """Get active rules relevant to *context*, deduplicated by rule_type, sorted by priority.

        Priority score = source_base_priority + success_rate * 30 + log(usage_count + 1) * 5

        Searches using individual keywords from the context for broader recall,
        then falls back to all RULE memories if keyword search yields nothing.
        """
        seen_ids: set[int] = set()
        raw: list[dict[str, Any]] = []

        # Try keyword-level search for broader recall.
        keywords = [w for w in context.split() if len(w) >= 2]
        for kw in keywords:
            for entry in self._mm.recall(kw, memory_type=MemoryType.RULE, limit=limit):
                eid = entry.get("id")
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    raw.append(entry)

        # Fallback: if keyword search found nothing, grab all rule memories.
        if not raw:
            store = self._mm.store
            rows = store._conn.execute(
                "SELECT * FROM memories WHERE type = ? ORDER BY importance DESC, created_at DESC LIMIT ?",
                (MemoryType.RULE.value, limit * 3),
            ).fetchall()
            for row in rows:
                entry = {
                    "id": row["id"],
                    "content": row["content"],
                    "importance": row["importance"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                raw.append(entry)

        rules: list[LearningRule] = []
        for entry in raw:
            meta = entry.get("metadata", {})
            if isinstance(meta, str):
                meta = json.loads(meta) if meta else {}
            if meta.get("status", "active") != "active":
                continue
            rules.append(self._metadata_to_rule(entry.get("id"), entry.get("content", ""), meta))

        # Deduplicate by rule_type, keeping highest-priority rule per type.
        by_type: dict[str, LearningRule] = {}
        for rule in rules:
            key = rule.rule_type
            if key not in by_type or self._score(rule) > self._score(by_type[key]):
                by_type[key] = rule

        deduped = list(by_type.values())
        deduped.sort(key=lambda r: self._score(r), reverse=True)
        return deduped[:limit]

    def update_rule_stats(self, rule_id: int, confidence: float | None = None) -> None:
        """Increment usage_count; if confidence is provided, also increment success_count."""
        store = self._mm.store
        memory = store.get_memory(rule_id)
        if not memory or memory.type != MemoryType.RULE:
            return

        meta = memory.metadata
        if isinstance(meta, str):
            meta = json.loads(meta) if meta else {}

        meta["usage_count"] = meta.get("usage_count", 0) + 1
        if confidence is not None:
            meta["success_count"] = meta.get("success_count", 0) + 1
            meta["confidence"] = confidence
        meta["updated_at"] = datetime.now().isoformat()

        # Persist via update_memory (content unchanged, importance tracks confidence).
        store.update_memory(
            rule_id,
            importance=meta.get("confidence", memory.importance),
        )
        # Write back metadata directly since MemoryStore.update_memory doesn't touch metadata.
        store._conn.execute(
            "UPDATE memories SET metadata = ?, updated_at = ? WHERE id = ?",
            (json.dumps(meta), meta["updated_at"], rule_id),
        )
        store._conn.commit()

    def archive_rule(self, rule_id: int) -> None:
        """Soft-delete a rule by setting its status to 'archived'."""
        store = self._mm.store
        memory = store.get_memory(rule_id)
        if not memory or memory.type != MemoryType.RULE:
            return

        meta = memory.metadata
        if isinstance(meta, str):
            meta = json.loads(meta) if meta else {}

        meta["status"] = "archived"
        meta["updated_at"] = datetime.now().isoformat()

        store._conn.execute(
            "UPDATE memories SET metadata = ?, updated_at = ? WHERE id = ?",
            (json.dumps(meta), meta["updated_at"], rule_id),
        )
        store._conn.commit()

    def cleanup_expired_rules(self) -> int:
        """Archive rules that haven't been used in EXPIRY_DAYS. Returns count archived."""
        store = self._mm.store
        cutoff = (datetime.now() - timedelta(days=self.EXPIRY_DAYS)).isoformat()

        rows = store._conn.execute(
            "SELECT id, metadata FROM memories WHERE type = ?",
            (MemoryType.RULE.value,),
        ).fetchall()

        archived = 0
        for row in rows:
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            if meta.get("status") != "active":
                continue
            updated_at = meta.get("updated_at", meta.get("created_at", ""))
            if updated_at and updated_at < cutoff:
                meta["status"] = "archived"
                meta["updated_at"] = datetime.now().isoformat()
                store._conn.execute(
                    "UPDATE memories SET metadata = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(meta), meta["updated_at"], row["id"]),
                )
                archived += 1

        if archived:
            store._conn.commit()
        return archived

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score(rule: LearningRule) -> float:
        """Compute composite priority score for a rule."""
        import math

        base = SOURCE_PRIORITY.get(rule.source, 50)
        success_rate = (rule.success_count / rule.usage_count) if rule.usage_count > 0 else 0.0
        usage_bonus = math.log(rule.usage_count + 1) * 5
        return base + success_rate * 30 + usage_bonus

    @staticmethod
    def _rule_to_metadata(rule: LearningRule) -> dict[str, Any]:
        """Convert a LearningRule to a flat metadata dict for storage."""
        return {
            "rule_type": rule.rule_type,
            "pattern": rule.pattern,
            "confidence": rule.confidence,
            "source": rule.source,
            "created_at": rule.created_at,
            "updated_at": rule.updated_at,
            "usage_count": rule.usage_count,
            "success_count": rule.success_count,
            "status": rule.status,
            "examples": [
                {
                    "input": ex.input,
                    "expected_output": ex.expected_output,
                    "actual_output": ex.actual_output,
                    "was_correct": ex.was_correct,
                }
                for ex in rule.examples
            ],
            **rule.metadata,
        }

    @staticmethod
    def _metadata_to_rule(memory_id: int | None, content: str, meta: dict[str, Any]) -> LearningRule:
        """Reconstruct a LearningRule from stored metadata."""
        examples_raw = meta.get("examples", [])
        examples = [RuleExample(**ex) for ex in examples_raw] if examples_raw else []
        return LearningRule(
            id=memory_id,
            rule_type=meta.get("rule_type", ""),
            pattern=meta.get("pattern", ""),
            logic=content,
            confidence=meta.get("confidence", 0.5),
            source=meta.get("source", "local_pattern"),
            created_at=meta.get("created_at", ""),
            updated_at=meta.get("updated_at", ""),
            usage_count=meta.get("usage_count", 0),
            success_count=meta.get("success_count", 0),
            status=meta.get("status", "active"),
            examples=examples,
            metadata={k: v for k, v in meta.items() if k not in {
                "rule_type", "pattern", "confidence", "source",
                "created_at", "updated_at", "usage_count", "success_count",
                "status", "examples",
            }},
        )
