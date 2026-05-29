# Hybrid Learning Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hybrid learning agent that combines local model real-time processing with remote model async learning, featuring session-level sanitization, dynamic rule management, and confidence-based learning triggers.

**Architecture:** Strategy pattern with 6 components: SessionSanitizer (privacy), RuleManager (knowledge), ConfidenceEvaluator (decision), LocalProcessor (inference), RemoteAnalyzer (learning), LearningController (orchestration). HybridAgent ties them together and inherits tool dispatch from existing Agent.

**Tech Stack:** Python (FastAPI, asyncio, sqlite-vec), existing backends (MLX models), existing memory system (MemoryManager), existing sanitizer (DataSanitizer)

---

## File Structure

| File | Role |
|------|------|
| `server/session_sanitizer.py` | Session-level sanitizer with consistent entity mapping |
| `server/rule_manager.py` | Rule storage, retrieval, priority, lifecycle |
| `server/confidence_evaluator.py` | Multi-factor confidence scoring |
| `server/local_processor.py` | Local model inference with rules and memory |
| `server/remote_analyzer.py` | Remote LLM analysis for learning |
| `server/learning_controller.py` | Learning trigger, budget, concurrency control |
| `server/hybrid_agent.py` | Main entry point, orchestrates all components |
| `memory/memory.py` | Extended with RULE MemoryType |
| `tests/python/test_session_sanitizer.py` | 5 tests |
| `tests/python/test_rule_manager.py` | 5 tests |
| `tests/python/test_confidence_evaluator.py` | 5 tests |
| `tests/python/test_local_processor.py` | 5 tests |
| `tests/python/test_remote_analyzer.py` | 3 tests |
| `tests/python/test_learning_controller.py` | 3 tests |
| `tests/python/test_hybrid_agent.py` | 4 tests |

---

### Task 1: Extend memory/memory.py with RULE type

**Files:**
- Modify: `memory/memory.py:22-27`

- [ ] **Step 1: Add RULE to MemoryType enum**

```python
class MemoryType(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    EXPERIENCE = "experience"
    CONVERSATION = "conversation"
    RULE = "rule"  # NEW
```

- [ ] **Step 2: Run existing memory tests**

Run: `python -m pytest memory/test_memory.py -v`
Expected: All existing tests PASS (no regressions)

- [ ] **Step 3: Commit**

```bash
git add memory/memory.py
git commit -m "feat: add RULE MemoryType to memory system"
```

---

### Task 2: Create SessionSanitizer

**Files:**
- Create: `server/session_sanitizer.py`
- Create: `tests/python/test_session_sanitizer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/python/test_session_sanitizer.py
import pytest
from server.session_sanitizer import SessionSanitizer


class TestSessionSanitizer:
    def test_consistent_entity_mapping(self):
        """Same entity gets same placeholder across calls."""
        s = SessionSanitizer()
        r1 = s.sanitize("张总说下周要开会")
        r2 = s.sanitize("张总要项目报告")
        assert "[PERSON_0]" in r1.sanitized
        assert "[PERSON_0]" in r2.sanitized
        assert r1.mapping["[PERSON_0]"] == r2.mapping["[PERSON_0]"]

    def test_multiple_entities_different_placeholders(self):
        """Different entities get different placeholders."""
        s = SessionSanitizer()
        r = s.sanitize("张总和王总都来了")
        assert "[PERSON_0]" in r.sanitized
        assert "[PERSON_1]" in r.sanitized
        assert r.mapping["[PERSON_0]"] != r.mapping["[PERSON_1]"]

    def test_desanitize_restores_original(self):
        """desanitize() restores placeholders to original text."""
        s = SessionSanitizer()
        r = s.sanitize("张总的邮箱是zhang@test.com")
        restored = s.desanitize(r.sanitized)
        assert "张总" in restored
        assert "zhang@test.com" in restored

    def test_get_sanitized_history_uses_same_mapping(self):
        """History sanitization reuses existing entity mappings."""
        s = SessionSanitizer()
        s.sanitize("张总说下周要开会")
        history = ["张总昨天说要项目报告"]
        sanitized = s.get_sanitized_history(history)
        assert "[PERSON_0]" in sanitized[0]

    def test_get_entity_types_only(self):
        """get_entity_types_only returns types without names."""
        s = SessionSanitizer()
        s.sanitize("张总和项目A")
        types = s.get_entity_types_only()
        assert len(types) >= 1
        for t in types:
            assert "type" in t
            assert "placeholder" in t
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/python/test_session_sanitizer.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement SessionSanitizer**

```python
# server/session_sanitizer.py
"""Session-level sanitizer with consistent entity mapping.

Ensures the same entity gets the same placeholder across all
sanitization calls within a session, so remote LLM sees consistent data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from server.sanitizer import SanitizedText, SensitivityLevel, SensitivityDetector


@dataclass
class EntityMapping:
    """Mapping for a single entity."""
    original: str
    placeholder: str
    entity_type: str  # PERSON, THING, PLACE, TOPIC
    created_at: datetime


class SessionSanitizer:
    """Session-level sanitizer - consistent entity mapping within a session."""

    # Chinese name pattern: 2-4 characters, common surname + given name
    _NAME_PATTERN = re.compile(
        r'(?:[一-十]{1})(?:[一-龥]{1,3})'
        r'(?=[\s,，。.、：:；;！!？?\n]|$)'
    )

    # Thing/topic patterns (quoted text, project names)
    _THING_PATTERN = re.compile(
        r'["“]([^"”]{2,20})["”]'
    )

    def __init__(self) -> None:
        self._entity_to_placeholder: dict[str, EntityMapping] = {}
        self._placeholder_to_entity: dict[str, EntityMapping] = {}
        self._counters: dict[str, int] = {
            "PERSON": 0, "THING": 0, "PLACE": 0, "TOPIC": 0
        }
        self._structured_detector = SensitivityDetector()

    def sanitize(self, text: str) -> SanitizedText:
        """Sanitize text, reusing existing entity mappings."""
        sanitized = text
        mapping: dict[str, str] = {}

        # 1. Replace known entities first
        for original, em in sorted(
            self._entity_to_placeholder.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        ):
            if original in sanitized:
                sanitized = sanitized.replace(original, em.placeholder)
                mapping[em.placeholder] = original

        # 2. Detect and map new person names
        for match in self._NAME_PATTERN.finditer(text):
            name = match.group(0)
            if name not in self._entity_to_placeholder and name in sanitized:
                placeholder = self._make_placeholder("PERSON")
                em = EntityMapping(name, placeholder, "PERSON", datetime.now())
                self._entity_to_placeholder[name] = em
                self._placeholder_to_entity[placeholder] = em
                sanitized = sanitized.replace(name, placeholder)
                mapping[placeholder] = name

        # 3. Detect and map new things (quoted text)
        for match in self._THING_PATTERN.finditer(text):
            thing = match.group(1)
            if thing not in self._entity_to_placeholder and thing in sanitized:
                placeholder = self._make_placeholder("THING")
                em = EntityMapping(thing, placeholder, "THING", datetime.now())
                self._entity_to_placeholder[thing] = em
                self._placeholder_to_entity[placeholder] = em
                sanitized = sanitized.replace(thing, placeholder)
                mapping[placeholder] = thing

        # 4. Sanitize structured data (email, phone, etc.)
        result = self._structured_detector.detect(sanitized)
        for match_type, original, placeholder in result.matches:
            if original in sanitized:
                sanitized = sanitized.replace(original, placeholder, 1)
                mapping[placeholder] = original

        return SanitizedText(
            original=text,
            sanitized=sanitized,
            mapping=mapping,
            level=SensitivityLevel.PERSONAL,
        )

    def desanitize(self, text: str) -> str:
        """Restore placeholders to original entity names."""
        result = text
        for placeholder, em in sorted(
            self._placeholder_to_entity.items(),
            key=lambda x: len(x[0]),
            reverse=True,
        ):
            result = result.replace(placeholder, em.original)
        return result

    def get_sanitized_history(self, history: list[str]) -> list[str]:
        """Sanitize history messages using the same entity mappings."""
        return [self.sanitize(msg).sanitized for msg in history]

    def get_entity_types_only(self) -> list[dict[str, str]]:
        """Return entity types without names (for sending to remote)."""
        seen = set()
        result = []
        for em in self._entity_to_placeholder.values():
            if em.placeholder not in seen:
                seen.add(em.placeholder)
                result.append({"type": em.entity_type, "placeholder": em.placeholder})
        return result

    def _make_placeholder(self, entity_type: str) -> str:
        idx = self._counters[entity_type]
        self._counters[entity_type] += 1
        return f"[{entity_type}_{idx}]"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/python/test_session_sanitizer.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add server/session_sanitizer.py tests/python/test_session_sanitizer.py
git commit -m "feat: add SessionSanitizer with consistent entity mapping"
```

---

### Task 3: Create RuleManager

**Files:**
- Create: `server/rule_manager.py`
- Create: `tests/python/test_rule_manager.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/python/test_rule_manager.py
import pytest
from server.rule_manager import RuleManager, LearningRule
from memory.memory import MemoryManager, MemoryType


@pytest.fixture
def memory(tmp_path):
    m = MemoryManager(data_dir=str(tmp_path / "data"))
    m.connect()
    yield m
    m.disconnect()


@pytest.fixture
def manager(memory):
    return RuleManager(memory)


class TestRuleManager:
    @pytest.mark.asyncio
    async def test_add_and_retrieve_rule(self, manager):
        rule = await manager.add_rule(
            rule_type="pronoun_resolution",
            logic="'那个东西'指代最近讨论的话题",
            confidence=0.8,
        )
        assert rule.id.startswith("rule_")
        rules = await manager.get_relevant_rules("那个东西")
        assert len(rules) > 0

    @pytest.mark.asyncio
    async def test_dynamic_rule_types(self, manager):
        rule = await manager.add_rule(
            rule_type="location_query",
            logic="当用户提到吃饭时搜索地点",
            confidence=0.8,
            source="remote_feedback",
        )
        assert rule.rule_type == "location_query"

    @pytest.mark.asyncio
    async def test_rule_priority_ordering(self, manager):
        await manager.add_rule("test", "规则A", 0.9, source="remote_feedback")
        await manager.add_rule("test", "规则B", 0.7, source="local_pattern")
        rules = await manager.get_relevant_rules("test")
        if len(rules) >= 2:
            assert rules[0].confidence >= rules[1].confidence

    @pytest.mark.asyncio
    async def test_rule_deactivation(self, manager):
        rule = await manager.add_rule("test", "bad rule", 0.5)
        # Simulate multiple low-confidence uses
        for _ in range(6):
            await manager.update_rule_stats(rule.id, confidence=0.3)
        # Rule should be deactivated
        rules = await manager.get_relevant_rules("test")
        active = [r for r in rules if r.status == "active"]
        assert len(active) == 0

    @pytest.mark.asyncio
    async def test_only_active_rules_returned(self, manager):
        rule = await manager.add_rule("test", "archived rule", 0.5)
        await manager.archive_rule(rule.id)
        rules = await manager.get_relevant_rules("test")
        assert len(rules) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/python/test_rule_manager.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement RuleManager**

```python
# server/rule_manager.py
"""Rule storage, retrieval, priority, and lifecycle management."""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from memory.memory import MemoryManager, MemoryType

logger = logging.getLogger(__name__)


@dataclass
class RuleExample:
    input: str
    expected_output: str
    actual_output: str | None = None
    was_correct: bool = False


@dataclass
class LearningRule:
    id: str
    rule_type: str
    pattern: str
    logic: str
    confidence: float
    source: str  # remote_feedback, local_pattern, user_taught
    created_at: datetime
    updated_at: datetime
    usage_count: int = 0
    success_count: int = 0
    status: str = "active"  # active, inactive, archived
    examples: list[RuleExample] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class RuleManager:
    """Manages learned rules with priority, conflict resolution, and lifecycle."""

    def __init__(self, memory: MemoryManager, config: dict[str, Any] | None = None):
        self._memory = memory
        self._config = config or {}
        self._expire_days = self._config.get("expire_days", 90)
        self._min_success_rate = self._config.get("min_success_rate", 0.3)
        self._min_usage_for_deactivation = 5

    async def add_rule(
        self,
        rule_type: str,
        logic: str,
        confidence: float,
        source: str = "remote_feedback",
        example: dict[str, Any] | None = None,
    ) -> LearningRule:
        """Add a new rule of any type."""
        now = datetime.now()
        rule_id = f"rule_{uuid.uuid4().hex[:8]}"

        rule_data = {
            "rule_id": rule_id,
            "rule_type": rule_type,
            "pattern": self._extract_pattern(logic),
            "logic": logic,
            "confidence": confidence,
            "source": source,
            "usage_count": 0,
            "success_count": 0,
            "status": "active",
            "examples": [example] if example else [],
        }

        await self._memory.remember(
            content=logic,
            memory_type=MemoryType.RULE,
            metadata=rule_data,
        )

        return LearningRule(
            id=rule_id,
            rule_type=rule_type,
            pattern=rule_data["pattern"],
            logic=logic,
            confidence=confidence,
            source=source,
            created_at=now,
            updated_at=now,
        )

    async def get_relevant_rules(self, context: str) -> list[LearningRule]:
        """Get relevant rules, deduplicated by type, sorted by priority."""
        keywords = self._extract_keywords(context)
        if not keywords:
            return []

        raw_memories = await self._memory.search(
            query=" ".join(keywords),
            memory_type=MemoryType.RULE,
            limit=20,
        )

        rules = []
        for mem in raw_memories:
            meta = mem.metadata or {}
            if meta.get("status") != "active":
                continue
            rules.append(LearningRule(
                id=meta.get("rule_id", str(mem.id)),
                rule_type=meta.get("rule_type", "general"),
                pattern=meta.get("pattern", ""),
                logic=mem.content,
                confidence=meta.get("confidence", 0.5),
                source=meta.get("source", "unknown"),
                created_at=datetime.fromisoformat(mem.created_at) if mem.created_at else datetime.now(),
                updated_at=datetime.fromisoformat(mem.updated_at) if mem.updated_at else datetime.now(),
                usage_count=meta.get("usage_count", 0),
                success_count=meta.get("success_count", 0),
                status=meta.get("status", "active"),
            ))

        rules.sort(key=lambda r: self._get_priority(r), reverse=True)

        # Deduplicate by rule_type, keep highest priority
        seen_types: set[str] = set()
        deduplicated = []
        for rule in rules:
            if rule.rule_type not in seen_types:
                seen_types.add(rule.rule_type)
                deduplicated.append(rule)

        return deduplicated[:10]

    async def update_rule_stats(self, rule_id: str, confidence: float) -> None:
        """Update rule usage/success counts based on confidence."""
        memories = await self._memory.search(
            query=rule_id,
            memory_type=MemoryType.RULE,
            limit=1,
        )
        if not memories:
            return

        mem = memories[0]
        meta = dict(mem.metadata or {})
        meta["usage_count"] = meta.get("usage_count", 0) + 1
        if confidence > 0.8:
            meta["success_count"] = meta.get("success_count", 0) + 1

        # Deactivate if success rate too low
        usage = meta["usage_count"]
        success = meta["success_count"]
        if usage >= self._min_usage_for_deactivation:
            rate = success / usage
            if rate < self._min_success_rate:
                meta["status"] = "inactive"
                logger.info("Deactivated rule %s (success rate %.2f)", rule_id, rate)

        await self._memory.update_memory(mem.id, metadata=meta)

    async def archive_rule(self, rule_id: str) -> None:
        """Archive a rule (soft delete)."""
        memories = await self._memory.search(
            query=rule_id,
            memory_type=MemoryType.RULE,
            limit=1,
        )
        if memories:
            meta = dict(memories[0].metadata or {})
            meta["status"] = "archived"
            await self._memory.update_memory(memories[0].id, metadata=meta)

    async def cleanup_expired_rules(self) -> int:
        """Archive rules unused for expire_days. Returns count archived."""
        now = datetime.now()
        all_rules = await self._memory.search(
            query="",
            memory_type=MemoryType.RULE,
            limit=1000,
        )
        count = 0
        for mem in all_rules:
            meta = mem.metadata or {}
            if meta.get("status") != "active":
                continue
            updated = datetime.fromisoformat(mem.updated_at) if mem.updated_at else datetime.now()
            if now - updated > timedelta(days=self._expire_days):
                meta["status"] = "archived"
                await self._memory.update_memory(mem.id, metadata=meta)
                count += 1
        return count

    def _get_priority(self, rule: LearningRule) -> float:
        """Calculate rule priority with success rate + usage weighting."""
        base = {"user_taught": 100, "remote_feedback": 80, "local_pattern": 60}.get(
            rule.source, 40
        )
        if rule.usage_count == 0:
            return base * 0.6
        trust = min(rule.usage_count / 10, 1.0)
        success_rate = rule.success_count / rule.usage_count
        effective = success_rate * trust + 0.5 * (1 - trust)
        return base * effective

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords for rule matching."""
        keywords = re.findall(r'[一-龥]{2,4}', text)
        keywords.extend(re.findall(r'[a-zA-Z]+', text))
        return keywords[:10]

    def _extract_pattern(self, logic: str) -> str:
        """Extract a human-readable pattern from rule logic."""
        # Simple: first sentence or first 50 chars
        first_line = logic.split("\n")[0].strip()
        return first_line[:50] if len(first_line) > 50 else first_line
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/python/test_rule_manager.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add server/rule_manager.py tests/python/test_rule_manager.py
git commit -m "feat: add RuleManager with priority, lifecycle, conflict resolution"
```

---

### Task 4: Create ConfidenceEvaluator

**Files:**
- Create: `server/confidence_evaluator.py`
- Create: `tests/python/test_confidence_evaluator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/python/test_confidence_evaluator.py
import pytest
from server.confidence_evaluator import ConfidenceEvaluator


class TestConfidenceEvaluator:
    def setup_method(self):
        self.evaluator = ConfidenceEvaluator()

    def test_simple_query_high_confidence(self):
        """Simple query with no pronouns/time/ambiguity → high confidence."""
        score = self.evaluator.evaluate(
            user_input="今天天气怎么样",
            response={"pronouns": {}, "time": None, "has_ambiguity": False},
            rules=[],
            memories=["天气不错"],
        )
        assert score > 0.7

    def test_pronoun_unresolved_low_confidence(self):
        """Unresolved pronoun → lower confidence."""
        score = self.evaluator.evaluate(
            user_input="他说什么了",
            response={"pronouns": {"[PERSON_0]": None}, "time": None, "has_ambiguity": False},
            rules=[],
            memories=[],
        )
        assert score < 0.7

    def test_time_expression_reduces_confidence(self):
        """Time expression detected → slightly lower confidence."""
        score_no_time = self.evaluator.evaluate(
            user_input="今天开会吗",
            response={"pronouns": {}, "time": None, "has_ambiguity": False},
            rules=[],
            memories=[],
        )
        score_with_time = self.evaluator.evaluate(
            user_input="上周开会了吗",
            response={"pronouns": {}, "time": {"expression": "上周"}, "has_ambiguity": False},
            rules=[],
            memories=[],
        )
        assert score_no_time > score_with_time

    def test_ambiguity_gives_low_confidence(self):
        """Ambiguity detected → low confidence."""
        score = self.evaluator.evaluate(
            user_input="那个东西",
            response={"pronouns": {}, "time": None, "has_ambiguity": True},
            rules=[],
            memories=[],
        )
        assert score < 0.5

    def test_with_rules_boosts_confidence(self):
        """Matching rules → higher confidence."""
        from server.rule_manager import LearningRule
        from datetime import datetime

        rule = LearningRule(
            id="rule_test", rule_type="test", pattern="test",
            logic="test", confidence=0.9, source="remote_feedback",
            created_at=datetime.now(), updated_at=datetime.now(),
        )
        score = self.evaluator.evaluate(
            user_input="test query",
            response={"pronouns": {}, "time": None, "has_ambiguity": False},
            rules=[rule],
            memories=[],
        )
        assert score > 0.7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/python/test_confidence_evaluator.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement ConfidenceEvaluator**

```python
# server/confidence_evaluator.py
"""Multi-factor confidence scoring for local model responses."""

from __future__ import annotations

from typing import Any


class ConfidenceEvaluator:
    """Evaluate confidence of local model processing."""

    # Weights for each factor (must sum to 1.0)
    WEIGHTS = {
        "rule": 0.30,
        "pronoun": 0.20,
        "time": 0.15,
        "memory": 0.20,
        "ambiguity": 0.15,
    }

    def evaluate(
        self,
        user_input: str,
        response: dict[str, Any],
        rules: list[Any],
        memories: list[Any],
    ) -> float:
        """Evaluate confidence as weighted average of factors."""
        scores: list[tuple[str, float]] = []

        # 1. Rule matching
        if rules:
            rule_score = max(getattr(r, "confidence", 0.5) for r in rules)
        else:
            rule_score = 0.5
        scores.append(("rule", rule_score))

        # 2. Pronoun resolution
        pronouns = response.get("pronouns", {})
        if pronouns:
            resolved = sum(1 for v in pronouns.values() if v)
            pronoun_score = 0.8 if resolved == len(pronouns) else 0.4
        else:
            pronoun_score = 0.9  # No pronouns = no problem
        scores.append(("pronoun", pronoun_score))

        # 3. Time parsing
        time_data = response.get("time")
        time_score = 0.8 if time_data else 0.9
        scores.append(("time", time_score))

        # 4. Memory matching
        memory_score = 0.7 if memories else 0.5
        scores.append(("memory", memory_score))

        # 5. Ambiguity
        has_ambiguity = response.get("has_ambiguity", False)
        ambiguity_score = 0.3 if has_ambiguity else 0.9
        scores.append(("ambiguity", ambiguity_score))

        total = sum(score * self.WEIGHTS[name] for name, score in scores)
        return max(0.0, min(1.0, total))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/python/test_confidence_evaluator.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add server/confidence_evaluator.py tests/python/test_confidence_evaluator.py
git commit -m "feat: add ConfidenceEvaluator with multi-factor scoring"
```

---

### Task 5: Create LocalProcessor

**Files:**
- Create: `server/local_processor.py`
- Create: `tests/python/test_local_processor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/python/test_local_processor.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from server.local_processor import LocalProcessor
from server.rule_manager import RuleManager, LearningRule
from memory.memory import MemoryManager, MemoryType
from datetime import datetime


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.generate_async = AsyncMock(return_value='{"answer": "test answer", "reasoning": "test", "entities": [], "pronouns": {}, "time": null, "has_ambiguity": false}')
    return model


@pytest.fixture
def memory(tmp_path):
    m = MemoryManager(data_dir=str(tmp_path / "data"))
    m.connect()
    yield m
    m.disconnect()


@pytest.fixture
def rule_manager(memory):
    return RuleManager(memory)


class TestLocalProcessor:
    @pytest.mark.asyncio
    async def test_process_returns_local_result(self, mock_model, memory, rule_manager):
        processor = LocalProcessor(mock_model, memory, rule_manager)
        result = await processor.process("今天天气怎么样")
        assert result.answer == "test answer"
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_process_with_rules(self, mock_model, memory, rule_manager):
        await rule_manager.add_rule("test", "test rule", 0.9)
        processor = LocalProcessor(mock_model, memory, rule_manager)
        result = await processor.process("test query")
        assert len(result.rules_applied) >= 0

    @pytest.mark.asyncio
    async def test_extract_pronoun_pattern(self, mock_model, memory, rule_manager):
        processor = LocalProcessor(mock_model, memory, rule_manager)
        from server.local_processor import LocalResult
        result = LocalResult(
            answer="test", confidence=0.95, reasoning="",
            entities_found=[], pronouns_resolved={"[PERSON_0]": "张总"},
            time_parsed=None, rules_applied=[], ambiguity_detected=False,
            needs_remote_help=False,
        )
        pattern = processor._extract_success_pattern("他说什么了", result)
        assert pattern is not None
        assert pattern["type"] == "pronoun_resolution"

    @pytest.mark.asyncio
    async def test_parse_response_json(self, mock_model, memory, rule_manager):
        processor = LocalProcessor(mock_model, memory, rule_manager)
        parsed = processor._parse_response('{"answer": "hi", "reasoning": "r", "entities": [], "pronouns": {}, "time": null, "has_ambiguity": false}')
        assert parsed["answer"] == "hi"

    @pytest.mark.asyncio
    async def test_parse_response_invalid_json(self, mock_model, memory, rule_manager):
        processor = LocalProcessor(mock_model, memory, rule_manager)
        parsed = processor._parse_response("not json at all")
        assert parsed["has_ambiguity"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/python/test_local_processor.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement LocalProcessor**

```python
# server/local_processor.py
"""Local model processor with rules, memory, and confidence evaluation."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from server.confidence_evaluator import ConfidenceEvaluator
from server.rule_manager import LearningRule, RuleManager
from memory.memory import MemoryManager, MemoryType

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    name: str
    type: str
    confidence: float
    source: str


@dataclass
class TimeRange:
    start: datetime
    end: datetime
    expression: str


@dataclass
class LocalResult:
    answer: str
    confidence: float
    reasoning: str
    entities_found: list[Entity]
    pronouns_resolved: dict[str, str]
    time_parsed: TimeRange | None
    rules_applied: list[str]
    ambiguity_detected: bool
    needs_remote_help: bool


class LocalProcessor:
    """Processes user input using local model with rules and memory."""

    def __init__(self, model: Any, memory: MemoryManager, rule_manager: RuleManager):
        self._model = model
        self._memory = memory
        self._rule_manager = rule_manager
        self._confidence = ConfidenceEvaluator()

    async def process(self, user_input: str) -> LocalResult:
        """Process user input and return structured result."""
        # 1. Get relevant rules
        rules = await self._rule_manager.get_relevant_rules(user_input)

        # 2. Get relevant memories
        memories = await self._memory.recall(user_input, limit=5)

        # 3. Get recent conversation history
        history = await self._memory.get_recent_conversations(limit=5)

        # 4. Build prompt
        prompt = self._build_prompt(user_input, rules, memories, history)

        # 5. Generate response
        response_text = await self._model.generate_async(prompt)

        # 6. Parse JSON response
        parsed = self._parse_response(response_text)

        # 7. Evaluate confidence
        confidence = self._confidence.evaluate(
            user_input=user_input,
            response=parsed,
            rules=rules,
            memories=memories,
        )

        result = LocalResult(
            answer=parsed.get("answer", ""),
            confidence=confidence,
            reasoning=parsed.get("reasoning", ""),
            entities_found=[
                Entity(**e) for e in parsed.get("entities", [])
            ],
            pronouns_resolved=parsed.get("pronouns", {}),
            time_parsed=self._parse_time(parsed.get("time")),
            rules_applied=[r.id for r in rules],
            ambiguity_detected=parsed.get("has_ambiguity", False),
            needs_remote_help=confidence < 0.9,
        )

        # 8. Extract and store pattern if successful
        if confidence > 0.9:
            await self._extract_and_store_pattern(user_input, result)

        return result

    def _build_prompt(
        self,
        user_input: str,
        rules: list[LearningRule],
        memories: list[Any],
        history: list[dict],
    ) -> str:
        """Build prompt with rules, memory, and history context."""
        rules_text = "\n".join(f"- {r.logic}" for r in rules[:5]) if rules else "无"
        memories_text = "\n".join(f"- {m.content}" for m in memories[:5]) if memories else "无"
        history_text = "\n".join(
            f"{h.get('role', 'user')}: {h.get('content', '')}" for h in history[-5:]
        ) if history else "无"

        return f"""你是一个个人 AI 助手。请基于以下信息处理用户输入。

## 当前时间
{datetime.now().isoformat()}

## 用户输入
{user_input}

## 相关规则 (从历史学习)
{rules_text}

## 相关记忆
{memories_text}

## 最近对话
{history_text}

## 处理要求
1. 识别代词 (他/她/它/这个/那个) 并根据上下文解析
2. 识别时间表达 (上周/昨天/三天前) 并转换为具体日期
3. 从记忆中查找相关信息
4. 生成简洁的回答

## 输出格式
请以 JSON 格式返回:
{{"answer": "你的回答", "reasoning": "推理过程", "entities": [{{"name": "实体名", "type": "person/thing", "confidence": 0.9, "source": "memory/history"}}], "pronouns": {{"[PERSON_0]": "张总"}}, "time": {{"expression": "上周", "start": "2024-01-01", "end": "2024-01-07"}} 或 null, "has_ambiguity": false}}"""

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse JSON response from local model."""
        try:
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            pass
        return {
            "answer": response,
            "reasoning": "无法解析结构化响应",
            "entities": [],
            "pronouns": {},
            "time": None,
            "has_ambiguity": True,
        }

    def _parse_time(self, time_data: Any) -> TimeRange | None:
        """Parse time data into TimeRange."""
        if not time_data or not isinstance(time_data, dict):
            return None
        try:
            return TimeRange(
                start=datetime.fromisoformat(time_data["start"]),
                end=datetime.fromisoformat(time_data["end"]),
                expression=time_data.get("expression", ""),
            )
        except (KeyError, ValueError):
            return None

    async def _extract_and_store_pattern(self, user_input: str, result: LocalResult) -> None:
        """Extract and store successful processing patterns."""
        pattern = self._extract_success_pattern(user_input, result)
        if pattern:
            await self._rule_manager.add_rule(
                rule_type=pattern["type"],
                logic=pattern["logic"],
                confidence=0.7,
                source="local_pattern",
                example={"input": user_input, "output": result.answer},
            )

    def _extract_success_pattern(self, user_input: str, result: LocalResult) -> dict | None:
        """Extract pattern from successful processing."""
        if result.pronouns_resolved:
            return {"type": "pronoun_resolution", "logic": "当用户输入包含代词时，查找最近提到的实体"}
        if result.time_parsed:
            return {"type": "time_parsing", "logic": "当用户输入包含时间表达时，转换为具体日期范围"}
        if result.entities_found:
            return {"type": "entity_matching", "logic": "当用户输入包含实体时，从记忆中查找相关信息"}
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/python/test_local_processor.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add server/local_processor.py tests/python/test_local_processor.py
git commit -m "feat: add LocalProcessor with rules, memory, and pattern extraction"
```

---

### Task 6: Create RemoteAnalyzer

**Files:**
- Create: `server/remote_analyzer.py`
- Create: `tests/python/test_remote_analyzer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/python/test_remote_analyzer.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from server.remote_analyzer import RemoteAnalyzer, SanitizedContext
from datetime import datetime


@pytest.fixture
def mock_remote():
    remote = MagicMock()
    remote.generate_async = AsyncMock(return_value='{"is_correct": true, "confidence": 0.9, "reasoning": "test", "better_answer": null, "logic_rule": null, "rule_type": null, "corrections": []}')
    return remote


class TestRemoteAnalyzer:
    @pytest.mark.asyncio
    async def test_analyze_returns_feedback(self, mock_remote):
        analyzer = RemoteAnalyzer(mock_remote)
        context = SanitizedContext(
            user_input="test", history=[], entities=[],
            local_answer="answer", confidence=0.8, reasoning="r",
            rules_applied=[], timestamp=datetime.now(),
        )
        feedback = await analyzer.analyze(context)
        assert feedback.is_correct is True
        assert feedback.confidence == 0.9

    @pytest.mark.asyncio
    async def test_analyze_handles_invalid_json(self, mock_remote):
        mock_remote.generate_async = AsyncMock(return_value="not json")
        analyzer = RemoteAnalyzer(mock_remote)
        context = SanitizedContext(
            user_input="test", history=[], entities=[],
            local_answer="answer", confidence=0.8, reasoning="r",
            rules_applied=[], timestamp=datetime.now(),
        )
        feedback = await analyzer.analyze(context)
        assert feedback.is_correct is False
        assert feedback.confidence == 0.0

    @pytest.mark.asyncio
    async def test_analyze_extracts_rule(self, mock_remote):
        mock_remote.generate_async = AsyncMock(return_value='{"is_correct": false, "confidence": 0.85, "reasoning": "wrong", "better_answer": "correct", "logic_rule": "when user says X, do Y", "rule_type": "custom_type", "corrections": []}')
        analyzer = RemoteAnalyzer(mock_remote)
        context = SanitizedContext(
            user_input="test", history=[], entities=[],
            local_answer="wrong", confidence=0.5, reasoning="r",
            rules_applied=[], timestamp=datetime.now(),
        )
        feedback = await analyzer.analyze(context)
        assert feedback.logic_rule == "when user says X, do Y"
        assert feedback.rule_type == "custom_type"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/python/test_remote_analyzer.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement RemoteAnalyzer**

```python
# server/remote_analyzer.py
"""Remote LLM analyzer for learning from local model failures."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Correction:
    field: str
    original: str
    corrected: str
    reason: str


@dataclass
class RemoteFeedback:
    is_correct: bool
    confidence: float
    reasoning: str
    better_answer: str | None
    logic_rule: str | None
    rule_type: str | None
    corrections: list[Correction]


@dataclass
class SanitizedContext:
    user_input: str
    history: list[str]
    entities: list[dict]
    local_answer: str
    confidence: float
    reasoning: str
    rules_applied: list[str]
    timestamp: datetime


class RemoteAnalyzer:
    """Analyzes local model results using remote LLM."""

    def __init__(self, remote_backend: Any):
        self._remote = remote_backend

    async def analyze(self, context: SanitizedContext) -> RemoteFeedback:
        """Analyze local processing result via remote LLM."""
        prompt = self._build_prompt(context)
        try:
            response = await self._remote.generate_async(prompt)
            return self._parse_feedback(response)
        except Exception as e:
            logger.error("Remote analysis failed: %s", e)
            return RemoteFeedback(
                is_correct=True, confidence=0.0,
                reasoning=f"Remote analysis failed: {e}",
                better_answer=None, logic_rule=None,
                rule_type=None, corrections=[],
            )

    def _build_prompt(self, ctx: SanitizedContext) -> str:
        history_text = "\n".join(ctx.history) if ctx.history else "无"
        entities_text = json.dumps(ctx.entities, ensure_ascii=False) if ctx.entities else "无"
        rules_text = "\n".join(ctx.rules_applied) if ctx.rules_applied else "无"

        return f"""分析以下对话处理是否正确，并给出改进建议。

## 用户输入 (已脱敏)
{ctx.user_input}

## 对话历史 (已脱敏)
{history_text}

## 实体类型 (不含具体名称)
{entities_text}

## 本地回答
{ctx.local_answer}

## 本地置信度
{ctx.confidence}

## 本地推理过程
{ctx.reasoning}

## 应用的规则
{rules_text}

## 分析要求
1. 判断本地回答是否正确
2. 分析推理过程是否有逻辑错误
3. 如果有错误，给出正确的回答
4. 提取可以通用化的规则

## 输出格式
{{"is_correct": true/false, "confidence": 0.0-1.0, "reasoning": "详细分析", "better_answer": "如果本地错了，正确答案是什么", "logic_rule": "可以学习的通用规则", "rule_type": "规则类型", "corrections": [{{"field": "字段", "original": "原值", "corrected": "正确值", "reason": "原因"}}]}}

注意:
- rule_type 可以是任何你认为合适的类型
- 规则要通用化，不要针对具体实体
- 鼓励发现新的模式和规则类型"""

    def _parse_feedback(self, response: str) -> RemoteFeedback:
        """Parse remote LLM feedback JSON."""
        try:
            data = json.loads(response)
            corrections = [
                Correction(**c) for c in data.get("corrections", [])
            ]
            return RemoteFeedback(
                is_correct=data.get("is_correct", False),
                confidence=data.get("confidence", 0.5),
                reasoning=data.get("reasoning", ""),
                better_answer=data.get("better_answer"),
                logic_rule=data.get("logic_rule"),
                rule_type=data.get("rule_type"),
                corrections=corrections,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return RemoteFeedback(
                is_correct=False, confidence=0.0,
                reasoning=f"Failed to parse feedback: {e}",
                better_answer=None, logic_rule=None,
                rule_type=None, corrections=[],
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/python/test_remote_analyzer.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add server/remote_analyzer.py tests/python/test_remote_analyzer.py
git commit -m "feat: add RemoteAnalyzer for async learning from remote LLM"
```

---

### Task 7: Create LearningController

**Files:**
- Create: `server/learning_controller.py`
- Create: `tests/python/test_learning_controller.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/python/test_learning_controller.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from server.learning_controller import LearningController
from server.local_processor import LocalResult
from server.remote_analyzer import RemoteFeedback
from server.session_sanitizer import SessionSanitizer
from server.rule_manager import RuleManager
from memory.memory import MemoryManager
from datetime import datetime


@pytest.fixture
def components(tmp_path):
    memory = MemoryManager(data_dir=str(tmp_path / "data"))
    memory.connect()
    rule_manager = RuleManager(memory)
    sanitizer = SessionSanitizer()
    remote = MagicMock()
    remote.analyze = AsyncMock(return_value=RemoteFeedback(
        is_correct=False, confidence=0.85, reasoning="test",
        better_answer="better", logic_rule="when X do Y",
        rule_type="test_type", corrections=[],
    ))
    yield memory, rule_manager, sanitizer, remote
    memory.disconnect()


class TestLearningController:
    @pytest.mark.asyncio
    async def test_learn_stores_rule(self, components):
        memory, rule_manager, sanitizer, remote = components
        controller = LearningController(sanitizer, remote, rule_manager, {})
        result = LocalResult(
            answer="wrong", confidence=0.5, reasoning="r",
            entities_found=[], pronouns_resolved={}, time_parsed=None,
            rules_applied=[], ambiguity_detected=True, needs_remote_help=True,
        )
        await controller.learn("test input", result)
        rules = await rule_manager.get_relevant_rules("test")
        assert len(rules) > 0

    @pytest.mark.asyncio
    async def test_budget_blocks_learning(self, components):
        memory, rule_manager, sanitizer, remote = components
        controller = LearningController(sanitizer, remote, rule_manager, {"hourly_limit": 0})
        result = LocalResult(
            answer="test", confidence=0.5, reasoning="r",
            entities_found=[], pronouns_resolved={}, time_parsed=None,
            rules_applied=[], ambiguity_detected=True, needs_remote_help=True,
        )
        await controller.learn("test", result)
        rules = await rule_manager.get_relevant_rules("test")
        assert len(rules) == 0

    @pytest.mark.asyncio
    async def test_duplicate_learning_skipped(self, components):
        memory, rule_manager, sanitizer, remote = components
        controller = LearningController(sanitizer, remote, rule_manager, {})
        result = LocalResult(
            answer="test", confidence=0.5, reasoning="r",
            entities_found=[], pronouns_resolved={}, time_parsed=None,
            rules_applied=[], ambiguity_detected=True, needs_remote_help=True,
        )
        # First call
        await controller.learn("duplicate test", result)
        # Second call with same input
        await controller.learn("duplicate test", result)
        # Should only have 1 rule, not 2
        rules = await rule_manager.get_relevant_rules("duplicate")
        assert len(rules) <= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/python/test_learning_controller.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement LearningController**

```python
# server/learning_controller.py
"""Learning controller with budget, concurrency, and session sanitization."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Callable

from server.local_processor import LocalResult
from server.remote_analyzer import RemoteAnalyzer, SanitizedContext
from server.rule_manager import RuleManager
from server.session_sanitizer import SessionSanitizer

logger = logging.getLogger(__name__)


class LearningController:
    """Controls when and how to learn from remote LLM."""

    def __init__(
        self,
        session_sanitizer: SessionSanitizer,
        remote_analyzer: RemoteAnalyzer,
        rule_manager: RuleManager,
        config: dict[str, Any],
    ):
        self._sanitizer = session_sanitizer
        self._remote = remote_analyzer
        self._rule_manager = rule_manager
        self._config = config

        # Budget tracking
        self._hourly_count = 0
        self._daily_count = 0
        self._last_hourly_reset = datetime.now()
        self._last_daily_reset = datetime.now()
        self._hourly_limit = config.get("hourly_limit", 10)
        self._daily_limit = config.get("daily_limit", 100)
        self._min_remote_confidence = config.get("min_remote_confidence", 0.7)

        # Concurrency control
        self._lock = asyncio.Lock()
        self._pending: dict[str, bool] = {}

    async def learn(self, user_input: str, result: LocalResult) -> None:
        """Execute learning flow with budget and concurrency control."""
        key = self._make_key(user_input)

        # Skip if same learning already pending
        if key in self._pending:
            logger.debug("Skipping duplicate learning: %s", key)
            return

        # Check budget
        if not self._check_budget():
            logger.debug("Learning budget exhausted")
            return

        self._pending[key] = True
        try:
            async with self._lock:
                context = self._prepare_context(user_input, result)
                feedback = await self._remote.analyze(context)

                if feedback.logic_rule and feedback.confidence >= self._min_remote_confidence:
                    await self._rule_manager.add_rule(
                        rule_type=feedback.rule_type or "general",
                        logic=feedback.logic_rule,
                        confidence=feedback.confidence,
                        source="remote_feedback",
                        example={
                            "input": context.user_input,
                            "expected": feedback.better_answer,
                            "actual": result.answer,
                            "was_correct": feedback.is_correct,
                        },
                    )
                    logger.info("Learned new rule: %s", feedback.rule_type)

                self._hourly_count += 1
                self._daily_count += 1
        except Exception as e:
            logger.error("Learning failed: %s", e)
        finally:
            self._pending.pop(key, None)

    def _prepare_context(self, user_input: str, result: LocalResult) -> SanitizedContext:
        sanitized = self._sanitizer.sanitize(user_input)
        entity_types = self._sanitizer.get_entity_types_only()

        return SanitizedContext(
            user_input=sanitized.sanitized,
            history=[],
            entities=entity_types,
            local_answer=result.answer,
            confidence=result.confidence,
            reasoning=result.reasoning,
            rules_applied=result.rules_applied,
            timestamp=datetime.now(),
        )

    def _check_budget(self) -> bool:
        now = datetime.now()
        if now - self._last_hourly_reset > timedelta(hours=1):
            self._hourly_count = 0
            self._last_hourly_reset = now
        if now - self._last_daily_reset > timedelta(days=1):
            self._daily_count = 0
            self._last_daily_reset = now
        return self._hourly_count < self._hourly_limit and self._daily_count < self._daily_limit

    def _make_key(self, user_input: str) -> str:
        return hashlib.md5(user_input[:50].encode()).hexdigest()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/python/test_learning_controller.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add server/learning_controller.py tests/python/test_learning_controller.py
git commit -m "feat: add LearningController with budget, concurrency, sanitization"
```

---

### Task 8: Create HybridAgent (main entry point)

**Files:**
- Create: `server/hybrid_agent.py`
- Create: `tests/python/test_hybrid_agent.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/python/test_hybrid_agent.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from server.hybrid_agent import HybridAgent, AgentResponse


@pytest.fixture
def config():
    return {
        "local": {"default_model": "test-model"},
        "remote": {"enabled": True, "model": "test-remote"},
        "learning": {"enabled": True, "confidence_threshold": 0.9},
        "rules": {"expire_days": 90},
        "privacy": {"sanitize_before_remote": True},
    }


class TestHybridAgent:
    @pytest.mark.asyncio
    async def test_run_returns_agent_response(self, config):
        with patch("server.hybrid_agent.ModelRegistry"), \
             patch("server.hybrid_agent.MemoryManager"), \
             patch("server.hybrid_agent.LocalProcessor") as MockLP:
            mock_result = MagicMock()
            mock_result.answer = "test answer"
            mock_result.confidence = 0.95
            mock_result.rules_applied = []
            MockLP.return_value.process = AsyncMock(return_value=mock_result)

            agent = HybridAgent(config)
            response = await agent.run("hello")
            assert isinstance(response, AgentResponse)
            assert response.answer == "test answer"

    @pytest.mark.asyncio
    async def test_followup_question_detected(self, config):
        with patch("server.hybrid_agent.ModelRegistry"), \
             patch("server.hybrid_agent.MemoryManager"), \
             patch("server.hybrid_agent.LocalProcessor"):
            agent = HybridAgent(config)
            assert agent._is_followup_question("他说什么了") is True
            assert agent._is_followup_question("今天天气怎么样") is False

    @pytest.mark.asyncio
    async def test_run_with_progress_callback(self, config):
        with patch("server.hybrid_agent.ModelRegistry"), \
             patch("server.hybrid_agent.MemoryManager"), \
             patch("server.hybrid_agent.LocalProcessor") as MockLP:
            mock_result = MagicMock()
            mock_result.answer = "test"
            mock_result.confidence = 0.95
            mock_result.rules_applied = []
            MockLP.return_value.process = AsyncMock(return_value=mock_result)

            agent = HybridAgent(config)
            messages = []
            result = await agent.run_with_progress("hello", lambda m: messages.append(m))
            assert result == "test"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/python/test_hybrid_agent.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement HybridAgent**

```python
# server/hybrid_agent.py
"""Hybrid Learning Agent - main entry point.

Combines local model real-time processing with remote model async learning.
Inherits tool dispatch from existing Agent pattern.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable

from backends.registry import ModelRegistry
from memory.manager import MemoryManager
from memory.memory import MemoryType
from server.local_processor import LocalProcessor, LocalResult
from server.remote_analyzer import RemoteAnalyzer
from server.rule_manager import RuleManager
from server.learning_controller import LearningController
from server.session_sanitizer import SessionSanitizer

logger = logging.getLogger(__name__)


class AgentResponse:
    """Response from HybridAgent.run()."""

    def __init__(
        self,
        answer: str,
        confidence: float,
        is_learning: bool = False,
        learning_task: asyncio.Task | None = None,
    ):
        self.answer = answer
        self.confidence = confidence
        self.is_learning = is_learning
        self.learning_task = learning_task


class HybridAgent:
    """Hybrid learning agent with local processing and remote learning.

    Args:
        config: Configuration dict with keys: local, remote, learning, rules, privacy.
        data_dir: Directory for memory data files.
    """

    # Pronouns that indicate follow-up questions
    _PRONOUNS = {"他", "她", "它", "这个", "那个", "这", "那"}

    def __init__(self, config: dict[str, Any], data_dir: str = "memory/data") -> None:
        self._config = config

        # Core components
        self._registry = ModelRegistry()
        self._registry.register_defaults()

        self._memory = MemoryManager(data_dir=data_dir)
        self._memory.connect()

        self._sanitizer = SessionSanitizer()
        self._rule_manager = RuleManager(self._memory, config.get("rules"))

        # Model (lazy-loaded)
        self._model: Any = None
        default_model = config.get("local", {}).get("default_model", "gemma-4-e2b-it-4bit")
        self._local_processor = LocalProcessor(self._get_model(default_model), self._memory, self._rule_manager)

        # Remote analyzer
        remote_config = config.get("remote", {})
        self._remote_analyzer: RemoteAnalyzer | None = None
        if remote_config.get("enabled"):
            remote_model = self._get_model(remote_config.get("model", "mimo-v2.5-pro"))
            self._remote_analyzer = RemoteAnalyzer(remote_model)

        # Learning controller
        self._learning_controller = LearningController(
            self._sanitizer,
            self._remote_analyzer,
            self._rule_manager,
            config.get("learning", {}),
        )

    async def run(self, user_input: str) -> AgentResponse:
        """Process user input with optional async learning."""
        # 1. Local processing
        result = await self._local_processor.process(user_input)

        # 2. Trigger async learning if confidence low
        learning_task = None
        if result.confidence < 0.9 and self._remote_analyzer:
            learning_task = asyncio.create_task(
                self._learning_controller.learn(user_input, result)
            )

            # If follow-up question, wait for learning to complete
            if self._is_followup_question(user_input):
                await learning_task
                learning_task = None
                result = await self._local_processor.process(user_input)

        # 3. Update rule stats
        for rule_id in result.rules_applied:
            await self._rule_manager.update_rule_stats(rule_id, result.confidence)

        # 4. Save conversation
        await self._save_conversation(user_input, result.answer)

        return AgentResponse(
            answer=result.answer,
            confidence=result.confidence,
            is_learning=learning_task is not None,
            learning_task=learning_task,
        )

    async def run_with_progress(
        self,
        user_input: str,
        callback: Callable[[str], None],
    ) -> str:
        """Process with progress callbacks for UI."""
        result = await self._local_processor.process(user_input)

        if result.confidence < 0.9 and self._remote_analyzer:
            callback("正在分析这个问题...")
            learning_task = asyncio.create_task(
                self._learning_controller.learn(user_input, result)
            )

            if self._is_followup_question(user_input):
                await learning_task
                callback("分析完成，正在重新处理...")
                result = await self._local_processor.process(user_input)

        for rule_id in result.rules_applied:
            await self._rule_manager.update_rule_stats(rule_id, result.confidence)

        await self._save_conversation(user_input, result.answer)
        return result.answer

    def _is_followup_question(self, user_input: str) -> bool:
        """Check if input contains pronouns suggesting follow-up."""
        return any(p in user_input for p in self._PRONOUNS)

    async def _save_conversation(self, user_input: str, answer: str) -> None:
        """Save conversation to memory."""
        await self._memory.remember(
            content=f"用户: {user_input}\n助手: {answer}",
            memory_type=MemoryType.CONVERSATION,
            metadata={"timestamp": datetime.now().isoformat()},
        )

    def _get_model(self, model_name: str) -> Any:
        """Get or load a model from registry."""
        if self._model is None:
            self._model = self._registry.get_or_load(model_name)
        return self._model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/python/test_hybrid_agent.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Run all tests**

Run: `python -m pytest tests/python/ -v`
Expected: All tests PASS (no regressions)

- [ ] **Step 6: Commit**

```bash
git add server/hybrid_agent.py tests/python/test_hybrid_agent.py
git commit -m "feat: add HybridAgent main entry point with learning integration"
```

---

### Task 9: Update config.yaml with hybrid_agent section

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Add hybrid_agent configuration**

Add to `config.yaml`:

```yaml
# ====== Hybrid Agent 配置 ======
hybrid_agent:
  local:
    default_model: minicpm-v-4.6
    max_tokens: 2048
    temperature: 0.7

  remote:
    enabled: true
    model: mimo-v2.5-pro
    api_key_env: MIMO_API_KEY
    base_url: https://token-plan-cn.xiaomimimo.com/v1
    timeout: 30

  learning:
    enabled: true
    confidence_threshold: 0.9
    hourly_limit: 10
    daily_limit: 100
    min_remote_confidence: 0.7

  rules:
    max_rules: 1000
    expire_days: 90
    min_success_rate: 0.3

  privacy:
    sanitize_before_remote: true
```

- [ ] **Step 2: Commit**

```bash
git add config.yaml
git commit -m "feat: add hybrid_agent configuration to config.yaml"
```

---

### Task 10: Final verification

- [ ] **Step 1: Run all Python tests**

Run: `python -m pytest tests/python/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run all JS tests**

Run: `npm test`
Expected: All tests PASS

- [ ] **Step 3: Verify imports work**

Run: `python -c "from server.hybrid_agent import HybridAgent; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Final commit if needed**

```bash
git add -A
git commit -m "chore: verify hybrid learning agent implementation"
```
