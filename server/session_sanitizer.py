"""Session sanitizer - 会话级脱敏，同一实体在同一会话内使用相同占位符

在会话生命周期内维护实体映射，确保：
- 同一人名始终映射到同一个占位符
- 不同实体使用不同占位符
- 可批量脱敏对话历史
- 可还原模型响应中的占位符
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from server.sanitizer import (
    SanitizedText,
    SensitivityDetector,
    SensitivityLevel,
)


@dataclass
class EntityMapping:
    """实体映射记录"""
    original: str
    placeholder: str
    entity_type: str
    created_at: float = field(default_factory=time.time)


class SessionSanitizer:
    """会话级脱敏器

    在会话内维护一致的实体→占位符映射。
    同一实体在多次 sanitize 调用中始终使用相同占位符。
    """

    # 中文人名模式：2-4个中文字符，在非中文字符之后（lazy 匹配优先短序列）
    CHINESE_NAME_PATTERN = re.compile(r'(?<![一-鿿])([一-鿿]{2,4}?)')

    # 常见动词/介词/连词首字（用于第二轮检测：动词后面可能跟人名）
    VERB_PREFIX_CHARS: set[str] = {
        "帮", "给", "把", "被", "让", "叫", "请", "跟", "和", "与",
        "同", "对", "向", "从", "到", "在", "是", "有", "说", "想",
        "能", "可", "会", "要", "也", "都", "不", "就", "还", "又",
        "再", "才", "找", "问", "给", "替", "为", "带", "送", "接",
    }

    # 常见非人名中文词（排除列表）
    EXCLUDED_WORDS: set[str] = {
        "我们", "他们", "她们", "它们", "大家", "自己", "什么",
        "怎么", "如何", "为什么", "可以", "应该", "需要",
        "已经", "正在", "可能", "不可能", "但是", "而且",
        "因为", "所以", "如果", "虽然", "不过", "然后",
        "今天", "明天", "昨天", "现在", "这里", "那里",
        "这个", "那个", "这些", "那些", "所有", "一些",
        "邮箱", "电话", "手机", "地址", "名字", "密码",
        "公司", "部门", "项目", "产品", "系统", "服务",
        "客户", "同事", "领导", "老板", "经理", "总监",
        "你好", "请问", "谢谢", "不客气", "好的", "明白",
        "回复", "记录", "帮忙", "确认", "检查", "处理",
        "信息", "数据", "文件", "报告", "邮件", "消息",
        "请假", "开会", "出差", "加班", "休息", "下班",
    }

    # 引号内文本模式（检测引号包裹的内容作为"事物"）
    QUOTED_TEXT_PATTERN = re.compile(
        r'[""「」『』【】《》](.+?)[""「」『』【】《》]'
    )

    def __init__(self, detector: SensitivityDetector | None = None):
        self.detector = detector or SensitivityDetector()
        # placeholder -> EntityMapping
        self._entity_map: dict[str, EntityMapping] = {}
        # original -> placeholder (reverse lookup for consistency)
        self._original_to_placeholder: dict[str, str] = {}
        # Counters for each entity type
        self._counters: dict[str, int] = {}

    def _next_placeholder(self, entity_type: str) -> str:
        """Generate next placeholder for a given entity type."""
        count = self._counters.get(entity_type, 0)
        self._counters[entity_type] = count + 1
        return f"[{entity_type}_{count}]"

    def _get_or_create_placeholder(self, original: str, entity_type: str) -> str:
        """Get existing placeholder for an entity, or create a new one."""
        if original in self._original_to_placeholder:
            return self._original_to_placeholder[original]

        placeholder = self._next_placeholder(entity_type)
        self._original_to_placeholder[original] = placeholder
        self._entity_map[placeholder] = EntityMapping(
            original=original,
            placeholder=placeholder,
            entity_type=entity_type,
        )
        return placeholder

    def _detect_chinese_names(self, text: str) -> list[tuple[str, str, str]]:
        """Detect Chinese names in text. Returns (entity_type, original, placeholder).

        Uses a two-pass approach:
        1. Match 2-4 char Chinese sequences at word boundaries (not preceded by Chinese).
           Skip sequences that start with a verb/particle character.
        2. For each verb/particle character, check if the next 2-3 chars form a
           potential name (all Chinese, not in exclusion list).
        """
        results = []
        seen: set[str] = set()

        # Pass 1: Names at word boundaries (not preceded by Chinese char)
        for match in self.CHINESE_NAME_PATTERN.finditer(text):
            name = match.group(1)
            if name in self.EXCLUDED_WORDS or name in seen:
                continue
            # Skip if the name starts with a verb/particle (likely not a name)
            if name[0] in self.VERB_PREFIX_CHARS:
                continue
            seen.add(name)
            placeholder = self._get_or_create_placeholder(name, "PERSON")
            results.append(("PERSON", name, placeholder))

        # Pass 2: Names preceded by a verb/particle character
        # Catches cases like "帮张伟回复" where "张伟" follows "帮"
        for i, ch in enumerate(text):
            if ch not in self.VERB_PREFIX_CHARS:
                continue
            for length in (2, 3):
                end = i + 1 + length
                if end > len(text):
                    continue
                candidate = text[i + 1 : end]
                if not all("一" <= c <= "鿿" for c in candidate):
                    continue
                if candidate in self.EXCLUDED_WORDS or candidate in seen:
                    continue
                # Skip 3-char candidate if its 2-char prefix is already a name
                # to avoid "张伟回" shadowing "张伟"
                if length == 3 and candidate[:2] in seen:
                    continue
                seen.add(candidate)
                placeholder = self._get_or_create_placeholder(candidate, "PERSON")
                results.append(("PERSON", candidate, placeholder))

        return results

    def _detect_quoted_text(self, text: str) -> list[tuple[str, str, str]]:
        """Detect quoted text as 'things'. Returns (entity_type, original, placeholder)."""
        results = []
        seen = set()
        for match in self.QUOTED_TEXT_PATTERN.finditer(text):
            quoted = match.group(0)  # including quotes
            inner = match.group(1)
            if quoted in seen or len(inner) < 2:
                continue
            seen.add(quoted)
            placeholder = self._get_or_create_placeholder(quoted, "THING")
            results.append(("THING", quoted, placeholder))
        return results

    def sanitize(self, text: str) -> SanitizedText:
        """Sanitize text with consistent entity mapping within the session.

        Same entity always gets the same placeholder across calls.
        """
        all_matches: list[tuple[str, str, str]] = []

        # Layer 1: Structured data via existing detector (email, phone, etc.)
        detection = self.detector.detect(text)
        for entity_type, original, _auto_placeholder in detection.matches:
            placeholder = self._get_or_create_placeholder(original, entity_type)
            all_matches.append((entity_type, original, placeholder))

        # Layer 2: Chinese name detection
        name_matches = self._detect_chinese_names(text)
        # Filter out names that overlap with already-detected structured data
        detected_originals = {m[1] for m in all_matches}
        for match in name_matches:
            if match[1] not in detected_originals:
                all_matches.append(match)

        # Layer 3: Quoted text detection
        quoted_matches = self._detect_quoted_text(text)
        for match in quoted_matches:
            if match[1] not in detected_originals:
                all_matches.append(match)

        # Sort by length descending to avoid short matches breaking long ones
        sorted_matches = sorted(all_matches, key=lambda x: len(x[1]), reverse=True)

        sanitized = text
        mapping: dict[str, str] = {}

        for entity_type, original, placeholder in sorted_matches:
            if original in sanitized:
                sanitized = sanitized.replace(original, placeholder, 1)
                mapping[placeholder] = original

        # Determine max sensitivity level
        level = detection.level
        if name_matches:
            level = max(level, SensitivityLevel.PERSONAL, key=lambda x: x.value)

        return SanitizedText(
            original=text,
            sanitized=sanitized,
            mapping=mapping,
            level=level,
        )

    def desanitize(self, text: str) -> str:
        """Restore all placeholders in text back to their original values."""
        result = text
        # Sort by placeholder length descending to avoid partial replacements
        for placeholder in sorted(self._entity_map, key=len, reverse=True):
            mapping = self._entity_map[placeholder]
            result = result.replace(placeholder, mapping.original)
        return result

    def get_sanitized_history(self, history: list[str]) -> list[str]:
        """Sanitize a conversation history using the same entity mapping.

        All entries share the same placeholder mapping, ensuring consistency
        across the entire conversation.
        """
        return [self.sanitize(text).sanitized for text in history]

    def get_entity_types_only(self) -> list[dict]:
        """Return entity types and placeholders without leaking original values.

        Useful for logging/auditing what was sanitized without exposing the data.
        """
        return [
            {
                "entity_type": mapping.entity_type,
                "placeholder": mapping.placeholder,
            }
            for mapping in self._entity_map.values()
        ]
