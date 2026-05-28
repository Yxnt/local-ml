"""Agent Soul - personality, values, and behavioral guidelines.

The soul defines WHO the agent is and HOW it should behave.
It's the core identity that persists across all interactions.

Supports both JSON and Markdown formats:
- soul.json: Structured data
- soul.md: Human-readable, editable
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Soul:
    """Agent personality and behavioral guidelines."""

    # Core identity
    name: str = "Local ML Assistant"
    personality: str = "友好、专业、简洁"
    values: list[str] = field(default_factory=lambda: ["隐私保护", "数据安全", "用户优先"])
    communication_style: str = "直接、有条理、不废话"

    # Capabilities and expertise
    expertise: list[str] = field(default_factory=lambda: ["编程", "AI", "智能家居"])
    languages: list[str] = field(default_factory=lambda: ["中文", "English"])

    # Behavioral rules
    rules: list[str] = field(default_factory=lambda: [
        "保护用户隐私，不泄露敏感信息",
        "数据存储在本地，不上传到外部服务",
        "回答简洁直接，避免冗余",
        "不确定时主动询问，不猜测",
        "尊重用户偏好，记住用户习惯",
    ])

    # Response templates
    greeting: str = "你好！有什么可以帮你的？"
    farewell: str = "好的，有需要随时找我。"
    uncertainty: str = "这个我不太确定，让我查一下。"

    # Metadata
    version: str = "1.0.0"
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "personality": self.personality,
            "values": self.values,
            "communication_style": self.communication_style,
            "expertise": self.expertise,
            "languages": self.languages,
            "rules": self.rules,
            "greeting": self.greeting,
            "farewell": self.farewell,
            "uncertainty": self.uncertainty,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Soul:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_markdown(self) -> str:
        """Convert to human-readable markdown."""
        values_md = "\n".join(f"- {v}" for v in self.values)
        expertise_md = ", ".join(self.expertise)
        languages_md = ", ".join(self.languages)
        rules_md = "\n".join(f"- {r}" for r in self.rules)

        return f"""# Soul - {self.name}

## 基本信息

- **名称**: {self.name}
- **版本**: {self.version}
- **更新时间**: {self.updated_at or datetime.now().isoformat()}

## 性格特点

{self.personality}

## 核心价值

{values_md}

## 沟通风格

{self.communication_style}

## 专业领域

{expertise_md}

## 语言能力

{languages_md}

## 行为准则

{rules_md}

## 响应模板

- **问候**: {self.greeting}
- **告别**: {self.farewell}
- **不确定**: {self.uncertainty}
"""

    @classmethod
    def from_markdown(cls, content: str) -> Soul:
        """Parse markdown content into Soul."""
        soul = cls()

        # Parse name from title
        title_match = re.search(r"^#\s+.*?-\s*(.+)$", content, re.MULTILINE)
        if title_match:
            soul.name = title_match.group(1).strip()

        # Parse version
        version_match = re.search(r"\*\*版本\*\*:\s*(.+)", content)
        if version_match:
            soul.version = version_match.group(1).strip()

        # Parse personality
        personality_match = re.search(r"## 性格特点\n\n(.+?)(?:\n##|\Z)", content, re.DOTALL)
        if personality_match:
            soul.personality = personality_match.group(1).strip()

        # Parse communication style
        style_match = re.search(r"## 沟通风格\n\n(.+?)(?:\n##|\Z)", content, re.DOTALL)
        if style_match:
            soul.communication_style = style_match.group(1).strip()

        # Parse values (bullet list)
        values_section = re.search(r"## 核心价值\n\n(.*?)(?:\n##|\Z)", content, re.DOTALL)
        if values_section:
            soul.values = cls._parse_bullet_list(values_section.group(1))

        # Parse expertise (comma separated)
        expertise_match = re.search(r"## 专业领域\n\n(.+?)(?:\n##|\Z)", content, re.DOTALL)
        if expertise_match:
            soul.expertise = [e.strip() for e in expertise_match.group(1).split(",")]

        # Parse rules (bullet list)
        rules_section = re.search(r"## 行为准则\n\n(.*?)(?:\n##|\Z)", content, re.DOTALL)
        if rules_section:
            soul.rules = cls._parse_bullet_list(rules_section.group(1))

        # Parse response templates
        greeting_match = re.search(r"\*\*问候\*\*:\s*(.+)", content)
        if greeting_match:
            soul.greeting = greeting_match.group(1).strip()

        farewell_match = re.search(r"\*\*告别\*\*:\s*(.+)", content)
        if farewell_match:
            soul.farewell = farewell_match.group(1).strip()

        uncertainty_match = re.search(r"\*\*不确定\*\*:\s*(.+)", content)
        if uncertainty_match:
            soul.uncertainty = uncertainty_match.group(1).strip()

        return soul

    @staticmethod
    def _parse_bullet_list(text: str) -> list[str]:
        """Parse markdown bullet list."""
        items = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("- "):
                items.append(line[2:].strip())
            elif line.startswith("* "):
                items.append(line[2:].strip())
        return items

    def save(self, path: str | Path) -> None:
        """Save soul to JSON or Markdown file based on extension."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".md":
            self.updated_at = datetime.now().isoformat()
            path.write_text(self.to_markdown(), encoding="utf-8")
        else:
            path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Soul:
        """Load soul from JSON or Markdown file."""
        path = Path(path)
        if not path.exists():
            return cls()

        content = path.read_text(encoding="utf-8")

        if path.suffix == ".md":
            return cls.from_markdown(content)
        else:
            return cls.from_dict(json.loads(content))

    def get_system_prompt(self) -> str:
        """Generate system prompt from soul definition."""
        rules_text = "\n".join(f"- {r}" for r in self.rules)
        values_text = ", ".join(self.values)
        expertise_text = ", ".join(self.expertise)

        return f"""你是{self.name}。

性格特点：{self.personality}
核心价值：{values_text}
沟通风格：{self.communication_style}
专业领域：{expertise_text}

行为准则：
{rules_text}

请始终保持这些特质，为用户提供最佳体验。"""
