"""Agent Soul - personality, values, and behavioral guidelines.

The soul defines WHO the agent is and HOW it should behave.
It's the core identity that persists across all interactions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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

    def save(self, path: str | Path) -> None:
        """Save soul to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Soul:
        """Load soul from JSON file."""
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls.from_dict(data)

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
