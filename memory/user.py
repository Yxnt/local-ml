"""User profile - preferences, habits, and context.

The user profile stores information about the user that helps
the agent provide personalized responses.

Supports both JSON and Markdown formats:
- user.json: Structured data
- user.md: Human-readable, editable
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class UserProfile:
    """User profile with preferences and context."""

    # Basic info
    name: str = ""
    role: str = ""  # e.g., "软件工程师"
    timezone: str = "Asia/Shanghai"
    language: str = "中文"

    # Preferences
    preferences: dict[str, Any] = field(default_factory=lambda: {
        "coding_style": "简洁、实用",
        "communication": "直接、不要废话",
        "response_length": "concise",  # concise, detailed, adaptive
        "show_reasoning": False,
    })

    # Context
    context: dict[str, Any] = field(default_factory=lambda: {
        "devices": ["Mac Mini", "iPhone", "iPad"],
        "smart_home": "小米生态",
        "work_hours": "9:00-18:00",
        "interests": ["编程", "AI", "智能家居"],
    })

    # Relationships
    relationships: dict[str, str] = field(default_factory=dict)
    # e.g., {"同事": "张三", "家人": "李四"}

    # Metadata
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "role": self.role,
            "timezone": self.timezone,
            "language": self.language,
            "preferences": self.preferences,
            "context": self.context,
            "relationships": self.relationships,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserProfile:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_markdown(self) -> str:
        """Convert to human-readable markdown."""
        prefs_md = "\n".join(f"- **{k}**: {v}" for k, v in self.preferences.items())
        ctx_md = "\n".join(f"- **{k}**: {v}" for k, v in self.context.items())
        rel_md = "\n".join(f"- **{k}**: {v}" for k, v in self.relationships.items()) if self.relationships else "- 无"

        return f"""# User Profile

## 基本信息

- **姓名**: {self.name or '未设置'}
- **角色**: {self.role or '未设置'}
- **时区**: {self.timezone}
- **语言**: {self.language}
- **更新时间**: {self.updated_at or datetime.now().isoformat()}

## 偏好设置

{prefs_md}

## 环境信息

{ctx_md}

## 人际关系

{rel_md}
"""

    @classmethod
    def from_markdown(cls, content: str) -> UserProfile:
        """Parse markdown content into UserProfile."""
        profile = cls()

        # Parse basic info
        name_match = re.search(r"\*\*姓名\*\*:\s*(.+)", content)
        if name_match and name_match.group(1).strip() != "未设置":
            profile.name = name_match.group(1).strip()

        role_match = re.search(r"\*\*角色\*\*:\s*(.+)", content)
        if role_match and role_match.group(1).strip() != "未设置":
            profile.role = role_match.group(1).strip()

        timezone_match = re.search(r"\*\*时区\*\*:\s*(.+)", content)
        if timezone_match:
            profile.timezone = timezone_match.group(1).strip()

        language_match = re.search(r"\*\*语言\*\*:\s*(.+)", content)
        if language_match:
            profile.language = language_match.group(1).strip()

        # Parse preferences section
        prefs_section = re.search(r"## 偏好设置\n\n(.*?)(?:\n##|\Z)", content, re.DOTALL)
        if prefs_section:
            profile.preferences = cls._parse_key_value_list(prefs_section.group(1))

        # Parse context section
        ctx_section = re.search(r"## 环境信息\n\n(.*?)(?:\n##|\Z)", content, re.DOTALL)
        if ctx_section:
            profile.context = cls._parse_key_value_list(ctx_section.group(1))

        # Parse relationships section
        rel_section = re.search(r"## 人际关系\n\n(.*?)(?:\n##|\Z)", content, re.DOTALL)
        if rel_section:
            rels = cls._parse_key_value_list(rel_section.group(1))
            if rels and "无" not in rels:
                profile.relationships = rels

        return profile

    @staticmethod
    def _parse_key_value_list(text: str) -> dict[str, Any]:
        """Parse markdown key-value list."""
        result = {}
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith("- ") and ":" in line[2:]:
                key, value = line[2:].split(":", 1)
                key = key.strip().strip("*")
                value = value.strip()
                # Try to parse as list
                if value.startswith("[") and value.endswith("]"):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                result[key] = value
        return result

    def save(self, path: str | Path) -> None:
        """Save profile to JSON or Markdown file based on extension."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".md":
            self.updated_at = datetime.now().isoformat()
            path.write_text(self.to_markdown(), encoding="utf-8")
        else:
            path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> UserProfile:
        """Load profile from JSON or Markdown file."""
        path = Path(path)
        if not path.exists():
            return cls()

        content = path.read_text(encoding="utf-8")

        if path.suffix == ".md":
            return cls.from_markdown(content)
        else:
            return cls.from_dict(json.loads(content))

    def get_context_summary(self) -> str:
        """Generate a summary of user context for the agent."""
        parts = []
        if self.name:
            parts.append(f"用户：{self.name}")
        if self.role:
            parts.append(f"角色：{self.role}")
        if self.preferences:
            prefs = ", ".join(f"{k}: {v}" for k, v in self.preferences.items() if v)
            parts.append(f"偏好：{prefs}")
        if self.context:
            ctx = ", ".join(f"{k}: {v}" for k, v in self.context.items() if v)
            parts.append(f"环境：{ctx}")
        return "\n".join(parts)

    def update_preference(self, key: str, value: Any) -> None:
        """Update a single preference."""
        self.preferences[key] = value

    def update_context(self, key: str, value: Any) -> None:
        """Update a single context item."""
        self.context[key] = value
