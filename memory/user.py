"""User profile - preferences, habits, and context.

The user profile stores information about the user that helps
the agent provide personalized responses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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

    def save(self, path: str | Path) -> None:
        """Save profile to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> UserProfile:
        """Load profile from JSON file."""
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls.from_dict(data)

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
