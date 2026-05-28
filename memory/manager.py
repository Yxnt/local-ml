"""Memory Manager - unified interface for soul, user, and memory.

Provides a single entry point for the agent to access all memory components.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from memory.soul import Soul
from memory.user import UserProfile
from memory.memory import MemoryStore, MemoryType


class MemoryManager:
    """Unified memory management for the agent.

    Usage:
        async with MemoryManager() as manager:
            # Get system prompt (includes soul + user context)
            prompt = manager.get_system_prompt()

            # Remember something
            manager.remember("用户喜欢简洁的回答", MemoryType.PREFERENCE)

            # Recall memories
            memories = manager.recall("用户偏好")
    """

    def __init__(self, data_dir: str = "memory/data"):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._soul_path = self._data_dir / "soul.json"
        self._user_path = self._data_dir / "user.json"
        self._db_path = str(self._data_dir / "assistant.db")

        self._soul: Soul | None = None
        self._user: UserProfile | None = None
        self._store: MemoryStore | None = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()

    def connect(self) -> None:
        """Initialize all memory components."""
        # Load or create soul
        self._soul = Soul.load(self._soul_path)

        # Load or create user profile
        self._user = UserProfile.load(self._user_path)

        # Initialize memory store
        self._store = MemoryStore(self._db_path)
        self._store.connect()
        self._store.init_tables()

    def disconnect(self) -> None:
        """Save and close all memory components."""
        if self._soul:
            self._soul.save(self._soul_path)
        if self._user:
            self._user.save(self._user_path)
        if self._store:
            self._store.disconnect()

    @property
    def soul(self) -> Soul:
        if not self._soul:
            raise RuntimeError("Not connected")
        return self._soul

    @property
    def user(self) -> UserProfile:
        if not self._user:
            raise RuntimeError("Not connected")
        return self._user

    @property
    def store(self) -> MemoryStore:
        if not self._store:
            raise RuntimeError("Not connected")
        return self._store

    def get_system_prompt(self) -> str:
        """Generate complete system prompt from soul + user context."""
        parts = []

        # Soul personality
        if self._soul:
            parts.append(self._soul.get_system_prompt())

        # User context
        if self._user:
            user_context = self._user.get_context_summary()
            if user_context:
                parts.append(f"\n用户信息：\n{user_context}")

        # Recent memories summary
        if self._store:
            recent = self._store.get_recent_memories(limit=5)
            if recent:
                memories_text = "\n".join(f"- {m.content}" for m in recent)
                parts.append(f"\n最近记忆：\n{memories_text}")

        return "\n\n".join(parts)

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Store a new memory."""
        if not self._store:
            raise RuntimeError("Not connected")
        return self._store.add_memory(content, memory_type, importance, metadata=metadata)

    def recall(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search memories by text query."""
        if not self._store:
            raise RuntimeError("Not connected")

        memories = self._store.search_memories(query, memory_type, limit)
        return [m.to_dict() for m in memories]

    def get_tools(self) -> list[dict[str, Any]]:
        """Return memory-related tool definitions for the agent."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "memory_remember",
                    "description": "Store a new memory for future reference.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The memory content to store.",
                            },
                            "type": {
                                "type": "string",
                                "enum": ["fact", "preference", "experience", "conversation"],
                                "description": "Type of memory.",
                            },
                            "importance": {
                                "type": "number",
                                "description": "Importance score (0.0-1.0, default: 0.5).",
                            },
                        },
                        "required": ["content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_recall",
                    "description": "Search memories by query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query.",
                            },
                            "type": {
                                "type": "string",
                                "enum": ["fact", "preference", "experience", "conversation"],
                                "description": "Filter by memory type.",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max results (default: 10).",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_stats",
                    "description": "Get memory statistics.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
