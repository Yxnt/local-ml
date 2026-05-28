"""Long-term memory with vector search using sqlite-vec.

Memory types:
- fact: Objective facts (e.g., "用户的生日是 1990-01-01")
- preference: User preferences (e.g., "喜欢简洁的回答")
- experience: Past experiences (e.g., "上次配置 HomeKit 失败了")
- conversation: Important conversation summaries
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    EXPERIENCE = "experience"
    CONVERSATION = "conversation"


@dataclass
class Memory:
    """A single memory entry."""
    id: int | None = None
    type: MemoryType = MemoryType.FACT
    content: str = ""
    importance: float = 0.5  # 0.0 - 1.0
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "importance": self.importance,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class MemoryStore:
    """Long-term memory store with vector search.

    Uses sqlite-vec for semantic search across memories.
    """

    def __init__(self, db_path: str = "memory/assistant.db", embedding_dim: int = 768):
        self._db_path = db_path
        self._embedding_dim = embedding_dim
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Connect to the database."""
        import importlib.resources

        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row

        # Load sqlite-vec extension
        try:
            ext_path = importlib.resources.files("sqlite_vector.binaries") / "vector"
            self._conn.enable_load_extension(True)
            self._conn.load_extension(str(ext_path))
            self._conn.enable_load_extension(False)
        except Exception:
            # sqlite-vec not available, vector search disabled
            pass

    def disconnect(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def init_tables(self) -> None:
        """Initialize database tables."""
        if not self._conn:
            raise RuntimeError("Not connected")

        self._conn.executescript("""
            -- Long-term memory
            CREATE TABLE IF NOT EXISTS memories (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                type       TEXT NOT NULL DEFAULT 'fact',
                content    TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                metadata   TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            -- Conversation history (short-term)
            CREATE TABLE IF NOT EXISTS conversations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            -- Index for faster lookups
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
            CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
        """)

        # Vector table (sqlite-vec syntax)
        try:
            self._conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec
                USING vec0(
                    memory_id INTEGER PRIMARY KEY,
                    embedding FLOAT[{self._embedding_dim}]
                );
            """)
        except Exception:
            # sqlite-vec not available
            pass

        self._conn.commit()

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        importance: float = 0.5,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a new memory."""
        if not self._conn:
            raise RuntimeError("Not connected")

        now = datetime.now().isoformat()
        cursor = self._conn.execute(
            """INSERT INTO memories (type, content, importance, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (memory_type.value, content, importance, json.dumps(metadata or {}), now, now)
        )
        memory_id = cursor.lastrowid

        # Store embedding if available
        if embedding and len(embedding) == self._embedding_dim:
            try:
                self._conn.execute(
                    "INSERT INTO memories_vec (memory_id, embedding) VALUES (?, ?)",
                    (memory_id, self._pack_embedding(embedding))
                )
            except Exception:
                pass

        self._conn.commit()
        return memory_id

    def search_memories(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        limit: int = 10,
    ) -> list[Memory]:
        """Search memories by content (text search)."""
        if not self._conn:
            raise RuntimeError("Not connected")

        sql = "SELECT * FROM memories WHERE content LIKE ?"
        params: list[Any] = [f"%{query}%"]

        if memory_type:
            sql += " AND type = ?"
            params.append(memory_type.value)

        sql += " ORDER BY importance DESC, created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def search_by_vector(
        self,
        embedding: list[float],
        limit: int = 10,
        threshold: float = 0.8,
    ) -> list[Memory]:
        """Search memories by vector similarity."""
        if not self._conn:
            raise RuntimeError("Not connected")

        try:
            rows = self._conn.execute(
                """SELECT m.*, v.distance
                   FROM memories_vec v
                   JOIN memories m ON m.id = v.memory_id
                   WHERE v.embedding MATCH ? AND v.distance < ?
                   ORDER BY v.distance
                   LIMIT ?""",
                (self._pack_embedding(embedding), threshold, limit)
            ).fetchall()
            return [self._row_to_memory(row) for row in rows]
        except Exception:
            return []

    def get_recent_memories(self, limit: int = 20) -> list[Memory]:
        """Get recent memories."""
        if not self._conn:
            raise RuntimeError("Not connected")

        rows = self._conn.execute(
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def get_memory(self, memory_id: int) -> Memory | None:
        """Get a specific memory by ID."""
        if not self._conn:
            raise RuntimeError("Not connected")

        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return self._row_to_memory(row) if row else None

    def update_memory(self, memory_id: int, content: str | None = None, importance: float | None = None) -> None:
        """Update an existing memory."""
        if not self._conn:
            raise RuntimeError("Not connected")

        updates = []
        params: list[Any] = []

        if content is not None:
            updates.append("content = ?")
            params.append(content)
        if importance is not None:
            updates.append("importance = ?")
            params.append(importance)

        if not updates:
            return

        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(memory_id)

        self._conn.execute(
            f"UPDATE memories SET {', '.join(updates)} WHERE id = ?",
            params
        )
        self._conn.commit()

    def delete_memory(self, memory_id: int) -> None:
        """Delete a memory."""
        if not self._conn:
            raise RuntimeError("Not connected")

        self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        try:
            self._conn.execute("DELETE FROM memories_vec WHERE memory_id = ?", (memory_id,))
        except Exception:
            pass
        self._conn.commit()

    def add_conversation(self, session_id: str, role: str, content: str) -> int:
        """Add a conversation entry."""
        if not self._conn:
            raise RuntimeError("Not connected")

        now = datetime.now().isoformat()
        cursor = self._conn.execute(
            "INSERT INTO conversations (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now)
        )
        self._conn.commit()
        return cursor.lastrowid

    def get_conversation_history(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Get conversation history for a session."""
        if not self._conn:
            raise RuntimeError("Not connected")

        rows = self._conn.execute(
            "SELECT role, content, created_at FROM conversations WHERE session_id = ? ORDER BY created_at LIMIT ?",
            (session_id, limit)
        ).fetchall()
        return [dict(row) for row in rows]

    def get_stats(self) -> dict[str, int]:
        """Get memory statistics."""
        if not self._conn:
            raise RuntimeError("Not connected")

        stats = {}
        stats["total_memories"] = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        stats["total_conversations"] = self._conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]

        for memory_type in MemoryType:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE type = ?",
                (memory_type.value,)
            ).fetchone()[0]
            stats[f"memories_{memory_type.value}"] = count

        return stats

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        """Convert a database row to a Memory object."""
        return Memory(
            id=row["id"],
            type=MemoryType(row["type"]),
            content=row["content"],
            importance=row["importance"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _pack_embedding(embedding: list[float]) -> bytes:
        """Pack embedding vector into bytes for sqlite-vec."""
        import struct
        return struct.pack(f"{len(embedding)}f", *embedding)

    @staticmethod
    def _unpack_embedding(data: bytes) -> list[float]:
        """Unpack embedding vector from bytes."""
        import struct
        count = len(data) // 4
        return list(struct.unpack(f"{count}f", data))
