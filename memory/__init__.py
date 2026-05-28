"""Agent memory system with soul, user profile, and long-term memory.

Components:
- Soul: Agent personality, values, behavioral guidelines
- User: User profile, preferences, context
- Memory: Long-term memory with vector search (sqlite-vec)
"""

from memory.soul import Soul
from memory.user import UserProfile
from memory.memory import MemoryStore
from memory.manager import MemoryManager

__all__ = ["Soul", "UserProfile", "MemoryStore", "MemoryManager"]
