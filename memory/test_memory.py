"""Tests for memory system."""

import json
import pytest
from pathlib import Path

from memory.soul import Soul
from memory.user import UserProfile
from memory.memory import MemoryStore, MemoryType
from memory.manager import MemoryManager


class TestSoul:
    def test_default_soul(self):
        soul = Soul()
        assert soul.name == "Local ML Assistant"
        assert len(soul.rules) > 0

    def test_save_load(self, tmp_path):
        soul = Soul(name="Test Bot", personality="测试")
        path = tmp_path / "soul.json"
        soul.save(path)

        loaded = Soul.load(path)
        assert loaded.name == "Test Bot"
        assert loaded.personality == "测试"

    def test_system_prompt(self):
        soul = Soul()
        prompt = soul.get_system_prompt()
        assert "Local ML Assistant" in prompt
        assert "隐私保护" in prompt


class TestUserProfile:
    def test_default_user(self):
        user = UserProfile()
        assert user.timezone == "Asia/Shanghai"

    def test_save_load(self, tmp_path):
        user = UserProfile(name="Yxnt", role="工程师")
        path = tmp_path / "user.json"
        user.save(path)

        loaded = UserProfile.load(path)
        assert loaded.name == "Yxnt"
        assert loaded.role == "工程师"

    def test_context_summary(self):
        user = UserProfile(name="Yxnt")
        summary = user.get_context_summary()
        assert "Yxnt" in summary


class TestMemoryStore:
    def test_add_and_search(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        store = MemoryStore(db_path)
        store.connect()
        store.init_tables()

        # Add memories
        id1 = store.add_memory("用户喜欢 Python", MemoryType.PREFERENCE, 0.8)
        id2 = store.add_memory("用户的生日是 1990-01-01", MemoryType.FACT, 0.9)

        assert id1 > 0
        assert id2 > 0

        # Search
        results = store.search_memories("Python")
        assert len(results) == 1
        assert results[0].content == "用户喜欢 Python"

        store.disconnect()

    def test_get_recent(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        store = MemoryStore(db_path)
        store.connect()
        store.init_tables()

        store.add_memory("记忆1")
        store.add_memory("记忆2")
        store.add_memory("记忆3")

        recent = store.get_recent_memories(limit=2)
        assert len(recent) == 2

        store.disconnect()

    def test_stats(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        store = MemoryStore(db_path)
        store.connect()
        store.init_tables()

        store.add_memory("fact1", MemoryType.FACT)
        store.add_memory("pref1", MemoryType.PREFERENCE)

        stats = store.get_stats()
        assert stats["total_memories"] == 2
        assert stats["memories_fact"] == 1
        assert stats["memories_preference"] == 1

        store.disconnect()


class TestMemoryManager:
    def test_context_manager(self, tmp_path):
        with MemoryManager(str(tmp_path / "data")) as manager:
            # Test remember
            manager.remember("测试记忆", MemoryType.FACT)

            # Test recall
            results = manager.recall("测试")
            assert len(results) == 1

    def test_system_prompt(self, tmp_path):
        with MemoryManager(str(tmp_path / "data")) as manager:
            manager.soul.name = "测试助手"
            manager.user.name = "Yxnt"

            prompt = manager.get_system_prompt()
            assert "测试助手" in prompt
            assert "Yxnt" in prompt

    def test_tools(self, tmp_path):
        with MemoryManager(str(tmp_path / "data")) as manager:
            tools = manager.get_tools()
            assert len(tools) == 3
            names = [t["function"]["name"] for t in tools]
            assert "memory_remember" in names
            assert "memory_recall" in names
            assert "memory_stats" in names
