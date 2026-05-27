import importlib.resources
import json
import sqlite3
import struct
from datetime import datetime

DB_PATH = "memory/assistant.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    ext_path = importlib.resources.files("sqlite_vector.binaries") / "vector"
    conn.enable_load_extension(True)
    conn.load_extension(str(ext_path))
    conn.enable_load_extension(False)
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        -- 对话历史表（短期记忆归档）
        CREATE TABLE IF NOT EXISTS conversations (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role      TEXT NOT NULL,       -- user / assistant
            content   TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        -- 长期记忆表
        CREATE TABLE IF NOT EXISTS memories (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            content   TEXT NOT NULL,       -- 记忆原文
            created_at TEXT NOT NULL
        );
    """)

    # 向量表单独建（sqlite-vec 语法）
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec
        USING vec0(
            memory_id INTEGER PRIMARY KEY,
            embedding  FLOAT[768]         -- 维度按你用的 embedding 模型决定
        );
    """)

    conn.commit()
    conn.close()
    print("DB initialized.")


if __name__ == "__main__":
    import os

    os.makedirs("memory", exist_ok=True)
    init_db()