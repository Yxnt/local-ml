"""Versioned prompt storage with rollback.

Stores optimized prompts in a SQLite ``prompt_versions`` table with full
version history.  Each version has a status lifecycle:

    candidate -> active -> archived
                  |  ^
                  v  |
                rejected

Only one version of a given prompt can be ``active`` at a time.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PromptVersion:
    """A single versioned snapshot of a prompt."""

    prompt_name: str
    version: int
    content: str
    optimizer: str  # "manual" | "gepa" | "dspy"
    score: float | None
    eval_summary: str
    created_at: str
    status: str  # "candidate" | "active" | "rejected" | "archived"


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class PromptStore:
    """Versioned prompt storage backed by SQLite.

    Args:
        db_path: Path to the SQLite database file.  The directory is
            created automatically if it does not exist.
    """

    _DDL = """
    CREATE TABLE IF NOT EXISTS prompt_versions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        prompt_name     TEXT NOT NULL,
        version         INTEGER NOT NULL,
        content         TEXT NOT NULL,
        optimizer       TEXT NOT NULL DEFAULT 'manual',
        score           REAL,
        eval_summary    TEXT DEFAULT '',
        created_at      TEXT NOT NULL,
        status          TEXT NOT NULL DEFAULT 'candidate',
        UNIQUE(prompt_name, version)
    );
    CREATE INDEX IF NOT EXISTS idx_pv_name ON prompt_versions(prompt_name);
    CREATE INDEX IF NOT EXISTS idx_pv_name_status ON prompt_versions(prompt_name, status);
    """

    def __init__(self, db_path: str = "memory/usage.db") -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    # -- lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        """Open (or create) the database and ensure the table exists."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _init_tables(self) -> None:
        assert self._conn is not None
        self._conn.executescript(self._DDL)
        self._conn.commit()

    # -- public API ----------------------------------------------------------

    def save(
        self,
        prompt_name: str,
        content: str,
        optimizer: str,
        score: float | None = None,
        eval_summary: str = "",
    ) -> PromptVersion:
        """Save a new candidate version of a prompt.

        The new version is always created with ``status="candidate"``.
        Auto-increments the version number.
        """
        assert self._conn is not None, "Not connected — call .connect() first"

        next_ver = self._next_version(prompt_name)
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """
            INSERT INTO prompt_versions
                (prompt_name, version, content, optimizer, score, eval_summary, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'candidate')
            """,
            (prompt_name, next_ver, content, optimizer, score, eval_summary, now),
        )
        self._conn.commit()

        logger.info(
            "Saved prompt %r version %d (optimizer=%s, score=%s)",
            prompt_name, next_ver, optimizer, score,
        )

        return PromptVersion(
            prompt_name=prompt_name,
            version=next_ver,
            content=content,
            optimizer=optimizer,
            score=score,
            eval_summary=eval_summary,
            created_at=now,
            status="candidate",
        )

    def get_active(self, prompt_name: str) -> PromptVersion | None:
        """Return the currently active version, or ``None``."""
        assert self._conn is not None
        row = self._conn.execute(
            """
            SELECT * FROM prompt_versions
            WHERE prompt_name = ? AND status = 'active'
            ORDER BY version DESC LIMIT 1
            """,
            (prompt_name,),
        ).fetchone()
        return self._row_to_pv(row) if row else None

    def get_version(self, prompt_name: str, version: int) -> PromptVersion | None:
        """Return a specific version, or ``None``."""
        assert self._conn is not None
        row = self._conn.execute(
            """
            SELECT * FROM prompt_versions
            WHERE prompt_name = ? AND version = ?
            """,
            (prompt_name, version),
        ).fetchone()
        return self._row_to_pv(row) if row else None

    def list_versions(self, prompt_name: str) -> list[PromptVersion]:
        """Return all versions of a prompt, newest first."""
        assert self._conn is not None
        rows = self._conn.execute(
            """
            SELECT * FROM prompt_versions
            WHERE prompt_name = ?
            ORDER BY version DESC
            """,
            (prompt_name,),
        ).fetchall()
        return [self._row_to_pv(r) for r in rows]

    def promote(self, prompt_name: str, version: int) -> bool:
        """Promote a candidate version to active.

        The previously active version (if any) is set to ``archived``.
        Returns ``True`` on success, ``False`` if the version is not found
        or not a candidate.
        """
        assert self._conn is not None
        target = self.get_version(prompt_name, version)
        if target is None:
            logger.warning("promote: version %d of %r not found", version, prompt_name)
            return False
        if target.status not in ("candidate",):
            logger.warning(
                "promote: version %d of %r has status %r (expected 'candidate')",
                version, prompt_name, target.status,
            )
            return False

        # Archive current active
        self._conn.execute(
            """
            UPDATE prompt_versions
            SET status = 'archived'
            WHERE prompt_name = ? AND status = 'active'
            """,
            (prompt_name,),
        )
        # Promote target
        self._conn.execute(
            """
            UPDATE prompt_versions
            SET status = 'active'
            WHERE prompt_name = ? AND version = ?
            """,
            (prompt_name, version),
        )
        self._conn.commit()
        logger.info("Promoted %r version %d to active", prompt_name, version)
        return True

    def rollback(self, prompt_name: str) -> bool:
        """Revert to the previously active version.

        Finds the most recent ``archived`` version and makes it ``active``.
        The current active version (if any) is set to ``rejected``.
        Returns ``True`` on success.
        """
        assert self._conn is not None

        # Find current active
        current = self.get_active(prompt_name)

        # Find most recent archived
        row = self._conn.execute(
            """
            SELECT * FROM prompt_versions
            WHERE prompt_name = ? AND status = 'archived'
            ORDER BY version DESC LIMIT 1
            """,
            (prompt_name,),
        ).fetchone()

        if row is None:
            logger.warning("rollback: no archived version of %r found", prompt_name)
            return False

        prev = self._row_to_pv(row)

        # Demote current active to rejected
        if current:
            self._conn.execute(
                """
                UPDATE prompt_versions
                SET status = 'rejected'
                WHERE prompt_name = ? AND version = ?
                """,
                (prompt_name, current.version),
            )

        # Restore previous
        self._conn.execute(
            """
            UPDATE prompt_versions
            SET status = 'active'
            WHERE prompt_name = ? AND version = ?
            """,
            (prompt_name, prev.version),
        )
        self._conn.commit()
        logger.info("Rolled back %r to version %d", prompt_name, prev.version)
        return True

    def reject(self, prompt_name: str, version: int) -> bool:
        """Mark a version as rejected.

        Returns ``True`` on success, ``False`` if not found.
        """
        assert self._conn is not None
        target = self.get_version(prompt_name, version)
        if target is None:
            return False

        self._conn.execute(
            """
            UPDATE prompt_versions
            SET status = 'rejected'
            WHERE prompt_name = ? AND version = ?
            """,
            (prompt_name, version),
        )
        self._conn.commit()
        logger.info("Rejected %r version %d", prompt_name, version)
        return True

    # -- internals -----------------------------------------------------------

    def _next_version(self, prompt_name: str) -> int:
        """Return the next version number for a prompt name."""
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT MAX(version) as max_ver FROM prompt_versions WHERE prompt_name = ?",
            (prompt_name,),
        ).fetchone()
        max_ver = row["max_ver"] if row and row["max_ver"] is not None else 0
        return max_ver + 1

    @staticmethod
    def _row_to_pv(row: sqlite3.Row) -> PromptVersion:
        return PromptVersion(
            prompt_name=row["prompt_name"],
            version=row["version"],
            content=row["content"],
            optimizer=row["optimizer"],
            score=row["score"],
            eval_summary=row["eval_summary"] or "",
            created_at=row["created_at"],
            status=row["status"],
        )
