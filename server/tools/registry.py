"""Unified Tool Registry — single source of truth for all tool definitions.

Replaces the previous pattern of ``memory.get_tools() + integrations.get_tools() + COMPUTER_USE_TOOL``
with a central registry that tools are registered into.  The Agent queries the
Registry instead of assembling tools itself.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

from server.tools.spec import (
    RiskLevel,
    ToolContext,
    ToolResult,
    ToolRuntime,
    ToolSpec,
    ToolStatus,
)
from server.tools.router import ToolRuntimeRouter
from server.tools.telemetry import TelemetryService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_TOOLS_DDL = """
CREATE TABLE IF NOT EXISTS tools (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    version         TEXT NOT NULL DEFAULT '1.0.0',
    description     TEXT NOT NULL,
    input_schema    TEXT NOT NULL,
    output_schema   TEXT,
    openai_schema   TEXT NOT NULL,
    runtime         TEXT NOT NULL,
    provider        TEXT,
    entrypoint      TEXT,
    risk_level      TEXT NOT NULL DEFAULT 'L0',
    privacy_scope   TEXT NOT NULL DEFAULT 'local_only',
    status          TEXT NOT NULL DEFAULT 'active',
    tags            TEXT DEFAULT '[]',
    metadata        TEXT DEFAULT '{}',
    embedding       BLOB,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    UNIQUE(name, version)
)
"""

_TOOLS_VEC_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS tools_vec USING vec0(
    name TEXT PRIMARY KEY,
    embedding float[768]
)
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_tools_name ON tools(name)",
    "CREATE INDEX IF NOT EXISTS idx_tools_status ON tools(status)",
    "CREATE INDEX IF NOT EXISTS idx_tools_runtime ON tools(runtime)",
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Central store for tool definitions with SQLite persistence and dynamic dispatch.

    Usage::

        registry = ToolRegistry(db_path="memory/usage.db", router=router, telemetry=telemetry)
        registry.connect()
        registry.register(spec)
        openai_tools = registry.list_openai_tools()
        result = await registry.dispatch("obsidian_search", {"query": "todo"}, ctx)
    """

    def __init__(
        self,
        db_path: str = "memory/usage.db",
        router: ToolRuntimeRouter | None = None,
        telemetry: TelemetryService | None = None,
    ) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._router = router
        self._telemetry = telemetry
        # Fast in-memory cache: name -> list[ToolSpec] (one per version)
        self._cache: dict[str, list[ToolSpec]] = {}

    # -- lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_tables()
        self._load_cache()

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _init_tables(self) -> None:
        assert self._conn is not None
        self._conn.execute(_TOOLS_DDL)
        for idx in _INDEXES:
            self._conn.execute(idx)
        # Attempt to create the vec0 virtual table (requires sqlite-vec).
        try:
            self._conn.execute(_TOOLS_VEC_DDL)
        except Exception as e:
            logger.warning("sqlite-vec not available for tools_vec: %s", e)
        self._conn.commit()

    def _load_cache(self) -> None:
        assert self._conn is not None
        rows = self._conn.execute("SELECT * FROM tools WHERE status != 'archived'").fetchall()
        self._cache.clear()
        for row in rows:
            spec = ToolSpec.from_db_row(dict(row))
            self._cache.setdefault(spec.name, []).append(spec)

    # -- registration --------------------------------------------------------

    def register(self, spec: ToolSpec) -> None:
        """Register (or upsert) a tool.  Writes to DB and updates the cache."""
        assert self._conn is not None, "Not connected"

        now = datetime.now(timezone.utc).isoformat()
        spec.updated_at = now
        row = spec.to_db_row()

        self._conn.execute(
            """
            INSERT INTO tools (name, version, description, input_schema, output_schema,
                               openai_schema, runtime, provider, entrypoint, risk_level,
                               privacy_scope, status, tags, metadata, embedding,
                               created_at, updated_at)
            VALUES (:name, :version, :description, :input_schema, :output_schema,
                    :openai_schema, :runtime, :provider, :entrypoint, :risk_level,
                    :privacy_scope, :status, :tags, :metadata, :embedding,
                    :created_at, :updated_at)
            ON CONFLICT(name, version) DO UPDATE SET
                description = excluded.description,
                input_schema = excluded.input_schema,
                output_schema = excluded.output_schema,
                openai_schema = excluded.openai_schema,
                runtime = excluded.runtime,
                provider = excluded.provider,
                entrypoint = excluded.entrypoint,
                risk_level = excluded.risk_level,
                privacy_scope = excluded.privacy_scope,
                status = excluded.status,
                tags = excluded.tags,
                metadata = excluded.metadata,
                embedding = excluded.embedding,
                updated_at = excluded.updated_at
            """,
            row,
        )

        # Upsert into tools_vec for semantic search
        if spec.embedding is not None:
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO tools_vec (name, embedding) VALUES (?, ?)",
                    (spec.name, spec.embedding),
                )
            except Exception:
                pass  # sqlite-vec may not be available

        self._conn.commit()

        # Update cache
        versions = self._cache.get(spec.name, [])
        for i, v in enumerate(versions):
            if v.version == spec.version:
                versions[i] = spec
                break
        else:
            versions.append(spec)
        self._cache[spec.name] = versions

        if self._telemetry:
            self._telemetry.record_tool_registered(spec.name, spec.version)

        logger.info("Registered tool: %s v%s (%s)", spec.name, spec.version, spec.runtime.value)

    def unregister(self, name: str, version: str | None = None) -> None:
        """Deprecate a tool (or all versions)."""
        assert self._conn is not None
        if version:
            self._conn.execute(
                "UPDATE tools SET status = 'deprecated', updated_at = ? WHERE name = ? AND version = ?",
                (datetime.now(timezone.utc).isoformat(), name, version),
            )
        else:
            self._conn.execute(
                "UPDATE tools SET status = 'deprecated', updated_at = ? WHERE name = ?",
                (datetime.now(timezone.utc).isoformat(), name),
            )
        self._conn.commit()
        self._load_cache()
        if self._telemetry:
            self._telemetry.record_tool_deprecated(name, version or "")

    # -- queries -------------------------------------------------------------

    def get_tool(self, name: str, version: str | None = None) -> ToolSpec | None:
        """Look up a tool by name (and optional version)."""
        versions = self._cache.get(name, [])
        if not versions:
            return None
        if version:
            return next((v for v in versions if v.version == version), None)
        # Return the first active version
        return next((v for v in versions if v.status == ToolStatus.ACTIVE), versions[0])

    def list_tools(self, *, status: ToolStatus | None = ToolStatus.ACTIVE) -> list[ToolSpec]:
        """List all registered tools, optionally filtered by status."""
        result: list[ToolSpec] = []
        for versions in self._cache.values():
            for spec in versions:
                if status is None or spec.status == status:
                    result.append(spec)
        return result

    def list_openai_tools(self, *, status: ToolStatus | None = ToolStatus.ACTIVE) -> list[dict[str, Any]]:
        """Return tools in OpenAI function-calling format."""
        return [s.to_openai_schema() for s in self.list_tools(status=status)]

    def get_tool_stats_from_telemetry(self, name: str) -> dict[str, int]:
        """Delegate to telemetry for invocation/success/failure counts."""
        if self._telemetry:
            return self._telemetry.get_tool_stats(name)
        return {}

    def set_embedding(self, name: str, embedding: list[float]) -> None:
        """Store an embedding vector for a tool and update tools_vec."""
        assert self._conn is not None
        import struct

        blob = struct.pack(f"{len(embedding)}f", *embedding)
        self._conn.execute(
            "UPDATE tools SET embedding = ?, updated_at = ? WHERE name = ?",
            (blob, datetime.now(timezone.utc).isoformat(), name),
        )
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO tools_vec (name, embedding) VALUES (?, ?)",
                (name, embedding),
            )
        except Exception:
            pass
        self._conn.commit()
        # Update cache
        for spec in self._cache.get(name, []):
            spec.embedding = embedding

    def search_by_embedding(self, embedding: list[float], limit: int = 10) -> list[tuple[ToolSpec, float]]:
        """Semantic search over tool embeddings using sqlite-vec.

        Returns list of (ToolSpec, distance) sorted by distance ascending.
        """
        assert self._conn is not None
        try:
            rows = self._conn.execute(
                "SELECT name, distance FROM tools_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                (embedding, limit),
            ).fetchall()
        except Exception:
            return []

        results: list[tuple[ToolSpec, float]] = []
        for row in rows:
            spec = self.get_tool(row[0])  # type: ignore[index]
            if spec is not None:
                results.append((spec, row[1]))  # type: ignore[index]
        return results

    # -- dispatch ------------------------------------------------------------

    async def dispatch(self, name: str, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Look up a tool by name and execute it via the runtime router."""
        spec = self.get_tool(name)
        if spec is None:
            return ToolResult(
                content=f"未知工具: {name}",
                success=False,
                error_type="tool_not_found",
            )

        if spec.status == ToolStatus.DEPRECATED:
            return ToolResult(
                content=f"工具已弃用: {name}",
                success=False,
                error_type="tool_deprecated",
            )

        if spec.status == ToolStatus.BLOCKED:
            return ToolResult(
                content=f"工具已禁用: {name}",
                success=False,
                error_type="tool_blocked",
            )

        if self._router is None:
            return ToolResult(
                content="ToolRuntimeRouter 未配置",
                success=False,
                error_type="no_router",
            )

        return await self._router.dispatch(spec, arguments, ctx)
