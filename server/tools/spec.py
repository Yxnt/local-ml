"""Canonical tool specification and supporting types.

Every tool in the system — memory, integration, generated, computer-use — is
described by a ``ToolSpec``.  Two adapters convert a ToolSpec to the wire
formats consumed by different clients:

* ``to_openai_schema``  → OpenAI function-calling dict
* ``to_define_tool_arg`` → pi-coding-agent / TypeBox-compatible dict
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RiskLevel(str, Enum):
    """Permission tier for tool execution."""

    L0 = "L0"  # Pure computation, string processing, format conversion
    L1 = "L1"  # Read temp dirs / whitelisted files
    L2 = "L2"  # Network read / read Obsidian / read email / read calendar
    L3 = "L3"  # Write files / modify calendar / modify notes
    L4 = "L4"  # Mouse & keyboard / smart home / desktop automation
    L5 = "L5"  # Shell / delete files / arbitrary network / privilege escalation


class ToolRuntime(str, Enum):
    """Execution backend for a tool."""

    MEMORY_METHOD = "memory_method"
    INTEGRATION_METHOD = "integration_method"
    PYTHON_GENERATED = "python_generated"
    JS_TOOL = "js_tool"
    COMPUTER_USE = "computer_use"


class ToolStatus(str, Enum):
    """Lifecycle status."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    CANDIDATE = "candidate"  # generated but not yet proven
    BLOCKED = "blocked"


# ---------------------------------------------------------------------------
# Core spec
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    """Canonical description of a single tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    runtime: ToolRuntime = ToolRuntime.MEMORY_METHOD
    provider: str = ""  # e.g. "obsidian", "calendar", "memory"
    entrypoint: str = ""  # e.g. "integrations.obsidian.integration:search"
    risk_level: RiskLevel = RiskLevel.L0
    privacy_scope: str = "local_only"
    version: str = "1.0.0"
    status: ToolStatus = ToolStatus.ACTIVE
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None  # 768-dim vector for semantic search
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    # -- Serialisation helpers ------------------------------------------------

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to the OpenAI function-calling wire format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def to_db_row(self) -> dict[str, Any]:
        """Flatten for SQLite INSERT."""
        import struct

        embedding_blob: bytes | None = None
        if self.embedding is not None:
            embedding_blob = struct.pack(f"{len(self.embedding)}f", *self.embedding)

        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "input_schema": json.dumps(self.input_schema, ensure_ascii=False),
            "output_schema": json.dumps(self.output_schema, ensure_ascii=False) if self.output_schema else None,
            "openai_schema": json.dumps(self.to_openai_schema(), ensure_ascii=False),
            "runtime": self.runtime.value,
            "provider": self.provider,
            "entrypoint": self.entrypoint,
            "risk_level": self.risk_level.value,
            "privacy_scope": self.privacy_scope,
            "status": self.status.value,
            "tags": json.dumps(self.tags, ensure_ascii=False),
            "metadata": json.dumps(self.metadata, ensure_ascii=False),
            "embedding": embedding_blob,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_db_row(cls, row: dict[str, Any]) -> ToolSpec:
        """Reconstruct from a SQLite row."""
        import struct

        embedding: list[float] | None = None
        if row.get("embedding"):
            blob = row["embedding"]
            if isinstance(blob, bytes) and len(blob) > 0:
                embedding = list(struct.unpack(f"{len(blob) // 4}f", blob))

        return cls(
            name=row["name"],
            description=row["description"],
            input_schema=json.loads(row["input_schema"]),
            output_schema=json.loads(row["output_schema"]) if row.get("output_schema") else None,
            runtime=ToolRuntime(row["runtime"]),
            provider=row.get("provider", ""),
            entrypoint=row.get("entrypoint", ""),
            risk_level=RiskLevel(row.get("risk_level", "L0")),
            privacy_scope=row.get("privacy_scope", "local_only"),
            version=row.get("version", "1.0.0"),
            status=ToolStatus(row.get("status", "active")),
            tags=json.loads(row.get("tags", "[]")),
            metadata=json.loads(row.get("metadata", "{}")),
            embedding=embedding,
            created_at=row.get("created_at", ""),
            updated_at=row.get("updated_at", ""),
        )


# ---------------------------------------------------------------------------
# Execution context & result
# ---------------------------------------------------------------------------


@dataclass
class ToolContext:
    """Runtime context passed to every tool execution."""

    session_id: str = ""
    task_id: str = ""
    user_input: str = ""
    model_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Standardised return value from tool execution."""

    content: str
    success: bool = True
    error_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_tool_message(self, tool_call_id: str, name: str) -> dict[str, Any]:
        """Convert to an OpenAI-style tool result message."""
        return {
            "tool_call_id": tool_call_id,
            "name": name,
            "content": self.content,
        }


# ---------------------------------------------------------------------------
# Tool Request (dry-run)
# ---------------------------------------------------------------------------


@dataclass
class ToolRequest:
    """A request for a tool that doesn't yet exist.

    Generated when the agent fails or produces low-confidence output because
    no suitable tool was available.  Recorded for analysis — no code is
    generated at this stage (dry-run).
    """

    task_id: str = ""
    session_id: str = ""
    reason: str = ""  # human-readable: why existing tools were insufficient
    missing_capability: str = ""  # what the tool should do
    candidate_name: str = ""  # suggested tool name
    candidate_description: str = ""
    candidate_input_schema: dict[str, Any] = field(default_factory=dict)
    candidate_output_schema: dict[str, Any] = field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.L0
    privacy_notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "reason": self.reason,
            "missing_capability": self.missing_capability,
            "candidate_name": self.candidate_name,
            "candidate_description": self.candidate_description,
            "candidate_input_schema": self.candidate_input_schema,
            "candidate_output_schema": self.candidate_output_schema,
            "risk_level": self.risk_level.value,
            "privacy_notes": self.privacy_notes,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }
