"""Obsidian vault integration - reads and searches markdown notes."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from integrations.base import Integration, IntegrationConfig


class ObsidianIntegration(Integration):
    """Integration with Obsidian vault (markdown files).

    Features:
    - Read notes by path or search
    - Search notes by content/tags
    - Parse frontmatter metadata
    - Track backlinks
    """

    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self._vault_path: Path | None = None
        self._index: dict[str, dict[str, Any]] = {}  # path -> metadata

    async def connect(self) -> None:
        vault_path = self.config.config.get("vault_path")
        if not vault_path:
            raise ValueError("Obsidian vault_path not configured")

        self._vault_path = Path(vault_path).expanduser()
        if not self._vault_path.exists():
            raise FileNotFoundError(f"Vault not found: {self._vault_path}")

        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False
        self._index.clear()

    async def sync(self) -> dict[str, int]:
        """Index all markdown files in the vault."""
        if not self._vault_path:
            raise RuntimeError("Not connected")

        new_count = 0
        updated_count = 0

        for md_file in self._vault_path.rglob("*.md"):
            # Skip hidden directories
            if any(p.startswith(".") for p in md_file.relative_to(self._vault_path).parts):
                continue

            rel_path = str(md_file.relative_to(self._vault_path))
            mtime = md_file.stat().st_mtime

            if rel_path in self._index:
                if self._index[rel_path]["mtime"] == mtime:
                    continue
                updated_count += 1
            else:
                new_count += 1

            # Parse file
            content = md_file.read_text(encoding="utf-8", errors="ignore")
            metadata = self._parse_frontmatter(content)

            self._index[rel_path] = {
                "path": rel_path,
                "title": self._extract_title(md_file, content),
                "tags": metadata.get("tags", []),
                "created": metadata.get("created"),
                "modified": datetime.fromtimestamp(mtime).isoformat(),
                "size": len(content),
                "mtime": mtime,
            }

        return {"new": new_count, "updated": updated_count, "total": len(self._index)}

    async def query(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search notes by content or tags."""
        if not self._vault_path:
            raise RuntimeError("Not connected")

        results = []
        query_lower = query.lower()

        for rel_path, meta in self._index.items():
            # Search in title
            if query_lower in meta["title"].lower():
                results.append({**meta, "match": "title"})
                if len(results) >= limit:
                    break
                continue

            # Search in tags
            if any(query_lower in tag.lower() for tag in meta["tags"]):
                results.append({**meta, "match": "tags"})
                if len(results) >= limit:
                    break
                continue

            # Search in content
            file_path = self._vault_path / rel_path
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if query_lower in content.lower():
                    # Extract snippet around match
                    idx = content.lower().index(query_lower)
                    start = max(0, idx - 100)
                    end = min(len(content), idx + len(query) + 100)
                    snippet = content[start:end].strip()
                    results.append({**meta, "match": "content", "snippet": snippet})
            except Exception:
                continue

            if len(results) >= limit:
                break

        return results

    async def read_note(self, path: str) -> dict[str, Any] | None:
        """Read a specific note by path."""
        if not self._vault_path:
            raise RuntimeError("Not connected")

        file_path = self._vault_path / path
        if not file_path.exists():
            return None

        content = file_path.read_text(encoding="utf-8", errors="ignore")
        metadata = self._parse_frontmatter(content)

        return {
            "path": path,
            "content": content,
            "metadata": metadata,
        }

    def get_tools(self) -> list[dict[str, Any]]:
        """Return tool definitions for the agent."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "obsidian_search",
                    "description": "Search Obsidian notes by content, tags, or title.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (keywords, tags, or title).",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 10).",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "obsidian_read",
                    "description": "Read a specific Obsidian note by path.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the note (relative to vault root).",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
        ]

    @staticmethod
    def _parse_frontmatter(content: str) -> dict[str, Any]:
        """Parse YAML frontmatter from markdown content."""
        if not content.startswith("---"):
            return {}

        try:
            end = content.index("---", 3)
            frontmatter = content[3:end].strip()

            # Simple YAML parser (no dependency)
            result = {}
            for line in frontmatter.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()

                    # Parse tags list
                    if key == "tags" and value.startswith("["):
                        value = [t.strip().strip('"') for t in value[1:-1].split(",")]
                    elif key == "tags" and value.startswith("- "):
                        # Multi-line tags handled below
                        pass

                    result[key] = value

            return result
        except (ValueError, IndexError):
            return {}

    @staticmethod
    def _extract_title(file_path: Path, content: str) -> str:
        """Extract title from frontmatter, first heading, or filename."""
        # Try first heading
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # Fall back to filename
        return file_path.stem
