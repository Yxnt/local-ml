"""Offline fixtures for local_ml_eval live execution.

These fixtures let the eval harness exercise the real Agent / ToolRegistry /
Telemetry path without requiring live calendar, email, or Obsidian accounts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from server.tools.spec import RiskLevel, ToolRuntime, ToolSpec


class OfflineObsidianIntegration:
    """Small in-memory Obsidian fixture."""

    def __init__(self) -> None:
        self._connected = False
        self._notes = [
            {
                "path": "README.md",
                "title": "README",
                "content": "# Machine Learning\nThis vault contains machine learning notes.",
            },
            {
                "path": "Projects/ML.md",
                "title": "ML Project",
                "content": "项目里有机器学习实验和向量检索记录。",
            },
        ]

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def sync(self) -> dict[str, int]:
        return {"notes": len(self._notes)}

    async def query(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        query_lower = query.lower()
        results = []
        for note in self._notes:
            haystack = f"{note['title']} {note['content']}".lower()
            if query_lower in haystack:
                results.append({
                    "path": note["path"],
                    "title": note["title"],
                    "match": "content",
                    "snippet": note["content"][:120],
                })
            if len(results) >= limit:
                break
        return results

    async def read_note(self, path: str) -> dict[str, Any] | None:
        for note in self._notes:
            if note["path"] == path:
                return {"path": note["path"], "content": note["content"], "metadata": {}}
        return None

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "obsidian_search",
                    "description": "Search Obsidian notes by content, tags, or title.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query."},
                            "limit": {"type": "integer", "description": "Max results."},
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
                            "path": {"type": "string", "description": "Relative note path."},
                        },
                        "required": ["path"],
                    },
                },
            },
        ]


class OfflineCalendarIntegration:
    """Small in-memory calendar fixture."""

    def __init__(self) -> None:
        self._connected = False
        self._events = [
            {"summary": "Today planning", "description": "today agenda and planning", "start": "2026-06-09T09:00:00"},
            {"summary": "Team sync", "description": "weekly meeting", "start": "2026-06-10T14:00:00"},
        ]

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def sync(self) -> dict[str, int]:
        return {"events": len(self._events)}

    async def query(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        query_lower = query.lower()
        results = []
        for event in self._events:
            if query_lower in event["summary"].lower() or query_lower in event["description"].lower():
                results.append(event)
            if len(results) >= limit:
                break
        return results

    async def get_upcoming(self, days: int = 7) -> list[dict[str, Any]]:
        return self._events[:days]

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "calendar_search",
                    "description": "Search calendar events by keyword.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search keyword."},
                            "days": {"type": "integer", "description": "Search ahead N days."},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calendar_upcoming",
                    "description": "Get upcoming calendar events.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "days": {"type": "integer", "description": "Number of days to look ahead."},
                        },
                    },
                },
            },
        ]


class OfflineEmailIntegration:
    """Small in-memory email fixture."""

    def __init__(self) -> None:
        self._connected = False
        self._emails = [
            {"from": "alex@example.com", "subject": "Recent ML update", "date": "2026-06-08"},
            {"from": "meetings@example.com", "subject": "会议安排", "date": "2026-06-07"},
        ]

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def sync(self) -> dict[str, int]:
        return {"emails": len(self._emails)}

    async def query(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        query_lower = query.lower()
        results = []
        for email in self._emails:
            if query_lower in email["subject"].lower():
                results.append(email)
            if len(results) >= limit:
                break
        return results

    async def get_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._emails[:limit]

    def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "email_search",
                    "description": "Search emails by subject or keyword.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search keyword."},
                            "limit": {"type": "integer", "description": "Max results."},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "email_recent",
                    "description": "Get recent emails from inbox.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "description": "Number of emails."},
                        },
                    },
                },
            },
        ]


_GENERATED_TOOL_FIXTURES: dict[str, dict[str, Any]] = {
    "text_count_words": {
        "description": "Count words in input text.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "output_schema": {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        },
        "source": """from pydantic import BaseModel\n\nclass InputModel(BaseModel):\n    text: str\n\nclass OutputModel(BaseModel):\n    count: int\n\ndef run(input: InputModel) -> OutputModel:\n    return OutputModel(count=len(input.text.split()))\n""",
    },
    "extract_emails": {
        "description": "Extract all email addresses from input text.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "output_schema": {
            "type": "object",
            "properties": {"emails": {"type": "array", "items": {"type": "string"}}},
            "required": ["emails"],
        },
        "source": """import re\nfrom pydantic import BaseModel\n\nclass InputModel(BaseModel):\n    text: str\n\nclass OutputModel(BaseModel):\n    emails: list[str]\n\ndef run(input: InputModel) -> OutputModel:\n    emails = re.findall(r\"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\\\.[A-Za-z]{2,}\", input.text)\n    return OutputModel(emails=emails)\n""",
    },
    "json_pretty_print": {
        "description": "Format JSON text with indentation.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "output_schema": {
            "type": "object",
            "properties": {"pretty_json": {"type": "string"}},
            "required": ["pretty_json"],
        },
        "source": """import json\nfrom pydantic import BaseModel\n\nclass InputModel(BaseModel):\n    text: str\n\nclass OutputModel(BaseModel):\n    pretty_json: str\n\ndef run(input: InputModel) -> OutputModel:\n    parsed = json.loads(input.text)\n    return OutputModel(pretty_json=json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True))\n""",
    },
    "date_diff_days": {
        "description": "Compute the day difference between two ISO dates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
            },
            "required": ["start_date", "end_date"],
        },
        "output_schema": {
            "type": "object",
            "properties": {"days": {"type": "integer"}},
            "required": ["days"],
        },
        "source": """from datetime import date\nfrom pydantic import BaseModel\n\nclass InputModel(BaseModel):\n    start_date: str\n    end_date: str\n\nclass OutputModel(BaseModel):\n    days: int\n\ndef run(input: InputModel) -> OutputModel:\n    start = date.fromisoformat(input.start_date)\n    end = date.fromisoformat(input.end_date)\n    return OutputModel(days=(end - start).days)\n""",
    },
    "markdown_to_text": {
        "description": "Convert Markdown text to simplified plain text.",
        "input_schema": {
            "type": "object",
            "properties": {"markdown": {"type": "string"}},
            "required": ["markdown"],
        },
        "output_schema": {
            "type": "object",
            "properties": {"plain_text": {"type": "string"}},
            "required": ["plain_text"],
        },
        "source": """import re\nfrom pydantic import BaseModel\n\nclass InputModel(BaseModel):\n    markdown: str\n\nclass OutputModel(BaseModel):\n    plain_text: str\n\ndef run(input: InputModel) -> OutputModel:\n    text = re.sub(r\"`([^`]*)`\", r\"\\\\1\", input.markdown)\n    text = re.sub(r\"\\[(.*?)\\]\\((.*?)\\)\", r\"\\\\1\", text)\n    text = re.sub(r\"^#+\\s*\", \"\", text, flags=re.MULTILINE)\n    text = re.sub(r\"[*_>-]\", \" \", text)\n    text = re.sub(r\"\\s+\", \" \", text).strip()\n    return OutputModel(plain_text=text)\n""",
    },
}


async def install_fixture_integrations(agent: Any, mode: str = "offline") -> None:
    """Connect offline fixture integrations for eval tasks."""
    if mode == "none":
        return
    if mode != "offline":
        raise ValueError(f"Unknown integration fixture mode: {mode}")

    for name, integration in {
        "obsidian": OfflineObsidianIntegration(),
        "calendar": OfflineCalendarIntegration(),
        "email": OfflineEmailIntegration(),
    }.items():
        await agent.connect_integration(name, integration)


def install_generated_tool_fixtures(agent: Any) -> None:
    """Register and materialize generated tool fixtures for generated_tool tasks."""
    registry = getattr(agent, "_tool_registry", None)
    router = getattr(agent, "_tool_router", None)
    if registry is None or router is None:
        raise RuntimeError("Agent tool registry is not available")

    generated_executor = router._executors[ToolRuntime.PYTHON_GENERATED]
    sandbox_dir = Path(generated_executor._sandbox_dir)
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    for name, fixture in _GENERATED_TOOL_FIXTURES.items():
        (sandbox_dir / f"{name}.py").write_text(fixture["source"], encoding="utf-8")
        if registry.get_tool(name) is not None:
            continue

        registry.register(
            ToolSpec(
                name=name,
                description=fixture["description"],
                input_schema=fixture["input_schema"],
                output_schema=fixture["output_schema"],
                runtime=ToolRuntime.PYTHON_GENERATED,
                provider="generated_fixture",
                risk_level=RiskLevel.L0,
            )
        )
