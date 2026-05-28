"""Calendar integration via CalDAV.

Supports:
- Apple Calendar (via CalDAV)
- Google Calendar (via CalDAV)
- Any CalDAV-compatible server
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from integrations.base import Integration, IntegrationConfig


class CalendarIntegration(Integration):
    """Calendar integration via CalDAV protocol."""

    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self._client = None
        self._calendar = None

    async def connect(self) -> None:
        """Connect to CalDAV server."""
        try:
            import caldav
        except ImportError:
            raise ImportError("pip install caldav")

        url = self.config.config.get("url")
        username = self.config.config.get("username")
        password = self.config.config.get("password")

        if not url:
            raise ValueError("Calendar URL not configured")

        self._client = caldav.DAVClient(url, username=username, password=password)
        principal = self._client.principal()

        calendar_name = self.config.config.get("calendar", "calendar")
        calendars = principal.calendars()
        self._calendar = next((c for c in calendars if calendar_name in c.name), calendars[0] if calendars else None)

        if not self._calendar:
            raise RuntimeError("No calendar found")

        self._connected = True

    async def disconnect(self) -> None:
        self._client = None
        self._calendar = None
        self._connected = False

    async def sync(self) -> dict[str, int]:
        """Fetch upcoming events."""
        if not self._calendar:
            raise RuntimeError("Not connected")

        # Fetch events for the next 30 days
        now = datetime.now()
        end = now + timedelta(days=30)

        events = self._calendar.date_search(now, end)
        return {"events": len(events)}

    async def query(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search events by summary/description."""
        if not self._calendar:
            raise RuntimeError("Not connected")

        now = datetime.now()
        end = now + timedelta(days=90)

        events = self._calendar.date_search(now, end)
        results = []

        for event in events:
            vevent = event.vobject_instance.vevent
            summary = vevent.summary.value if hasattr(vevent, "summary") else ""
            description = vevent.description.value if hasattr(vevent, "description") else ""

            if query.lower() in summary.lower() or query.lower() in description.lower():
                results.append(self._format_event(event))
                if len(results) >= limit:
                    break

        return results

    async def get_upcoming(self, days: int = 7) -> list[dict[str, Any]]:
        """Get upcoming events for the next N days."""
        if not self._calendar:
            raise RuntimeError("Not connected")

        now = datetime.now()
        end = now + timedelta(days=days)

        events = self._calendar.date_search(now, end)
        return [self._format_event(e) for e in events]

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
                            "days": {"type": "integer", "description": "Search ahead N days (default: 90)."},
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
                            "days": {"type": "integer", "description": "Number of days to look ahead (default: 7)."},
                        },
                    },
                },
            },
        ]

    @staticmethod
    def _format_event(event) -> dict[str, Any]:
        """Format a CalDAV event into a dict."""
        vevent = event.vobject_instance.vevent

        result = {
            "uid": vevent.uid.value if hasattr(vevent, "uid") else "",
            "summary": vevent.summary.value if hasattr(vevent, "summary") else "",
            "description": vevent.description.value if hasattr(vevent, "description") else "",
        }

        if hasattr(vevent, "dtstart"):
            result["start"] = vevent.dtstart.value.isoformat()
        if hasattr(vevent, "dtend"):
            result["end"] = vevent.dtend.value.isoformat()
        if hasattr(vevent, "location"):
            result["location"] = vevent.location.value

        return result
