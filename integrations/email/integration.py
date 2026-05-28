"""Email integration via IMAP.

Supports:
- Gmail (via IMAP)
- Outlook (via IMAP)
- Any IMAP-compatible server
"""

from __future__ import annotations

import email
from datetime import datetime, timedelta
from email.header import decode_header
from typing import Any

from integrations.base import Integration, IntegrationConfig


class EmailIntegration(Integration):
    """Email integration via IMAP protocol."""

    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self._imap = None

    async def connect(self) -> None:
        """Connect to IMAP server."""
        import imaplib

        host = self.config.config.get("host")
        username = self.config.config.get("username")
        password = self.config.config.get("password")
        port = self.config.config.get("port", 993)

        if not host or not username or not password:
            raise ValueError("Email host/username/password not configured")

        self._imap = imaplib.IMAP4_SSL(host, port)
        self._imap.login(username, password)
        self._imap.select("INBOX")
        self._connected = True

    async def disconnect(self) -> None:
        if self._imap:
            try:
                self._imap.logout()
            except Exception:
                pass
        self._imap = None
        self._connected = False

    async def sync(self) -> dict[str, int]:
        """Fetch recent emails."""
        if not self._imap:
            raise RuntimeError("Not connected")

        # Search for emails from the last 7 days
        since = (datetime.now() - timedelta(days=7)).strftime("%d-%b-%Y")
        status, messages = self._imap.search(None, f'(SINCE "{since}")')

        if status != "OK":
            return {"emails": 0}

        email_ids = messages[0].split()
        return {"emails": len(email_ids)}

    async def query(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search emails by subject/body."""
        if not self._imap:
            raise RuntimeError("Not connected")

        # Search by subject
        status, messages = self._imap.search(None, f'(SUBJECT "{query}")')
        if status != "OK":
            return []

        email_ids = messages[0].split()[-limit:]
        results = []

        for eid in email_ids:
            status, msg_data = self._imap.fetch(eid, "(RFC822)")
            if status != "OK":
                continue

            msg = email.message_from_bytes(msg_data[0][1])
            results.append(self._format_email(msg))

        return results

    async def get_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent emails."""
        if not self._imap:
            raise RuntimeError("Not connected")

        status, messages = self._imap.search(None, "ALL")
        if status != "OK":
            return []

        email_ids = messages[0].split()[-limit:]
        results = []

        for eid in email_ids:
            status, msg_data = self._imap.fetch(eid, "(RFC822)")
            if status != "OK":
                continue

            msg = email.message_from_bytes(msg_data[0][1])
            results.append(self._format_email(msg))

        return results

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
                            "limit": {"type": "integer", "description": "Max results (default: 10)."},
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
                            "limit": {"type": "integer", "description": "Number of emails (default: 20)."},
                        },
                    },
                },
            },
        ]

    @staticmethod
    def _format_email(msg) -> dict[str, Any]:
        """Format an email message into a dict."""
        subject = msg.get("Subject", "")
        if subject:
            decoded = decode_header(subject)
            subject = decoded[0][0]
            if isinstance(subject, bytes):
                subject = subject.decode(decoded[0][1] or "utf-8", errors="ignore")

        return {
            "from": msg.get("From", ""),
            "to": msg.get("To", ""),
            "subject": subject,
            "date": msg.get("Date", ""),
            "message_id": msg.get("Message-ID", ""),
        }
