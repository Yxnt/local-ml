"""Tool definitions for Apple Photos agent integration."""

from __future__ import annotations

from typing import Any


def get_photos_tools() -> list[dict[str, Any]]:
    """Return tool definitions for the Photos agent."""
    return [
        {
            "type": "function",
            "function": {
                "name": "photos_search",
                "description": "Search Apple Photos by keyword, date range, album, or tags.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (keywords, tags, location, description).",
                        },
                        "date_from": {
                            "type": "string",
                            "description": "Start date in ISO format (YYYY-MM-DD).",
                        },
                        "date_to": {
                            "type": "string",
                            "description": "End date in ISO format (YYYY-MM-DD).",
                        },
                        "album": {
                            "type": "string",
                            "description": "Filter by album name.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default: 10).",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "photos_list_albums",
                "description": "List all albums in the Apple Photos library.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "photos_describe",
                "description": "Describe the content of a photo using the vision model.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "photo_id": {
                            "type": "string",
                            "description": "UUID of the photo to describe.",
                        },
                        "question": {
                            "type": "string",
                            "description": "Optional question about the photo content.",
                        },
                    },
                    "required": ["photo_id"],
                },
            },
        },
    ]
