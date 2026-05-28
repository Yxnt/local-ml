"""Data source integrations for the local AI agent.

Each integration provides:
- Connection/authentication
- Data fetching and indexing
- Search and retrieval
- Tool definitions for the agent
"""

from integrations.base import Integration, IntegrationConfig

__all__ = ["Integration", "IntegrationConfig"]
