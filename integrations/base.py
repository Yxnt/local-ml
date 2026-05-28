"""Base class for data source integrations."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class IntegrationConfig:
    """Configuration for an integration."""
    name: str
    enabled: bool = True
    sync_interval: int = 3600  # seconds
    config: dict[str, Any] = field(default_factory=dict)


class Integration(abc.ABC):
    """Abstract base for data source integrations.

    Lifecycle: configure() -> connect() -> sync() -> query() -> disconnect()
    """

    def __init__(self, config: IntegrationConfig):
        self.config = config
        self._connected = False

    @property
    def name(self) -> str:
        return self.config.name

    @abc.abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data source."""

    @abc.abstractmethod
    async def sync(self) -> dict[str, int]:
        """Sync data from the source. Returns stats (e.g., {"new": 10, "updated": 5})."""

    @abc.abstractmethod
    async def query(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search the data source. Returns list of results."""

    @abc.abstractmethod
    def get_tools(self) -> list[dict[str, Any]]:
        """Return tool definitions for the agent."""

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.disconnect()
