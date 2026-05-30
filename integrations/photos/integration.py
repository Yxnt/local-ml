"""Apple Photos integration - ties together indexer, VLM analyzer, and tools."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from integrations.base import Integration, IntegrationConfig
from integrations.photos.indexer import PhotosIndexer
from integrations.photos.tools import get_photos_tools
from integrations.photos.vlm_analyzer import VLMAnalyzer


class PhotosIntegration(Integration):
    """Integration with the Apple Photos library.

    Features:
    - Index photo metadata from the macOS Photos library
    - Search photos by keyword, date range, album, or tags
    - Describe photo contents using a vision-language model

    Required config keys:
        photos_library: Path to the Photos library (e.g. ~/Pictures/Photos Library.photoslibrary)
        db_path:        Path to the SQLite cache database (default: ./photos.db)
        model_name:     Name of the VLM model for describe (default: minicpm-v-4.6)
    """

    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self._indexer: PhotosIndexer | None = None
        self._vlm: VLMAnalyzer | None = None

    async def connect(self) -> None:
        """Initialize the indexer and VLM analyzer from config."""
        library_path = self.config.config.get("photos_library")
        if not library_path:
            raise ValueError("photos_library not configured")

        db_path = self.config.config.get("db_path", "photos.db")
        model_name = self.config.config.get("model_name", "minicpm-v-4.6")

        self._indexer = PhotosIndexer(library_path=library_path, db_path=db_path)
        self._indexer.init_db()

        self._vlm = VLMAnalyzer(model_name=model_name, db_path=db_path)
        self._vlm.init_db()

        self._connected = True

    async def disconnect(self) -> None:
        """Close database connections."""
        if self._indexer:
            self._indexer.close()
            self._indexer = None
        if self._vlm:
            self._vlm.close()
            self._vlm = None
        self._connected = False

    async def sync(self) -> dict[str, int]:
        """Sync photo metadata from the Photos library."""
        if not self._indexer:
            raise RuntimeError("Not connected")
        return self._indexer.sync_sync()

    async def query(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search cached photo metadata."""
        if not self._indexer:
            raise RuntimeError("Not connected")
        return self._indexer.search_sync(query=query, limit=limit)

    def get_tools(self) -> list[dict[str, Any]]:
        """Return tool definitions for the agent."""
        return get_photos_tools()

    # ------------------------------------------------------------------
    # tool handlers
    # ------------------------------------------------------------------

    async def handle_search(self, **kwargs: Any) -> dict[str, Any]:
        """Handle a photos_search tool call."""
        if not self._indexer:
            raise RuntimeError("Not connected")

        query = kwargs.get("query")
        date_from_str = kwargs.get("date_from")
        date_to_str = kwargs.get("date_to")
        album = kwargs.get("album")
        limit = kwargs.get("limit", 10)

        date_from = datetime.fromisoformat(date_from_str) if date_from_str else None
        date_to = datetime.fromisoformat(date_to_str) if date_to_str else None

        results = self._indexer.search_sync(
            query=query,
            date_from=date_from,
            date_to=date_to,
            album=album,
            limit=limit,
        )
        return {"results": results, "count": len(results)}

    async def handle_list_albums(self, **kwargs: Any) -> dict[str, Any]:
        """Handle a photos_list_albums tool call."""
        if not self._indexer:
            raise RuntimeError("Not connected")

        albums = self._indexer.list_albums_sync()
        return {"albums": albums, "count": len(albums)}

    async def handle_describe(self, **kwargs: Any) -> dict[str, Any]:
        """Handle a photos_describe tool call."""
        if not self._indexer:
            raise RuntimeError("Not connected")
        if not self._vlm:
            raise RuntimeError("Not connected")

        photo_id = kwargs.get("photo_id")
        if not photo_id:
            raise ValueError("photo_id is required")

        question = kwargs.get("question", "Describe this photo in detail.")

        photo_path = self._indexer.get_photo_path(photo_id)
        if not photo_path:
            return {"error": f"Photo not found: {photo_id}"}

        analysis = self._vlm.analyze(photo_path, question)
        return {"photo_id": photo_id, "analysis": analysis}
