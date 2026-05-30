# Apple Photos Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate Apple Photos into the local AI agent with incremental metadata indexing, VLM-based image analysis, and tool definitions for the agent.

**Architecture:** PhotosIntegration extends the existing Integration base class. PhotosIndexer uses osxphotos to sync metadata into a local SQLite cache. VLMAnalyzer provides image understanding with result caching. Three tools are exposed: photos_search, photos_list_albums, photos_describe.

**Tech Stack:** Python (osxphotos, sqlite3, mlx_vlm), existing Integration pattern

---

## File Structure

| File | Role |
|------|------|
| `integrations/photos/__init__.py` | Package exports |
| `integrations/photos/integration.py` | PhotosIntegration (main entry) |
| `integrations/photos/indexer.py` | PhotosIndexer (osxphotos + SQLite cache) |
| `integrations/photos/vlm_analyzer.py` | VLMAnalyzer (image analysis + cache) |
| `integrations/photos/tools.py` | Tool definitions |
| `integrations/photos/test_integration.py` | Integration tests |

---

### Task 1: Create indexer.py (metadata indexing)

**Files:**
- Create: `integrations/photos/indexer.py`
- Create: `integrations/photos/test_integration.py`

- [ ] **Step 1: Write failing tests**

```python
# integrations/photos/test_integration.py
"""Tests for Apple Photos integration."""

import json
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from integrations.photos.indexer import PhotosIndexer


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "photos.db")


@pytest.fixture
def indexer(db_path):
    return PhotosIndexer(library_path="/fake/path", db_path=db_path)


class TestPhotosIndexer:
    def test_init_creates_database(self, indexer, db_path):
        """Indexer initializes and creates SQLite database."""
        indexer.init_db()
        conn = sqlite3.connect(db_path)
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = [t[0] for t in tables]
        assert "photos" in table_names
        assert "photo_analyses" in table_names
        conn.close()

    def test_insert_and_get(self, indexer, db_path):
        """Indexer can insert and retrieve photos."""
        indexer.init_db()
        indexer._insert({
            "uuid": "test-uuid-1",
            "path": "/photos/test.jpg",
            "filename": "test.jpg",
            "created_at": "2024-01-01T00:00:00",
            "modified_at": "2024-01-01T00:00:00",
            "latitude": 35.6762,
            "longitude": 139.6503,
            "location_name": "Tokyo",
            "album": "Travel",
            "description": "A photo in Tokyo",
            "tags": '["travel", "japan"]',
        })
        result = indexer._get_cached("test-uuid-1")
        assert result is not None
        assert result["filename"] == "test.jpg"
        assert result["location_name"] == "Tokyo"

    def test_search_by_location(self, indexer, db_path):
        """Indexer can search by location name."""
        indexer.init_db()
        indexer._insert({
            "uuid": "uuid-1",
            "path": "/p1.jpg",
            "filename": "p1.jpg",
            "created_at": "2024-01-01",
            "modified_at": "2024-01-01",
            "latitude": 35.67,
            "longitude": 139.65,
            "location_name": "Tokyo, Japan",
            "album": "Travel",
            "description": "",
            "tags": "[]",
        })
        indexer._insert({
            "uuid": "uuid-2",
            "path": "/p2.jpg",
            "filename": "p2.jpg",
            "created_at": "2024-02-01",
            "modified_at": "2024-02-01",
            "latitude": 48.85,
            "longitude": 2.35,
            "location_name": "Paris, France",
            "album": "Travel",
            "description": "",
            "tags": "[]",
        })
        results = indexer.search_sync("Tokyo")
        assert len(results) == 1
        assert results[0]["location_name"] == "Tokyo, Japan"

    def test_search_by_date_range(self, indexer, db_path):
        """Indexer can search by date range."""
        indexer.init_db()
        indexer._insert({
            "uuid": "uuid-1",
            "path": "/p1.jpg",
            "filename": "p1.jpg",
            "created_at": "2024-01-15",
            "modified_at": "2024-01-15",
            "latitude": None,
            "longitude": None,
            "location_name": None,
            "album": "",
            "description": "",
            "tags": "[]",
        })
        indexer._insert({
            "uuid": "uuid-2",
            "path": "/p2.jpg",
            "filename": "p2.jpg",
            "created_at": "2024-06-15",
            "modified_at": "2024-06-15",
            "latitude": None,
            "longitude": None,
            "location_name": None,
            "album": "",
            "description": "",
            "tags": "[]",
        })
        results = indexer.search_sync("", date_from="2024-01-01", date_to="2024-03-01")
        assert len(results) == 1
        assert results[0]["uuid"] == "uuid-1"

    def test_search_by_tags(self, indexer, db_path):
        """Indexer can search by tags."""
        indexer.init_db()
        indexer._insert({
            "uuid": "uuid-1",
            "path": "/p1.jpg",
            "filename": "p1.jpg",
            "created_at": "2024-01-01",
            "modified_at": "2024-01-01",
            "latitude": None,
            "longitude": None,
            "location_name": None,
            "album": "",
            "description": "",
            "tags": '["cat", "pet"]',
        })
        results = indexer.search_sync("cat")
        assert len(results) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest integrations/photos/test_integration.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement indexer**

```python
# integrations/photos/indexer.py
"""Incremental metadata indexer for Apple Photos using osxphotos."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PhotosIndexer:
    """Indexes Apple Photos metadata into a local SQLite cache."""

    def __init__(self, library_path: str, db_path: str):
        self._library_path = library_path
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def init_db(self) -> None:
        """Initialize the SQLite database."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS photos (
                uuid TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                filename TEXT,
                created_at TEXT,
                modified_at TEXT,
                latitude REAL,
                longitude REAL,
                location_name TEXT,
                album TEXT,
                description TEXT,
                tags TEXT,
                has_vlm_analysis BOOLEAN DEFAULT FALSE
            );
            CREATE TABLE IF NOT EXISTS photo_analyses (
                photo_id TEXT NOT NULL,
                question TEXT DEFAULT '',
                analysis TEXT NOT NULL,
                created_at TEXT,
                PRIMARY KEY (photo_id, question)
            );
            CREATE INDEX IF NOT EXISTS idx_photos_location ON photos(location_name);
            CREATE INDEX IF NOT EXISTS idx_photos_created ON photos(created_at);
        """)
        self._conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def sync_sync(self) -> dict[str, int]:
        """Synchronous sync using osxphotos."""
        import osxphotos

        if not self._conn:
            self.init_db()

        photosdb = osxphotos.PhotosDB(self._library_path)
        photos = photosdb.photos()

        new_count = 0
        updated_count = 0

        for photo in photos:
            existing = self._get_cached(photo.uuid)
            modified_at = str(photo.date_modified) if photo.date_modified else str(photo.date)

            if existing and existing["modified_at"] == modified_at:
                continue

            metadata = {
                "uuid": photo.uuid,
                "path": photo.path or "",
                "filename": photo.filename or "",
                "created_at": str(photo.date),
                "modified_at": modified_at,
                "latitude": photo.latitude,
                "longitude": photo.longitude,
                "location_name": photo.place.name if photo.place else None,
                "album": ", ".join(a.title for a in photo.albums) if photo.albums else "",
                "description": photo.description or "",
                "tags": json.dumps(photo.keywords or []),
            }

            if existing:
                self._update(metadata)
                updated_count += 1
            else:
                self._insert(metadata)
                new_count += 1

        return {"new": new_count, "updated": updated_count, "total": len(photos)}

    def _get_cached(self, uuid: str) -> dict[str, Any] | None:
        """Get cached photo metadata."""
        row = self._conn.execute("SELECT * FROM photos WHERE uuid = ?", (uuid,)).fetchone()
        return dict(row) if row else None

    def _insert(self, metadata: dict[str, Any]) -> None:
        """Insert new photo metadata."""
        self._conn.execute(
            """INSERT INTO photos (uuid, path, filename, created_at, modified_at,
               latitude, longitude, location_name, album, description, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (metadata["uuid"], metadata["path"], metadata["filename"],
             metadata["created_at"], metadata["modified_at"],
             metadata["latitude"], metadata["longitude"],
             metadata["location_name"], metadata["album"],
             metadata["description"], metadata["tags"])
        )
        self._conn.commit()

    def _update(self, metadata: dict[str, Any]) -> None:
        """Update existing photo metadata."""
        self._conn.execute(
            """UPDATE photos SET path=?, filename=?, created_at=?, modified_at=?,
               latitude=?, longitude=?, location_name=?, album=?, description=?, tags=?
               WHERE uuid=?""",
            (metadata["path"], metadata["filename"],
             metadata["created_at"], metadata["modified_at"],
             metadata["latitude"], metadata["longitude"],
             metadata["location_name"], metadata["album"],
             metadata["description"], metadata["tags"],
             metadata["uuid"])
        )
        self._conn.commit()

    def search_sync(
        self,
        query: str = "",
        date_from: str | None = None,
        date_to: str | None = None,
        album: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search photos by metadata."""
        if not self._conn:
            self.init_db()

        conditions = []
        params: list[Any] = []

        if query:
            conditions.append(
                "(filename LIKE ? OR description LIKE ? OR location_name LIKE ? OR tags LIKE ?)"
            )
            pattern = f"%{query}%"
            params.extend([pattern, pattern, pattern, pattern])

        if date_from:
            conditions.append("created_at >= ?")
            params.append(date_from)

        if date_to:
            conditions.append("created_at <= ?")
            params.append(date_to)

        if album:
            conditions.append("album LIKE ?")
            params.append(f"%{album}%")

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM photos WHERE {where} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def get_photo_path(self, uuid: str) -> str | None:
        """Get photo file path by UUID."""
        if not self._conn:
            return None
        row = self._conn.execute("SELECT path FROM photos WHERE uuid = ?", (uuid,)).fetchone()
        return row["path"] if row else None

    def list_albums_sync(self) -> list[str]:
        """List all unique album names."""
        if not self._conn:
            self.init_db()
        rows = self._conn.execute("SELECT DISTINCT album FROM photos WHERE album != ''").fetchall()
        albums = set()
        for row in rows:
            for album in row["album"].split(", "):
                if album.strip():
                    albums.add(album.strip())
        return sorted(albums)

    def get_stats(self) -> dict[str, int]:
        """Get indexer statistics."""
        if not self._conn:
            return {"total": 0}
        total = self._conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
        analyzed = self._conn.execute("SELECT COUNT(*) FROM photos WHERE has_vlm_analysis = TRUE").fetchone()[0]
        return {"total": total, "analyzed": analyzed}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest integrations/photos/test_integration.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add integrations/photos/indexer.py integrations/photos/test_integration.py
git commit -m "feat: add PhotosIndexer with incremental metadata indexing"
```

---

### Task 2: Create vlm_analyzer.py (image analysis)

**Files:**
- Create: `integrations/photos/vlm_analyzer.py`
- Modify: `integrations/photos/test_integration.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to integrations/photos/test_integration.py

from integrations.photos.vlm_analyzer import VLMAnalyzer


class TestVLMAnalyzer:
    def test_cache_and_retrieve(self, db_path):
        """VLM analyzer caches and retrieves results."""
        analyzer = VLMAnalyzer(model_name="test", db_path=db_path)
        analyzer.init_db()

        # Cache a result
        analyzer._cache_result("photo-1", "Is there a cat?", "Yes, there is a cat in the photo.")

        # Retrieve it
        cached = analyzer._get_cached("photo-1", "Is there a cat?")
        assert cached == "Yes, there is a cat in the photo."

    def test_cache_different_questions(self, db_path):
        """Different questions for same photo are cached separately."""
        analyzer = VLMAnalyzer(model_name="test", db_path=db_path)
        analyzer.init_db()

        analyzer._cache_result("photo-1", "What is this?", "A cat.")
        analyzer._cache_result("photo-1", "What color?", "Orange.")

        assert analyzer._get_cached("photo-1", "What is this?") == "A cat."
        assert analyzer._get_cached("photo-1", "What color?") == "Orange."

    def test_cache_miss(self, db_path):
        """Cache miss returns None."""
        analyzer = VLMAnalyzer(model_name="test", db_path=db_path)
        analyzer.init_db()

        assert analyzer._get_cached("nonexistent", "question") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest integrations/photos/test_integration.py::TestVLMAnalyzer -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement VLMAnalyzer**

```python
# integrations/photos/vlm_analyzer.py
"""VLM-based image analysis with caching."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class VLMAnalyzer:
    """Analyzes photos using a VLM model with result caching."""

    def __init__(self, model_name: str, db_path: str):
        self._model_name = model_name
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._backend: Any = None

    def init_db(self) -> None:
        """Initialize database connection."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS photo_analyses (
                photo_id TEXT NOT NULL,
                question TEXT DEFAULT '',
                analysis TEXT NOT NULL,
                created_at TEXT,
                PRIMARY KEY (photo_id, question)
            );
        """)
        self._conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _get_cached(self, photo_id: str, question: str) -> str | None:
        """Get cached analysis result."""
        if not self._conn:
            return None
        row = self._conn.execute(
            "SELECT analysis FROM photo_analyses WHERE photo_id = ? AND question = ?",
            (photo_id, question or "")
        ).fetchone()
        return row["analysis"] if row else None

    def _cache_result(self, photo_id: str, question: str, analysis: str) -> None:
        """Cache analysis result."""
        if not self._conn:
            self.init_db()
        self._conn.execute(
            "INSERT OR REPLACE INTO photo_analyses (photo_id, question, analysis, created_at) VALUES (?, ?, ?, ?)",
            (photo_id, question or "", analysis, datetime.now().isoformat())
        )
        self._conn.commit()

    async def analyze(self, photo_path: str, question: str | None = None) -> str:
        """Analyze a photo using the VLM model.

        Args:
            photo_path: Path to the photo file.
            question: Optional question about the photo.

        Returns:
            Analysis text from the VLM.
        """
        if not self._backend:
            from backends.registry import ModelRegistry
            registry = ModelRegistry()
            registry.register_defaults()
            self._backend = await registry.get_or_load(self._model_name)

        prompt = question or "Describe this photo in detail."
        result = self._backend.generate(
            prompt=f"{prompt}\n\n[Image: {photo_path}]",
            max_tokens=512,
        )
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest integrations/photos/test_integration.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add integrations/photos/vlm_analyzer.py integrations/photos/test_integration.py
git commit -m "feat: add VLMAnalyzer with result caching"
```

---

### Task 3: Create tools.py (tool definitions)

**Files:**
- Create: `integrations/photos/tools.py`
- Modify: `integrations/photos/test_integration.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to integrations/photos/test_integration.py

from integrations.photos.tools import get_photos_tools


class TestPhotosTools:
    def test_tools_count(self):
        """Returns 3 tool definitions."""
        tools = get_photos_tools()
        assert len(tools) == 3

    def test_search_tool(self):
        """photos_search tool has correct schema."""
        tools = get_photos_tools()
        search = next(t for t in tools if t["function"]["name"] == "photos_search")
        params = search["function"]["parameters"]["properties"]
        assert "query" in params
        assert "date_from" in params
        assert "date_to" in params
        assert "album" in params
        assert "limit" in params

    def test_list_albums_tool(self):
        """photos_list_albums tool has correct schema."""
        tools = get_photos_tools()
        albums = next(t for t in tools if t["function"]["name"] == "photos_list_albums")
        assert "description" in albums["function"]

    def test_describe_tool(self):
        """photos_describe tool has correct schema."""
        tools = get_photos_tools()
        describe = next(t for t in tools if t["function"]["name"] == "photos_describe")
        params = describe["function"]["parameters"]["properties"]
        assert "photo_id" in params
        assert "question" in params
        assert "photo_id" in describe["function"]["parameters"]["required"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest integrations/photos/test_integration.py::TestPhotosTools -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement tools**

```python
# integrations/photos/tools.py
"""Tool definitions for Apple Photos integration."""

from __future__ import annotations

from typing import Any


def get_photos_tools() -> list[dict[str, Any]]:
    """Return tool definitions for the agent."""
    return [
        {
            "type": "function",
            "function": {
                "name": "photos_search",
                "description": "Search photos by time, location, tags, or description.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search keyword",
                        },
                        "date_from": {
                            "type": "string",
                            "description": "Start date (YYYY-MM-DD)",
                        },
                        "date_to": {
                            "type": "string",
                            "description": "End date (YYYY-MM-DD)",
                        },
                        "album": {
                            "type": "string",
                            "description": "Album name",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default: 10)",
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
                "description": "List all photo albums.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "photos_describe",
                "description": "Analyze photo content using VLM.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "photo_id": {
                            "type": "string",
                            "description": "Photo UUID from search results",
                        },
                        "question": {
                            "type": "string",
                            "description": "Question about the photo (optional)",
                        },
                    },
                    "required": ["photo_id"],
                },
            },
        },
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest integrations/photos/test_integration.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add integrations/photos/tools.py integrations/photos/test_integration.py
git commit -m "feat: add Photos tool definitions"
```

---

### Task 4: Create integration.py (main entry point)

**Files:**
- Create: `integrations/photos/integration.py`
- Create: `integrations/photos/__init__.py`
- Modify: `integrations/photos/test_integration.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to integrations/photos/test_integration.py

from integrations.base import IntegrationConfig
from integrations.photos.integration import PhotosIntegration


class TestPhotosIntegration:
    def test_get_tools(self):
        """Integration returns tools."""
        config = IntegrationConfig(name="photos", config={"db_path": ":memory:"})
        integration = PhotosIntegration(config)
        tools = integration.get_tools()
        assert len(tools) == 3
        assert tools[0]["function"]["name"] == "photos_search"

    def test_connect_without_library(self):
        """Connect raises if library_path not configured."""
        config = IntegrationConfig(name="photos", config={})
        integration = PhotosIntegration(config)
        with pytest.raises(ValueError, match="library_path"):
            import asyncio
            asyncio.run(integration.connect())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest integrations/photos/test_integration.py::TestPhotosIntegration -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement integration**

```python
# integrations/photos/integration.py
"""Apple Photos integration - search, browse, and analyze photos."""

from __future__ import annotations

import logging
from typing import Any

from integrations.base import Integration, IntegrationConfig
from integrations.photos.indexer import PhotosIndexer
from integrations.photos.vlm_analyzer import VLMAnalyzer
from integrations.photos.tools import get_photos_tools

logger = logging.getLogger(__name__)


class PhotosIntegration(Integration):
    """Apple Photos integration with metadata indexing and VLM analysis."""

    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self._indexer: PhotosIndexer | None = None
        self._vlm: VLMAnalyzer | None = None

    async def connect(self) -> None:
        """Initialize indexer and VLM analyzer."""
        library_path = self.config.config.get("library_path")
        if not library_path:
            raise ValueError("photos library_path not configured")

        db_path = self.config.config.get("db_path", "memory/photos.db")
        self._indexer = PhotosIndexer(library_path, db_path)
        self._indexer.init_db()

        vlm_model = self.config.config.get("vlm_model")
        if vlm_model:
            self._vlm = VLMAnalyzer(vlm_model, db_path)
            self._vlm.init_db()

        self._connected = True
        logger.info("Photos integration connected: %s", library_path)

    async def disconnect(self) -> None:
        """Cleanup."""
        if self._indexer:
            self._indexer.close()
        if self._vlm:
            self._vlm.close()
        self._indexer = None
        self._vlm = None
        self._connected = False

    async def sync(self) -> dict[str, int]:
        """Incremental sync metadata from Photos.app."""
        if not self._indexer:
            raise RuntimeError("Not connected")
        return self._indexer.sync_sync()

    async def query(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search photos by metadata."""
        if not self._indexer:
            raise RuntimeError("Not connected")
        return self._indexer.search_sync(query, limit=limit)

    def get_tools(self) -> list[dict[str, Any]]:
        """Return tool definitions for the agent."""
        return get_photos_tools()

    async def handle_search(self, **kwargs) -> list[dict[str, Any]]:
        """Handle photos_search tool call."""
        if not self._indexer:
            return [{"error": "Photos integration not connected"}]
        return self._indexer.search_sync(
            query=kwargs.get("query", ""),
            date_from=kwargs.get("date_from"),
            date_to=kwargs.get("date_to"),
            album=kwargs.get("album"),
            limit=kwargs.get("limit", 10),
        )

    async def handle_list_albums(self, **kwargs) -> list[str]:
        """Handle photos_list_albums tool call."""
        if not self._indexer:
            return []
        return self._indexer.list_albums_sync()

    async def handle_describe(self, **kwargs) -> str:
        """Handle photos_describe tool call."""
        photo_id = kwargs.get("photo_id")
        question = kwargs.get("question")

        if not photo_id:
            return "Error: photo_id is required"

        if not self._vlm:
            return "Error: VLM model not configured"

        # Check cache first
        cached = self._vlm._get_cached(photo_id, question or "")
        if cached:
            return cached

        # Get photo path
        photo_path = self._indexer.get_photo_path(photo_id)
        if not photo_path:
            return f"Error: Photo {photo_id} not found"

        # Analyze with VLM
        result = await self._vlm.analyze(photo_path, question)

        # Cache result
        self._vlm._cache_result(photo_id, question or "", result)

        return result
```

- [ ] **Step 4: Create __init__.py**

```python
# integrations/photos/__init__.py
"""Apple Photos integration."""

from integrations.photos.integration import PhotosIntegration

__all__ = ["PhotosIntegration"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest integrations/photos/test_integration.py -v`
Expected: All 14 tests PASS

- [ ] **Step 6: Commit**

```bash
git add integrations/photos/integration.py integrations/photos/__init__.py integrations/photos/test_integration.py
git commit -m "feat: add PhotosIntegration main entry point"
```

---

### Task 5: Update config.yaml

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Add photos configuration**

Add to the `integrations` section of `config.yaml`:

```yaml
  # Apple Photos
  photos:
    enabled: true
    library_path: ~/Pictures/Photos Library.photoslibrary
    db_path: memory/photos.db
    vlm_model: minicpm-v-4.6
    cache_analyses: true
    max_results: 20
```

- [ ] **Step 2: Commit**

```bash
git add config.yaml
git commit -m "feat: add Apple Photos configuration to config.yaml"
```

---

### Task 6: Final verification

- [ ] **Step 1: Run all Python tests**

Run: `python -m pytest tests/python/ integrations/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run JS tests**

Run: `npm test`
Expected: All tests PASS

- [ ] **Step 3: Verify imports**

Run: `python -c "from integrations.photos import PhotosIntegration; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit if needed**

```bash
git add -A
git commit -m "chore: verify Apple Photos integration"
```
